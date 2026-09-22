

import itertools
import json
import queue
import threading
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from context.session_manager import SessionContextManager, Turn
from llm.orchestrator import TRIGGER_HOTKEY, LLMOrchestrator, Suggestion
from llm.providers import ProviderError, build_provider

load_dotenv()

TRIGGER_SOURCES = {"both": None, "system": ("system",), "mic": ("mic",)}


class SessionConfig(BaseModel):
    """The CLI flags main.py used to apply locally, sent over the wire instead."""

    provider: str = "mock"
    model: Optional[str] = None
    max_tokens: int = 300
    debounce_ms: float = 500.0
    min_interval_ms: Optional[float] = None
    max_context_tokens: int = 1500
    trigger_on: str = "both"


class TurnEvent(BaseModel):
    source: str
    event: dict


class FlushRequest(BaseModel):
    source: str


class _Broadcaster:
    """Fan-out to every connected overlay.

    publish() runs on the orchestrator's request thread, which must never
    block on a slow reader -- a stalled HTTP client would otherwise apply
    backpressure all the way into the token loop. A full queue drops instead.
    """

    def __init__(self, maxsize: int = 512):
        self._lock = threading.Lock()
        self._subscribers: set = set()
        self._maxsize = maxsize

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def publish(self, payload: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


class Brain:
    """Owns the session context and the orchestrator for one live session."""

    def __init__(self):
        self._lock = threading.RLock()
        self.events = _Broadcaster()
        self.session: Optional[SessionContextManager] = None
        self.orchestrator: Optional[LLMOrchestrator] = None
        self.provider = None
        self.config: Optional[SessionConfig] = None
        self._generations = itertools.count(1)
        # Same problem main.py had: the three callbacks for one request share
        # a thread, but a superseded request can still be draining on another.
        self._request_state = threading.local()

    # --- lifecycle ---------------------------------------------------------

    def configure(self, config: SessionConfig) -> dict:
        if config.trigger_on not in TRIGGER_SOURCES:
            raise HTTPException(
                status_code=400, detail=f"unknown trigger_on {config.trigger_on!r}"
            )

        with self._lock:
            self._shutdown_locked()

            provider_kwargs = {}
            if config.model and config.provider != "mock":
                provider_kwargs["model"] = config.model
            try:
                provider = build_provider(config.provider, **provider_kwargs)
            except (ProviderError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            session = SessionContextManager(
                max_tokens=config.max_context_tokens,
                on_turn_complete=self._on_turn_complete,
            )

            # The rate floor belongs to the backend, not the scheduling policy.
            min_interval_ms = config.min_interval_ms
            if min_interval_ms is None:
                min_interval_ms = getattr(provider, "suggested_min_interval_ms", 2000.0)

            orchestrator = LLMOrchestrator(
                provider,
                session,
                debounce_ms=config.debounce_ms,
                min_interval_ms=min_interval_ms,
                max_tokens=config.max_tokens,
                trigger_sources=TRIGGER_SOURCES[config.trigger_on],
                on_start=self._on_start,
                on_token=self._on_token,
                on_done=self._on_done,
            )
            orchestrator.start()

            self.provider = provider
            self.session = session
            self.orchestrator = orchestrator
            self.config = config

            return {
                "provider": provider.name,
                "min_interval_ms": min_interval_ms,
                "debounce_ms": config.debounce_ms,
            }

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown_locked()

    def _shutdown_locked(self) -> None:
        if self.orchestrator is not None:
            self.orchestrator.stop()
            self.orchestrator = None
        if self.provider is not None and hasattr(self.provider, "close"):
            self.provider.close()
        self.provider = None
        self.session = None
        self.config = None

    def require_session(self):
        session, orchestrator = self.session, self.orchestrator
        if session is None or orchestrator is None:
            raise HTTPException(status_code=409, detail="no session; POST /session first")
        return session, orchestrator

    # --- orchestrator callbacks -> SSE ------------------------------------

    def _on_turn_complete(self, turn: Turn) -> None:
        self.events.publish({"type": "turn", "source": turn.source, "text": turn.text})
        if self.orchestrator is not None:
            self.orchestrator.notify_turn(turn)

    def _on_start(self, reason: str) -> None:
        generation = next(self._generations)
        self._request_state.generation = generation
        self._request_state.reason = reason
        self.events.publish({"type": "start", "generation": generation, "reason": reason})

    def _on_token(self, text: str) -> None:
        generation = getattr(self._request_state, "generation", None)
        if generation is None:
            return
        self.events.publish({"type": "token", "generation": generation, "text": text})

    def _on_done(self, suggestion: Suggestion) -> None:
        generation = getattr(self._request_state, "generation", None)
        if generation is None:
            return
        # Every field of Suggestion travels, so the host can rebuild the exact
        # dataclass its overlay callback already expects.
        self.events.publish(
            {
                "type": "done",
                "generation": generation,
                "text": suggestion.text,
                "reason": suggestion.reason,
                "provider": suggestion.provider,
                "ttft_ms": suggestion.ttft_ms,
                "total_ms": suggestion.total_ms,
                "cancelled": suggestion.cancelled,
                "error": suggestion.error,
            }
        )


brain = Brain()
app = FastAPI(title="Meeting_Copilot brain", version="1.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "session": brain.config is not None,
        "provider": brain.provider.name if brain.provider else None,
    }


@app.post("/session")
def open_session(config: SessionConfig) -> dict:
    return brain.configure(config)


@app.delete("/session")
def close_session() -> dict:
    brain.shutdown()
    return {"status": "closed"}


@app.post("/events")
def post_event(payload: TurnEvent) -> dict:
    session, _ = brain.require_session()
    session.handle_turn_event(payload.source, payload.event)
    return {"ok": True}


@app.post("/flush")
def post_flush(payload: FlushRequest) -> dict:
    """The host lost its STT connection; finalize whatever was mid-turn."""
    session, _ = brain.require_session()
    session.flush_pending(payload.source)
    return {"ok": True}


@app.get("/context")
def get_context() -> dict:
    """The assembled transcript, as the prompt builder would see it."""
    session, _ = brain.require_session()
    return {"context": session.get_context_window()}


@app.post("/trigger")
def post_trigger() -> dict:
    """Ctrl+Alt+J on the host. Bypasses debounce and the rate floor."""
    _, orchestrator = brain.require_session()
    orchestrator.trigger_now(TRIGGER_HOTKEY)
    return {"ok": True}


@app.get("/stream")
def stream() -> StreamingResponse:
    q = brain.events.subscribe()

    # A sync generator, so FastAPI runs it on a worker thread and the blocking
    # queue.get() below never stalls the event loop.
    def generate():
        try:
            yield ": connected\n\n"
            while True:
                try:
                    payload = q.get(timeout=15.0)
                except queue.Empty:
                    # Proxies and idle NAT mappings drop a silent connection.
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            brain.events.unsubscribe(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
