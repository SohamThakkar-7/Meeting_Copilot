

import json
import queue
import threading
from typing import Callable, Optional

import httpx

from llm.orchestrator import Suggestion

_STOP = object()


class BrainUnavailable(RuntimeError):
    """The service could not be reached or refused the session."""


class RemoteBrain:
    def __init__(
        self,
        base_url: str,
        config: dict,
        on_turn: Optional[Callable[[str, str], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[Suggestion], None]] = None,
        queue_size: int = 256,
        timeout: float = 5.0,
    ):
        self._base = base_url.rstrip("/")
        self._config = config
        self._on_turn = on_turn
        self._on_start = on_start
        self._on_token = on_token
        self._on_done = on_done

        self._client = httpx.Client(timeout=timeout)
        # Separate client for the stream: it is a long-lived read with no
        # timeout, and sharing one would let a slow POST sit behind it.
        self._stream_client = httpx.Client(timeout=httpx.Timeout(None, connect=timeout))

        self._outbox: queue.Queue = queue.Queue(maxsize=queue_size)
        self._sender: Optional[threading.Thread] = None
        self._listener: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._dropped = 0
        self._final_context: Optional[str] = None

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> dict:
        try:
            response = self._client.post(f"{self._base}/session", json=self._config)
        except httpx.HTTPError as exc:
            raise BrainUnavailable(f"cannot reach brain at {self._base}: {exc}") from exc
        if response.status_code >= 400:
            raise BrainUnavailable(f"brain refused the session: {response.text}")
        info = response.json()

        self._stop.clear()
        self._sender = threading.Thread(target=self._send_loop, daemon=True)
        self._listener = threading.Thread(target=self._listen_loop, daemon=True)
        self._sender.start()
        self._listener.start()
        return info

    def stop(self) -> None:
        self._stop.set()
        try:
            self._outbox.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._sender is not None:
            self._sender.join(timeout=2.0)

        # main.py prints the final transcript after shutting the brain down,
        # and DELETE /session discards it. Grab it while it still exists.
        self._final_context = self._fetch_context()

        try:
            self._client.delete(f"{self._base}/session")
        except httpx.HTTPError:
            pass
        self._client.close()
        self._stream_client.close()

    @property
    def dropped_events(self) -> int:
        return self._dropped

    # --- outbound ----------------------------------------------------------

    def handle_turn_event(self, source_label: str, event: dict) -> None:
        self._enqueue("/events", {"source": source_label, "event": event})

    def flush_pending(self, source_label: str) -> None:
        self._enqueue("/flush", {"source": source_label})

    def trigger_now(self) -> None:
        # The hotkey is a direct user action and is rare, so it jumps the
        # queue -- waiting behind a backlog of transcript events would make
        # the one explicitly requested suggestion the slowest one.
        try:
            self._client.post(f"{self._base}/trigger")
        except httpx.HTTPError as exc:
            print(f"[brain] trigger failed: {exc}")

    def get_context_window(self) -> str:
        if self._final_context is not None:
            return self._final_context
        return self._fetch_context()

    def _fetch_context(self) -> str:
        try:
            response = self._client.get(f"{self._base}/context")
            response.raise_for_status()
            return response.json().get("context", "")
        except httpx.HTTPError as exc:
            return f"[context unavailable: {exc}]"

    def _enqueue(self, path: str, payload: dict) -> None:
        try:
            self._outbox.put_nowait((path, payload))
        except queue.Full:
            # Never block the audio/STT thread. Drop the oldest instead, so a
            # stalled link costs us history rather than live capture.
            self._dropped += 1
            try:
                self._outbox.get_nowait()
                self._outbox.put_nowait((path, payload))
            except (queue.Empty, queue.Full):
                pass

    def _send_loop(self) -> None:
        while True:
            item = self._outbox.get()
            if item is _STOP:
                return
            path, payload = item
            try:
                self._client.post(f"{self._base}{path}", json=payload)
            except httpx.HTTPError as exc:
                print(f"[brain] POST {path} failed: {exc}")
            if self._stop.is_set() and self._outbox.empty():
                return

    # --- inbound -----------------------------------------------------------

    def _listen_loop(self) -> None:
        while not self._stop.is_set():
            try:
                with self._stream_client.stream("GET", f"{self._base}/stream") as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if self._stop.is_set():
                            return
                        if not line or line.startswith(":"):
                            continue  # keepalive
                        if not line.startswith("data: "):
                            continue
                        try:
                            self._dispatch(json.loads(line[6:]))
                        except json.JSONDecodeError:
                            continue
            except httpx.HTTPError as exc:
                if self._stop.is_set():
                    return
                # The overlay stays up across a brain restart; reconnecting
                # beats tearing down a live meeting over one dropped socket.
                print(f"[brain] stream dropped ({exc}); reconnecting")
                if self._stop.wait(1.0):
                    return

    def _dispatch(self, payload: dict) -> None:
        kind = payload.get("type")

        if kind == "turn":
            if self._on_turn:
                self._on_turn(payload.get("source", "?"), payload.get("text", ""))
            return

        if kind == "start":
            if self._on_start:
                self._on_start(payload.get("reason", "turn"))
            return

        if kind == "token":
            if self._on_token:
                self._on_token(payload.get("text", ""))
            return

        if kind == "done":
            if self._on_done:
                self._on_done(
                    Suggestion(
                        text=payload.get("text", ""),
                        reason=payload.get("reason", "turn"),
                        provider=payload.get("provider", "remote"),
                        ttft_ms=payload.get("ttft_ms"),
                        total_ms=payload.get("total_ms", 0.0),
                        cancelled=payload.get("cancelled", False),
                        error=payload.get("error"),
                    )
                )
