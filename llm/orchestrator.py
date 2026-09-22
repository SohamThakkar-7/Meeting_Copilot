

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .prompt_builder import build_prompt
from .providers.base import LLMProvider, ProviderError

TRIGGER_TURN = "turn"
TRIGGER_HOTKEY = "hotkey"


@dataclass
class Suggestion:
    """The outcome of one request, delivered to on_done whether it succeeded,
    was superseded, or failed."""

    text: str
    reason: str
    provider: str
    ttft_ms: Optional[float]  # None if it was cancelled or failed before any token
    total_ms: float
    cancelled: bool = False
    error: Optional[str] = None


class LLMOrchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        context,
        debounce_ms: float = 500.0,
        # Gemini's free tier allows 15 requests/minute -- one every 4000ms.
        # 5000 leaves headroom for hotkey presses, which bypass this floor
        # entirely and would otherwise eat into the same quota. Raise the
        # rate here, not the limit there: on a paid tier drop this to ~1500.
        min_interval_ms: float = 5000.0,
        max_tokens: int = 300,
        trigger_sources: Optional[tuple] = None,
        on_start: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[Suggestion], None]] = None,
    ):
        """
        Parameters
        ----------
        provider : LLMProvider
        context : SessionContextManager
            Read at fire time, not at trigger time -- the extra debounce
            milliseconds often buy another few words of transcript.
        trigger_sources : tuple, optional
            Which sources' turns arm the timer, e.g. ("system",) to only
            suggest after the other person speaks. None means all sources.
        """
        self._provider = provider
        self._context = context
        self._debounce = debounce_ms / 1000.0
        self._min_interval = min_interval_ms / 1000.0
        self._max_tokens = max_tokens
        self._trigger_sources = trigger_sources

        self._on_start = on_start
        self._on_token = on_token
        self._on_done = on_done

        self._cv = threading.Condition()
        self._deadline: Optional[float] = None
        self._reason: Optional[str] = None
        self._last_fired = float("-inf")

        self._stop = threading.Event()
        self._scheduler: Optional[threading.Thread] = None

        self._inflight_lock = threading.Lock()
        self._inflight_cancel: Optional[threading.Event] = None
        self._inflight_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._scheduler = threading.Thread(
            target=self._run, daemon=True, name="llm-scheduler"
        )
        self._scheduler.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self._cancel_inflight()
        if self._scheduler is not None:
            self._scheduler.join(timeout=2.0)
            self._scheduler = None
        with self._inflight_lock:
            thread = self._inflight_thread
        if thread is not None:
            thread.join(timeout=2.0)

    def notify_turn(self, turn) -> None:
        """Wire this to SessionContextManager's on_turn_complete. Arms (or
        re-arms) the debounce timer."""
        if self._trigger_sources and turn.source not in self._trigger_sources:
            return
        with self._cv:
            self._deadline = time.perf_counter() + self._debounce
            self._reason = TRIGGER_TURN
            self._cv.notify_all()

    def trigger_now(self, reason: str = TRIGGER_HOTKEY) -> None:
        """Fire immediately, skipping debounce and the rate-limit floor."""
        with self._cv:
            self._deadline = time.perf_counter()
            self._reason = reason
            self._cv.notify_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            reason = self._wait_for_trigger()
            if reason is None:
                continue
            self._fire(reason)

    def _wait_for_trigger(self) -> Optional[str]:
        with self._cv:
            while not self._stop.is_set():
                if self._deadline is None:
                    self._cv.wait(0.25)
                    continue

                remaining = self._deadline - time.perf_counter()
                if remaining > 0:
                    self._cv.wait(remaining)
                    continue

                reason = self._reason
                if reason != TRIGGER_HOTKEY:
                    # Rate-limit floor. Push the timer out rather than dropping
                    # the trigger -- the conversation still deserves an answer,
                    # just not this instant, and by then the transcript is
                    # fuller anyway.
                    earliest = self._last_fired + self._min_interval
                    if time.perf_counter() < earliest:
                        self._deadline = earliest
                        continue

                self._deadline = None
                self._reason = None
                self._last_fired = time.perf_counter()
                return reason
        return None

    def _fire(self, reason: str) -> None:
        self._cancel_inflight()

        # Snapshot the context here, on the scheduler thread, so the prompt
        # reflects the moment we decided to fire.
        system, user = build_prompt(self._context.get_context_window())

        cancel = threading.Event()
        thread = threading.Thread(
            target=self._request,
            args=(reason, system, user, cancel),
            daemon=True,
            name="llm-request",
        )
        with self._inflight_lock:
            self._inflight_cancel = cancel
            self._inflight_thread = thread
        thread.start()

    def _cancel_inflight(self) -> None:
        with self._inflight_lock:
            cancel = self._inflight_cancel
        if cancel is not None:
            cancel.set()

    def _request(
        self, reason: str, system: str, user: str, cancel: threading.Event
    ) -> None:
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        chunks: list[str] = []
        error: Optional[str] = None

        self._emit(self._on_start, reason)

        try:
            for chunk in self._provider.stream(system, user, self._max_tokens, cancel):
                if cancel.is_set():
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                chunks.append(chunk)
                self._emit(self._on_token, chunk)
        except ProviderError as exc:
            error = str(exc)
        except Exception as exc:  # a provider bug must not kill the pipeline
            error = f"{type(exc).__name__}: {exc}"

        self._emit(
            self._on_done,
            Suggestion(
                text="".join(chunks).strip(),
                reason=reason,
                provider=self._provider.name,
                ttft_ms=(first_token_at - started) * 1000.0 if first_token_at else None,
                total_ms=(time.perf_counter() - started) * 1000.0,
                cancelled=cancel.is_set(),
                error=error,
            ),
        )

    @staticmethod
    def _emit(callback, payload) -> None:
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:
            # A misbehaving consumer (or an overlay that just went away) must
            # never take down a request thread.
            pass
