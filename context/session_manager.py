import bisect
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

# Uses perf_counter(), not monotonic(): monotonic() has 15.6ms resolution on
# Windows, so turns starting in the same tick get equal timestamps and
# bisect.insort (insort_right) puts them in arrival order instead.


DEFAULT_SPEAKER_LABELS = {"mic": "You", "system": "Them"}


CHARS_PER_TOKEN = 4


_LINE_OVERHEAD_TOKENS = 4


@dataclass
class Turn:
    """One completed utterance by one speaker."""

    source: str
    speaker: str
    text: str
    started_at: float
    ended_at: float

    @property
    def approx_tokens(self) -> int:
        return len(self.text) // CHARS_PER_TOKEN + _LINE_OVERHEAD_TOKENS

    def render(self) -> str:
        return f"{self.speaker}: {self.text}"


@dataclass
class _PendingTurn:
    """A turn still in progress. Flux resends the whole turn transcript with
    every event, so `text` is replaced on each update, not appended to."""

    started_at: float
    text: str = ""
    eager: bool = False  # EagerEndOfTurn seen; may still be TurnResumed


class SessionContextManager:
    def __init__(
        self,
        max_tokens: int = 1500,
        speaker_labels: Optional[dict] = None,
        on_turn_complete: Optional[Callable[[Turn], None]] = None,
        track_active_window: bool = True,
    ):
        self._max_tokens = max_tokens
        self._labels = dict(speaker_labels or DEFAULT_SPEAKER_LABELS)
        self._on_turn_complete = on_turn_complete

        self._lock = threading.Lock()
        self._turns: list[Turn] = []
        self._pending: dict[str, _PendingTurn] = {}
        self._dropped_turns = 0

        self._window = None
        if track_active_window:
            from .active_window import ActiveWindowTracker

            self._window = ActiveWindowTracker()

    def handle_turn_event(self, source_label: str, event: dict) -> None:
        kind = event.get("event")
        transcript = (event.get("transcript") or "").strip()
        now = time.perf_counter()

        completed: Optional[Turn] = None

        with self._lock:
            if kind == "StartOfTurn":
                # A fresh StartOfTurn supersedes any pending turn -- if the
                # previous one never got an EndOfTurn we lost it to a
                # reconnect, and its words are already stale.
                self._pending[source_label] = _PendingTurn(started_at=now)
                return

            pending = self._pending.get(source_label)
            if pending is None:
                # Update/EndOfTurn with no StartOfTurn: we joined mid-turn,
                # almost always just after a reconnect. Open one lazily
                # rather than dropping the words.
                pending = _PendingTurn(started_at=now)
                self._pending[source_label] = pending

            if kind in ("Update", "EagerEndOfTurn"):
                if transcript:
                    pending.text = transcript
                pending.eager = kind == "EagerEndOfTurn"
                return

            if kind == "TurnResumed":
                # The eager end-of-turn guess was wrong; they kept talking.
                pending.eager = False
                return

            if kind == "EndOfTurn":
                if transcript:
                    pending.text = transcript
                self._pending.pop(source_label, None)
                completed = self._finalize(source_label, pending, now)

        if completed is not None and self._on_turn_complete is not None:
            self._on_turn_complete(completed)

    def flush_pending(self, source_label: str) -> None:
        """Finalize whatever a source had mid-turn. Called when that source's
        STT connection drops, so a half-spoken turn lands in the transcript
        instead of dangling."""
        with self._lock:
            pending = self._pending.pop(source_label, None)
            if pending is None:
                return
            completed = self._finalize(source_label, pending, time.perf_counter())

        if completed is not None and self._on_turn_complete is not None:
            self._on_turn_complete(completed)

    def _finalize(
        self, source_label: str, pending: _PendingTurn, now: float
    ) -> Optional[Turn]:
        """Caller must hold the lock."""
        if not pending.text:
            return None

        turn = Turn(
            source=source_label,
            speaker=self._labels.get(source_label, source_label),
            text=pending.text,
            started_at=pending.started_at,
            ended_at=now,
        )
        # Insert by start time, not arrival time -- overlapping speech on two
        # connections can otherwise land out of order.
        bisect.insort(self._turns, turn, key=lambda t: t.started_at)
        self._trim()
        return turn

    def _trim(self) -> None:
        """Caller must hold the lock."""
        total = sum(turn.approx_tokens for turn in self._turns)
        while self._turns and total > self._max_tokens:
            total -= self._turns.pop(0).approx_tokens
            self._dropped_turns += 1

    def get_context_window(
        self, include_partial: bool = True, include_window: bool = True
    ) -> str:
        """The LLM-ready transcript, optionally including still-in-progress
        utterances and the active window title."""
        with self._lock:
            dropped = self._dropped_turns
            lines = [turn.render() for turn in self._turns]
            partials = sorted(
                (pending.started_at, source, pending.text)
                for source, pending in self._pending.items()
                if pending.text
            )

        header = []
        if include_window and self._window is not None:
            title = self._window.get_title()
            if title:
                header.append(f"[Active window: {title}]")

        body = []
        if dropped:
            body.append(f"[... {dropped} earlier turn(s) trimmed ...]")
        body.extend(lines)

        if include_partial:
            for _, source, text in partials:
                speaker = self._labels.get(source, source)
                body.append(f"{speaker} (still speaking): {text}")

        return "\n".join(header + body)

    @property
    def active_window_available(self) -> bool:
        """False when window tracking is off or pywin32 is missing."""
        return self._window is not None and self._window.available

    @property
    def turns(self) -> list[Turn]:
        with self._lock:
            return list(self._turns)

    @property
    def last_turn(self) -> Optional[Turn]:
        with self._lock:
            return self._turns[-1] if self._turns else None

    @property
    def dropped_turns(self) -> int:
        with self._lock:
            return self._dropped_turns

    def approx_tokens(self) -> int:
        with self._lock:
            return sum(turn.approx_tokens for turn in self._turns)

    def reset(self) -> None:
        """Start a new session -- new meeting, new conversation."""
        with self._lock:
            self._turns.clear()
            self._pending.clear()
            self._dropped_turns = 0
