

import threading
import time
from typing import Iterator


class MockProvider:
    name = "mock"
    suggested_min_interval_ms = 0.0       # nothing to rate limit

    def __init__(self, ttft_ms: float = 250.0, token_ms: float = 20.0):
        """
        Parameters
        ----------
        ttft_ms : float
            Simulated time to first token. Set this near what you expect from
            the real model so the overlay's pacing gets tested honestly.
        token_ms : float
            Simulated delay between tokens.
        """
        self._ttft = ttft_ms / 1000.0
        self._token_delay = token_ms / 1000.0

    def stream(
        self,
        system: str,
        user: str,
        max_tokens: int,
        cancel: threading.Event,
    ) -> Iterator[str]:
        if cancel.wait(self._ttft):
            return

        last = _last_spoken_line(user)
        reply = f"[mock] heard: {last}" if last else "[mock] no transcript in context"

        for word in reply.split(" "):
            if cancel.is_set():
                return
            yield word + " "
            if cancel.wait(self._token_delay):
                return


def _last_spoken_line(user: str) -> str:
    """The most recent You:/Them: line, ignoring the bracketed system context
    and the instruction wrapper around the transcript."""
    for line in reversed(user.splitlines()):
        line = line.strip()
        if line.startswith(("You:", "Them:", "You (", "Them (")):
            return line
    return ""
