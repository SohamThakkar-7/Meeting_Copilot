# active window tracker -- which application the user is actually looking at right now

import time

try:
    import win32gui

    _HAS_WIN32 = True
except ImportError:  # pywin32 not installed, or non-Windows
    _HAS_WIN32 = False


class ActiveWindowTracker:
    def __init__(self, ttl_seconds: float = 1.0):
        """
        Parameters
        ----------
        ttl_seconds : float
            How long a looked-up title stays fresh. Anything under a
            second or so is wasted work -- nobody alt-tabs that fast.
        """
        self._ttl = ttl_seconds
        self._cached = ""
        self._checked_at = float("-inf")

    @property
    def available(self) -> bool:
        """False when pywin32 is missing -- callers can decide whether to
        mention window context in the prompt at all."""
        return _HAS_WIN32

    def get_title(self) -> str:
        """Foreground window title, or "" if unavailable."""
        if not _HAS_WIN32:
            return ""

        now = time.perf_counter()
        if now - self._checked_at < self._ttl:
            return self._cached

        self._checked_at = now
        try:
            hwnd = win32gui.GetForegroundWindow()
            self._cached = win32gui.GetWindowText(hwnd) if hwnd else ""
        except Exception:
            self._cached = ""
        return self._cached
