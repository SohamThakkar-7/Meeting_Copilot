"""Converts the raw per-source turn events into one coherent, time-ordered,
token-budgeted conversation transcript."""

from .session_manager import SessionContextManager, Turn
from .active_window import ActiveWindowTracker

__all__ = ["SessionContextManager", "Turn", "ActiveWindowTracker"]
