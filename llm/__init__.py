"""
llm
===

When to ask the model (orchestrator), what to ask it (prompt_builder), and
which model to ask (providers). Nothing in here imports a vendor SDK above the
providers package -- swapping backends is a one-file change.
"""

from .orchestrator import LLMOrchestrator, Suggestion, TRIGGER_HOTKEY, TRIGGER_TURN
from .prompt_builder import NOTHING_TO_SAY, SYSTEM_PROMPT, build_prompt
from .providers import build_provider

__all__ = [
    "LLMOrchestrator",
    "Suggestion",
    "TRIGGER_HOTKEY",
    "TRIGGER_TURN",
    "NOTHING_TO_SAY",
    "SYSTEM_PROMPT",
    "build_prompt",
    "build_provider",
]
