

from typing import Callable, Optional

from context.session_manager import SessionContextManager, Turn
from llm.orchestrator import LLMOrchestrator, Suggestion
from llm.providers import build_provider

TRIGGER_SOURCES = {"both": None, "system": ("system",), "mic": ("mic",)}


class LocalBrain:
    def __init__(
        self,
        config: dict,
        on_turn: Optional[Callable[[str, str], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[Suggestion], None]] = None,
    ):
        self._on_turn = on_turn

        provider_kwargs = {}
        if config.get("model") and config["provider"] != "mock":
            provider_kwargs["model"] = config["model"]
        self.provider = build_provider(config["provider"], **provider_kwargs)

        self.session = SessionContextManager(
            max_tokens=config["max_context_tokens"],
            on_turn_complete=self._on_turn_complete,
        )

        # The rate floor belongs to the backend, not the scheduling policy --
        # Gemini's free tier needs 4.5s between requests, Groq tolerates 1.5s.
        min_interval_ms = config.get("min_interval_ms")
        if min_interval_ms is None:
            min_interval_ms = getattr(self.provider, "suggested_min_interval_ms", 2000.0)
        self._min_interval_ms = min_interval_ms

        self.orchestrator = LLMOrchestrator(
            self.provider,
            self.session,
            debounce_ms=config["debounce_ms"],
            min_interval_ms=min_interval_ms,
            max_tokens=config["max_tokens"],
            trigger_sources=TRIGGER_SOURCES[config["trigger_on"]],
            on_start=on_start,
            on_token=on_token,
            on_done=on_done,
        )

    def start(self) -> dict:
        self.orchestrator.start()
        return {
            "provider": self.provider.name,
            "min_interval_ms": self._min_interval_ms,
        }

    def stop(self) -> None:
        self.orchestrator.stop()
        if hasattr(self.provider, "close"):
            self.provider.close()

    def handle_turn_event(self, source_label: str, event: dict) -> None:
        self.session.handle_turn_event(source_label, event)

    def flush_pending(self, source_label: str) -> None:
        self.session.flush_pending(source_label)

    def trigger_now(self) -> None:
        self.orchestrator.trigger_now()

    def _on_turn_complete(self, turn: Turn) -> None:
        if self._on_turn:
            self._on_turn(turn.source, turn.text)
        self.orchestrator.notify_turn(turn)

    def get_context_window(self) -> str:
        return self.session.get_context_window()
