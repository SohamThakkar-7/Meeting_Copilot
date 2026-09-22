from .base import LLMProvider, ProviderError

__all__ = ["LLMProvider", "ProviderError", "build_provider"]


def build_provider(name: str, **kwargs) -> LLMProvider:
    if name == "mock":
        from .mock import MockProvider

        return MockProvider(**kwargs)
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(**kwargs)
    if name == "groq":
        from .groq import GroqProvider

        return GroqProvider(**kwargs)
    raise ValueError(f"Unknown provider {name!r}. Known: mock, gemini, groq.")
