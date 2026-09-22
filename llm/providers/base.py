import threading
from typing import Iterator, Protocol


class ProviderError(RuntimeError):
    """Anything that went wrong talking to the model. The orchestrator catches
    this and surfaces it as a failed suggestion rather than letting it kill the
    request thread silently."""


class LLMProvider(Protocol):
    #: Short identifier, printed in logs so you can tell which backend ran.
    name: str

    #: Floor between requests this backend's rate limit can sustain. It's a
    #: property of the provider, not of the scheduling policy -- Gemini's free
    #: tier allows 15/min, Groq's is far more generous -- so the orchestrator
    #: takes its default from here rather than hardcoding one number that is
    #: wrong for every backend but one.
    suggested_min_interval_ms: float

    def stream(
        self,
        system: str,
        user: str,
        max_tokens: int,
        cancel: threading.Event,
    ) -> Iterator[str]:
        """Yield response text incrementally.

        `system` is the static, cacheable persona -- byte-identical on every
        call. `user` is the volatile part (the rolling transcript). Providers
        that support prompt caching should mark the boundary between them.

        `cancel` is set when the suggestion has been superseded by a newer
        turn. Check it between chunks and stop yielding; the caller stops
        consuming either way, but checking lets you close the socket early.

        Raise ProviderError on failure.
        """
        ...
