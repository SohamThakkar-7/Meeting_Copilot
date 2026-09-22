
import json
import os
import threading
from typing import Iterator, Optional

import httpx

from .base import ProviderError

_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Chosen by measurement, not by reputation. Over 5 runs each on a real
# transcript: 3.5-flash-lite at minimal thinking held a median 812ms TTFT
# (704-844ms), plain 3.5-flash-lite 890ms, and full 3.5-flash 5-19s -- the
# bigger model spends its time thinking, which is exactly the wrong trade for
# a suggestion that has to land while someone is still drawing breath.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Gemini 3.x reasons before answering unless told not to. Left on, it costs
# seconds. Note the config shape is model-dependent: 3.x takes thinkingLevel,
# older models took thinkingBudget and 400 on this key -- so it's optional.
DEFAULT_THINKING_LEVEL = "minimal"


class GeminiProvider:
    name = "gemini"
    suggested_min_interval_ms = 4500.0   # free tier is 15 requests/minute

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 30.0,
        thinking_level: Optional[str] = DEFAULT_THINKING_LEVEL,
    ):
        self._model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        self._key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv(
            "GOOGLE_API_KEY"
        )
        if not self._key:
            raise ProviderError(
                "No Gemini API key. Set GEMINI_API_KEY (get one free at "
                "https://aistudio.google.com/apikey)."
            )
        self._temperature = temperature
        self._thinking_level = thinking_level
        self.name = f"gemini:{self._model}"

        # One connection, reused across suggestions -- see module docstring.
        self._client = httpx.Client(timeout=timeout)

    def stream(
        self,
        system: str,
        user: str,
        max_tokens: int,
        cancel: threading.Event,
    ) -> Iterator[str]:
        url = f"{_BASE}/models/{self._model}:streamGenerateContent"
        generation_config = {
            "maxOutputTokens": max_tokens,
            "temperature": self._temperature,
        }
        if self._thinking_level:
            generation_config["thinkingConfig"] = {
                "thinkingLevel": self._thinking_level
            }
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }

        try:
            with self._client.stream(
                "POST",
                url,
                params={"alt": "sse", "key": self._key},
                json=body,
            ) as response:
                if response.status_code == 429:
                    # The raw body is a 400-character JSON blob that would be
                    # rendered verbatim into a 420px overlay. Say the useful
                    # part instead.
                    raise ProviderError(
                        "rate limited (free tier is 15 requests/minute) -- "
                        "raise min_interval_ms or use a paid key"
                    )
                if response.status_code != 200:
                    detail = self._scrub(response.read().decode("utf-8", "replace"))
                    raise ProviderError(
                        f"Gemini HTTP {response.status_code}: {detail[:200]}"
                    )

                for line in response.iter_lines():
                    if cancel.is_set():
                        return
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    for text in self._texts(payload):
                        yield text
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini request failed: {self._scrub(str(exc))}")

    def _texts(self, payload: str) -> list[str]:
        """Pull the text out of one SSE chunk.

        Chunks are not uniform: some carry only a finishReason or safety
        metadata and no parts at all, so every level here is optional. A
        chunk we can't read is skipped rather than raised -- one odd frame
        shouldn't abort a suggestion that's already half on screen.
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return []

        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        if blocked:
            raise ProviderError(f"Gemini blocked the prompt: {blocked}")

        out = []
        for candidate in data.get("candidates") or []:
            for part in ((candidate.get("content") or {}).get("parts")) or []:
                text = part.get("text")
                if text:
                    out.append(text)
        return out

    def list_models(self) -> list[str]:
        """Model ids this key can actually use. Free-tier availability moves
        around, so check here rather than trusting a hardcoded default."""
        try:
            response = self._client.get(f"{_BASE}/models", params={"key": self._key})
            if response.status_code != 200:
                raise ProviderError(
                    f"Gemini HTTP {response.status_code}: "
                    f"{self._scrub(response.text)[:400]}"
                )
            data = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini request failed: {self._scrub(str(exc))}")

        names = []
        for model in data.get("models") or []:
            methods = model.get("supportedGenerationMethods") or []
            if "generateContent" in methods or not methods:
                names.append((model.get("name") or "").removeprefix("models/"))
        return [name for name in names if name]

    def close(self) -> None:
        self._client.close()

    def _scrub(self, text: str) -> str:
        """The key travels in the query string, so it can surface in error text
        and exception reprs. Never let it reach a log."""
        return text.replace(self._key, "<key>") if self._key else text
