

import json
import os
import threading
from typing import Iterator, Optional

import httpx

from .base import ProviderError

_BASE = "https://api.groq.com/openai/v1"


DEFAULT_MODEL = "qwen/qwen3.8-27b"


class GroqProvider:
    name = "groq"
    suggested_min_interval_ms = 1500.0   # token-based limits, far more headroom

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 30.0,
    ):
        self._model = model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
        self._key = api_key or os.getenv("GROQ_API_KEY")
        if not self._key:
            raise ProviderError(
                "No Groq API key. Set GROQ_API_KEY in .env "
                "(get one free at https://console.groq.com/keys)."
            )
        self._temperature = temperature
        self.name = f"groq:{self._model}"

        # Bearer auth, so unlike Gemini the key never enters a URL.
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {self._key}"},
        )

    def stream(
        self,
        system: str,
        user: str,
        max_tokens: int,
        cancel: threading.Event,
    ) -> Iterator[str]:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "stream": True,
        }

        try:
            with self._client.stream(
                "POST", f"{_BASE}/chat/completions", json=body
            ) as response:
                if response.status_code == 429:
                    raise ProviderError(
                        "rate limited by Groq -- raise min_interval_ms"
                    )
                if response.status_code != 200:
                    detail = response.read().decode("utf-8", "replace")
                    raise ProviderError(
                        f"Groq HTTP {response.status_code}: {detail[:200]}"
                    )

                for line in response.iter_lines():
                    if cancel.is_set():
                        return
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    text = self._delta(payload)
                    if text:
                        yield text
        except httpx.HTTPError as exc:
            raise ProviderError(f"Groq request failed: {exc}")

    def _delta(self, payload: str) -> str:
        """Text out of one SSE frame.

        Frames are not uniform -- the first carries a role and no content, the
        last carries a finish_reason and no content -- so every level is
        optional. An unreadable frame is skipped rather than raised: one odd
        frame shouldn't kill a suggestion that's already half on screen.
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return ""

        for choice in data.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if content:
                return content
        return ""

    def list_models(self) -> list[str]:
        """Model ids this key can use. Groq deprecates ids regularly, so check
        here rather than trusting the default."""
        try:
            response = self._client.get(f"{_BASE}/models")
            if response.status_code != 200:
                raise ProviderError(
                    f"Groq HTTP {response.status_code}: {response.text[:200]}"
                )
            data = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Groq request failed: {exc}")

        return sorted(
            model["id"] for model in (data.get("data") or []) if model.get("id")
        )

    def close(self) -> None:
        self._client.close()
