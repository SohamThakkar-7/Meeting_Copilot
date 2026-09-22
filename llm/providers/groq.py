"""
Groq provider, over the OpenAI-compatible chat-completions endpoint.

Groq runs models on LPUs rather than GPUs, which for our purposes means one
thing: very low time-to-first-token. That plus a free tier measured in
thousands of requests per day -- rather than Gemini's 15 per minute -- makes
it a better fit for a copilot that fires on every turn of a live conversation.

Same shape as gemini.py deliberately: raw HTTP over the shared httpx.Client,
no vendor SDK, one connection reused across suggestions so a TLS handshake
never lands inside the latency budget.

The wire format is OpenAI's, so `messages` carries the system prompt as its
own entry rather than a separate field. That keeps the static/volatile split
intact -- system first, transcript second -- which is what any prefix cache
downstream will want.
"""

import json
import os
import threading
from typing import Iterator, Optional

import httpx

from .base import ProviderError

_BASE = "https://api.groq.com/openai/v1"

# Groq retires model ids fairly often. If this 404s, run with --list-models
# and pick a current one; the id is a constructor arg for exactly that reason.
#
# Chosen by measurement over 5 runs each: qwen3.8-27b held a 312ms median with
# the tightest spread (250-375ms) against gpt-oss-20b's 375ms/188-1063ms. For
# live use, predictable beats occasionally-faster. It was also the only model
# tested that gave advice instead of inventing precise figures the transcript
# never contained -- the GPT-OSS pair both fabricated latency numbers.
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
