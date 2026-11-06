"""Minimal client for any OpenAI-compatible chat endpoint.

Works with Groq (GPT-OSS, Qwen, Llama), OpenRouter (incl. `:free` models),
Together, Moonshot's own API, or a local Ollama — just change LLM_BASE_URL / LLM_MODEL.
"""
from __future__ import annotations

import threading
import time

import httpx

from . import config


class LLMError(RuntimeError):
    pass


class _DailyCounter:
    def __init__(self) -> None:
        self.day = time.strftime("%Y-%m-%d")
        self.count = 0
        self.lock = threading.Lock()

    def take(self) -> None:
        with self.lock:
            today = time.strftime("%Y-%m-%d")
            if today != self.day:
                self.day, self.count = today, 0
            if self.count >= config.LLM_DAILY_CALL_LIMIT:
                raise LLMError("The site reached its daily AI limit. Please try again tomorrow.")
            self.count += 1


_daily = _DailyCounter()


def chat(messages: list[dict], temperature: float = 0.3) -> str:
    if not config.LLM_API_KEY:
        raise LLMError("The server has no LLM_API_KEY configured.")
    _daily.take()
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}", "Content-Type": "application/json"}
    if "openrouter.ai" in config.LLM_BASE_URL:
        headers["X-Title"] = "Manim Studio"
    body = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": config.LLM_MAX_TOKENS,
    }
    last_err = ""
    for attempt in range(3):
        try:
            r = httpx.post(f"{config.LLM_BASE_URL}/chat/completions", json=body, headers=headers,
                           timeout=config.LLM_TIMEOUT_SECONDS)
        except httpx.HTTPError as e:
            last_err = f"network error: {e}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            retry = r.headers.get("retry-after")
            time.sleep(min(float(retry) if retry and retry.replace(".", "").isdigit() else 10 * (attempt + 1), 60))
            continue
        if r.status_code != 200:
            raise LLMError(f"AI provider error (HTTP {r.status_code}): {r.text[:300]}")
        try:
            content = r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError):
            raise LLMError(f"Unexpected AI response: {r.text[:300]}")
        if not content.strip():
            last_err = "empty response"
            continue
        return content
    raise LLMError(f"AI provider is busy or unreachable ({last_err}). Try again in a minute.")
