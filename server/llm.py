"""OpenAI-compatible chat client with an automatic chain of free providers.

Providers are tried in the order of config.PROVIDERS (Gemini → NVIDIA → Z.ai → OVHcloud).
When one answers "rate limited" / "quota exhausted" / server error / timeout, it is put on a
cooldown and the next provider is used, so the site keeps working when one free quota runs out.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import httpx

from . import config


class LLMError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    provider: str
    model: str


_cooldown_until: dict[str, float] = {}
_lock = threading.Lock()
_daily = {"day": "", "count": 0}

DAILY_QUOTA_RE = re.compile(r"per.?day|daily|quota|RESOURCE_EXHAUSTED|exceeded your current", re.I)


def _cool(name: str, seconds: float) -> None:
    with _lock:
        _cooldown_until[name] = max(_cooldown_until.get(name, 0), time.time() + seconds)


def _available(name: str) -> bool:
    return _cooldown_until.get(name, 0) <= time.time()


def _count_call() -> None:
    if config.LLM_DAILY_CALL_LIMIT <= 0:
        return
    with _lock:
        today = time.strftime("%Y-%m-%d")
        if _daily["day"] != today:
            _daily.update(day=today, count=0)
        if _daily["count"] >= config.LLM_DAILY_CALL_LIMIT:
            raise LLMError("The site reached its daily AI limit. Please try again tomorrow.")
        _daily["count"] += 1


def status() -> list[dict]:
    """Provider chain + whether each is currently cooling down (for /api/health)."""
    now = time.time()
    return [{"provider": n, "model": m, "cooling_down_s": max(0, int(_cooldown_until.get(n, 0) - now))}
            for n, _, _, m in config.ACTIVE_PROVIDERS]


def _call(name: str, base: str, key: str, model: str, messages: list[dict], temperature: float) -> str:
    headers = {"Content-Type": "application/json"}
    if key and key != "anonymous":
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": config.LLM_MAX_TOKENS}
    r = httpx.post(f"{base}/chat/completions", json=body, headers=headers, timeout=config.LLM_TIMEOUT_SECONDS)
    if r.status_code == 429:
        # Daily quota → long cooldown; per-minute limit → short one.
        retry = r.headers.get("retry-after", "")
        wait = float(retry) if retry.replace(".", "").isdigit() else 30
        daily = bool(DAILY_QUOTA_RE.search(r.text)) and "minute" not in r.text.lower()
        _cool(name, 3600 if daily else min(max(wait, 15), 300))
        raise LLMError(f"{name}: rate limited")
    if r.status_code in (401, 403):
        _cool(name, 3600)
        raise LLMError(f"{name}: key rejected (HTTP {r.status_code})")
    if r.status_code >= 400:
        _cool(name, 120 if r.status_code >= 500 else 600)
        raise LLMError(f"{name}: HTTP {r.status_code} {r.text[:200]}")
    try:
        content = r.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError):
        _cool(name, 120)
        raise LLMError(f"{name}: unexpected response {r.text[:200]}")
    if not content.strip():
        raise LLMError(f"{name}: empty response")
    return content


def chat(messages: list[dict], temperature: float = 0.3, prefer: str | None = None) -> Reply:
    """Send a chat request to the first available provider in the chain.
    `prefer` puts a provider first (used so a fix round stays on the model that wrote the code)."""
    if not config.ACTIVE_PROVIDERS:
        raise LLMError("No AI provider is configured on the server.")
    _count_call()
    chain = sorted(config.ACTIVE_PROVIDERS, key=lambda p: p[0] != prefer) if prefer else config.ACTIVE_PROVIDERS
    errors = []
    for _ in range(2):  # second pass: retry anything whose short cooldown has expired
        for name, base, key, model in chain:
            if not _available(name):
                continue
            try:
                return Reply(_call(name, base, key, model, messages, temperature), name, model)
            except httpx.HTTPError as e:
                _cool(name, 60)
                errors.append(f"{name}: network error {type(e).__name__}")
            except LLMError as e:
                errors.append(str(e))
        soonest = min((_cooldown_until.get(n, 0) for n, *_ in chain), default=0) - time.time()
        if soonest > 25:
            break
        time.sleep(max(soonest, 1))
    print("[llm] all providers failed:", "; ".join(errors), flush=True)
    raise LLMError("All free AI providers are busy right now. Please try again in a few minutes.")
