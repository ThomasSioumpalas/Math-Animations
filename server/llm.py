"""OpenAI-compatible chat client with an automatic chain of free providers.

Providers are tried in the order of config.PROVIDERS (gpt-oss-120b → Gemini → NVIDIA → Qwen → GLM).
When one answers "rate limited" / "quota exhausted" / server error / timeout, it is put on a
cooldown and the next provider is used, so the site keeps working when one free quota runs out.
"""
from __future__ import annotations

import json
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


RETRY_TEXT_RE = re.compile(r"retry in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", re.I)
RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"([\d.]+)s"')


def _provider_wait(text: str) -> float | None:
    """Seconds the provider asked us to wait (Gemini: 'Please retry in 11h47m29s' / retryDelay)."""
    m = RETRY_DELAY_RE.search(text)
    if m:
        return float(m.group(1))
    m = RETRY_TEXT_RE.search(text)
    if m and any(m.groups()):
        h, mi, s = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mi * 60 + s
    return None


def _fail_on_status(name: str, status: int, text: str, retry_after: str) -> None:
    if status == 429:
        # Use the provider's own wait time when it gives one (a daily quota can mean ~12 h);
        # otherwise: daily quota -> 1 h, per-minute limit -> a short pause.
        asked = _provider_wait(text)
        if asked is None and retry_after.replace(".", "").isdigit():
            asked = float(retry_after)
        daily = bool(DAILY_QUOTA_RE.search(text)) and "minute" not in text.lower()
        if asked is not None:
            _cool(name, min(max(asked, 15), 86400))
        else:
            _cool(name, 3600 if daily else 30)
        raise LLMError(f"{name}: rate limited" + (" (daily quota used up)" if daily else ""))
    if status in (401, 403):
        _cool(name, 3600)
        raise LLMError(f"{name}: key rejected (HTTP {status})")
    if status == 503:
        _cool(name, 90)  # "high demand": usually brief
        raise LLMError(f"{name}: overloaded (HTTP 503)")
    if status >= 400:
        _cool(name, 120 if status >= 500 else 600)
        raise LLMError(f"{name}: HTTP {status} {text[:200]}")


def _call(name: str, base: str, key: str, model: str, messages: list[dict], temperature: float,
          should_stop=None, effort: str | None = None) -> str:
    """Streamed request. Fails fast when the provider produces nothing within
    LLM_FIRST_TOKEN_SECONDS (overloaded free tiers often just hang), so the chain moves on."""
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key and key != "anonymous":
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": config.LLM_MAX_TOKENS, "stream": True}
    if effort:
        body["reasoning_effort"] = effort
    timeout = httpx.Timeout(connect=20, read=config.LLM_FIRST_TOKEN_SECONDS, write=30, pool=20)
    deadline = time.time() + config.LLM_TIMEOUT_SECONDS
    parts: list[str] = []
    try:
        with httpx.stream("POST", f"{base}/chat/completions", json=body, headers=headers, timeout=timeout) as r:
            if r.status_code != 200:
                text = r.read().decode("utf-8", "replace")
                _fail_on_status(name, r.status_code, text, r.headers.get("retry-after", ""))
            for line in r.iter_lines():
                if should_stop is not None and should_stop():
                    raise LLMError("cancelled")
                if time.time() > deadline:
                    _cool(name, 120)
                    raise LLMError(f"{name}: answer took longer than {config.LLM_TIMEOUT_SECONDS}s")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if isinstance(chunk, list):  # Gemini sometimes wraps errors in a list
                    chunk = chunk[0] if chunk else {}
                if "error" in chunk:
                    err = chunk["error"]
                    _fail_on_status(name, int(err.get("code", 500)) if isinstance(err, dict) else 500, str(err), "")
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or (choice.get("message") or {}).get("content")
                    if piece:
                        parts.append(piece)
    except httpx.ReadTimeout:
        _cool(name, 180)
        raise LLMError(f"{name}: no output for {config.LLM_FIRST_TOKEN_SECONDS}s (overloaded?)")
    content = "".join(parts)
    if not content.strip():
        _cool(name, 60)
        raise LLMError(f"{name}: empty response")
    return content


def chat(messages: list[dict], temperature: float = 0.3, prefer: str | None = None,
         should_stop=None, effort: str | None = None) -> Reply:
    """Send a chat request to the first available provider in the chain.
    `prefer` puts a provider first (used so a fix round stays on the model that wrote the code)."""
    if not config.ACTIVE_PROVIDERS:
        raise LLMError("No AI provider is configured on the server.")
    _count_call()
    chain = sorted(config.ACTIVE_PROVIDERS, key=lambda p: p[0] != prefer) if prefer else config.ACTIVE_PROVIDERS
    errors = []
    for _ in range(2):  # second pass: retry anything whose short cooldown has expired
        for name, base, key, model in chain:
            if should_stop is not None and should_stop():
                raise LLMError("cancelled")
            if not _available(name):
                continue
            t0 = time.time()
            try:
                text = _call(name, base, key, model, messages, temperature, should_stop, effort)
                print(f"[llm] {name} ({model}) answered in {time.time() - t0:.0f}s", flush=True)
                return Reply(text, name, model)
            except httpx.HTTPError as e:
                _cool(name, 60)
                errors.append(f"{name}: network error {type(e).__name__}")
                print(f"[llm] {errors[-1]} -> trying next provider", flush=True)
            except LLMError as e:
                if str(e) == "cancelled":
                    raise
                errors.append(str(e))
                print(f"[llm] {e} (after {time.time() - t0:.0f}s) -> trying next provider", flush=True)
        soonest = min((_cooldown_until.get(n, 0) for n, *_ in chain), default=0) - time.time()
        if soonest > 25:
            break
        time.sleep(max(soonest, 1))
    print("[llm] all providers failed:", "; ".join(errors), flush=True)
    raise LLMError("All free AI providers are busy right now. Please try again in a few minutes.")
