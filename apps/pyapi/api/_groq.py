"""Bounded OpenAI-compatible client. Legacy module name retained for callers."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

_DEADLINE = ContextVar("llm_deadline", default=None)
LOG = logging.getLogger("myanalyst.llm")
_BASES = {"groq": "https://api.groq.com/openai/v1", "openai": "https://api.openai.com/v1",
          "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
          "openrouter": "https://openrouter.ai/api/v1"}
_MODELS = {"groq": "openai/gpt-oss-120b", "openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash",
           "openrouter": "meta-llama/llama-3.3-70b-instruct"}


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "groq").strip() or "groq"


def available() -> bool:
    return bool(os.environ.get("LLM_API_KEY", "").strip())


@contextmanager
def budget(seconds: float):
    current = _DEADLINE.get()
    token = _DEADLINE.set(min(current, time.monotonic() + seconds) if current else time.monotonic() + seconds)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def chat(messages: list[dict], *, json_mode: bool = True, temperature: float = 0.3, max_tokens: int = 1600) -> str | None:
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not key:
        return None
    name = provider()
    model = os.environ.get("LLM_MODEL", "").strip() or _MODELS.get(name)
    base = (os.environ.get("LLM_BASE_URL", "").strip() or _BASES.get(name, "")).rstrip("/")
    parsed_url = urlparse(base)
    if not model or parsed_url.scheme != "https" or not parsed_url.hostname or parsed_url.username or parsed_url.password:
        LOG.warning("llm_configuration_invalid provider=%s", name)
        return None
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if name == "groq" and "gpt-oss" in model:
        body["reasoning_effort"] = "low"
        body["reasoning_format"] = "hidden"
        body["max_tokens"] = max(max_tokens, 1400)
    serialized = json.dumps(body, ensure_ascii=False).encode("utf-8")
    if len(serialized) > 120_000:
        return None
    req = urllib.request.Request(f"{base}/chat/completions", data=serialized,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": "MyAnalyst/1.0 (+https://myanalyst.net)"}, method="POST")
    deadline = _DEADLINE.get() or time.monotonic() + 25
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining < 1:
            return None
        try:
            with urllib.request.urlopen(req, timeout=min(18, remaining)) as resp:
                data = resp.read(128_001)
                if len(data) > 128_000:
                    return None
                parsed = json.loads(data)
                choice = parsed["choices"][0]
                content = choice["message"].get("content")
                if choice.get("finish_reason") in {"length", "content_filter"}:
                    return None
                return content if isinstance(content, str) and content.strip() else None
        except urllib.error.HTTPError as exc:
            retry = exc.code in {429, 500, 502, 503, 504}
            try:
                wait = float(exc.headers.get("Retry-After", "0.4"))
            except (TypeError, ValueError):
                wait = 0.4
            exc.close()
            LOG.warning("llm_http_failure provider=%s status=%d", name, exc.code)
            if attempt or not retry or not (0 <= wait <= 2) or deadline - time.monotonic() <= wait + 1:
                return None
            time.sleep(wait)
        except Exception as exc:
            LOG.warning("llm_failure provider=%s exception=%s", name, type(exc).__name__)
            return None
    return None
