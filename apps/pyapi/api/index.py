"""Vercel entrypoint shared with the local server; bounded input and private responses."""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _groq  # noqa: E402
from _engine import analyze  # noqa: E402
from _conclude import generate_conclusions  # noqa: E402
from _ask import answer_question  # noqa: E402
from _validation import InputError, MAX_BODY_BYTES, dataframe, records, text  # noqa: E402

LOG = logging.getLogger("myanalyst.api")
if not LOG.handlers:
    LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)
LOG.propagate = False
# Resource guard for a warm instance. This is NOT distributed rate limiting or authentication.
_COMPUTE = threading.BoundedSemaphore(2)
_DEFAULT_ORIGINS = "https://myanalyst.net,https://www.myanalyst.net,http://localhost:3000,http://127.0.0.1:3000"


def _jsonable(obj):
    """Normalize recursively BEFORE json.dumps (default does not see built-in NaN floats)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def _reject_constant(_):
    raise InputError("JSON numbers must be finite.")


class handler(BaseHTTPRequestHandler):
    def _fn(self):
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        if q.get("fn"):
            return q["fn"][0]
        return parsed.path.rstrip("/").rsplit("/", 1)[-1]

    def _origin(self):
        origin = self.headers.get("Origin")
        allowed = {o.strip() for o in os.environ.get("API_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()}
        return origin if origin in allowed else None

    def _start(self):
        self._request_id = uuid.uuid4().hex
        self._started = time.monotonic()
        if self.headers.get("Origin") and not self._origin():
            self._send(403, {"error": "This origin is not allowed."})
            return False
        if self._fn() not in {"analyze", "conclude", "ask", "health", "index"}:
            self._send(404, {"error": "Unknown API endpoint."})
            return False
        return True

    def do_GET(self):
        if self._start():
            self._send(200, {"ok": True, "engine": "python-pandas", "fn": self._fn(), "llm": _groq.available()})

    def do_POST(self):
        if not self._start():
            return
        fn = self._fn()
        if fn not in {"analyze", "conclude", "ask"}:
            return self._send(405, {"error": "Use POST /api/analyze, /api/conclude, or /api/ask."})
        try:
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise InputError("Content-Type must be application/json.", 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise InputError("Invalid Content-Length.") from exc
            if length <= 0:
                raise InputError("Provide a JSON request body.")
            if length > MAX_BODY_BYTES:
                raise InputError("Request body exceeds 4 MB. Upload a smaller dataset.", 413)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise InputError("Incomplete request body.")
            try:
                payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
                raise InputError("Provide a valid UTF-8 JSON object.") from exc
            if not isinstance(payload, dict):
                raise InputError("Request body must be a JSON object.")
            if fn == "conclude":
                facts = records(payload.get("facts"), "facts", ("text",))
                if not facts:
                    raise InputError("Provide 'facts'.")
                args = dict(domain=text(payload.get("domain", "generic"), "domain", 100),
                            user_context=text(payload.get("userContext"), "userContext", 2000),
                            templated_fallback=text(payload.get("narrative"), "narrative", 16000),
                            kpis=records(payload.get("kpis"), "kpis", ("name", "value"), 30),
                            chart_readings=records(payload.get("chartReadings"), "chartReadings", ("title", "reading"), 30))
                if len(json.dumps([facts, args], ensure_ascii=False)) > 100_000:
                    raise InputError("Conclusion context is too large.", 413)
                if not _COMPUTE.acquire(blocking=False):
                    return self._send(429, {"error": "The analysis service is busy. Try again shortly."})
                try:
                    with _groq.budget(40):
                        return self._send(200, generate_conclusions(facts, **args))
                finally:
                    _COMPUTE.release()
            question = text(payload.get("question"), "question", 2000, required=True) if fn == "ask" else None
            df = dataframe(payload)
            source_rows = payload.get("sourceRowCount", len(df))
            if type(source_rows) is not int or source_rows < len(df):
                raise InputError("'sourceRowCount' must be an integer at least as large as the submitted row count.")
            if not _COMPUTE.acquire(blocking=False):
                return self._send(429, {"error": "The analysis service is busy. Try again shortly."})
            try:
                if fn == "ask":
                    with _groq.budget(40):
                        result = answer_question(df, question)
                else:
                    currency = payload.get("currency")
                    if currency is not None:
                        if not isinstance(currency, dict):
                            raise InputError("'currency' must contain a symbol and code.")
                        currency = {"symbol": text(currency.get("symbol"), "currency.symbol", 8, required=True),
                                    "code": text(currency.get("code"), "currency.code", 8, required=True)}
                    result = analyze(df, currency=currency)
                result["scope"] = {"sourceRows": source_rows, "analyzedRows": len(df), "sampled": source_rows > len(df)}
                return self._send(200, result)
            finally:
                _COMPUTE.release()
        except InputError as exc:
            self._send(exc.status, {"error": str(exc)})
        except Exception as exc:
            # No request contents, provider response, keys, or exception messages in logs.
            LOG.error("request_failed id=%s endpoint=%s exception=%s", self._request_id, fn, type(exc).__name__)
            self._send(500, {"error": "Analysis could not be completed. Please try again."})

    def do_OPTIONS(self):
        if self._start():
            self._send(204, None)

    def log_message(self, format, *args):
        # BaseHTTPRequestHandler logs full request URLs; query strings may contain user data.
        pass

    def _send(self, code, obj):
        if isinstance(obj, dict) and code >= 400:
            obj = {**obj, "requestId": self._request_id}
        body = b"" if obj is None else json.dumps(_jsonable(obj), allow_nan=False, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Request-ID", self._request_id)
        self.send_header("Vary", "Origin")
        if self._origin():
            self.send_header("Access-Control-Allow-Origin", self._origin())
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Expose-Headers", "X-Request-ID, Retry-After")
        if code == 429:
            self.send_header("Retry-After", "2")
        self.end_headers()
        if body:
            self.wfile.write(body)
        endpoint = self._fn()
        if endpoint not in {"analyze", "ask", "conclude", "health", "index"}:
            endpoint = "unknown"
        LOG.info("request_completed id=%s endpoint=%s status=%s duration_ms=%d", self._request_id,
                 endpoint, code, (time.monotonic() - self._started) * 1000)
