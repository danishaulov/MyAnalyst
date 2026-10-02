"""Conservative numeric evidence checks. These do not prove semantic correctness."""
from __future__ import annotations

import re

_ORDINAL = re.compile(r"(?m)(?:^|[;\n])\s*\d{1,2}\s*[.):\-]\s+")
_NUMBER = re.compile(
    r"(?<![\w.])(?P<currency>[$€£₪])?\s*(?P<number>-?\d+(?:,\d{3})*(?:\.\d+)?|[-]?\.\d+)"
    r"\s*(?P<scale>trillion|billion|million|thousand|[KMBT]\b)?\s*(?P<percent>%|percent\b)?",
    re.I,
)
_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6,
           "b": 1e9, "billion": 1e9, "t": 1e12, "trillion": 1e12}


def claims(text: str):
    text = _ORDINAL.sub(" ", text)
    for match in _NUMBER.finditer(text):
        number = match["number"].replace(",", "")
        scale = _SCALES.get((match["scale"] or "").lower(), 1)
        unit = "%" if match["percent"] else match["currency"] or "number"
        value = float(number) * scale
        decimals = len(number.split(".")[1]) if "." in number else 0
        # Round to the precision actually written, capped to prevent coarse claims swallowing small values.
        tolerance = min(0.5 * 10 ** -decimals * scale, max(abs(value) * 0.02, 1e-9))
        yield value, unit, tolerance, match.group().strip()


def check_grounding(answer: str, facts: list[dict]) -> dict:
    evidence = [c for f in facts for c in claims(str(f.get("text", "")))]
    unverified = []
    for value, unit, tolerance, token in claims(answer):
        # A neutral number can describe a named KPI whose units live in the column name.
        # A percent or currency claim must have a compatible unit in the source.
        matches = [c for c in evidence if unit == "number" or c[1] == unit]
        if any(abs(value - source[0]) <= max(tolerance, source[2], 1e-9) for source in matches):
            continue
        # "fell 33.8%" commonly describes the signed -33.8% trend emitted by the engine.
        if value >= 0 and unit == "%" and any(abs(value + c[0]) <= max(tolerance, c[2], 1e-9) for c in matches):
            continue
        unverified.append(token)
    return {"grounded": not unverified, "unverified": sorted(set(unverified))[:6]}
