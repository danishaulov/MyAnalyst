"""Bounded, explicit contracts for the public compute API. No SDK dependencies."""
from __future__ import annotations

import csv
from io import StringIO
import math

import pandas as pd

MAX_BODY_BYTES = 4_000_000
MAX_ROWS = 100_000
MAX_COLUMNS = 100
MAX_CELLS = 5_000_000


class InputError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def text(value, name: str, limit: int, *, required: bool = False) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise InputError(f"'{name}' must be {'a non-empty' if required else 'a'} string of at most {limit} characters.")
    return value.strip()


def records(value, name: str, fields: tuple[str, ...], limit: int = 200) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > limit:
        raise InputError(f"'{name}' must be a list of at most {limit} objects.")
    out = []
    for item in value:
        if not isinstance(item, dict):
            raise InputError(f"Each '{name}' entry must be an object.")
        clean = {}
        for f in fields:
            value = item.get(f)
            if f == "value" and type(value) in (int, float) and math.isfinite(value):
                value = str(value)
            clean[f] = text(value, f"{name}.{f}", 4000, required=True)
        out.append(clean)
    return out


def dataframe(payload: dict) -> pd.DataFrame:
    raw_csv = payload.get("csv")
    if raw_csv is not None:
        if payload.get("rows") is not None:
            raise InputError("Provide either 'csv' or 'rows', not both.")
        raw_csv = text(raw_csv, "csv", MAX_BODY_BYTES, required=True)
        try:
            reader = csv.reader(StringIO(raw_csv))
            header = next(reader)
            _columns(header)
            for number, row in enumerate(reader, 1):
                if not row:  # pandas skips blank lines
                    continue
                if number > MAX_ROWS or number * len(header) > MAX_CELLS:
                    raise InputError("The CSV exceeds the row or cell limit.", 413)
                if len(row) != len(header):
                    raise InputError("CSV rows must match the number of header columns.")
            df = pd.read_csv(StringIO(raw_csv), nrows=MAX_ROWS + 1)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, StopIteration) as exc:
            raise InputError("The CSV could not be parsed. Check its header and row structure.") from exc
    else:
        rows, columns = payload.get("rows"), payload.get("columns")
        if not isinstance(rows, list) or not rows:
            raise InputError("Provide a non-empty 'csv' or 'rows' array.")
        if len(rows) > MAX_ROWS:
            raise InputError(f"At most {MAX_ROWS} rows are supported per request.", 413)
        if columns is None and isinstance(rows[0], dict):
            columns = list(rows[0])
        _columns(columns)
        if len(rows) * len(columns) > MAX_CELLS:
            raise InputError(f"At most {MAX_CELLS} cells are supported per request.", 413)
        keys = set(columns)
        for row in rows:
            if isinstance(row, dict):
                if set(row) - keys:
                    raise InputError("Row keys must match the declared columns.")
                cells = row.values()
            elif isinstance(row, list) and len(row) == len(columns):
                cells = row
            else:
                raise InputError("Each row must be an object or an array matching the columns.")
            for cell in cells:
                if isinstance(cell, (list, dict)) or (isinstance(cell, float) and not math.isfinite(cell)):
                    raise InputError("Cells must be finite numbers, strings, booleans, or null.")
        df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        raise InputError("No data rows were provided.")
    if len(df) > MAX_ROWS or len(df.columns) > MAX_COLUMNS or df.size > MAX_CELLS:
        raise InputError("The dataset exceeds the supported row, column, or cell limit.", 413)
    return df.replace([float("inf"), -float("inf")], float("nan"))


def _columns(columns):
    if not isinstance(columns, list) or not columns or len(columns) > MAX_COLUMNS:
        raise InputError(f"Provide between 1 and {MAX_COLUMNS} columns.")
    if any(not isinstance(c, str) or not c.strip() or len(c) > 200 for c in columns):
        raise InputError("Column names must be non-empty strings of at most 200 characters.")
    if len(set(columns)) != len(columns):
        raise InputError("Column names must be unique.")
