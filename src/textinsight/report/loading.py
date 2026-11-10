"""Read texts from CSV, JSON-lines or plain text files for the command line."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..core.exceptions import InvalidInputError

__all__ = ["read_texts"]

_DEFAULT_COLUMNS = ("text", "review", "comment", "content")


def _choose(columns: list[str], column: str | None, path: Path) -> str:
    if column is not None:
        if column not in columns:
            raise InvalidInputError(f"{path} has no column {column!r}. Found: {', '.join(columns)}")
        return column
    if len(columns) == 1:
        return columns[0]
    for name in _DEFAULT_COLUMNS:
        if name in columns:
            return name
    raise InvalidInputError(f"Choose a column with --column. Found: {', '.join(columns)}")


def _read_csv(path: Path, column: str | None) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise InvalidInputError(f"{path} has no header row")
        chosen = _choose(list(reader.fieldnames), column, path)
        return [row[chosen] or "" for row in reader]


def _read_jsonl(path: Path, column: str | None) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise InvalidInputError(f"{path}, line {number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise InvalidInputError(f"{path}, line {number}: expected a JSON object")
            texts.append(str(record[_choose(list(record), column, path)] or ""))
    return texts


def read_texts(path: Path, column: str | None = None) -> list[str]:
    """Return one text per CSV row, JSON line, or non-empty line of plain text."""
    if path.suffix.lower() == ".csv":
        return _read_csv(path, column)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return _read_jsonl(path, column)
    return [line.strip() for line in path.read_text("utf-8-sig").splitlines() if line.strip()]
