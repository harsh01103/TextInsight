"""Input validation helpers shared by every public function."""

from __future__ import annotations

from collections.abc import Collection, Iterable

from .config import config
from .exceptions import InvalidInputError

__all__ = ["ensure_choice", "ensure_positive_int", "ensure_ratio", "ensure_text", "ensure_texts"]


def ensure_text(text: object, *, name: str = "text", allow_empty: bool = True) -> str:
    """Return ``text`` if it is a valid string, otherwise raise :class:`InvalidInputError`."""
    if text is None:
        raise InvalidInputError(f"{name} must be a string, got None")
    if isinstance(text, (bytes, bytearray)):
        raise InvalidInputError(
            f"{name} must be a str, got bytes. Decode it first, e.g. data.decode('utf-8')"
        )
    if not isinstance(text, str):
        raise InvalidInputError(f"{name} must be a str, got {type(text).__name__}")
    if not allow_empty and not text.strip():
        raise InvalidInputError(f"{name} must not be empty")
    limit = config.max_input_chars
    if len(text) > limit:
        raise InvalidInputError(
            f"{name} has {len(text):,} characters, above the limit of {limit:,}. "
            "Split the text, or raise textinsight.config.max_input_chars."
        )
    return text


def ensure_texts(
    texts: object, *, name: str = "text", allow_empty: bool = True
) -> tuple[list[str], bool]:
    """Normalize a single string or an iterable of strings to a list.

    Returns ``(texts_as_list, was_single)`` so callers can return a single result
    for a single input and a list of results for a list input.
    """
    if isinstance(texts, str):
        return [ensure_text(texts, name=name, allow_empty=allow_empty)], True
    if texts is None or isinstance(texts, (bytes, bytearray)):
        ensure_text(texts, name=name)  # raises with a helpful message
    if isinstance(texts, dict) or not isinstance(texts, Iterable):
        raise InvalidInputError(
            f"{name} must be a str or a list of str, got {type(texts).__name__}"
        )
    items = list(texts)
    checked = [
        ensure_text(item, name=f"{name}[{i}]", allow_empty=allow_empty)
        for i, item in enumerate(items)
    ]
    return checked, False


def ensure_choice(value: object, choices: Collection[str], *, name: str) -> str:
    """Check that ``value`` is one of ``choices`` (case-insensitive) and return it lowercased."""
    if not isinstance(value, str) or value.lower() not in {c.lower() for c in choices}:
        raise InvalidInputError(
            f"{name} must be one of {', '.join(sorted(choices))}; got {value!r}"
        )
    return value.lower()


def ensure_positive_int(value: object, *, name: str) -> int:
    """Check that ``value`` is an int greater than zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidInputError(f"{name} must be a positive integer, got {value!r}")
    return value


def ensure_ratio(value: object, *, name: str) -> float:
    """Check that ``value`` is a number with 0 < value <= 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise InvalidInputError(f"{name} must be a number in (0, 1], got {value!r}")
    return float(value)
