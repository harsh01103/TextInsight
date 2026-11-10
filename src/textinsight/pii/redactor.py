"""Find and hide personal information with patterns and checksums."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import overload

from ..core.results import PIIEntity, PIIResult
from ..core.validation import ensure_choice, ensure_texts
from ..preprocess._patterns import EMAIL, URL
from .validators import is_valid_aadhaar, is_valid_card, is_valid_pan

__all__ = ["KINDS", "redact"]

_NO_DIGIT_BEFORE = r"(?<![\d+])(?<!\d[ -])"
_NO_DIGIT_AFTER = r"(?![ -]?\d)"

Validator = Callable[[str], bool]

# Earlier kinds win when two matches overlap, so more specific checks come first.
_RULES: tuple[tuple[str, re.Pattern[str], Validator | None], ...] = (
    ("URL", re.compile(URL, re.IGNORECASE), None),
    ("EMAIL", re.compile(EMAIL), None),
    ("UPI", re.compile(r"(?<![\w.+-])[\w.-]{2,}@[A-Za-z]{2,}(?![\w@-]|\.\w)"), None),
    (
        "PAN",
        re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{5}\d{4}[A-Za-z](?![A-Za-z0-9])"),
        is_valid_pan,
    ),
    (
        "CARD",
        re.compile(rf"{_NO_DIGIT_BEFORE}\d(?:[ -]?\d){{12,18}}{_NO_DIGIT_AFTER}"),
        is_valid_card,
    ),
    (
        "AADHAAR",
        re.compile(rf"{_NO_DIGIT_BEFORE}[2-9]\d{{3}}[ -]?\d{{4}}[ -]?\d{{4}}{_NO_DIGIT_AFTER}"),
        is_valid_aadhaar,
    ),
    (
        "PHONE",
        re.compile(
            rf"{_NO_DIGIT_BEFORE}(?:"
            r"(?:\+91[ -]?|91[ -]|0)?(?:[6-9]\d{4}[ -]?\d{5}|[6-9]\d{2}[ -]?\d{3}[ -]?\d{4})"
            r"|\+(?!91)\d{1,3}[ -]?\d(?:[ -]?\d){6,13}"
            rf"){_NO_DIGIT_AFTER}"
        ),
        None,
    ),
)
KINDS = tuple(kind for kind, _, _ in _RULES)


def _find(text: str, kinds: Iterable[str]) -> list[PIIEntity]:
    wanted = set(kinds)
    chosen: list[PIIEntity] = []
    for kind, pattern, validator in _RULES:
        if kind not in wanted:
            continue
        for match in pattern.finditer(text):
            start, end = match.span()
            if (validator is None or validator(match.group(0))) and not any(
                start < other.end and other.start < end for other in chosen
            ):
                chosen.append(PIIEntity(kind=kind, start=start, end=end, value=match.group(0)))
    return sorted(chosen, key=lambda entity: entity.start)


def _redact_one(text: str, kinds: Iterable[str]) -> PIIResult:
    entities = _find(text, kinds)
    pieces: list[str] = []
    position = 0
    for entity in entities:
        pieces += [text[position : entity.start], f"[{entity.kind}]"]
        position = entity.end
    pieces.append(text[position:])
    return PIIResult(text="".join(pieces), entities=tuple(entities))


@overload
def redact(text: str, *, kinds: Iterable[str] | None = ...) -> PIIResult: ...
@overload
def redact(text: list[str], *, kinds: Iterable[str] | None = ...) -> list[PIIResult]: ...


def redact(
    text: str | list[str], *, kinds: Iterable[str] | None = None
) -> PIIResult | list[PIIResult]:
    """Replace personal information with placeholders such as ``[EMAIL]``.

    Finds URLs, emails, UPI IDs, PAN, payment cards, Aadhaar numbers and phone
    numbers. Cards must pass Luhn, Aadhaar must pass Verhoeff, and PAN must match
    its format.

    Args:
        text: A string (returns one result) or a list of strings (returns a list).
        kinds: Only look for these, for example ``["EMAIL", "PHONE"]``.

    This finds patterns only, so it cannot find names or addresses, and any
    10-digit number starting with 6 to 9 looks like a phone number. Do not rely on
    it as the only safeguard for sensitive data.

    >>> redact("Mail me at a@b.co").text
    'Mail me at [EMAIL]'
    """
    texts, single = ensure_texts(text)
    if kinds is None:
        selected = list(KINDS)
    else:
        requested = [kinds] if isinstance(kinds, str) else list(kinds)
        selected = [ensure_choice(kind, KINDS, name="kinds").upper() for kind in requested]
    results = [_redact_one(item, selected) for item in texts]
    return results[0] if single else results
