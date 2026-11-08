"""Text cleaning.

Phase 0 placeholder: only whitespace normalization. The full configurable
cleaner (URLs, HTML, emojis, punctuation) arrives in Phase 2.
"""

from __future__ import annotations

import re
from typing import overload

from ..core.validation import ensure_texts

__all__ = ["clean"]

_WHITESPACE = re.compile(r"\s+")


def _clean_one(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


@overload
def clean(text: str) -> str: ...
@overload
def clean(text: list[str]) -> list[str]: ...


def clean(text: str | list[str]) -> str | list[str]:
    """Collapse runs of whitespace and trim the ends.

    Accepts a single string (returns a string) or a list of strings (returns a list).

    >>> clean("  Hello    world!\\n")
    'Hello world!'
    """
    texts, single = ensure_texts(text)
    cleaned = [_clean_one(t) for t in texts]
    return cleaned[0] if single else cleaned
