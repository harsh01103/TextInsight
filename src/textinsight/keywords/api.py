"""Public keyword extraction function."""

from __future__ import annotations

from typing import cast, overload

from ..core.registry import AUTO, get_backend
from ..core.results import KeywordResult
from ..core.validation import ensure_positive_int, ensure_texts
from ..langid.grouping import group_by_language
from .base import DEFAULT_TOP_N, KeywordBackend

__all__ = ["keywords"]


@overload
def keywords(
    text: str, *, top_n: int = ..., method: str = ..., language: str | None = ...
) -> KeywordResult: ...
@overload
def keywords(
    text: list[str], *, top_n: int = ..., method: str = ..., language: str | None = ...
) -> list[KeywordResult]: ...


def keywords(
    text: str | list[str],
    *,
    top_n: int = DEFAULT_TOP_N,
    method: str = AUTO,
    language: str | None = None,
) -> KeywordResult | list[KeywordResult]:
    """Extract the most relevant words and phrases, for English or Hinglish text.

    In Hinglish, spelling variants ("accha", "acha") count as one word.

    Args:
        text: A string (returns one result) or a list of strings (returns a list).
            With ``"tfidf"``, the documents in a list are compared with each other.
        top_n: The most keywords to return per text.
        method: ``"auto"``, ``"textrank"``, ``"rake"`` or ``"tfidf"``.
        language: Skip detection and treat all input as this language code.

    Raises:
        UnsupportedLanguageError: for text in an unsupported language, such as
            Hindi written in Devanagari.

    Each ``score`` is relative to the best keyword of that text, which scores 1.

    >>> keywords("The battery life is great. Battery life matters.", top_n=1).texts
    ['battery life']
    """
    texts, single = ensure_texts(text)
    ensure_positive_int(top_n, name="top_n")

    results: list[KeywordResult | None] = [None] * len(texts)
    for code, indices in group_by_language(texts, language).items():
        backend = cast(KeywordBackend, get_backend("keywords", method, language=code))
        extracted = backend.extract([texts[i] for i in indices], top_n=top_n, language=code)
        for index, result in zip(indices, extracted, strict=True):
            results[index] = result

    final = [result for result in results if result is not None]
    return final[0] if single else final
