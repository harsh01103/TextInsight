"""Public aspect sentiment function."""

from __future__ import annotations

from typing import cast, overload

from ..core.registry import AUTO, get_backend
from ..core.results import AspectResult
from ..core.validation import ensure_positive_int, ensure_texts
from ..langid.grouping import group_by_language
from .extractor import RuleAspects

__all__ = ["aspects"]


@overload
def aspects(
    text: str, *, min_mentions: int = ..., method: str = ..., language: str | None = ...
) -> AspectResult: ...
@overload
def aspects(
    text: list[str], *, min_mentions: int = ..., method: str = ..., language: str | None = ...
) -> list[AspectResult]: ...


def aspects(
    text: str | list[str],
    *,
    min_mentions: int = 1,
    method: str = AUTO,
    language: str | None = None,
) -> AspectResult | list[AspectResult]:
    """Find what a review praises or criticizes, in English or Hinglish.

    Args:
        text: A string (returns one result) or a list of strings (returns a list).
        min_mentions: Ignore aspects mentioned fewer times than this in a text.
        method: ``"auto"`` or ``"rules"``.
        language: Skip detection and treat all input as this language code.

    Raises:
        UnsupportedLanguageError: for text in an unsupported language, such as
            Hindi written in Devanagari.

    >>> aspects("camera mast hai but battery bekaar hai").labels
    {'camera': 'positive', 'battery': 'negative'}
    """
    texts, single = ensure_texts(text)
    ensure_positive_int(min_mentions, name="min_mentions")

    results: list[AspectResult | None] = [None] * len(texts)
    for code, indices in group_by_language(texts, language).items():
        backend = cast(RuleAspects, get_backend("aspects", method, language=code))
        extracted = backend.extract(
            [texts[i] for i in indices], min_mentions=min_mentions, language=code
        )
        for index, result in zip(indices, extracted, strict=True):
            results[index] = result

    final = [result for result in results if result is not None]
    return final[0] if single else final
