"""Public summarization function."""

from __future__ import annotations

from typing import cast, overload

from ..core.registry import AUTO, get_backend
from ..core.results import SummaryResult
from ..core.validation import ensure_positive_int, ensure_ratio, ensure_texts
from ..langid.grouping import group_by_language
from .extractive import DEFAULT_RATIO, SummaryBackend

__all__ = ["summarize"]


@overload
def summarize(
    text: str,
    *,
    ratio: float = ...,
    max_sentences: int | None = ...,
    method: str = ...,
    language: str | None = ...,
) -> SummaryResult: ...
@overload
def summarize(
    text: list[str],
    *,
    ratio: float = ...,
    max_sentences: int | None = ...,
    method: str = ...,
    language: str | None = ...,
) -> list[SummaryResult]: ...


def summarize(
    text: str | list[str],
    *,
    ratio: float = DEFAULT_RATIO,
    max_sentences: int | None = None,
    method: str = AUTO,
    language: str | None = None,
) -> SummaryResult | list[SummaryResult]:
    """Pick the most representative sentences of English or Hinglish text.

    Args:
        text: A string (returns one result) or a list of strings (returns a list).
        ratio: Share of sentences to keep, above 0 and at most 1. At least one
            sentence is always kept.
        max_sentences: Upper limit on the sentences kept, whatever ``ratio`` says.
        method: ``"auto"``, ``"textrank"``, ``"frequency"``, or ``"abstractive"`` for
            English summaries written by a model (needs the ``transformers`` extra).
        language: Skip detection and treat all input as this language code.

    Raises:
        UnsupportedLanguageError: for text in an unsupported language, such as
            Hindi written in Devanagari.

    Sentences are found from punctuation, so Hinglish typed without any full
    stops counts as a single sentence and is returned unchanged.

    >>> summarize("Only one sentence here.").summary
    'Only one sentence here.'
    """
    texts, single = ensure_texts(text)
    ensure_ratio(ratio, name="ratio")
    if max_sentences is not None:
        ensure_positive_int(max_sentences, name="max_sentences")

    results: list[SummaryResult | None] = [None] * len(texts)
    for code, indices in group_by_language(texts, language).items():
        backend = cast(SummaryBackend, get_backend("summarize", method, language=code))
        summaries = backend.summarize(
            [texts[i] for i in indices], ratio=ratio, max_sentences=max_sentences, language=code
        )
        for index, result in zip(indices, summaries, strict=True):
            results[index] = result

    final = [result for result in results if result is not None]
    return final[0] if single else final
