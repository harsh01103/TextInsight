"""Public sentiment function."""

from __future__ import annotations

from dataclasses import replace
from typing import overload

from ..core.registry import AUTO, get_backend
from ..core.results import SentimentResult
from ..core.validation import ensure_texts
from ..langid import detect_language

__all__ = ["sentiment"]


@overload
def sentiment(text: str, *, backend: str = ..., language: str | None = ...) -> SentimentResult: ...
@overload
def sentiment(
    text: list[str], *, backend: str = ..., language: str | None = ...
) -> list[SentimentResult]: ...


def sentiment(
    text: str | list[str], *, backend: str = AUTO, language: str | None = None
) -> SentimentResult | list[SentimentResult]:
    """Analyse sentiment of English or Hinglish text.

    Args:
        text: A string (returns one result) or a list of strings (returns a list).
        backend: ``"auto"``, ``"lexicon"``, ``"classical"``, ``"hybrid"``, ``"onnx"``
            (English only) or ``"llm"`` (sends your text to Anthropic). With
            ``"auto"``, each text goes to the best bundled backend for its language.
            ``"hybrid"`` is the most accurate but needs the ``transformers`` extra.
        language: Skip detection and treat all input as this language code.

    Raises:
        UnsupportedLanguageError: for text in an unsupported language, such as
            Hindi written in Devanagari.

    ``score`` runs from -1 (negative) to +1 (positive) and its exact meaning
    depends on the backend. It is not a calibrated probability.

    >>> sentiment("yeh phone bahut accha hai", backend="lexicon").label
    'positive'
    """
    texts, single = ensure_texts(text)
    if language is not None:
        languages = [language] * len(texts)
    else:
        languages = [result.language for result in detect_language(texts)]

    by_language: dict[str, list[int]] = {}
    for index, code in enumerate(languages):
        by_language.setdefault(code, []).append(index)

    results: list[SentimentResult | None] = [None] * len(texts)
    for code, indices in by_language.items():
        predictions = get_backend("sentiment", backend, language=code)([texts[i] for i in indices])
        for index, prediction in zip(indices, predictions, strict=True):
            results[index] = replace(prediction, language=code)

    final = [result for result in results if result is not None]
    return final[0] if single else final
