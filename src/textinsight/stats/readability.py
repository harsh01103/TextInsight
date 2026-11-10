"""Counts, lexical diversity, reading time and Flesch reading ease."""

from __future__ import annotations

import re
from typing import overload

from ..core.results import TextStats
from ..core.validation import ensure_texts
from ..langid import detect_language
from ..preprocess.hinglish import canonical_form
from ..preprocess.tokenizer import iter_tokens, sentences

__all__ = ["count_syllables", "stats"]

WORDS_PER_MINUTE = 238  # average silent reading speed in English (Brysbaert, 2019)
_WORD_KINDS = {"word", "number", "hashtag"}
_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def count_syllables(word: str) -> int:
    """Estimate the syllables of an English word from its spelling."""
    word = word.lower()
    count = len(_VOWEL_GROUP.findall(word))
    if (
        (word.endswith("e") and not word.endswith("le"))
        or (word.endswith("es") and not word.endswith(("ses", "xes", "zes", "ches", "shes")))
    ) or (word.endswith("ed") and not word.endswith(("ted", "ded"))):
        count -= 1
    return max(1, count)


def _flesch(words: list[str], sentence_count: int) -> float:
    syllables = sum(count_syllables(word) for word in words if word.isalpha())
    per_sentence = len(words) / sentence_count
    per_word = syllables / len(words)
    return round(206.835 - 1.015 * per_sentence - 84.6 * per_word, 2)


def _stats_one(text: str, language: str) -> TextStats:
    words = [token.lstrip("#").lower() for token, kind in iter_tokens(text) if kind in _WORD_KINDS]
    # Counting spelling variants once keeps Hinglish diversity scores meaningful.
    distinct = {canonical_form(word) if language == "hi-en" else word for word in words}
    sentence_count = len(sentences(text)) if words else 0
    return TextStats(
        characters=len(text),
        words=len(words),
        sentences=sentence_count,
        unique_words=len(distinct),
        type_token_ratio=round(len(distinct) / len(words), 4) if words else 0.0,
        average_word_length=round(sum(map(len, words)) / len(words), 2) if words else 0.0,
        average_sentence_length=round(len(words) / sentence_count, 2) if words else 0.0,
        flesch_reading_ease=_flesch(words, sentence_count) if words and language == "en" else None,
        reading_time_seconds=round(len(words) / WORDS_PER_MINUTE * 60, 1),
        language=language,
    )


@overload
def stats(text: str, *, language: str | None = ...) -> TextStats: ...
@overload
def stats(text: list[str], *, language: str | None = ...) -> list[TextStats]: ...


def stats(text: str | list[str], *, language: str | None = None) -> TextStats | list[TextStats]:
    """Measure the size, diversity and readability of text.

    Args:
        text: A string (returns one result) or a list of strings (returns a list).
        language: Skip detection and treat all input as this language code.

    Flesch reading ease is given only for English. Syllables are estimated from
    spelling, so the score is approximate. Reading time assumes 238 words a minute.

    >>> stats("The cat sat.", language="en").words
    3
    """
    texts, single = ensure_texts(text)
    codes = (
        [language] * len(texts)
        if language is not None
        else [result.language for result in detect_language(texts)]
    )
    results = [_stats_one(item, code) for item, code in zip(texts, codes, strict=True)]
    return results[0] if single else results
