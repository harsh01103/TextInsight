"""TF-IDF keyword extraction.

With several documents, inverse document frequency comes from those documents. A
single document has no collection to compare against, so its sentences are used.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

from ..core.registry import register_backend
from .base import Document, KeywordBackend, Phrase, ngrams, phrase_key

__all__ = ["TfidfKeywords"]


def _words(units: Iterable[list[list[str]]]) -> list[set[str]]:
    return [{key for run in unit for key in run} for unit in units]


def _inverse_frequencies(documents: Sequence[Document]) -> dict[str, float]:
    if len(documents) == 1:
        units = [[[term.key for term in run] for run in sentence] for sentence in documents[0]]
    else:
        units = [
            [[term.key for term in run] for sentence in document for run in sentence]
            for document in documents
        ]
    frequencies: Counter[str] = Counter()
    for words in _words(units):
        frequencies.update(words)
    total = len(units)
    return {word: math.log((1 + total) / (1 + df)) + 1.0 for word, df in frequencies.items()}


@register_backend
class TfidfKeywords(KeywordBackend):
    """Scores a phrase by adding up the TF-IDF of its words."""

    name = "tfidf"
    priority = 5

    def score(self, documents: Sequence[Document]) -> list[dict[Phrase, float]]:
        idf = _inverse_frequencies(documents)
        results: list[dict[Phrase, float]] = []
        for document in documents:
            runs = [run for sentence in document for run in sentence]
            counts = Counter(term.key for run in runs for term in run)
            scores: dict[Phrase, float] = {}
            for run in runs:
                for terms in ngrams(run):
                    phrase = phrase_key(terms)
                    scores[phrase] = sum(counts[key] * idf[key] for key in phrase)
            results.append(scores)
        return results
