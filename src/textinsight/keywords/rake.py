"""RAKE keyword extraction (Rose et al., 2010).

Each unbroken run of content words is a candidate. A word scores its degree (how
much it co-occurs inside candidates) divided by its frequency, and a candidate
scores the sum of its words.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ..core.registry import register_backend
from .base import Document, KeywordBackend, Phrase, candidate_phrases

__all__ = ["RakeKeywords"]


@register_backend
class RakeKeywords(KeywordBackend):
    """Rapid Automatic Keyword Extraction."""

    name = "rake"
    priority = 7

    def score(self, documents: Sequence[Document]) -> list[dict[Phrase, float]]:
        results: list[dict[Phrase, float]] = []
        for document in documents:
            candidates = list(candidate_phrases(document))
            frequency: Counter[str] = Counter()
            degree: Counter[str] = Counter()
            for candidate in candidates:
                for key in candidate:
                    frequency[key] += 1
                    degree[key] += len(candidate)
            results.append(
                {
                    candidate: sum(degree[key] / frequency[key] for key in candidate)
                    for candidate in candidates
                }
            )
        return results
