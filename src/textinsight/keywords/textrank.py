"""TextRank keyword extraction, written from scratch (Mihalcea & Tarau, 2004).

Content words become graph nodes, linked when they appear close together. PageRank
finds the most central words, and each phrase scores the words it contains.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from ..core.graph import pagerank
from ..core.registry import register_backend
from .base import Document, KeywordBackend, Phrase, candidate_phrases

__all__ = ["TextRankKeywords"]

WINDOW = 10  # chosen on Inspec train and validation documents


def build_graph(document: Document, window: int = WINDOW) -> dict[str, dict[str, float]]:
    """Link words that occur within ``window`` content words of each other."""
    graph: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for sentence in document:
        keys = [term.key for run in sentence for term in run]
        for index, key in enumerate(keys):
            graph[key]  # creating the entry gives a lone word its node
            for other in keys[index + 1 : index + window]:
                if other != key:
                    graph[key][other] += 1.0
                    graph[other][key] += 1.0
    return {key: dict(edges) for key, edges in graph.items()}


@register_backend
class TextRankKeywords(KeywordBackend):
    """Graph-based keywords; needs no training data or other documents."""

    name = "textrank"
    priority = 10

    def score(self, documents: Sequence[Document]) -> list[dict[Phrase, float]]:
        return [self._score_one(document) for document in documents]

    def _score_one(self, document: Document) -> dict[Phrase, float]:
        ranks = pagerank(build_graph(document))
        # Dividing by the square root stops long phrases winning just by being long.
        return {
            phrase: sum(ranks[key] for key in phrase) / math.sqrt(len(phrase))
            for phrase in candidate_phrases(document)
        }
