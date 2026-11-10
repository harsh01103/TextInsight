"""Extractive summarization: choose the most representative sentences of a text."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import ClassVar

from ..core.base import Backend
from ..core.graph import pagerank
from ..core.registry import register_backend
from ..core.results import SummaryResult
from ..preprocess.content import content_runs
from ..preprocess.tokenizer import sentences as split_sentences

__all__ = ["DEFAULT_RATIO", "FrequencySummarizer", "SummaryBackend", "TextRankSummarizer"]

DEFAULT_RATIO = 0.3
POSITION_BIAS = 1.0  # chosen on CNN/DailyMail validation articles


class SummaryBackend(Backend[SummaryResult]):
    """Base class: subclasses only decide how sentences are scored."""

    task: ClassVar[str] = "summarize"
    languages: ClassVar[frozenset[str]] = frozenset({"en", "hi-en"})

    @abstractmethod
    def rank(self, sentences: Sequence[Sequence[str]]) -> list[float]:
        """Return one score per sentence, given each sentence's content words."""

    def summarize(
        self,
        texts: Sequence[str],
        *,
        ratio: float,
        max_sentences: int | None,
        language: str,
    ) -> list[SummaryResult]:
        """Summarize each text, keeping the chosen sentences in their original order."""
        self.ensure_loaded()
        return [self._summarize_one(text, ratio, max_sentences, language) for text in texts]

    def predict(self, texts: Sequence[str]) -> list[SummaryResult]:
        return self.summarize(texts, ratio=DEFAULT_RATIO, max_sentences=None, language="hi-en")

    def _summarize_one(
        self, text: str, ratio: float, max_sentences: int | None, language: str
    ) -> SummaryResult:
        originals = split_sentences(text)
        if not originals:
            return SummaryResult(summary="", method=self.name)
        words = [
            [term.key for run in content_runs(sentence, language) for term in run]
            for sentence in originals
        ]
        scores = self.rank(words)
        count = max(1, round(len(originals) * ratio))
        if max_sentences is not None:
            count = min(count, max_sentences)
        # sorted() is stable, so ties favour earlier sentences.
        best = sorted(range(len(originals)), key=lambda index: -scores[index])[:count]
        chosen = tuple(sorted(best))
        return SummaryResult(
            summary=" ".join(originals[index] for index in chosen),
            method=self.name,
            sentence_indices=chosen,
        )


def similarity(first: set[str], second: set[str]) -> float:
    """TextRank sentence similarity: shared words, normalized by sentence length."""
    shared = len(first & second)
    return shared / (math.log(1 + len(first)) + math.log(1 + len(second))) if shared else 0.0


@register_backend
class TextRankSummarizer(SummaryBackend):
    """Ranks sentences with PageRank over a sentence-similarity graph.

    Random jumps favour earlier sentences, because writers tend to state the main
    point near the start.
    """

    name = "textrank"
    priority = 10

    def rank(self, sentences: Sequence[Sequence[str]]) -> list[float]:
        word_sets = [set(words) for words in sentences]
        postings: defaultdict[str, list[int]] = defaultdict(list)
        for index, words in enumerate(word_sets):
            for word in words:
                postings[word].append(index)

        # Only sentences that share a word can be linked, which saves comparing every pair.
        pairs = {
            (first, second)
            for indices in postings.values()
            for position, first in enumerate(indices)
            for second in indices[position + 1 :]
        }
        graph: dict[int, dict[int, float]] = {index: {} for index in range(len(sentences))}
        for first, second in pairs:
            graph[first][second] = graph[second][first] = similarity(
                word_sets[first], word_sets[second]
            )
        weights = [1.0 / (index + 1) ** POSITION_BIAS for index in range(len(sentences))]
        total = sum(weights)
        scores = pagerank(graph, personalization={i: w / total for i, w in enumerate(weights)})
        return [scores[index] for index in range(len(sentences))]


@register_backend
class FrequencySummarizer(SummaryBackend):
    """Scores a sentence by the average document frequency of its content words."""

    name = "frequency"
    priority = 5

    def rank(self, sentences: Sequence[Sequence[str]]) -> list[float]:
        counts = Counter(word for words in sentences for word in words)
        top = max(counts.values(), default=1)
        return [
            sum(counts[word] / top for word in words) / len(words) if words else 0.0
            for words in sentences
        ]
