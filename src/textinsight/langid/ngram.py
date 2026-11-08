"""A small character n-gram Naive Bayes classifier, implemented from scratch."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Iterator

__all__ = ["CharNgramClassifier"]


class CharNgramClassifier:
    """Classifies single words by their letter patterns.

    Each word is broken into overlapping character n-grams, with ``^`` and ``$``
    marking its boundaries so that word beginnings and endings count as features.
    Classes are given equal priors, so an unbalanced word list does not bias
    predictions. Unseen n-grams are handled with add-``alpha`` smoothing.
    """

    def __init__(self, sizes: tuple[int, ...] = (2, 3, 4), alpha: float = 0.5) -> None:
        self.sizes = sizes
        self.alpha = alpha
        self._counts: dict[str, Counter[str]] = {}
        self._totals: dict[str, int] = {}
        self._vocabulary_size = 0

    def _ngrams(self, word: str) -> Iterator[str]:
        marked = f"^{word.lower()}$"
        for size in self.sizes:
            for i in range(len(marked) - size + 1):
                yield marked[i : i + size]

    def fit(self, samples: Iterable[tuple[str, str]]) -> CharNgramClassifier:
        """Train from ``(word, label)`` pairs."""
        self._counts = {}
        for word, label in samples:
            self._counts.setdefault(label, Counter()).update(self._ngrams(word))
        if len(self._counts) < 2:
            raise ValueError("CharNgramClassifier needs samples from at least two classes")
        self._totals = {label: sum(c.values()) for label, c in self._counts.items()}
        vocabulary: set[str] = set()
        for counts in self._counts.values():
            vocabulary.update(counts)
        self._vocabulary_size = len(vocabulary)
        return self

    def predict(self, word: str) -> tuple[str, float]:
        """Return ``(label, margin)``.

        ``margin`` is the average per-n-gram log-likelihood lead of the best class
        over the runner-up. Larger means more decisive; it is not a probability.
        """
        if not self._counts:
            raise RuntimeError("Call fit() before predict()")
        grams = list(self._ngrams(word))
        scores = sorted(
            ((self._log_likelihood(grams, label), label) for label in self._counts),
            reverse=True,
        )
        (best_score, best_label), (second_score, _) = scores[0], scores[1]
        return best_label, (best_score - second_score) / max(len(grams), 1)

    def _log_likelihood(self, grams: list[str], label: str) -> float:
        counts = self._counts[label]
        denominator = self._totals[label] + self.alpha * self._vocabulary_size
        return math.fsum(math.log((counts[g] + self.alpha) / denominator) for g in grams)
