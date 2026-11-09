"""Bundled sentiment classifier for tweets in Hinglish and English.

A Naive Bayes-weighted logistic regression (NB-LR) over word, character and
Hinglish spelling features. It is trained with scikit-learn (see
``training/train_sentiment.py``), but its weights are exported to JSON and applied
here in plain Python. That keeps scikit-learn out of the runtime dependencies and
avoids loading pickle files, which can execute arbitrary code.

The bundled weights derive from the SentiMix dataset (CC-BY-4.0, Patwa et al. 2020).
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from importlib import resources
from typing import Any

from ..core.base import Backend
from ..core.registry import register_backend
from ..core.results import SentimentResult
from .features import FeatureOptions, extract_features
from .lexicon import score_text

__all__ = ["ClassicalSentiment", "LinearModel", "lexicon_backoff", "load_bundled_model", "softmax"]

MODEL_FILE = "sentiment_hinglish.json.gz"

# A model trained on tweets reads short, plain statements as neutral far too often,
# so a clear lexicon verdict wins there. See the model card for the measurements.
BACKOFF_THRESHOLD = 0.3
BACKOFF_MAX_WORDS = 12
_TWEET_MARKS = re.compile(r"[#\U0001F300-\U0001FAFF\u2600-\u27BF]")


def softmax(scores: Sequence[float]) -> list[float]:
    """Convert raw scores to probabilities that sum to one."""
    peak = max(scores)
    exponentials = [math.exp(score - peak) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def lexicon_backoff(text: str, label: str, score: float) -> tuple[str, float]:
    """Replace a "neutral" prediction with a clear lexicon verdict on short, plain text."""
    if label != "neutral" or len(text.split()) > BACKOFF_MAX_WORDS or _TWEET_MARKS.search(text):
        return label, score
    lexicon_score, lexicon_label = score_text(text)
    if lexicon_label == "neutral" or abs(lexicon_score) < BACKOFF_THRESHOLD:
        return label, score
    return lexicon_label, lexicon_score


class LinearModel:
    """A linear classifier over sublinear TF-IDF features."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.classes: list[str] = payload["classes"]
        settings: dict[str, Any] = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in payload["options"].items()
        }
        self.options = FeatureOptions(**settings)
        self.vocabulary: dict[str, int] = {
            feature: index for index, feature in enumerate(payload["features"])
        }
        self.idf: list[float] = payload["idf"]
        self.coefficients: list[list[float]] = payload["coefficients"]
        self.intercepts: list[float] = payload["intercepts"]

    def probabilities(self, text: str) -> list[float]:
        """Return one score per class, in the order of ``classes``, summing to one."""
        counts = Counter(
            self.vocabulary[f] for f in extract_features(text, self.options) if f in self.vocabulary
        )
        weights = {i: (1.0 + math.log(n)) * self.idf[i] for i, n in counts.items()}
        norm = math.sqrt(sum(v * v for v in weights.values()))
        if norm:
            weights = {i: v / norm for i, v in weights.items()}
        return softmax(
            [
                intercept + sum(row[i] * v for i, v in weights.items())
                for row, intercept in zip(self.coefficients, self.intercepts, strict=True)
            ]
        )


@lru_cache(maxsize=1)
def load_bundled_model() -> LinearModel:
    """Load the model shipped inside the package (cached after the first call)."""
    resource = resources.files("textinsight._data").joinpath(MODEL_FILE)
    with resource.open("rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as handle:
        return LinearModel(json.load(handle))


@register_backend
class ClassicalSentiment(Backend[SentimentResult]):
    """Classifier trained on real code-mixed tweets; bundled, so it needs no download."""

    task = "sentiment"
    name = "classical"
    # English is included because the classifier beat the lexicon on English-looking
    # tweets in the dev split. It has only been measured on tweets.
    languages = frozenset({"en", "hi-en"})
    priority = 20

    def __init__(self) -> None:
        super().__init__()
        self._model: LinearModel | None = None

    def load(self) -> None:
        self._model = load_bundled_model()

    def predict(self, texts: Sequence[str]) -> list[SentimentResult]:
        model = self._model
        assert model is not None  # __call__ runs load() first
        results = []
        for text in texts:
            probabilities = dict(zip(model.classes, model.probabilities(text), strict=True))
            label = max(probabilities, key=lambda name: probabilities[name])
            score = probabilities["positive"] - probabilities["negative"]
            label, score = lexicon_backoff(text, label, score)
            results.append(SentimentResult(label=label, score=round(score, 4), backend=self.name))
        return results
