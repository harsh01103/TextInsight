"""A scikit-learn transformer that turns texts into numeric features.

Install with ``pip install "textinsight-nlp[sklearn]"``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..sentiment import sentiment
from ..stats import stats

__all__ = ["FEATURE_NAMES", "TextInsightFeatures"]

FEATURE_NAMES = (
    "sentiment_score",
    "words",
    "sentences",
    "type_token_ratio",
    "average_word_length",
    "average_sentence_length",
)


class TextInsightFeatures(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Sentiment and statistics features, for use inside a scikit-learn pipeline.

    Args:
        language: Treat all input as this language code instead of detecting it.
    """

    def __init__(self, language: str | None = None) -> None:
        self.language = language

    def fit(self, X: Iterable[str], y: Any = None) -> TextInsightFeatures:  # noqa: N803
        """Do nothing, because these features need no training."""
        return self

    def transform(self, X: Iterable[str]) -> Any:  # noqa: N803
        """Return one row per text and one column per name in ``FEATURE_NAMES``."""
        texts = list(X)
        rows = [
            [
                mood.score,
                numbers.words,
                numbers.sentences,
                numbers.type_token_ratio,
                numbers.average_word_length,
                numbers.average_sentence_length,
            ]
            for mood, numbers in zip(
                sentiment(texts, language=self.language),
                stats(texts, language=self.language),
                strict=True,
            )
        ]
        return np.array(rows, dtype=float).reshape(len(texts), len(FEATURE_NAMES))

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        """Return the output column names."""
        return np.array(FEATURE_NAMES, dtype=object)
