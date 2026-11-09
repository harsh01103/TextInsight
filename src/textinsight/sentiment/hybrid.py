"""Optional, most accurate backend: a Hinglish transformer blended with the classifier.

Install with ``pip install "textinsight[transformers]"``. It must be requested
with ``backend="hybrid"`` because first use downloads a 438 MB model, and
``backend="auto"`` should never cause that without being asked.

The encoder, l3cube-pune/hing-bert (CC-BY-4.0, Joshi 2022), is not fine-tuned.
Only a small classifier on top of its embeddings was trained, and that classifier
ships inside the package.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from importlib import resources
from typing import Any

from ..core.base import Backend
from ..core.config import config
from ..core.exceptions import ModelDownloadError
from ..core.registry import register_backend
from ..core.results import SentimentResult
from ..preprocess.cleaner import clean
from .classical import LinearModel, lexicon_backoff, load_bundled_model, softmax

__all__ = ["HybridSentiment"]

HEAD_FILE = "sentiment_hybrid_head.json.gz"
ENCODER = "l3cube-pune/hing-bert"
ENCODER_REVISION = "62cf31a13d65ae3b846456cede60c7d965e5fdd4"
MAX_LENGTH = 64
BATCH_SIZE = 32
TRANSFORMER_WEIGHT = 0.5  # chosen by 5-fold cross-validation


def _load_head() -> dict[str, Any]:
    resource = resources.files("textinsight._data").joinpath(HEAD_FILE)
    with resource.open("rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def head_probabilities(head: dict[str, Any], vector: Sequence[float]) -> list[float]:
    """Apply the trained scaler and classifier to one embedding."""
    scaled = [
        (value - mean) / scale
        for value, mean, scale in zip(vector, head["mean"], head["scale"], strict=True)
    ]
    scores = [
        intercept + sum(weight * value for weight, value in zip(row, scaled, strict=True))
        for row, intercept in zip(head["coefficients"], head["intercepts"], strict=True)
    ]
    return softmax(scores)


@register_backend
class HybridSentiment(Backend[SentimentResult]):
    """Averages transformer-embedding and word-feature predictions."""

    task = "sentiment"
    name = "hybrid"
    languages = frozenset({"en", "hi-en"})
    requires = ("torch", "transformers")
    extra = "transformers"
    priority = 5  # below the bundled model, so "auto" never triggers the large download

    def __init__(self) -> None:
        super().__init__()
        self._head: dict[str, Any] = {}
        self._classical: LinearModel | None = None
        self._tokenizer: Any = None
        self._encoder: Any = None

    def load(self) -> None:
        self._head = _load_head()
        self._classical = load_bundled_model()
        self._load_encoder()

    def _load_encoder(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        settings = {"revision": ENCODER_REVISION, "local_files_only": config.offline}
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(ENCODER, **settings)
            self._encoder = AutoModel.from_pretrained(
                ENCODER, use_safetensors=True, **settings
            ).eval()
        except OSError as error:
            hint = " Offline mode is on and it is not cached." if config.offline else ""
            raise ModelDownloadError(f"Could not load {ENCODER}.{hint} {error}") from error

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        import torch

        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            encoded = self._tokenizer(
                list(texts[start : start + BATCH_SIZE]),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            with torch.inference_mode():
                hidden = self._encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            vectors.extend(((hidden * mask).sum(1) / mask.sum(1)).tolist())
        return vectors

    def predict(self, texts: Sequence[str]) -> list[SentimentResult]:
        classical = self._classical
        assert classical is not None  # __call__ runs load() first
        prepared = [
            clean(text, mentions=True, html_tags=False, unicode_form=False) for text in texts
        ]
        results = []
        for text, vector in zip(texts, self._embed(prepared), strict=True):
            transformer = dict(
                zip(self._head["classes"], head_probabilities(self._head, vector), strict=True)
            )
            words = dict(zip(classical.classes, classical.probabilities(text), strict=True))
            blended = {
                name: TRANSFORMER_WEIGHT * transformer[name]
                + (1.0 - TRANSFORMER_WEIGHT) * words[name]
                for name in classical.classes
            }
            label = max(blended, key=lambda name: blended[name])
            score = blended["positive"] - blended["negative"]
            label, score = lexicon_backoff(text, label, score)
            results.append(SentimentResult(label=label, score=round(score, 4), backend=self.name))
        return results
