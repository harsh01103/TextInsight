"""Optional fast English sentiment using a small quantized ONNX model.

Install with ``pip install "textinsight-nlp[onnx]"`` and request it with
``backend="onnx"``. The first use downloads a 68 MB model. It was trained on English
movie reviews and has only positive and negative classes, so "neutral" here means
the model was unsure. It does not support Hinglish.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.base import Backend
from ..core.models import ModelSpec, models
from ..core.registry import register_backend
from ..core.results import SentimentResult

__all__ = ["OnnxSentiment"]

_REPOSITORY = "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
_REVISION = "0b6928efcb76139cae2c6881d49cda67fe119f42"
_BASE = f"https://huggingface.co/{_REPOSITORY}/resolve/{_REVISION}"
MODEL_FILE = ModelSpec(
    name="distilbert-sst2-onnx",
    version=_REVISION[:12],
    url=f"{_BASE}/onnx/model_quantized.onnx",
    sha256="79b45c7fd8c2673aa53615aba07caeb2de798f63a95a2393a365982429c94f7f",
)
TOKENIZER_FILE = ModelSpec(
    name="distilbert-sst2-onnx",
    version=_REVISION[:12],
    url=f"{_BASE}/tokenizer.json",
    sha256="d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
)
MAX_LENGTH = 256
BATCH_SIZE = 32
NEUTRAL_BAND = 0.95  # chosen on SentiMix dev; the model is very confident on most text


@register_backend
class OnnxSentiment(Backend[SentimentResult]):
    """Small English transformer run with ONNX Runtime, without PyTorch."""

    task = "sentiment"
    name = "onnx"
    languages = frozenset({"en"})
    requires = ("onnxruntime", "tokenizers")
    extra = "onnx"
    priority = 3  # below the bundled models, so "auto" never triggers the download

    def __init__(self) -> None:
        super().__init__()
        self._session: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        self._session = onnxruntime.InferenceSession(
            str(models.fetch(MODEL_FILE)), providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(models.fetch(TOKENIZER_FILE)))
        self._tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self._tokenizer.enable_padding()

    def _positive_margin(self, batch: list[str]) -> list[float]:
        import numpy as np

        encoded = self._tokenizer.encode_batch(batch)
        inputs = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
        }
        logits = self._session.run(None, inputs)[0]
        exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        return [float(value) for value in probabilities[:, 1] - probabilities[:, 0]]

    def predict(self, texts: Sequence[str]) -> list[SentimentResult]:
        margins: list[float] = []
        for start in range(0, len(texts), BATCH_SIZE):
            margins.extend(self._positive_margin(list(texts[start : start + BATCH_SIZE])))
        results = []
        for margin in margins:
            label = "positive" if margin > 0 else "negative"
            if abs(margin) < NEUTRAL_BAND:
                label = "neutral"
            results.append(SentimentResult(label=label, score=round(margin, 4), backend=self.name))
        return results
