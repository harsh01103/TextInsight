"""Optional abstractive summaries: a small T5 model writes new sentences.

Install with ``pip install "textinsight-nlp[transformers]"`` and request it with
``method="abstractive"``. The first use downloads a 242 MB model. It is English only
and, as with any generated summary, it can state things the text does not say.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from ..core.base import Backend
from ..core.config import config
from ..core.exceptions import ModelDownloadError
from ..core.registry import register_backend
from ..core.results import SummaryResult
from ..preprocess.tokenizer import sentences as split_sentences

__all__ = ["AbstractiveSummarizer"]

MODEL = "Falconsai/text_summarization"
REVISION = "6e505f907968c4a9360773ff57885cdc6dca4bfd"
_PREFIX = "summarize: "
_MAX_INPUT_TOKENS = 512
_MAX_SUMMARY_TOKENS = 120
_MIN_SUMMARY_TOKENS = 20
_BATCH_SIZE = 8
DEFAULT_RATIO = 0.3


@register_backend
class AbstractiveSummarizer(Backend[SummaryResult]):
    """Writes a new summary, so ``sentence_indices`` is always empty."""

    task: ClassVar[str] = "summarize"
    name: ClassVar[str] = "abstractive"
    languages: ClassVar[frozenset[str]] = frozenset({"en"})
    requires: ClassVar[tuple[str, ...]] = ("torch", "transformers")
    extra: ClassVar[str | None] = "transformers"
    priority: ClassVar[int] = 1  # lowest, so "auto" never triggers the download

    def __init__(self) -> None:
        super().__init__()
        self._tokenizer: Any = None
        self._model: Any = None

    def load(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        settings = {"revision": REVISION, "local_files_only": config.offline}
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL, **settings)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL, use_safetensors=True, **settings
            ).eval()
        except OSError as error:
            hint = " Offline mode is on and it is not cached." if config.offline else ""
            raise ModelDownloadError(f"Could not load {MODEL}.{hint} {error}") from error

    def _generate(self, batch: list[str]) -> list[str]:
        import torch

        encoded = self._tokenizer(
            [_PREFIX + text for text in batch],
            padding=True,
            truncation=True,
            max_length=_MAX_INPUT_TOKENS,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                max_new_tokens=_MAX_SUMMARY_TOKENS,
                min_new_tokens=_MIN_SUMMARY_TOKENS,
                num_beams=2,
            )
        return [
            str(text) for text in self._tokenizer.batch_decode(output, skip_special_tokens=True)
        ]

    def summarize(
        self,
        texts: Sequence[str],
        *,
        ratio: float,
        max_sentences: int | None,
        language: str,
    ) -> list[SummaryResult]:
        """Summarize each text; ``ratio`` does not apply to generated summaries."""
        self.ensure_loaded()
        written: list[str] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            written.extend(self._generate(list(texts[start : start + _BATCH_SIZE])))
        if max_sentences is not None:
            written = [" ".join(split_sentences(text)[:max_sentences]) for text in written]
        return [SummaryResult(summary=text, method=self.name) for text in written]

    def predict(self, texts: Sequence[str]) -> list[SummaryResult]:
        return self.summarize(texts, ratio=DEFAULT_RATIO, max_sentences=None, language="en")
