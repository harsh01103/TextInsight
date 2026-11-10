"""Optional sentiment backend that asks Claude. It sends your text to Anthropic.

Install with ``pip install "textinsight-nlp[llm]"`` and request it with
``backend="llm"``. ``backend="auto"`` never picks it, because each call costs money.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from ..core.base import Backend
from ..core.llm import structured_reply
from ..core.registry import register_backend
from ..core.results import SentimentResult

__all__ = ["LLMSentiment"]

_WORKERS = 4
_SYSTEM = (
    "You classify the sentiment of English and Hinglish (romanized Hindi-English) text. "
    "Give a label and a score from -1 (most negative) to 1 (most positive)."
)
_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]},
        "score": {"type": "number"},
    },
    "required": ["label", "score"],
    "additionalProperties": False,
}


@register_backend
class LLMSentiment(Backend[SentimentResult]):
    """Sentiment from Claude. Not benchmarked here, because every call costs money."""

    task = "sentiment"
    name = "llm"
    languages = frozenset({"en", "hi-en"})
    requires = ("anthropic",)
    extra = "llm"
    priority = 1

    def _one(self, text: str) -> SentimentResult:
        answer = structured_reply(_SYSTEM, text, _SCHEMA)
        score = max(-1.0, min(1.0, float(answer["score"])))
        return SentimentResult(label=answer["label"], score=round(score, 4), backend=self.name)

    def predict(self, texts: Sequence[str]) -> list[SentimentResult]:
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            return list(pool.map(self._one, texts))
