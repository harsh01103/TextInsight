"""Optional aspect backend that asks Claude. It sends your text to Anthropic.

Install with ``pip install "textinsight-nlp[llm]"`` and request it with
``method="llm"``. ``method="auto"`` never picks it, because each call costs money.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..core.base import Backend
from ..core.llm import structured_reply
from ..core.registry import register_backend
from ..core.results import Aspect, AspectResult

__all__ = ["LLMAspects"]

_WORKERS = 4
_SYSTEM = (
    "You find what English or Hinglish (romanized Hindi-English) text gives an opinion about, "
    "such as a phone's camera or a restaurant's service. For each, give its short name as "
    "written, a sentiment label, a score from -1 to 1, and how many times it is mentioned. "
    "Leave out anything without an opinion."
)
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "aspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "label": {"type": "string", "enum": ["positive", "negative", "mixed"]},
                    "score": {"type": "number"},
                    "mentions": {"type": "integer"},
                },
                "required": ["name", "label", "score", "mentions"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["aspects"],
    "additionalProperties": False,
}


@register_backend
class LLMAspects(Backend[AspectResult]):
    """Aspects from Claude, which needs no word lists."""

    task = "aspects"
    name = "llm"
    languages = frozenset({"en", "hi-en"})
    requires = ("anthropic",)
    extra = "llm"
    priority = 1

    def _one(self, text: str, min_mentions: int) -> AspectResult:
        answer = structured_reply(_SYSTEM, text, _SCHEMA)
        found = [
            Aspect(
                name=item["name"].lower(),
                label=item["label"],
                score=round(max(-1.0, min(1.0, float(item["score"]))), 4),
                mentions=max(1, int(item["mentions"])),
            )
            for item in answer["aspects"]
        ]
        kept = [aspect for aspect in found if aspect.mentions >= min_mentions]
        return AspectResult(aspects=tuple(kept), method=self.name)

    def extract(
        self, texts: Sequence[str], *, min_mentions: int, language: str
    ) -> list[AspectResult]:
        """Return the aspects of each text; ``language`` is not needed by the model."""
        self.ensure_loaded()
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            return list(pool.map(lambda text: self._one(text, min_mentions), texts))

    def predict(self, texts: Sequence[str]) -> list[AspectResult]:
        return self.extract(texts, min_mentions=1, language="hi-en")
