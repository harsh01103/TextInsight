"""Typed result objects returned by TextInsight functions.

All results are immutable dataclasses with a ``to_dict()`` method, so they are
easy to inspect, compare in tests, and serialize to JSON.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "SENTIMENT_LABELS",
    "Keyword",
    "KeywordResult",
    "LanguageResult",
    "Result",
    "SentimentResult",
    "SummaryResult",
]

SENTIMENT_LABELS = frozenset({"positive", "negative", "neutral", "mixed"})


@dataclass(frozen=True)
class Result:
    """Base class for every result type."""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-Python (JSON-friendly) representation."""
        return asdict(self)


@dataclass(frozen=True)
class LanguageResult(Result):
    """Detected language of a text.

    ``language`` is an ISO-style code such as ``"en"``, ``"hi"`` or ``"hi-en"``
    (code-mixed Hindi-English). ``score`` is the detector's score in [0, 1];
    it is only a probability if the detector's documentation says so.
    """

    language: str
    score: float
    token_languages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclass(frozen=True)
class SentimentResult(Result):
    """Sentiment of a text.

    ``score`` ranges from -1 (most negative) to +1 (most positive). Its exact
    meaning depends on the backend and is documented in each backend's model card.
    """

    label: str
    score: float
    backend: str
    language: str = "en"

    def __post_init__(self) -> None:
        if self.label not in SENTIMENT_LABELS:
            raise ValueError(f"label must be one of {sorted(SENTIMENT_LABELS)}, got {self.label!r}")
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [-1, 1], got {self.score}")


@dataclass(frozen=True)
class Keyword(Result):
    """One extracted keyword or keyphrase with its relevance score."""

    text: str
    score: float


@dataclass(frozen=True)
class KeywordResult(Result):
    """Ranked keywords, most relevant first."""

    keywords: tuple[Keyword, ...]
    method: str

    @property
    def texts(self) -> list[str]:
        """Just the keyword strings, in rank order."""
        return [k.text for k in self.keywords]

    def __iter__(self) -> Iterator[Keyword]:
        return iter(self.keywords)

    def __len__(self) -> int:
        return len(self.keywords)


@dataclass(frozen=True)
class SummaryResult(Result):
    """A summary plus the indices of source sentences used (for extractive methods)."""

    summary: str
    method: str
    sentence_indices: tuple[int, ...] = field(default_factory=tuple)
