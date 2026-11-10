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
    "Aspect",
    "AspectResult",
    "Keyword",
    "KeywordResult",
    "LanguageResult",
    "PIIEntity",
    "PIIResult",
    "Result",
    "SentimentResult",
    "SummaryResult",
    "TextStats",
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


@dataclass(frozen=True)
class Aspect(Result):
    """Sentiment towards one thing mentioned in a text, such as "camera"."""

    name: str
    label: str
    score: float
    mentions: int

    def __post_init__(self) -> None:
        if self.label not in SENTIMENT_LABELS:
            raise ValueError(f"label must be one of {sorted(SENTIMENT_LABELS)}, got {self.label!r}")


@dataclass(frozen=True)
class AspectResult(Result):
    """Aspects found in a text, most mentioned first."""

    aspects: tuple[Aspect, ...]
    method: str

    @property
    def labels(self) -> dict[str, str]:
        """Each aspect name with its sentiment label."""
        return {aspect.name: aspect.label for aspect in self.aspects}

    def __iter__(self) -> Iterator[Aspect]:
        return iter(self.aspects)

    def __len__(self) -> int:
        return len(self.aspects)


@dataclass(frozen=True)
class TextStats(Result):
    """Counts and readability of a text.

    ``flesch_reading_ease`` is ``None`` unless the text is English, because the
    formula was designed for English. Higher scores mean easier text.
    """

    characters: int
    words: int
    sentences: int
    unique_words: int
    type_token_ratio: float
    average_word_length: float
    average_sentence_length: float
    flesch_reading_ease: float | None
    reading_time_seconds: float
    language: str


@dataclass(frozen=True)
class PIIEntity(Result):
    """One piece of personal information. Its value is left out of ``repr()``."""

    kind: str
    start: int
    end: int
    value: str = field(repr=False)


@dataclass(frozen=True)
class PIIResult(Result):
    """Text with personal information replaced by placeholders such as ``[EMAIL]``."""

    text: str
    entities: tuple[PIIEntity, ...]

    def __iter__(self) -> Iterator[PIIEntity]:
        return iter(self.entities)

    def __len__(self) -> int:
        return len(self.entities)
