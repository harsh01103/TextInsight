"""Aspect-based sentiment: what is being praised or criticized in a text.

The text is cut into clauses, and each clause's opinion goes to the things named in
it. An aspect is reported only if some clause expresses an opinion about it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from .._data.sentiment_lexicon import ENGLISH, HINGLISH
from ..core.base import Backend
from ..core.registry import register_backend
from ..core.results import Aspect, AspectResult
from ..preprocess.content import Term, content_runs
from ..preprocess.tokenizer import sentences
from ..sentiment.lexicon import score_text

__all__ = ["RuleAspects", "split_clauses"]

_MAX_ASPECT_WORDS = 3
_MIXED_MIN_SHARE = 0.25  # the smaller side needs this share of the opinions to count as mixed
_NEUTRAL_BAND = 0.05

# Opinion words common in reviews that the sentiment lexicon does not list. They are
# kept here because changing that lexicon would change the published sentiment models.
_EXTRA_OPINIONS = {
    "delicious": 0.6,
    "tasty": 0.6,
    "swaadisht": 0.6,
    "swadisht": 0.6,
    "soft": 0.4,
    "kamzor": -0.5,
}
_OPINION_WORDS = frozenset(ENGLISH) | frozenset(HINGLISH) | frozenset(_EXTRA_OPINIONS)
# Common English verbs. Without part-of-speech tags they could look like aspects.
_VERBS = frozenset(
    """
    lasts last works work worked looks look looked feels feel felt seems seem seemed
    gets get got goes go went came come comes took take takes gave give gives keeps
    keep kept stopped stops broke breaks arrived loved love hated hate liked like
    bought buy need needs want wants
    """.split()
)
_CONTRAST = re.compile(r"\b(?:but|however|although|though|yet|lekin|magar|parantu|kintu)\b", re.I)
_SOFT_BREAK = re.compile(r"\b(?:and|aur|or)\b|[,;:]", re.I)
# "phone ki battery" is about the battery, so the owner before ka/ki/ke is removed.
_POSSESSIVE = re.compile(
    r"\b[\w-]+\s+(?:ka|ki|ke)\s+(?!liye\b|baad\b|pehle\b|saath\b|paas\b|bare\b|baare\b"
    r"|upar\b|niche\b|neeche\b|andar\b|bahar\b|wajah\b|karan\b|kaaran\b)",
    re.I,
)


def _opinion_score(text: str) -> float:
    score, label = score_text(text)
    if label != "neutral":
        return score
    words = [word.strip(".,!?").lower() for word in text.split()]
    return next((_EXTRA_OPINIONS[word] for word in words if word in _EXTRA_OPINIONS), score)


def _has_opinion(text: str) -> bool:
    return abs(_opinion_score(text)) >= _NEUTRAL_BAND


def split_clauses(sentence: str) -> list[str]:
    """Split at contrast words, and at "and" or commas when each side has its own opinion."""
    clauses: list[str] = []
    for piece in _CONTRAST.split(sentence):
        parts = [part for part in _SOFT_BREAK.split(piece) if part.strip()]
        if len(parts) > 1 and all(_has_opinion(part) for part in parts):
            clauses.extend(part.strip() for part in parts)
        elif piece.strip():
            clauses.append(piece.strip())
    return clauses


def _is_aspect_word(term: Term) -> bool:
    return (
        term.surface not in _OPINION_WORDS
        and term.key not in _OPINION_WORDS
        and term.surface not in _VERBS
    )


def _candidates(clause: str, language: str) -> list[list[Term]]:
    text = _POSSESSIVE.sub(" ", clause) if language == "hi-en" else clause
    found: list[list[Term]] = []
    for run in content_runs(text, language):
        current: list[Term] = []
        for term in [*run, None]:
            if term is not None and _is_aspect_word(term):
                current.append(term)
            else:
                if current:
                    found.append(current[-_MAX_ASPECT_WORDS:])
                current = []
    return found


@dataclass
class _Tally:
    scores: list[float] = field(default_factory=list)
    surfaces: Counter[str] = field(default_factory=Counter)

    def label(self) -> str:
        positive = sum(1 for score in self.scores if score >= _NEUTRAL_BAND)
        negative = sum(1 for score in self.scores if score <= -_NEUTRAL_BAND)
        if min(positive, negative) / (positive + negative) >= _MIXED_MIN_SHARE:
            return "mixed"
        return "positive" if positive > negative else "negative"

    def to_aspect(self) -> Aspect:
        return Aspect(
            name=self.surfaces.most_common(1)[0][0],
            label=self.label(),
            score=round(sum(self.scores) / len(self.scores), 4),
            mentions=len(self.scores),
        )


@register_backend
class RuleAspects(Backend[AspectResult]):
    """Finds aspects from word patterns and scores them with the sentiment lexicon."""

    task = "aspects"
    name = "rules"
    languages = frozenset({"en", "hi-en"})
    priority = 10

    def extract(
        self, texts: Sequence[str], *, min_mentions: int, language: str
    ) -> list[AspectResult]:
        """Return the aspects mentioned at least ``min_mentions`` times in each text."""
        self.ensure_loaded()
        return [self._extract_one(text, min_mentions, language) for text in texts]

    def predict(self, texts: Sequence[str]) -> list[AspectResult]:
        return self.extract(texts, min_mentions=1, language="hi-en")

    def _extract_one(self, text: str, min_mentions: int, language: str) -> AspectResult:
        tallies: dict[tuple[str, ...], _Tally] = {}
        for sentence in sentences(text):
            for clause in split_clauses(sentence):
                score = _opinion_score(clause)
                for terms in _candidates(clause, language):
                    tally = tallies.setdefault(tuple(t.key for t in terms), _Tally())
                    tally.scores.append(score)
                    tally.surfaces[" ".join(t.surface for t in terms)] += 1
        kept = [
            tally.to_aspect()
            for tally in tallies.values()
            if len(tally.scores) >= min_mentions
            and any(abs(score) >= _NEUTRAL_BAND for score in tally.scores)
        ]
        # sorted() is stable, so ties keep the order of first mention.
        ranked = sorted(kept, key=lambda aspect: (-aspect.mentions, -abs(aspect.score)))
        return AspectResult(aspects=tuple(ranked), method=self.name)
