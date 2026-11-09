"""Rule-and-lexicon sentiment for English and Hinglish, needing no model.

The method follows the idea behind VADER: add up word scores, adjust for
intensifiers, negation, capitals, exclamation marks and contrast words, then
squash the total into [-1, 1]. Hindi differs from English in one important way
handled here: negation usually comes after the word ("accha nahi hai").
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .._data.sentiment_lexicon import (
    CONTRAST,
    EMOJI,
    EMOTICONS,
    ENGLISH,
    HINGLISH,
    INTENSIFIERS,
    NEGATIONS_AFTER,
    NEGATIONS_BEFORE,
)
from ..core.base import Backend
from ..core.registry import register_backend
from ..core.results import SentimentResult
from ..preprocess.cleaner import clean
from ..preprocess.hinglish import canonical_form
from ..preprocess.normalizer import reduce_elongation
from ..preprocess.tokenizer import iter_tokens

__all__ = ["LexiconSentiment", "score_text"]

_NEGATION_FACTOR = -0.74
_NEGATION_WINDOW_BEFORE = 3
_NEGATION_WINDOW_AFTER = 2
_CAPS_BOOST = 0.7
_EXCLAMATION_BOOST = 0.2
_MAX_EXCLAMATIONS = 4
_BEFORE_CONTRAST = 0.5
_AFTER_CONTRAST = 1.5
_SQUASH = 15.0
_NEUTRAL_BAND = 0.05
_MIXED_BAND = 0.3
_MIXED_MIN_SIDE = 1.0
_KINDS = {"word", "hashtag", "emoji", "emoticon"}


def _normalize_token(token: str, kind: str) -> str:
    if kind == "hashtag":
        token = token[1:]
    if kind in {"word", "hashtag"}:
        return canonical_form(reduce_elongation(token.lower()))
    return token


def _base_valence(token: str, kind: str) -> float:
    if kind == "emoji":
        return EMOJI.get(token, 0.0)
    if kind == "emoticon":
        return EMOTICONS.get(token, 0.0)
    return ENGLISH.get(token, HINGLISH.get(token, 0.0))


def _adjusted_valence(words: list[str], index: int, base: float, raw: str, kind: str) -> float:
    valence = base
    sign = 1.0 if base > 0 else -1.0
    if index > 0 and words[index - 1] in INTENSIFIERS:
        valence += sign * INTENSIFIERS[words[index - 1]]
    if kind == "word" and raw.isupper() and len(raw) > 1:
        valence += sign * _CAPS_BOOST
    before = words[max(0, index - _NEGATION_WINDOW_BEFORE) : index]
    after = words[index + 1 : index + 1 + _NEGATION_WINDOW_AFTER]
    if NEGATIONS_BEFORE.intersection(before) or NEGATIONS_AFTER.intersection(after):
        valence *= _NEGATION_FACTOR
    return valence


def score_text(text: str) -> tuple[float, str]:
    """Return ``(score, label)``, with the score in [-1, 1]."""
    tokens = [(t, k) for t, k in iter_tokens(clean(text)) if k in _KINDS or t == "!"]
    exclamations = sum(1 for t, _ in tokens if t == "!")
    tokens = [(t, k) for t, k in tokens if k in _KINDS]
    words = [_normalize_token(t, k) for t, k in tokens]

    valences = []
    for index, (raw, kind) in enumerate(tokens):
        base = _base_valence(words[index], kind)
        valences.append(_adjusted_valence(words, index, base, raw, kind) if base else 0.0)

    # What follows "but" usually matters more than what comes before it.
    pivot = next((i for i, word in enumerate(words) if word in CONTRAST), None)
    if pivot is not None:
        valences = [
            v * (_BEFORE_CONTRAST if i < pivot else _AFTER_CONTRAST) for i, v in enumerate(valences)
        ]

    positive = sum(v for v in valences if v > 0)
    negative = -sum(v for v in valences if v < 0)
    total = positive - negative
    if total:
        total += math.copysign(min(exclamations, _MAX_EXCLAMATIONS) * _EXCLAMATION_BOOST, total)
    score = max(-1.0, min(1.0, total / math.sqrt(total * total + _SQUASH)))

    if min(positive, negative) >= _MIXED_MIN_SIDE and abs(score) < _MIXED_BAND:
        return score, "mixed"
    if score >= _NEUTRAL_BAND:
        return score, "positive"
    if score <= -_NEUTRAL_BAND:
        return score, "negative"
    return score, "neutral"


@register_backend
class LexiconSentiment(Backend[SentimentResult]):
    """Fast, transparent baseline for English and Hinglish."""

    task = "sentiment"
    name = "lexicon"
    languages = frozenset({"en", "hi-en", "und"})
    priority = 10

    def predict(self, texts: Sequence[str]) -> list[SentimentResult]:
        results = []
        for text in texts:
            score, label = score_text(text)
            results.append(SentimentResult(label=label, score=round(score, 4), backend=self.name))
        return results
