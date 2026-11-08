"""Language detection for English, Hindi and code-mixed Hinglish.

Labels returned for a text:

* ``"en"``: English.
* ``"hi"``: Hindi written mainly in Devanagari script.
* ``"hi-en"``: code-mixed text or romanized Hindi. Romanized Hindi counts here
  even with no English words in it, because it needs the Hinglish pipeline.
* ``"und"``: undetermined, for example only emojis or an unsupported script.

Each word is labelled by script, then by word lists, then by a character n-gram
classifier for words not in either list. The text label comes from those votes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import overload

from .._data.english import ENGLISH_COMMON
from ..core.base import ANY_LANGUAGE, Backend
from ..core.registry import AUTO, get_backend, register_backend
from ..core.results import LanguageResult
from ..core.validation import ensure_texts
from ..preprocess._patterns import DEVANAGARI_LETTERS
from ..preprocess.hinglish import HINGLISH_WORDS, squeeze_repeats
from ..preprocess.tokenizer import iter_tokens
from .ngram import CharNgramClassifier

__all__ = ["HeuristicLanguageDetector", "detect_language"]

EN, HI, MIXED, UNDETERMINED = "en", "hi", "hi-en", "und"
OTHER, AMBIGUOUS = "other", "ambiguous"

_DEVANAGARI = re.compile(f"[{DEVANAGARI_LETTERS}]")
_MIN_CLASSIFIED_LENGTH = 3
_CLASSIFIER_WEIGHT = 0.5  # a guess from letter patterns counts for less than a known word
_MIN_MARGIN = 0.12
_MIXED_THRESHOLD = 0.2  # minimum share of Hindi evidence for a text to count as mixed
_DEVANAGARI_THRESHOLD = 0.85
_AMBIGUOUS_ONLY_SCORE = 0.25


def _lexicon_label(word: str) -> str | None:
    """Label from the word lists, trying the word and its de-elongated forms."""
    for form in (word, squeeze_repeats(word, 2), squeeze_repeats(word, 1)):
        in_english, in_hindi = form in ENGLISH_COMMON, form in HINGLISH_WORDS
        if in_english and in_hindi:
            return AMBIGUOUS
        if in_english:
            return EN
        if in_hindi:
            return HI
    return None


@register_backend
class HeuristicLanguageDetector(Backend[LanguageResult]):
    """Rule-and-lexicon detector with an n-gram fallback; needs no model download."""

    task = "langid"
    name = "heuristic"
    languages = frozenset({ANY_LANGUAGE})
    priority = 10

    def __init__(self) -> None:
        super().__init__()
        self._classifier = CharNgramClassifier()

    def load(self) -> None:
        unambiguous_hindi = HINGLISH_WORDS - ENGLISH_COMMON
        unambiguous_english = ENGLISH_COMMON - HINGLISH_WORDS
        samples = [(w, EN) for w in unambiguous_english] + [(w, HI) for w in unambiguous_hindi]
        self._classifier.fit(samples)

    def predict(self, texts: Sequence[str]) -> list[LanguageResult]:
        return [self._detect(text) for text in texts]

    def _label_word(self, word: str) -> tuple[str, float]:
        """Return the word's label and how much its vote counts."""
        if _DEVANAGARI.search(word):
            return HI, 1.0
        if not word.isascii():
            return OTHER, 0.0
        lowered = word.lower()
        known = _lexicon_label(lowered)
        if known is not None:
            return known, 0.0 if known == AMBIGUOUS else 1.0
        if len(lowered) < _MIN_CLASSIFIED_LENGTH or not any(c.isalpha() for c in lowered):
            return AMBIGUOUS, 0.0
        label, margin = self._classifier.predict(lowered)
        if margin < _MIN_MARGIN:
            return AMBIGUOUS, 0.0
        return label, _CLASSIFIER_WEIGHT

    def _detect(self, text: str) -> LanguageResult:
        labels: list[str] = []
        votes = {EN: 0.0, HI: 0.0}
        devanagari = 0.0
        ambiguous = 0
        for token, kind in iter_tokens(text):
            if kind != "word":
                labels.append(OTHER)
                continue
            label, weight = self._label_word(token)
            labels.append(label)
            if label in votes:
                votes[label] += weight
                devanagari += 1.0 if _DEVANAGARI.search(token) else 0.0
            elif label == AMBIGUOUS:
                ambiguous += 1

        evidence = votes[EN] + votes[HI]
        hindi_share = votes[HI] / evidence if evidence else 0.0
        majority = HI if hindi_share >= 0.5 else EN
        resolved = tuple(majority if label == AMBIGUOUS else label for label in labels)

        if not evidence:
            if ambiguous:
                return LanguageResult(EN, _AMBIGUOUS_ONLY_SCORE, resolved)
            return LanguageResult(UNDETERMINED, 0.0, resolved)

        coverage = evidence / (evidence + ambiguous)
        if devanagari / evidence >= _DEVANAGARI_THRESHOLD:
            return LanguageResult(HI, round(hindi_share * coverage, 3), resolved)
        if hindi_share < _MIXED_THRESHOLD:
            return LanguageResult(EN, round((1 - hindi_share) * coverage, 3), resolved)
        return LanguageResult(MIXED, round(coverage, 3), resolved)


@overload
def detect_language(text: str, *, backend: str = ...) -> LanguageResult: ...
@overload
def detect_language(text: list[str], *, backend: str = ...) -> list[LanguageResult]: ...


def detect_language(
    text: str | list[str], *, backend: str = AUTO
) -> LanguageResult | list[LanguageResult]:
    """Detect whether text is English, Hindi or code-mixed Hinglish.

    ``score`` is a heuristic strength of evidence in [0, 1], not a probability.
    ``token_languages`` has one label per token of ``tokenize(text)``.

    >>> detect_language("yeh phone mast hai but battery bekaar").language
    'hi-en'
    """
    texts, single = ensure_texts(text)
    results: list[LanguageResult] = get_backend("langid", backend)(texts)
    return results[0] if single else results
