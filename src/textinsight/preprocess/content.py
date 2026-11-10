"""Content words grouped into runs, shared by keyword extraction and summarization.

Stopwords, negations, numbers and punctuation end a run. In Hinglish text each word
gets a phonetic key, so spellings such as "khana" and "khaana" count as one word.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._data.english import ENGLISH_COMMON, GENERIC_WORDS_EN, LY_NOUNS, STOPWORDS_EN
from .._data.hinglish import FUNCTION_WORDS_HINGLISH, STOPWORDS_HINGLISH
from .._data.sentiment_lexicon import ENGLISH as OPINIONS_EN
from .._data.sentiment_lexicon import HINGLISH as OPINIONS_HINGLISH
from .hinglish import canonical_form, phonetic_key
from .tokenizer import iter_tokens, sentences

__all__ = ["Term", "content_runs", "parse_document"]

_NEGATIONS = frozenset({"not", "no", "never", "nahi", "nahin", "nhi", "mat", "na"})
_MIN_WORD_LENGTH = 2
_STOPWORDS = {
    "en": STOPWORDS_EN | GENERIC_WORDS_EN,
    # Opinion words describe a topic instead of naming it, which matters in short reviews.
    "hi-en": STOPWORDS_EN
    | GENERIC_WORDS_EN
    | STOPWORDS_HINGLISH
    | FUNCTION_WORDS_HINGLISH
    | frozenset(OPINIONS_EN)
    | frozenset(OPINIONS_HINGLISH),
}


@dataclass(frozen=True)
class Term:
    """A content word as written (``surface``) and its matching form (``key``)."""

    surface: str
    key: str


def _singular(word: str) -> str:
    if len(word) <= 3 or not word.endswith("s") or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    return word[:-1]


def _key(word: str, language: str) -> str:
    if language != "hi-en":
        return _singular(word)
    singular = _singular(word)
    # Only English plurals lose their "s": "khaas" and "paas" are Hindi words.
    if singular != word and singular in ENGLISH_COMMON:
        return singular
    # The phonetic key also merges spellings the dictionary does not list (kitab, kitaab).
    return phonetic_key(canonical_form(word))


def _term(token: str, kind: str, language: str) -> Term | None:
    if kind not in {"word", "hashtag"}:
        return None
    word = token.lstrip("#").lower()
    if len(word) < _MIN_WORD_LENGTH or word in _NEGATIONS or not any(c.isalpha() for c in word):
        return None
    if word.endswith("ly") and word not in LY_NOUNS:
        return None  # adverbs describe how something happens, not what it is about
    # Variant spellings of a stopword or negation, such as "acha", are filtered too.
    known = canonical_form(word) if language == "hi-en" else _singular(word)
    if word in _STOPWORDS[language] or known in _STOPWORDS[language] or known in _NEGATIONS:
        return None
    return Term(word, _key(word, language))


def content_runs(text: str, language: str) -> list[list[Term]]:
    """Return unbroken runs of content words."""
    runs: list[list[Term]] = []
    current: list[Term] = []
    for token, kind in iter_tokens(text):
        term = _term(token, kind, language)
        if term is not None:
            current.append(term)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def parse_document(text: str, language: str) -> list[list[list[Term]]]:
    """Return the content runs of each sentence."""
    return [content_runs(sentence, language) for sentence in sentences(text)]
