"""General normalization steps: stopwords, elongation and a combined ``normalize``."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import overload

from .._data.english import STOPWORDS_EN
from .._data.hinglish import STOPWORDS_HINGLISH
from ..core.validation import ensure_choice, ensure_positive_int, ensure_texts
from .cleaner import clean
from .hinglish import canonical_form
from .tokenizer import iter_tokens

__all__ = ["normalize", "reduce_elongation", "remove_stopwords"]

_ELONGATED = re.compile(r"([^\W\d_])\1{2,}")
_LANGUAGES = ("en", "hi-en")


def reduce_elongation(text: str, max_repeat: int = 2) -> str:
    """Shorten letters repeated three or more times.

    >>> reduce_elongation("sooooo goooood")
    'soo good'
    """
    ensure_positive_int(max_repeat, name="max_repeat")
    return _ELONGATED.sub(lambda m: m.group(1) * max_repeat, text)


def remove_stopwords(
    tokens: Iterable[str],
    *,
    language: str = "en",
    keep: Iterable[str] = (),
) -> list[str]:
    """Remove stopwords from a list of tokens.

    Negations ("not", "nahi") are never in these lists, because removing them
    reverses the meaning of a sentence.

    Args:
        tokens: Tokens to filter; matching ignores case.
        language: ``"en"``, or ``"hi-en"`` to apply both lists.
        keep: Words that should survive even if they are stopwords.
    """
    chosen = ensure_choice(language, _LANGUAGES, name="language")
    stopwords = STOPWORDS_EN | STOPWORDS_HINGLISH if chosen == "hi-en" else STOPWORDS_EN
    protected = {word.lower() for word in keep}
    return [t for t in tokens if t.lower() not in stopwords or t.lower() in protected]


def _normalize_one(text: str, *, hinglish: bool, stopwords: bool, language: str) -> str:
    cleaned = reduce_elongation(clean(text, lowercase=True))
    pieces = [
        canonical_form(token) if hinglish and kind == "word" else token
        for token, kind in iter_tokens(cleaned)
    ]
    return " ".join(remove_stopwords(pieces, language=language) if stopwords else pieces)


@overload
def normalize(
    text: str, *, hinglish: bool = ..., stopwords: bool = ..., language: str = ...
) -> str: ...
@overload
def normalize(
    text: list[str], *, hinglish: bool = ..., stopwords: bool = ..., language: str = ...
) -> list[str]: ...


def normalize(
    text: str | list[str],
    *,
    hinglish: bool = False,
    stopwords: bool = False,
    language: str = "en",
) -> str | list[str]:
    """Prepare text for feature extraction.

    Cleans, lowercases, shortens elongated words and re-joins the tokens with
    single spaces. Optionally maps Hinglish spellings to canonical forms and
    removes stopwords.

    >>> normalize("Yeh SOOOO bohot acha hai!", hinglish=True)
    'yeh soo bahut accha hai !'
    """
    chosen = ensure_choice(language, _LANGUAGES, name="language")
    texts, single = ensure_texts(text)
    result = [
        _normalize_one(t, hinglish=hinglish, stopwords=stopwords, language=chosen) for t in texts
    ]
    return result[0] if single else result
