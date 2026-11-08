"""Rule-based word and sentence tokenizers, written for informal English and Hinglish."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import overload

from ..core.validation import ensure_texts
from ._patterns import EMAIL, EMOJI, EMOTICON, HASHTAG, MENTION, NUMBER, URL, WORD

__all__ = ["iter_tokens", "sentences", "tokenize"]

_TOKEN = re.compile(
    rf"(?P<url>{URL})"
    rf"|(?P<email>{EMAIL})"
    rf"|(?P<mention>{MENTION})"
    rf"|(?P<hashtag>{HASHTAG})"
    rf"|(?P<emoticon>{EMOTICON})"
    rf"|(?P<emoji>{EMOJI})"
    rf"|(?P<number>{NUMBER})"
    rf"|(?P<word>{WORD})"
    r"|(?P<punct>\S)",
    re.IGNORECASE,
)


def iter_tokens(text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(token, kind)`` pairs.

    ``kind`` is one of url, email, mention, hashtag, emoticon, emoji, number,
    word or punct. Each token's kind is what ``tokenize`` uses for filtering.
    """
    for match in _TOKEN.finditer(text):
        kind = match.lastgroup
        assert kind is not None  # every alternative is a named group
        yield match.group(0), kind


def _tokenize_one(text: str, *, lowercase: bool, punctuation: bool) -> list[str]:
    tokens: list[str] = []
    for token, kind in iter_tokens(text):
        if kind == "punct" and not punctuation:
            continue
        tokens.append(token.lower() if lowercase and kind == "word" else token)
    return tokens


@overload
def tokenize(text: str, *, lowercase: bool = ..., punctuation: bool = ...) -> list[str]: ...
@overload
def tokenize(
    text: list[str], *, lowercase: bool = ..., punctuation: bool = ...
) -> list[list[str]]: ...


def tokenize(
    text: str | list[str], *, lowercase: bool = False, punctuation: bool = True
) -> list[str] | list[list[str]]:
    """Split text into tokens.

    URLs, emails, mentions, hashtags, emoticons, emojis, numbers and contractions
    each stay in one piece, and Devanagari words are kept whole.

    Args:
        text: A string (returns a list of tokens) or a list of strings.
        lowercase: Lowercase word tokens only, so links keep their case.
        punctuation: Keep punctuation tokens.

    >>> tokenize("Don't miss it :) #sale")
    ["Don't", 'miss', 'it', ':)', '#sale']
    """
    texts, single = ensure_texts(text)
    result = [_tokenize_one(t, lowercase=lowercase, punctuation=punctuation) for t in texts]
    return result[0] if single else result


# ----------------------------------------------------------------- sentences
_TERMINATOR = re.compile("[.!?…।॥]+")
_PROTECTED = re.compile(rf"{URL}|{EMAIL}", re.IGNORECASE)
_PARAGRAPH = re.compile(r"\n\s*\n")
_MULTI_DOT_ABBREVIATION = re.compile(r"^(?:[a-z]\.)+[a-z]$")
_CLOSERS = "\"')]}”’»"
_OPENERS = "\"'([{“‘«"
_DANDAS = {"।", "॥"}
_LOOKBACK = 24

_NEVER_ENDS = frozenset(
    "mr mrs ms dr prof sr jr st vs no fig rs sh smt shri lt col gen capt mt ltd inc".split()
)
_MAY_END = frozenset({"etc"})


def _next_visible(text: str, position: int) -> str:
    for char in text[position : position + _LOOKBACK]:
        if not char.isspace():
            return char
    return ""


def _word_before(text: str, position: int) -> tuple[str, int]:
    """Return the whitespace-delimited word ending at ``position`` and its start."""
    start = position
    floor = max(0, position - _LOOKBACK)
    while start > floor and not text[start - 1].isspace():
        start -= 1
    return text[start:position].lstrip(_OPENERS), start


def _is_boundary(text: str, start: int, end: int, sentence_start: int) -> bool:
    terminator = text[start:end].rstrip(_CLOSERS)
    if _DANDAS & set(terminator):
        return True
    if end < len(text) and not text[end].isspace():
        return False  # 3.14, example.com, wow!great, or a comma after a closing quote
    if end == len(text):
        return True

    if "." not in terminator and "…" not in terminator:
        return True  # ! and ?
    following = _next_visible(text, end)
    if terminator != ".":  # an ellipsis ends a sentence only before a capital
        return following.isupper()

    word, word_start = _word_before(text, start)
    lowered = word.lower()
    if lowered in _NEVER_ENDS or _MULTI_DOT_ABBREVIATION.match(lowered):
        return False
    if lowered in _MAY_END:
        return following.isupper()
    if len(word) == 1 and word.isupper() and word != "I":
        return False  # an initial, as in "A. Sharma"
    # A list marker such as "1." at the start of a sentence.
    return not (word.isdigit() and word_start <= sentence_start)


def _split_paragraph(paragraph: str) -> list[str]:
    protected = [m.span() for m in _PROTECTED.finditer(paragraph)]
    results: list[str] = []
    sentence_start = 0
    span_index = 0
    for match in _TERMINATOR.finditer(paragraph):
        start, end = match.span()
        while span_index < len(protected) and protected[span_index][1] <= start:
            span_index += 1
        if span_index < len(protected) and protected[span_index][0] <= start:
            continue  # the terminator is inside a URL or email
        while end < len(paragraph) and paragraph[end] in _CLOSERS:
            end += 1
        if _is_boundary(paragraph, start, end, sentence_start):
            results.append(paragraph[sentence_start:end])
            sentence_start = end
            while sentence_start < len(paragraph) and paragraph[sentence_start].isspace():
                sentence_start += 1
    results.append(paragraph[sentence_start:])
    return [s.strip() for s in results if s.strip()]


def _sentences_one(text: str) -> list[str]:
    results: list[str] = []
    for paragraph in _PARAGRAPH.split(text):
        results.extend(_split_paragraph(paragraph.strip()))
    return results


@overload
def sentences(text: str) -> list[str]: ...
@overload
def sentences(text: list[str]) -> list[list[str]]: ...


def sentences(text: str | list[str]) -> list[str] | list[list[str]]:
    """Split text into sentences.

    Handles abbreviations, initials, decimals, URLs, ellipses and the Hindi
    danda. Lowercase after a full stop still starts a new sentence, because
    informal Hinglish rarely uses capitals.

    >>> sentences("Dr. Rao aaye. yeh accha hai! Price is 3.5 lakh.")
    ['Dr. Rao aaye.', 'yeh accha hai!', 'Price is 3.5 lakh.']
    """
    texts, single = ensure_texts(text)
    result = [_sentences_one(t) for t in texts]
    return result[0] if single else result
