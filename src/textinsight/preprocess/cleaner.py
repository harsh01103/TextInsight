"""Configurable text cleaning.

The defaults are deliberately conservative. Case, punctuation, emojis and
numbers all carry sentiment signal ("GREAT!!! 😀"), so they are kept unless you
ask for them to be removed.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import overload

from ..core.validation import ensure_choice, ensure_texts
from ._patterns import EMAIL, EMOJI, HASHTAG, MENTION, URL

__all__ = [
    "clean",
    "normalize_whitespace",
    "remove_emails",
    "remove_emojis",
    "remove_html",
    "remove_mentions",
    "remove_numbers",
    "remove_punctuation",
    "remove_urls",
]

_WHITESPACE = re.compile(r"\s+")
_INVISIBLE = re.compile("[​⁠﻿­]")  # keeps ZWJ/ZWNJ, which Indic scripts need
_URL = re.compile(URL, re.IGNORECASE)
_EMAIL = re.compile(EMAIL)
_MENTION = re.compile(MENTION)
_HASHTAG = re.compile(HASHTAG)
_EMOJI = re.compile(EMOJI)
_NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?![\w])")
_TAG = re.compile(r"</?[a-zA-Z!][^<>]*>")
_BLOCK_OPEN = re.compile(r"<!--|<(script|style)\b", re.IGNORECASE)
_BLOCK_CLOSE = {
    "comment": re.compile(r"-->"),
    "script": re.compile(r"</script\s*>", re.IGNORECASE),
    "style": re.compile(r"</style\s*>", re.IGNORECASE),
}
# Starts only at the beginning of a whitespace run, so it stays linear.
_SPACE_BEFORE_PUNCT = re.compile(r"(?<!\s)\s+([.,!?;:])")

_HASHTAG_MODES = ("keep", "strip_symbol", "remove")
_JOINERS = {"'", "’", "-"}


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to one space and trim the ends."""
    return _WHITESPACE.sub(" ", text).strip()


def _remove(pattern: re.Pattern[str], text: str) -> str:
    result, count = pattern.subn(" ", text)
    # Removing "see https://x.com." must not leave "see  ." behind.
    return _SPACE_BEFORE_PUNCT.sub(r"\1", result) if count else result


def remove_urls(text: str) -> str:
    """Remove http(s), ftp and www links, keeping punctuation that follows them."""
    return _remove(_URL, text)


def remove_emails(text: str) -> str:
    """Remove email addresses."""
    return _remove(_EMAIL, text)


def remove_mentions(text: str) -> str:
    """Remove @mentions (email addresses are left alone)."""
    return _remove(_MENTION, text)


def remove_emojis(text: str) -> str:
    """Remove emojis, including joined sequences, skin tones, flags and keycaps."""
    return _remove(_EMOJI, text)


def remove_numbers(text: str) -> str:
    """Remove standalone numbers. Digits inside words such as ``gr8`` are kept."""
    return _remove(_NUMBER, text)


def _strip_blocks(text: str) -> str:
    """Remove comments and script/style blocks, including their contents."""
    parts: list[str] = []
    position = 0
    while True:
        opening = _BLOCK_OPEN.search(text, position)
        if opening is None:
            break
        kind = (opening.group(1) or "comment").lower()
        closing = _BLOCK_CLOSE[kind].search(text, opening.end())
        if closing is None:  # unterminated: stop, rather than rescanning for every opener
            break
        parts += [text[position : opening.start()], " "]
        position = closing.end()
    parts.append(text[position:])
    return "".join(parts)


def remove_html(text: str) -> str:
    """Strip HTML tags, comments and script/style blocks, then decode entities.

    Tags are removed before entities are decoded, so escaped text such as
    ``&lt;b&gt;`` stays visible as the literal text ``<b>``.
    """
    return html.unescape(_TAG.sub(" ", _strip_blocks(text)))


def remove_punctuation(text: str) -> str:
    """Remove Unicode punctuation, keeping apostrophes and hyphens inside words."""
    out: list[str] = []
    last = len(text) - 1
    for i, char in enumerate(text):
        if not unicodedata.category(char).startswith("P") or (
            char in _JOINERS and 0 < i < last and text[i - 1].isalnum() and text[i + 1].isalnum()
        ):
            out.append(char)
        else:
            out.append(" ")
    return "".join(out)


def _clean_one(
    text: str,
    *,
    lowercase: bool,
    urls: bool,
    html_tags: bool,
    emails: bool,
    mentions: bool,
    hashtags: str,
    emojis: bool,
    numbers: bool,
    punctuation: bool,
    unicode_form: bool,
) -> str:
    if unicode_form:
        text = _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))
    if html_tags:
        text = remove_html(text)
    # URLs go first: a link can contain "@" or "#" that the later steps would match.
    if urls:
        text = remove_urls(text)
    if emails:
        text = remove_emails(text)
    if mentions:
        text = remove_mentions(text)
    if hashtags == "remove":
        text = _remove(_HASHTAG, text)
    elif hashtags == "strip_symbol":
        text = _HASHTAG.sub(lambda m: m.group(0)[1:], text)
    if emojis:
        text = remove_emojis(text)
    if numbers:
        text = remove_numbers(text)
    if punctuation:
        text = remove_punctuation(text)
    if lowercase:
        text = text.lower()
    return normalize_whitespace(text)


@overload
def clean(
    text: str,
    *,
    lowercase: bool = ...,
    urls: bool = ...,
    html_tags: bool = ...,
    emails: bool = ...,
    mentions: bool = ...,
    hashtags: str = ...,
    emojis: bool = ...,
    numbers: bool = ...,
    punctuation: bool = ...,
    unicode_form: bool = ...,
) -> str: ...
@overload
def clean(
    text: list[str],
    *,
    lowercase: bool = ...,
    urls: bool = ...,
    html_tags: bool = ...,
    emails: bool = ...,
    mentions: bool = ...,
    hashtags: str = ...,
    emojis: bool = ...,
    numbers: bool = ...,
    punctuation: bool = ...,
    unicode_form: bool = ...,
) -> list[str]: ...


def clean(
    text: str | list[str],
    *,
    lowercase: bool = False,
    urls: bool = True,
    html_tags: bool = True,
    emails: bool = False,
    mentions: bool = False,
    hashtags: str = "keep",
    emojis: bool = False,
    numbers: bool = False,
    punctuation: bool = False,
    unicode_form: bool = True,
) -> str | list[str]:
    """Clean a string or a list of strings.

    Args:
        text: A string (returns a string) or a list of strings (returns a list).
        lowercase: Convert to lowercase.
        urls: Remove links.
        html_tags: Remove HTML markup and decode entities.
        emails: Remove email addresses.
        mentions: Remove @mentions.
        hashtags: ``"keep"``, ``"strip_symbol"`` (``#great`` becomes ``great``) or ``"remove"``.
        emojis: Remove emojis.
        numbers: Remove standalone numbers.
        punctuation: Remove punctuation.
        unicode_form: Apply NFKC normalization and drop invisible characters.

    >>> clean("Hello!!!   Visit https://example.com now")
    'Hello!!! Visit now'
    >>> clean("Hello!!! Visit https://example.com", lowercase=True, punctuation=True)
    'hello visit'
    """
    hashtag_mode = ensure_choice(hashtags, _HASHTAG_MODES, name="hashtags")
    texts, single = ensure_texts(text)
    cleaned = [
        _clean_one(
            t,
            lowercase=lowercase,
            urls=urls,
            html_tags=html_tags,
            emails=emails,
            mentions=mentions,
            hashtags=hashtag_mode,
            emojis=emojis,
            numbers=numbers,
            punctuation=punctuation,
            unicode_form=unicode_form,
        )
        for t in texts
    ]
    return cleaned[0] if single else cleaned
