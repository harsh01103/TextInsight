"""Normalization of romanized Hindi spelling.

The same word is written many ways ("bahut", "bohot", "bahuttt"). Mapping these
to one canonical spelling lets later stages treat them as a single feature. A
word is resolved in this order:

1. an exact match in the seed variant list;
2. a match after squeezing repeated letters ("bahuttt" becomes "bahut");
3. a match on a phonetic key that ignores common spelling differences.

Common English words are never rewritten, even inside mixed sentences.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import overload

from .._data.english import ENGLISH_COMMON
from .._data.hinglish import GROUPS
from ..core.validation import ensure_texts
from ._patterns import EMAIL, HASHTAG, MENTION, URL

__all__ = [
    "canonical_form",
    "cluster_variants",
    "levenshtein",
    "normalize_hinglish",
    "phonetic_key",
    "squeeze_repeats",
]

_MIN_PHONETIC_LENGTH = 4
_REPEATS = re.compile(r"([a-z])\1+")
_ASPIRATED = re.compile(r"([bcdgjkpst])h")
_PHONETIC_SWAPS = (("ee", "i"), ("oo", "u"), ("ph", "f"), ("ck", "k"), ("w", "v"), ("z", "j"))

# Links and handles are matched first so they pass through untouched.
_SKIP_OR_WORD = re.compile(
    rf"{URL}|{EMAIL}|{MENTION}|{HASHTAG}|(?P<word>(?<![\w'])[A-Za-z]+(?![\w']))", re.IGNORECASE
)


def squeeze_repeats(word: str, max_repeat: int = 1) -> str:
    """Shorten runs of a repeated letter to ``max_repeat`` letters."""
    return _REPEATS.sub(lambda m: m.group(1) * max_repeat, word.lower())


def phonetic_key(word: str) -> str:
    """Reduce a romanized word to a key that ignores common spelling differences.

    >>> phonetic_key("khoobsoorat") == phonetic_key("khubsurat")
    True
    """
    key = word.lower()
    for old, new in _PHONETIC_SWAPS:
        key = key.replace(old, new)
    return _ASPIRATED.sub(r"\1", squeeze_repeats(key))


def _build_indexes() -> tuple[dict[str, str], dict[str, str]]:
    variants: dict[str, str] = {}
    keys: dict[str, str | None] = {}
    for canonical, spellings in GROUPS.items():
        for spelling in (canonical, *spellings):
            variants[spelling] = canonical
            key = phonetic_key(spelling)
            # A key shared by two different words is ambiguous, so it is unusable.
            keys[key] = canonical if keys.get(key, canonical) == canonical else None
    return variants, {key: value for key, value in keys.items() if value is not None}


_VARIANTS, _KEYS = _build_indexes()
HINGLISH_WORDS = frozenset(_VARIANTS)


def canonical_form(word: str) -> str:
    """Return the canonical spelling of ``word``, or ``word`` unchanged if unknown."""
    lowered = word.lower()
    if lowered in _VARIANTS:
        return _VARIANTS[lowered]
    if lowered in ENGLISH_COMMON:
        return word
    for form in (squeeze_repeats(lowered, 2), squeeze_repeats(lowered, 1)):
        if form in ENGLISH_COMMON:
            return word  # "gooood" stays English rather than being treated as Hindi
        if form in _VARIANTS:
            return _VARIANTS[form]
    if len(squeeze_repeats(lowered)) >= _MIN_PHONETIC_LENGTH:
        return _KEYS.get(phonetic_key(lowered), word)
    return word


def _normalize_one(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        word = match.group("word")
        return canonical_form(word) if word else match.group(0)

    return _SKIP_OR_WORD.sub(replace, text)


@overload
def normalize_hinglish(text: str) -> str: ...
@overload
def normalize_hinglish(text: list[str]) -> list[str]: ...


def normalize_hinglish(text: str | list[str]) -> str | list[str]:
    """Rewrite romanized Hindi words in their canonical spelling.

    Everything else, including English words, links and punctuation, is unchanged.

    >>> normalize_hinglish("yeh phone bohot acha hai but battery bekar")
    'yeh phone bahut accha hai but battery bekaar'
    """
    texts, single = ensure_texts(text)
    result = [_normalize_one(t) for t in texts]
    return result[0] if single else result


# ---------------------------------------------------- finding new variants
def levenshtein(first: str, second: str, max_distance: int | None = None) -> int:
    """Edit distance between two strings.

    With ``max_distance``, the search stops early and returns ``max_distance + 1``
    as soon as the true distance is known to be larger.
    """
    if len(first) < len(second):
        first, second = second, first
    if max_distance is not None and len(first) - len(second) > max_distance:
        return max_distance + 1
    previous = list(range(len(second) + 1))
    for i, char_a in enumerate(first, start=1):
        current = [i]
        for j, char_b in enumerate(second, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        if max_distance is not None and min(current) > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def cluster_variants(
    tokens: Iterable[str],
    *,
    max_distance: int = 1,
    min_length: int = 4,
    min_count: int = 2,
) -> dict[str, list[str]]:
    """Group likely spelling variants found in a corpus.

    This is a corpus-mining tool for growing the seed list, and its output should
    be reviewed by a person: edit distance alone also pairs unrelated words.
    Each cluster is named after its most frequent member, and common English
    words are ignored. Returns only clusters with at least two members.
    """
    counts = Counter(
        t.lower() for t in tokens if t.isascii() and t.isalpha() and len(t) >= min_length
    )
    candidates = [
        (word, n) for word, n in counts.items() if n >= min_count and word not in ENGLISH_COMMON
    ]
    candidates.sort(key=lambda pair: (-pair[1], pair[0]))

    clusters: dict[str, list[str]] = {}
    by_first_letter: dict[str, list[str]] = {}
    for word, _ in candidates:
        heads = by_first_letter.setdefault(word[0], [])
        word_key = phonetic_key(word)
        for head in heads:
            if (
                phonetic_key(head) == word_key
                or levenshtein(word, head, max_distance) <= max_distance
            ):
                clusters[head].append(word)
                break
        else:
            heads.append(word)
            clusters[word] = [word]
    return {head: members for head, members in clusters.items() if len(members) > 1}
