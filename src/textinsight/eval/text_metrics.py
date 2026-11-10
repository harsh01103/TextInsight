"""Porter stemming, keyphrase matching and ROUGE, written with no dependencies.

Keyphrase papers and ROUGE both compare stemmed words, so scores from these
functions can be compared with published ones only roughly. Differences in
tokenization and in the ROUGE-L variant can shift results a little.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

__all__ = ["keyphrase_scores", "porter_stem", "rouge_scores", "stemmed_words"]

_WORD = re.compile(r"[a-z0-9]+")

_STEP2 = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
)
_STEP3 = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)
_STEP4 = "al ance ence er ic able ible ant ement ment ent ou ism ate iti ous ive ize".split()


def _is_consonant(word: str, index: int) -> bool:
    letter = word[index]
    if letter in "aeiou":
        return False
    if letter == "y":
        return index == 0 or not _is_consonant(word, index - 1)
    return True


def _measure(stem: str) -> int:
    """Count vowel-then-consonant transitions, Porter's measure m."""
    pattern = [_is_consonant(stem, index) for index in range(len(stem))]
    return sum(1 for index in range(1, len(pattern)) if pattern[index] and not pattern[index - 1])


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, index) for index in range(len(stem)))


def _double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    if len(word) < 3:
        return False
    last = len(word) - 1
    return (
        _is_consonant(word, last)
        and not _is_consonant(word, last - 1)
        and _is_consonant(word, last - 2)
        and word[-1] not in "wxy"
    )


def _after_step1b(word: str) -> str:
    if word.endswith(("at", "bl", "iz")):
        return word + "e"
    if _double_consonant(word) and word[-1] not in "lsz":
        return word[:-1]
    if _measure(word) == 1 and _ends_cvc(word):
        return word + "e"
    return word


def _step1(word: str) -> str:
    if word.endswith("sses") or word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]

    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        for suffix in ("ed", "ing"):
            if word.endswith(suffix) and _has_vowel(word[: -len(suffix)]):
                word = _after_step1b(word[: -len(suffix)])
                break
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"
    return word


def _replace(word: str, rules: Sequence[tuple[str, str]]) -> str:
    # Porter applies only the longest matching suffix.
    for suffix, replacement in sorted(rules, key=lambda rule: -len(rule[0])):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            return stem + replacement if _measure(stem) > 0 else word
    return word


def _step4(word: str) -> str:
    for suffix in sorted([*_STEP4, "ion"], key=len, reverse=True):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if suffix == "ion" and not stem.endswith(("s", "t")):
                return word
            return stem if _measure(stem) > 1 else word
    return word


def porter_stem(word: str) -> str:
    """Return the Porter (1980) stem of a lowercase word."""
    if len(word) <= 2:
        return word
    word = _replace(_replace(_step1(word), _STEP2), _STEP3)
    word = _step4(word)
    if word.endswith("e"):
        stem = word[:-1]
        if _measure(stem) > 1 or (_measure(stem) == 1 and not _ends_cvc(stem)):
            word = stem
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


def stemmed_words(text: str) -> list[str]:
    """Lowercase ``text``, keep letters and digits, and stem each word."""
    return [porter_stem(word) for word in _WORD.findall(text.lower())]


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def keyphrase_scores(predicted: Sequence[str], gold: Iterable[str], k: int) -> dict[str, float]:
    """Precision, recall and F1 of the top ``k`` phrases, matched after stemming.

    Precision divides by the number of phrases returned, at most ``k``.
    """
    targets = {" ".join(stemmed_words(phrase)) for phrase in gold}
    targets.discard("")
    seen: set[str] = set()
    top: list[str] = []
    for phrase in predicted[:k]:
        stems = " ".join(stemmed_words(phrase))
        if stems and stems not in seen:
            seen.add(stems)
            top.append(stems)
    hits = sum(phrase in targets for phrase in top)
    precision = hits / len(top) if top else 0.0
    recall = hits / len(targets) if targets else 0.0
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def _ngram_counts(words: Sequence[str], size: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(words[i : i + size]) for i in range(len(words) - size + 1))


def _lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    row = [0] * (len(second) + 1)
    for word in first:
        diagonal = 0
        for index, other in enumerate(second, start=1):
            above = row[index]
            row[index] = diagonal + 1 if word == other else max(row[index], row[index - 1])
            diagonal = above
    return row[-1]


def _report(matches: int, candidate_total: int, reference_total: int) -> dict[str, float]:
    precision = matches / candidate_total if candidate_total else 0.0
    recall = matches / reference_total if reference_total else 0.0
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def rouge_scores(candidate: str, reference: str) -> dict[str, dict[str, float]]:
    """ROUGE-1, ROUGE-2 and ROUGE-L, with stemming.

    ROUGE-L here is the longest common subsequence over the whole text.
    """
    system, target = stemmed_words(candidate), stemmed_words(reference)
    results: dict[str, dict[str, float]] = {}
    for size in (1, 2):
        system_counts, target_counts = _ngram_counts(system, size), _ngram_counts(target, size)
        results[f"rouge-{size}"] = _report(
            sum((system_counts & target_counts).values()),
            sum(system_counts.values()),
            sum(target_counts.values()),
        )
    results["rouge-l"] = _report(_lcs_length(system, target), len(system), len(target))
    return results
