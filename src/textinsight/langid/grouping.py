"""Group texts by language, so that each group can go to a suitable backend."""

from __future__ import annotations

from collections.abc import Sequence

from .detector import detect_language

__all__ = ["group_by_language"]


def group_by_language(texts: Sequence[str], language: str | None) -> dict[str, list[int]]:
    """Map each language code to the positions of its texts.

    Text whose language cannot be determined is treated as English.
    """
    if language is not None:
        codes = [language] * len(texts)
    else:
        codes = [result.language for result in detect_language(list(texts))]
    groups: dict[str, list[int]] = {}
    for index, code in enumerate(codes):
        groups.setdefault("en" if code == "und" else code, []).append(index)
    return groups
