from __future__ import annotations

import pytest

import textinsight as ti


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Hello    world!  ", "Hello world!"),
        ("line one\n\nline two\t end", "line one line two end"),
        ("", ""),
        ("   ", ""),
        ("yeh phone mast hai 😀", "yeh phone mast hai 😀"),
    ],
)
def test_clean_single(raw: str, expected: str) -> None:
    assert ti.clean(raw) == expected


def test_clean_batch() -> None:
    assert ti.clean(["  a  b ", "c"]) == ["a b", "c"]


def test_clean_rejects_bad_input() -> None:
    with pytest.raises(ti.InvalidInputError):
        ti.clean(None)  # type: ignore[call-overload]
