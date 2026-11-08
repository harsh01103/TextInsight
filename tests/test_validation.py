from __future__ import annotations

import pytest

from textinsight.core import InvalidInputError, config
from textinsight.core.validation import (
    ensure_choice,
    ensure_positive_int,
    ensure_ratio,
    ensure_text,
    ensure_texts,
)


class TestEnsureText:
    def test_accepts_strings(self) -> None:
        assert ensure_text("hello") == "hello"
        assert ensure_text("") == ""

    @pytest.mark.parametrize("bad", [None, 5, 3.2, ["a"], {"a": 1}])
    def test_rejects_non_strings(self, bad: object) -> None:
        with pytest.raises(InvalidInputError, match="must be a str"):
            ensure_text(bad)

    def test_bytes_get_a_decode_hint(self) -> None:
        with pytest.raises(InvalidInputError, match="decode"):
            ensure_text(b"hello")

    def test_empty_can_be_disallowed(self) -> None:
        with pytest.raises(InvalidInputError, match="empty"):
            ensure_text("   ", allow_empty=False)

    def test_size_limit(self) -> None:
        config.max_input_chars = 10
        assert ensure_text("x" * 10) == "x" * 10
        with pytest.raises(InvalidInputError, match="limit"):
            ensure_text("x" * 11)

    def test_invalid_input_is_also_a_value_error(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011 - checking the built-in base class
            ensure_text(None)


class TestEnsureTexts:
    def test_single_string(self) -> None:
        assert ensure_texts("hi") == (["hi"], True)

    def test_list_and_tuple_and_generator(self) -> None:
        assert ensure_texts(["a", "b"]) == (["a", "b"], False)
        assert ensure_texts(("a",)) == (["a"], False)
        assert ensure_texts(t for t in ["x", "y"]) == (["x", "y"], False)

    def test_empty_list(self) -> None:
        assert ensure_texts([]) == ([], False)

    def test_bad_item_reports_its_index(self) -> None:
        with pytest.raises(InvalidInputError, match=r"text\[1\]"):
            ensure_texts(["ok", 3])

    @pytest.mark.parametrize("bad", [None, 5, b"bytes", {"a": "b"}])
    def test_rejects_non_text_containers(self, bad: object) -> None:
        with pytest.raises(InvalidInputError):
            ensure_texts(bad)


def test_ensure_choice() -> None:
    assert ensure_choice("TextRank", {"textrank", "rake"}, name="method") == "textrank"
    with pytest.raises(InvalidInputError, match="method must be one of rake, textrank"):
        ensure_choice("yake", {"textrank", "rake"}, name="method")
    with pytest.raises(InvalidInputError):
        ensure_choice(1, {"a"}, name="method")


def test_ensure_positive_int() -> None:
    assert ensure_positive_int(3, name="top_n") == 3
    for bad in (0, -1, 2.5, True, "3"):
        with pytest.raises(InvalidInputError):
            ensure_positive_int(bad, name="top_n")


def test_ensure_ratio() -> None:
    assert ensure_ratio(0.3, name="ratio") == 0.3
    assert ensure_ratio(1, name="ratio") == 1.0
    for bad in (0, -0.1, 1.5, True, "0.5"):
        with pytest.raises(InvalidInputError):
            ensure_ratio(bad, name="ratio")
