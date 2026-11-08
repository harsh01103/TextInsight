from __future__ import annotations

import time

import pytest

import textinsight as ti
from textinsight.preprocess import (
    normalize_whitespace,
    remove_emails,
    remove_emojis,
    remove_html,
    remove_mentions,
    remove_numbers,
    remove_punctuation,
    remove_urls,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("see https://example.com/a?b=1 now", "see now"),
        ("HTTPS://EXAMPLE.COM ok", "ok"),
        ("go to www.example.com today", "go to today"),
        ("Visit https://example.com.", "Visit."),
        ("Really https://example.com, yes", "Really, yes"),
        ("(see https://example.com)", "(see )"),
        ("wiki https://en.wikipedia.org/wiki/Foo_(bar) end", "wiki end"),
        ("no links in example dot com", "no links in example dot com"),
    ],
)
def test_remove_urls(raw: str, expected: str) -> None:
    assert normalize_whitespace(remove_urls(raw)) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<p>one</p><p>two</p>", "one two"),
        ("<script>alert(1)</script>text", "text"),
        ("<STYLE>p{}</STYLE>text", "text"),
        ("a<!-- hidden <b>x</b> -->b", "a b"),
        ("Tom &amp; Jerry", "Tom & Jerry"),
        ("&lt;b&gt;literal&lt;/b&gt;", "<b>literal</b>"),
        ("3 < 5 and 7 > 2", "3 < 5 and 7 > 2"),
        ("<script>never closed", "never closed"),
    ],
)
def test_remove_html(raw: str, expected: str) -> None:
    assert normalize_whitespace(remove_html(raw)) == expected


def test_remove_emails_and_mentions() -> None:
    assert normalize_whitespace(remove_emails("mail me@x.co.in please")) == "mail please"
    assert normalize_whitespace(remove_mentions("hi @rahul and me@x.com")) == "hi and me@x.com"


@pytest.mark.parametrize(
    "emoji",
    ["😀", "👍🏽", "👨‍👩‍👧", "🇮🇳", "❤️", "1️⃣", "⭐"],
)
def test_remove_emojis(emoji: str) -> None:
    assert normalize_whitespace(remove_emojis(f"good {emoji} day")) == "good day"


def test_remove_emojis_keeps_text_and_devanagari() -> None:
    assert remove_emojis("हिंदी भाषा ok") == "हिंदी भाषा ok"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("costs 1,000 rupees", "costs rupees"),
        ("rated 4.5 stars", "rated stars"),
        ("gr8 h2o 2nd", "gr8 h2o 2nd"),
    ],
)
def test_remove_numbers(raw: str, expected: str) -> None:
    assert normalize_whitespace(remove_numbers(raw)) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello, world!", "Hello world"),
        ("don't stop", "don't stop"),
        ("well-known fact", "well-known fact"),
        ("- dash at start -", "dash at start"),
        ("'quoted'", "quoted"),
        ("यह अच्छा है।", "यह अच्छा है"),
    ],
)
def test_remove_punctuation(raw: str, expected: str) -> None:
    assert normalize_whitespace(remove_punctuation(raw)) == expected


class TestCleanOptions:
    def test_defaults_keep_sentiment_signal(self) -> None:
        text = "GREAT!!! 100% 😀 #win @me"
        assert ti.clean(text) == text

    def test_lowercase_and_punctuation(self) -> None:
        result = ti.clean("Hello!!! Visit https://example.com", lowercase=True, punctuation=True)
        assert result == "hello visit"

    def test_each_option_can_be_turned_off(self) -> None:
        assert "https://a.com" in ti.clean("x https://a.com", urls=False)
        assert "<b>" in ti.clean("<b>x</b>", html_tags=False)

    def test_each_removal_option(self) -> None:
        assert ti.clean("a me@x.com", emails=True) == "a"
        assert ti.clean("a @b", mentions=True) == "a"
        assert ti.clean("a 😀", emojis=True) == "a"
        assert ti.clean("a 5", numbers=True) == "a"

    def test_hashtag_modes(self) -> None:
        assert ti.clean("a #Win") == "a #Win"
        assert ti.clean("a #Win", hashtags="strip_symbol") == "a Win"
        assert ti.clean("a #Win", hashtags="remove") == "a"
        assert ti.clean("#जीत", hashtags="strip_symbol") == "जीत"
        with pytest.raises(ti.InvalidInputError):
            ti.clean("a", hashtags="drop")

    def test_url_is_removed_before_mentions_and_hashtags(self) -> None:
        result = ti.clean("https://x.com/@user#frag ok", mentions=True, hashtags="remove")
        assert result == "ok"

    def test_unicode_normalization(self) -> None:
        assert ti.clean("ｆｕｌｌ​width") == "fullwidth"
        assert ti.clean("ｆｕｌｌ", unicode_form=False) == "ｆｕｌｌ"

    def test_keeps_zero_width_joiners_used_by_indic_scripts(self) -> None:
        assert "‍" in ti.clean("क्‍ष")

    def test_batch(self) -> None:
        assert ti.clean(["a  b", "<i>c</i>"]) == ["a b", "c"]


@pytest.mark.parametrize(
    "hostile",
    [
        " " * 500_000 + "x",
        "<script " * 100_000,
        "a" * 500_000,
        "<a" * 200_000,
        "http://" + "." * 200_000,
        "a.b@" * 100_000,
    ],
)
def test_large_hostile_input_stays_fast(hostile: str) -> None:
    start = time.perf_counter()
    ti.clean(hostile, emails=True, mentions=True, emojis=True, numbers=True)
    assert time.perf_counter() - start < 5
