from __future__ import annotations

import pytest

import textinsight as ti
from textinsight.preprocess import iter_tokens


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, world!", ["Hello", ",", "world", "!"]),
        ("don't can’t well-known", ["don't", "can’t", "well-known"]),
        ("", []),
        ("   ", []),
        ("3.14 1,000 10:30 12-05-2024", ["3.14", "1,000", "10:30", "12-05-2024"]),
        ("2nd gr8", ["2nd", "gr8"]),
        ("wow!!!", ["wow", "!", "!", "!"]),
    ],
)
def test_words_and_numbers(text: str, expected: list[str]) -> None:
    assert ti.tokenize(text) == expected


@pytest.mark.parametrize(
    "whole",
    [
        "https://example.com/a?b=1",
        "www.example.com",
        "me.you@example.co.in",
        "@rahul_01",
        "#ThrowbackThursday",
        "#जीत",
        ":)",
        ":-(",
        ";D",
        "<3",
        "xD",
        "T_T",
        "😀",
        "👍🏽",
        "👨‍👩‍👧",
        "🇮🇳",
        "1️⃣",
    ],
)
def test_special_tokens_stay_whole(whole: str) -> None:
    assert ti.tokenize(f"before {whole} after") == ["before", whole, "after"]


def test_url_does_not_swallow_following_punctuation() -> None:
    assert ti.tokenize("see https://a.com.") == ["see", "https://a.com", "."]


def test_adjacent_emojis_are_separate() -> None:
    assert ti.tokenize("😀😀") == ["😀", "😀"]


def test_time_is_not_an_emoticon() -> None:
    assert ti.tokenize("at 10:30") == ["at", "10:30"]


def test_devanagari_words_are_not_split_at_vowel_signs() -> None:
    assert ti.tokenize("हिंदी भाषा।") == ["हिंदी", "भाषा", "।"]


def test_other_indic_scripts_stay_whole() -> None:
    assert ti.tokenize("தமிழ் மொழி") == ["தமிழ்", "மொழி"]


def test_lowercase_affects_only_words() -> None:
    assert ti.tokenize("HELLO https://A.com #TAG", lowercase=True) == [
        "hello",
        "https://A.com",
        "#TAG",
    ]


def test_punctuation_can_be_dropped_but_emoticons_stay() -> None:
    assert ti.tokenize("Nice, right? :)", punctuation=False) == ["Nice", "right", ":)"]


def test_batch() -> None:
    assert ti.tokenize(["a b", "c"]) == [["a", "b"], ["c"]]


def test_iter_tokens_reports_kinds() -> None:
    kinds = [kind for _, kind in iter_tokens("hi 5 @a #b 😀 :) , me@x.com https://a.com")]
    assert kinds == [
        "word",
        "number",
        "mention",
        "hashtag",
        "emoji",
        "emoticon",
        "punct",
        "email",
        "url",
    ]


def test_invalid_input() -> None:
    with pytest.raises(ti.InvalidInputError):
        ti.tokenize(None)  # type: ignore[call-overload]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two! Three?", ["One.", "Two!", "Three?"]),
        ("", []),
        ("No terminator", ["No terminator"]),
        ("Dr. Rao came. He left.", ["Dr. Rao came.", "He left."]),
        ("Compare A vs. B now.", ["Compare A vs. B now."]),
        ("Use tools, e.g. a saw. Next one.", ["Use tools, e.g. a saw.", "Next one."]),
        ("The U.S. economy grew.", ["The U.S. economy grew."]),
        ("Met A. Sharma today.", ["Met A. Sharma today."]),
        ("So do I. Then we left.", ["So do I.", "Then we left."]),
        ("Pens, books, etc. and more.", ["Pens, books, etc. and more."]),
        ("Pens, books, etc. Then more.", ["Pens, books, etc.", "Then more."]),
        ("It costs 3.5 lakh. Too much.", ["It costs 3.5 lakh.", "Too much."]),
        ("See example.com for more. Thanks.", ["See example.com for more.", "Thanks."]),
        ("Go to https://a.com/b.html. Done.", ["Go to https://a.com/b.html.", "Done."]),
        ("Mail a.b@x.com. Done.", ["Mail a.b@x.com.", "Done."]),
        ("wait... what happened", ["wait... what happened"]),
        ("wait... What happened", ["wait...", "What happened"]),
        ("wait… What", ["wait…", "What"]),
        ('He said "Go!" Then left.', ['He said "Go!"', "Then left."]),
        ('"Go!", he said.', ['"Go!", he said.']),
        ("wow!great stuff", ["wow!great stuff"]),
        ("1. First item here.", ["1. First item here."]),
        ("I scored 90. Next test.", ["I scored 90.", "Next test."]),
        ("yeh accha hai. par mehenga hai.", ["yeh accha hai.", "par mehenga hai."]),
        ("यह अच्छा है। वह गया।", ["यह अच्छा है।", "वह गया।"]),
        ("यह अच्छा है।वह गया॥", ["यह अच्छा है।", "वह गया॥"]),
        ("First para\n\nSecond para", ["First para", "Second para"]),
        ("line one\nstill same.", ["line one\nstill same."]),
        ("Really?! Yes.", ["Really?!", "Yes."]),
    ],
)
def test_sentences(text: str, expected: list[str]) -> None:
    assert ti.sentences(text) == expected


def test_sentences_batch() -> None:
    assert ti.sentences(["Ab. Cd.", "Ef"]) == [["Ab.", "Cd."], ["Ef"]]
