from __future__ import annotations

import pytest

import textinsight as ti
from textinsight._data.english import ENGLISH_COMMON, STOPWORDS_EN
from textinsight._data.hinglish import GROUPS, STOPWORDS_HINGLISH
from textinsight.preprocess import (
    canonical_form,
    cluster_variants,
    levenshtein,
    phonetic_key,
    reduce_elongation,
)


class TestWordLists:
    def test_variants_never_collide_with_english(self) -> None:
        variants = {v for spellings in GROUPS.values() for v in spellings}
        assert variants & ENGLISH_COMMON == set()

    def test_no_variant_belongs_to_two_canonical_words(self) -> None:
        seen: dict[str, str] = {}
        for canonical, spellings in GROUPS.items():
            for spelling in (canonical, *spellings):
                assert seen.setdefault(spelling, canonical) == canonical, spelling

    def test_stopword_lists_exclude_negations(self) -> None:
        for word in ("not", "no", "never", "nor"):
            assert word not in STOPWORDS_EN
        for word in ("nahi", "na", "mat"):
            assert word not in STOPWORDS_HINGLISH


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("bohot", "bahut"),
        ("BOHOT", "bahut"),
        ("acha", "accha"),
        ("nhi", "nahi"),
        ("bahuttt", "bahut"),
        ("achaaaa", "accha"),
        ("khoobsoorat", "khubsurat"),
        ("zabardastt", "zabardast"),
        ("bahut", "bahut"),
    ],
)
def test_canonical_form(word: str, expected: str) -> None:
    assert canonical_form(word) == expected


@pytest.mark.parametrize("word", ["phone", "Battery", "gooood", "soooo", "Samsung", "xyzzy"])
def test_canonical_form_leaves_english_and_unknown_words(word: str) -> None:
    assert canonical_form(word) == word


def test_phonetic_key_groups_spellings() -> None:
    assert phonetic_key("bhai") == phonetic_key("bai")
    assert phonetic_key("zindagi") == phonetic_key("jindagi")
    assert phonetic_key("wala") == phonetic_key("vala")
    assert phonetic_key("phir") == phonetic_key("fir")


def test_normalize_hinglish_preserves_everything_else() -> None:
    text = "Yeh phone bohot acha hai, but https://bohot.com @bohot #bohot me@bohot.com don't"
    assert ti.normalize_hinglish(text) == (
        "yeh phone bahut accha hai, but https://bohot.com @bohot #bohot me@bohot.com don't"
    )


def test_normalize_hinglish_batch() -> None:
    assert ti.normalize_hinglish(["acha", "nhi"]) == ["accha", "nahi"]


def test_reduce_elongation() -> None:
    assert reduce_elongation("sooooo goood") == "soo good"
    assert reduce_elongation("sooooo", max_repeat=1) == "so"
    assert reduce_elongation("good 1000 !!!") == "good 1000 !!!"
    with pytest.raises(ti.InvalidInputError):
        reduce_elongation("x", max_repeat=0)


class TestRemoveStopwords:
    def test_english(self) -> None:
        assert ti.remove_stopwords(["The", "food", "was", "not", "good"]) == [
            "food",
            "not",
            "good",
        ]

    def test_hinglish_applies_both_lists(self) -> None:
        tokens = ["yeh", "the", "phone", "nahi", "hai"]
        assert ti.remove_stopwords(tokens, language="hi-en") == ["phone", "nahi"]

    def test_english_mode_keeps_hindi_words(self) -> None:
        assert ti.remove_stopwords(["hai"]) == ["hai"]

    def test_keep(self) -> None:
        assert ti.remove_stopwords(["the", "very"], keep=["Very"]) == ["very"]

    def test_invalid_language(self) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.remove_stopwords(["a"], language="fr")


class TestNormalize:
    def test_default(self) -> None:
        assert ti.normalize("Sooooo GOOD!!") == "soo good ! !"

    def test_hinglish(self) -> None:
        assert ti.normalize("Yeh bohot acha hai", hinglish=True) == "yeh bahut accha hai"

    def test_stopwords(self) -> None:
        assert ti.normalize("the food", stopwords=True) == "food"

    def test_hinglish_stopwords(self) -> None:
        result = ti.normalize("yeh bohot acha hai", hinglish=True, stopwords=True, language="hi-en")
        assert result == "bahut accha"

    def test_batch(self) -> None:
        assert ti.normalize(["A B", "C"]) == ["a b", "c"]

    def test_invalid_language(self) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.normalize("x", language="fr")


class TestLevenshtein:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("", "", 0),
            ("abc", "abc", 0),
            ("", "abc", 3),
            ("kitten", "sitting", 3),
            ("bahut", "bohot", 2),
        ],
    )
    def test_distance(self, a: str, b: str, expected: int) -> None:
        assert levenshtein(a, b) == expected
        assert levenshtein(b, a) == expected

    def test_early_exit_on_length(self) -> None:
        assert levenshtein("a", "abcdef", max_distance=2) == 3

    def test_early_exit_during_search(self) -> None:
        assert levenshtein("abcdef", "uvwxyz", max_distance=2) == 3

    def test_within_limit_returns_true_distance(self) -> None:
        assert levenshtein("bahut", "bahot", max_distance=2) == 1


class TestClusterVariants:
    def test_groups_variants_under_most_frequent_spelling(self) -> None:
        tokens = ["dhansu"] * 5 + ["dhansoo"] * 2 + ["dhaansu"] * 3 + ["other"] * 4
        clusters = cluster_variants(tokens, max_distance=1)
        assert clusters["dhansu"][0] == "dhansu"
        assert set(clusters["dhansu"]) == {"dhansu", "dhansoo", "dhaansu"}
        assert "other" not in clusters

    def test_ignores_rare_short_english_and_non_alpha(self) -> None:
        tokens = ["abc"] * 5 + ["battery"] * 5 + ["batteri"] * 1 + ["12345"] * 5 + ["हिंदी"] * 5
        assert cluster_variants(tokens) == {}

    def test_case_is_ignored(self) -> None:
        clusters = cluster_variants(["Dhansu", "dhansu", "DHANSOO", "dhansoo"])
        assert clusters == {"dhansoo": ["dhansoo", "dhansu"]}  # equal counts: alphabetical
