from __future__ import annotations

import pytest

import textinsight as ti
from textinsight.langid import CharNgramClassifier, HeuristicLanguageDetector


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("The food was great and the staff were friendly", "en"),
        ("yeh phone mast hai but battery bekaar", "hi-en"),
        ("bahut accha laga mujhe", "hi-en"),
        ("यह बहुत अच्छा है", "hi"),
        ("मुझे यह phone पसंद है", "hi-en"),
        ("मुझे यह नया phone बहुत ज़्यादा पसंद आया है", "hi"),
        ("यह phone delivery service quality bekar", "hi-en"),
        ("😀😀", "und"),
        ("", "und"),
        ("12345 !!!", "und"),
        ("Это русский текст", "und"),
        ("to me", "en"),
    ],
)
def test_language_labels(text: str, language: str) -> None:
    assert ti.detect_language(text).language == language


def test_elongated_words_are_recognised() -> None:
    result = ti.detect_language("bahuttt achaaaa")
    assert result.language == "hi-en"
    assert result.token_languages == ("hi", "hi")


def test_token_labels_align_with_tokenize() -> None:
    text = "yeh phone, bohot mast hai :) https://a.com"
    result = ti.detect_language(text)
    tokens = ti.tokenize(text)
    assert len(result.token_languages) == len(tokens)
    labels = dict(zip(tokens, result.token_languages, strict=True))
    assert labels["yeh"] == "hi"
    assert labels["phone"] == "en"
    assert labels[","] == "other"
    assert labels[":)"] == "other"
    assert labels["https://a.com"] == "other"


def test_ambiguous_words_follow_the_majority() -> None:
    hindi = ti.detect_language("main bahut khush hoon yaar")
    assert hindi.token_languages[0] == "hi"
    english = ti.detect_language("the main problem was the battery")
    assert english.token_languages[1] == "en"


def test_score_is_in_range_and_lower_without_evidence() -> None:
    strong = ti.detect_language("The food was great")
    weak = ti.detect_language("main")  # valid in both languages, so no real evidence
    assert weak.language == "en"
    assert 0.0 < weak.score < strong.score <= 1.0


def test_unknown_words_use_the_classifier() -> None:
    assert ti.detect_language("kharidne").token_languages == ("hi",)
    assert ti.detect_language("international").token_languages == ("en",)


def test_short_unknown_words_are_not_guessed() -> None:
    backend = HeuristicLanguageDetector()
    backend.ensure_loaded()
    assert backend._label_word("zq") == ("ambiguous", 0.0)
    assert backend._label_word("123") == ("ambiguous", 0.0)


def test_low_margin_words_are_ambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = HeuristicLanguageDetector()
    backend.ensure_loaded()
    monkeypatch.setattr(backend._classifier, "predict", lambda word: ("hi", 0.0))
    assert backend._label_word("qwertyish") == ("ambiguous", 0.0)


def test_batch() -> None:
    results = ti.detect_language(["hello there friend", "bahut accha"])
    assert [r.language for r in results] == ["en", "hi-en"]


def test_backend_is_registered() -> None:
    info = {i.name: i for i in ti.list_backends("langid")}
    assert info["heuristic"].available


def test_unknown_backend() -> None:
    with pytest.raises(ti.UnknownBackendError):
        ti.detect_language("hello", backend="nope")


def test_invalid_input() -> None:
    with pytest.raises(ti.InvalidInputError):
        ti.detect_language(None)  # type: ignore[call-overload]


class TestCharNgramClassifier:
    SAMPLES = (
        ("alpha", "x"),
        ("alpine", "x"),
        ("alps", "x"),
        ("zork", "y"),
        ("zorg", "y"),
        ("zoom", "y"),
    )

    def test_predicts_by_letter_patterns(self) -> None:
        model = CharNgramClassifier().fit(self.SAMPLES)
        label, margin = model.predict("alpaca")
        assert label == "x"
        assert margin > 0
        assert model.predict("zorb")[0] == "y"

    def test_needs_two_classes(self) -> None:
        with pytest.raises(ValueError, match="two classes"):
            CharNgramClassifier().fit([("a", "x")])

    def test_needs_fit(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            CharNgramClassifier().predict("a")

    def test_is_case_insensitive(self) -> None:
        model = CharNgramClassifier().fit(self.SAMPLES)
        assert model.predict("ALPACA") == model.predict("alpaca")
