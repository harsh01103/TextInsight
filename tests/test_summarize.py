from __future__ import annotations

import pytest

import textinsight as ti
from textinsight.eval.text_metrics import (
    keyphrase_scores,
    porter_stem,
    rouge_scores,
    stemmed_words,
)
from textinsight.summarize import TextRankSummarizer
from textinsight.summarize.extractive import similarity

ARTICLE = (
    "The city council approved a new solar power plan on Monday. "
    "The solar power plan will cut electricity costs for the city. "
    "My cat slept all afternoon. "
    "Council members said solar power would also create jobs in the city. "
    "Rain fell briefly."
)
HINGLISH = (
    "Yeh phone ki battery life bahut acchi hai. Battery life do din chalti hai. "
    "Mummy bazaar gayi hain. Phone ki battery bahut fast charge hoti hai."
)
METHODS = ("textrank", "frequency")


class TestSummarize:
    @pytest.mark.parametrize("method", METHODS)
    def test_keeps_central_sentences_and_drops_off_topic_ones(self, method: str) -> None:
        result = ti.summarize(ARTICLE, ratio=0.6, method=method, language="en")
        assert result.method == method
        assert "cat" not in result.summary
        assert "Rain" not in result.summary

    @pytest.mark.parametrize("method", METHODS)
    def test_hinglish(self, method: str) -> None:
        result = ti.summarize(HINGLISH, max_sentences=2, method=method, language="hi-en")
        assert "battery" in result.summary.lower()
        assert "Mummy" not in result.summary

    @pytest.mark.parametrize("method", METHODS)
    def test_sentences_stay_in_original_order(self, method: str) -> None:
        indices = ti.summarize(ARTICLE, ratio=0.6, method=method, language="en").sentence_indices
        assert list(indices) == sorted(indices)

    @pytest.mark.parametrize(
        ("ratio", "limit", "expected"),
        [(0.2, None, 1), (0.6, None, 3), (1.0, None, 5), (1.0, 2, 2), (0.01, None, 1)],
    )
    def test_length_options(self, ratio: float, limit: int | None, expected: int) -> None:
        result = ti.summarize(ARTICLE, ratio=ratio, max_sentences=limit, language="en")
        assert len(result.sentence_indices) == expected

    def test_summary_is_built_from_the_original_sentences(self) -> None:
        result = ti.summarize(ARTICLE, ratio=0.4, language="en")
        chosen = ti.sentences(ARTICLE)
        assert result.summary == " ".join(chosen[i] for i in result.sentence_indices)

    @pytest.mark.parametrize("method", METHODS)
    def test_empty_text(self, method: str) -> None:
        result = ti.summarize("", method=method, language="en")
        assert result.summary == ""
        assert result.sentence_indices == ()

    @pytest.mark.parametrize("method", METHODS)
    def test_sentences_without_content_do_not_crash(self, method: str) -> None:
        result = ti.summarize("The and of. It is. Was he?", method=method, language="en")
        assert result.sentence_indices == (0,)

    def test_single_sentence_is_returned_unchanged(self) -> None:
        text = "yeh phone accha hai aur battery bhi mast hai"
        assert ti.summarize(text).summary == text

    def test_auto_uses_textrank(self) -> None:
        assert ti.summarize(ARTICLE, language="en").method == "textrank"

    def test_language_is_detected(self) -> None:
        assert ti.summarize(HINGLISH, ratio=0.5).summary

    def test_list_input_returns_a_list(self) -> None:
        results = ti.summarize([ARTICLE, HINGLISH], ratio=0.5)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_devanagari_is_rejected(self) -> None:
        with pytest.raises(ti.UnsupportedLanguageError):
            ti.summarize("यह फोन बहुत अच्छा है। कैमरा भी अच्छा है।")

    @pytest.mark.parametrize("ratio", [0, 1.5, -0.1, True, "half"])
    def test_invalid_ratio(self, ratio: object) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.summarize("Some text.", ratio=ratio)  # type: ignore[call-overload]

    @pytest.mark.parametrize("limit", [0, -2, 1.5])
    def test_invalid_max_sentences(self, limit: object) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.summarize("Some text.", max_sentences=limit)  # type: ignore[call-overload]

    def test_backends_are_listed(self) -> None:
        assert {b.name for b in ti.list_backends("summarize")} == {*METHODS, "abstractive"}

    def test_backend_can_be_called_directly(self) -> None:
        assert TextRankSummarizer()([ARTICLE])[0].summary

    def test_similarity(self) -> None:
        assert similarity({"a", "b"}, {"c"}) == 0.0
        assert similarity({"a", "b"}, {"a", "c"}) > 0
        assert similarity({"a"}, {"a"}) > 0


class TestPorterStem:
    @pytest.mark.parametrize(
        ("word", "stem"),
        [
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("caress", "caress"),
            ("cats", "cat"),
            ("feed", "feed"),
            ("agreed", "agre"),
            ("plastered", "plaster"),
            ("bled", "bled"),
            ("motoring", "motor"),
            ("sing", "sing"),
            ("conflated", "conflat"),
            ("troubled", "troubl"),
            ("sized", "size"),
            ("hopping", "hop"),
            ("tanned", "tan"),
            ("falling", "fall"),
            ("hissing", "hiss"),
            ("fizzed", "fizz"),
            ("failing", "fail"),
            ("filing", "file"),
            ("happy", "happi"),
            ("sky", "sky"),
            ("relational", "relat"),
            ("conditional", "condit"),
            ("rational", "ration"),
            ("digitizer", "digit"),
            ("triplicate", "triplic"),
            ("formative", "form"),
            ("formalize", "formal"),
            ("electrical", "electr"),
            ("hopeful", "hope"),
            ("goodness", "good"),
            ("revival", "reviv"),
            ("allowance", "allow"),
            ("inference", "infer"),
            ("airliner", "airlin"),
            ("gyroscopic", "gyroscop"),
            ("adjustable", "adjust"),
            ("defensible", "defens"),
            ("irritant", "irrit"),
            ("replacement", "replac"),
            ("adjustment", "adjust"),
            ("dependent", "depend"),
            ("adoption", "adopt"),
            ("communism", "commun"),
            ("activate", "activ"),
            ("homologous", "homolog"),
            ("effective", "effect"),
            ("bowdlerize", "bowdler"),
            ("probate", "probat"),
            ("rate", "rate"),
            ("cease", "ceas"),
            ("controlling", "control"),
            ("roll", "roll"),
            ("at", "at"),
        ],
    )
    def test_published_examples(self, word: str, stem: str) -> None:
        assert porter_stem(word) == stem

    def test_stemmed_words(self) -> None:
        assert stemmed_words("Running cats, 2 dogs!") == ["run", "cat", "2", "dog"]


class TestKeyphraseScores:
    def test_matches_after_stemming(self) -> None:
        scores = keyphrase_scores(["neural networks"], ["neural network"], k=5)
        assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_partial_match(self) -> None:
        scores = keyphrase_scores(["a b", "c d"], ["a b", "e f", "g h", "i j"], k=2)
        assert scores["precision"] == 0.5
        assert scores["recall"] == 0.25
        assert scores["f1"] == pytest.approx(1 / 3)

    def test_only_top_k_count(self) -> None:
        assert keyphrase_scores(["x", "gold"], ["gold"], k=1)["precision"] == 0.0

    def test_duplicates_count_once(self) -> None:
        assert keyphrase_scores(["cat", "cats"], ["cat"], k=2)["precision"] == 1.0

    def test_nothing_predicted_or_no_gold(self) -> None:
        assert keyphrase_scores([], ["a"], k=5)["f1"] == 0.0
        assert keyphrase_scores(["a"], [], k=5)["recall"] == 0.0


class TestRouge:
    def test_identical_text_scores_one(self) -> None:
        scores = rouge_scores("the cat sat", "the cat sat")
        assert all(scores[name]["f1"] == 1.0 for name in ("rouge-1", "rouge-2", "rouge-l"))

    def test_no_overlap_scores_zero(self) -> None:
        assert rouge_scores("red", "blue")["rouge-1"]["f1"] == 0.0

    def test_known_values(self) -> None:
        scores = rouge_scores("the cat was found", "the cat was under the bed")
        assert scores["rouge-1"]["precision"] == 0.75
        assert scores["rouge-1"]["recall"] == 0.5
        assert scores["rouge-2"]["precision"] == pytest.approx(2 / 3)
        assert scores["rouge-l"]["recall"] == 0.5

    def test_counts_are_clipped(self) -> None:
        assert rouge_scores("the the the", "the cat")["rouge-1"]["precision"] == pytest.approx(
            1 / 3
        )

    def test_empty_text(self) -> None:
        assert rouge_scores("", "a b")["rouge-l"]["f1"] == 0.0
