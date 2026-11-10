from __future__ import annotations

import pytest

import textinsight as ti
from textinsight.core.graph import pagerank
from textinsight.keywords import RakeKeywords, TextRankKeywords
from textinsight.keywords.base import ngrams
from textinsight.keywords.textrank import build_graph
from textinsight.preprocess.content import Term, content_runs, parse_document

ENGLISH = (
    "Machine learning models need large amounts of training data. "
    "Deep learning models are a kind of machine learning model. "
    "Training data quality decides how well machine learning models perform."
)
HINGLISH = (
    "Yeh phone ki battery life bahut acchi hai. Camera quality bhi acha hai. "
    "Battery life ke liye yeh phone best hai, camera quality bhi mast hai."
)
METHODS = ("textrank", "rake", "tfidf")


class TestPageRank:
    def test_empty_graph(self) -> None:
        assert pagerank({}) == {}

    def test_scores_sum_to_one(self) -> None:
        scores = pagerank({"a": {"b": 1.0}, "b": {"a": 1.0, "c": 2.0}, "c": {"b": 2.0}, "d": {}})
        assert sum(scores.values()) == pytest.approx(1.0)

    def test_hub_outranks_leaves(self) -> None:
        scores = pagerank(
            {
                "hub": {"x": 1.0, "y": 1.0, "z": 1.0},
                "x": {"hub": 1.0},
                "y": {"hub": 1.0},
                "z": {"hub": 1.0},
            }
        )
        assert scores["hub"] == max(scores.values())

    def test_personalization_favours_its_target(self) -> None:
        graph = {"a": {"b": 1.0}, "b": {"a": 1.0}}
        scores = pagerank(graph, personalization={"a": 0.9, "b": 0.1})
        assert scores["a"] > scores["b"]
        assert sum(scores.values()) == pytest.approx(1.0)

    def test_symmetric_nodes_tie(self) -> None:
        scores = pagerank({"a": {"b": 1.0}, "b": {"a": 1.0}})
        assert scores["a"] == pytest.approx(scores["b"])

    def test_stops_at_iteration_limit(self) -> None:
        scores = pagerank({"a": {"b": 1.0}, "b": {"a": 1.0}, "c": {}}, max_iterations=1)
        assert sum(scores.values()) == pytest.approx(1.0)


class TestContentRuns:
    def test_stopwords_and_punctuation_break_runs(self) -> None:
        runs = content_runs("battery life of the phone, camera", "en")
        assert [[t.surface for t in run] for run in runs] == [
            ["battery", "life"],
            ["phone"],
            ["camera"],
        ]

    def test_english_plurals_share_a_key(self) -> None:
        assert content_runs("cameras", "en")[0][0].key == content_runs("camera", "en")[0][0].key

    def test_spelling_variants_share_a_key_in_hinglish(self) -> None:
        for variants in (("khana", "khaana"), ("kitab", "kitaab"), ("fone", "phone")):
            keys = {content_runs(word, "hi-en")[0][0].key for word in variants}
            assert len(keys) == 1, variants

    def test_hindi_words_keep_their_final_s(self) -> None:
        assert content_runs("khaas", "hi-en")[0][0].surface == "khaas"
        assert content_runs("cameras", "hi-en")[0][0].key == "camera"

    @pytest.mark.parametrize(
        "text",
        ["bahut", "lekin", "nahi", "accha", "acha", "1234", "x", "https://a.co", "\U0001f602"],
    )
    def test_non_topic_tokens_are_dropped_in_hinglish(self, text: str) -> None:
        assert content_runs(text, "hi-en") == []

    def test_adverbs_are_dropped_but_nouns_ending_in_ly_stay(self) -> None:
        assert content_runs("quickly", "en") == []
        assert content_runs("family", "en")[0][0].surface == "family"

    def test_generic_words_are_dropped(self) -> None:
        assert content_runs("using proposed", "en") == []

    def test_hashtag_loses_its_symbol(self) -> None:
        assert content_runs("#cricket", "en")[0][0].surface == "cricket"

    @pytest.mark.parametrize(
        ("word", "key"),
        [("cities", "city"), ("classes", "class"), ("bus", "bus"), ("glass", "glass")],
    )
    def test_singular_rules(self, word: str, key: str) -> None:
        assert content_runs(word, "en")[0][0].key == key

    def test_parse_document_groups_by_sentence(self) -> None:
        assert len(parse_document("Red car. Blue bus.", "en")) == 2


class TestKeywords:
    @pytest.mark.parametrize("method", METHODS)
    def test_finds_the_main_topic_in_english(self, method: str) -> None:
        result = ti.keywords(ENGLISH, method=method, top_n=5, language="en")
        assert result.method == method
        assert any("learning" in text for text in result.texts)

    @pytest.mark.parametrize("method", METHODS)
    def test_finds_topics_in_hinglish(self, method: str) -> None:
        texts = ti.keywords(HINGLISH, method=method, top_n=5, language="hi-en").texts
        assert "camera quality" in texts

    @pytest.mark.parametrize("method", METHODS)
    def test_spelling_variants_merge_into_one_keyword(self, method: str) -> None:
        text = "khana swaadisht tha. khaana swaadisht laga. kitab table par thi."
        phrases = ti.keywords(text, method=method, language="hi-en").texts
        assert len([p for p in phrases if p.startswith(("khana", "khaana"))]) == 1

    @pytest.mark.parametrize("method", METHODS)
    def test_scores_are_relative_to_the_best(self, method: str) -> None:
        keywords = ti.keywords(ENGLISH, method=method, language="en").keywords
        assert keywords[0].score == 1.0
        assert all(0 <= k.score <= 1 for k in keywords)
        assert [k.score for k in keywords] == sorted((k.score for k in keywords), reverse=True)

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("text", ["", "   ", "the and of", "123 456"])
    def test_text_without_content_gives_nothing(self, method: str, text: str) -> None:
        result = ti.keywords(text, method=method, language="en")
        assert len(result) == 0
        assert result.texts == []

    @pytest.mark.parametrize("method", METHODS)
    def test_top_n_limits_results(self, method: str) -> None:
        assert len(ti.keywords(ENGLISH, method=method, top_n=2, language="en")) <= 2

    def test_overlapping_phrases_are_removed(self) -> None:
        texts = ti.keywords(ENGLISH, method="tfidf", top_n=10, language="en").texts
        for first in texts:
            assert not any(first != second and first in second for second in texts)

    def test_auto_uses_textrank(self) -> None:
        assert ti.keywords(ENGLISH, language="en").method == "textrank"

    def test_language_is_detected(self) -> None:
        assert ti.keywords(HINGLISH, top_n=3).texts

    def test_undetermined_language_falls_back_to_english(self) -> None:
        assert ti.keywords("xyzzy qwerty").method == "textrank"

    def test_list_input_returns_a_list(self) -> None:
        results = ti.keywords([ENGLISH, HINGLISH], top_n=3)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_tfidf_compares_documents_in_a_list(self) -> None:
        documents = [
            "Solar panels and solar power reduce bills.",
            "Solar panels power homes in the village.",
            "Cricket matches excite the whole village.",
        ]
        first = ti.keywords(documents, method="tfidf", top_n=1, language="en")[0]
        assert "bill" in first.texts[0] or "reduce" in first.texts[0]

    def test_devanagari_is_rejected(self) -> None:
        with pytest.raises(ti.UnsupportedLanguageError):
            ti.keywords("यह फोन बहुत अच्छा है और कैमरा भी अच्छा है")

    @pytest.mark.parametrize("value", [0, -1, 1.5, True, "3"])
    def test_invalid_top_n(self, value: object) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.keywords("some text", top_n=value)  # type: ignore[call-overload]

    def test_invalid_input(self) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.keywords(None)  # type: ignore[call-overload]

    def test_unknown_method(self) -> None:
        with pytest.raises(ti.UnknownBackendError):
            ti.keywords("some text", method="nope", language="en")

    def test_backends_are_listed(self) -> None:
        names = {b.name for b in ti.list_backends("keywords")}
        assert names == set(METHODS)

    def test_result_serializes(self) -> None:
        data = ti.keywords(ENGLISH, top_n=1, language="en").to_dict()
        assert data["method"] == "textrank"
        assert data["keywords"][0]["score"] == 1.0

    def test_backend_can_be_called_directly(self) -> None:
        assert RakeKeywords()([ENGLISH])[0].texts


class TestInternals:
    def test_ngrams(self) -> None:
        run = [Term(w, w) for w in "a b c d".split()]
        grams = {tuple(t.key for t in g) for g in ngrams(run, longest=2)}
        assert ("a", "b") in grams
        assert ("a", "c") not in grams
        assert ("a", "b", "c") not in grams

    def test_graph_links_words_inside_the_window_only(self) -> None:
        graph = build_graph(parse_document("alpha beta gamma delta", "en"), window=2)
        assert "beta" in graph["alpha"]
        assert "gamma" not in graph["alpha"]

    def test_graph_ignores_links_across_sentences(self) -> None:
        graph = build_graph(parse_document("Alpha beta. Gamma delta.", "en"))
        assert "gamma" not in graph["beta"]

    def test_repeated_word_has_no_self_loop(self) -> None:
        graph = build_graph(parse_document("alpha alpha beta", "en"))
        assert "alpha" not in graph["alpha"]

    def test_lone_word_is_a_node(self) -> None:
        assert build_graph(parse_document("alpha", "en")) == {"alpha": {}}

    def test_long_phrases_are_capped(self) -> None:
        text = "alpha beta gamma delta epsilon zeta"
        assert all(len(p.split()) <= 3 for p in ti.keywords(text, language="en").texts)

    def test_textrank_keeps_several_words_in_short_text(self) -> None:
        document = parse_document("alpha beta gamma delta epsilon", "en")
        assert len(TextRankKeywords().score([document])[0]) >= 1
