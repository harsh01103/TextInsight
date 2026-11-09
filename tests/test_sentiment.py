from __future__ import annotations

import sys
import types
from typing import Any, ClassVar

import pytest

import textinsight as ti
from textinsight._data.sentiment_lexicon import ENGLISH, HINGLISH
from textinsight.preprocess import canonical_form
from textinsight.sentiment import ClassicalSentiment, HybridSentiment, hybrid
from textinsight.sentiment.classical import LinearModel, lexicon_backoff, load_bundled_model
from textinsight.sentiment.features import FeatureOptions, extract_features, feature_tokens
from textinsight.sentiment.lexicon import score_text


def lexicon(text: str) -> ti.SentimentResult:
    return ti.sentiment(text, backend="lexicon", language="hi-en")


class TestLexicon:
    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("The food was excellent", "positive"),
            ("This is a terrible phone", "negative"),
            ("The train leaves at noon", "neutral"),
            ("", "neutral"),
            ("yeh phone bahut accha hai", "positive"),
            ("service bekaar hai", "negative"),
            ("😍😍", "positive"),
            ("😡", "negative"),
            (":)", "positive"),
            (":(", "negative"),
        ],
    )
    def test_labels(self, text: str, label: str) -> None:
        assert lexicon(text).label == label

    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("This is not good", "negative"),
            ("not bad at all", "positive"),
            ("yeh accha nahi hai", "negative"),
            ("yeh acha nhi hai", "negative"),
            ("bekaar nahi hai", "positive"),
        ],
    )
    def test_negation_before_and_after(self, text: str, label: str) -> None:
        assert lexicon(text).label == label

    def test_spelling_variants_score_the_same(self) -> None:
        assert score_text("bohot acha") == score_text("bahut accha")
        assert score_text("achaaaa")[1] == "positive"

    def test_intensifier_strengthens_and_downtoner_weakens(self) -> None:
        plain = score_text("good")[0]
        assert score_text("very good")[0] > plain > score_text("slightly good")[0]

    def test_capitals_and_exclamations_strengthen(self) -> None:
        plain = score_text("good")[0]
        assert score_text("GOOD")[0] > plain
        assert score_text("good!!!")[0] > plain

    def test_negative_intensifier_makes_it_more_negative(self) -> None:
        assert score_text("very bad")[0] < score_text("bad")[0]

    def test_hashtag_counts(self) -> None:
        assert lexicon("#awesome").label == "positive"

    def test_contrast_favours_the_second_clause(self) -> None:
        assert score_text("good but terrible")[0] < score_text("terrible but good")[0]

    def test_mixed_label(self) -> None:
        assert score_text("good and bad")[1] == "mixed"

    def test_score_stays_in_range(self) -> None:
        for text in ("excellent " * 200, "terrible " * 200):
            assert -1.0 <= score_text(text)[0] <= 1.0

    def test_hinglish_lexicon_uses_canonical_spellings(self) -> None:
        for word in HINGLISH:
            assert canonical_form(word) == word, word

    def test_lexicon_words_do_not_conflict(self) -> None:
        shared = set(ENGLISH) & set(HINGLISH)
        assert all(ENGLISH[w] * HINGLISH[w] > 0 for w in shared)


class TestFeatures:
    def test_tokens_are_normalized(self) -> None:
        assert feature_tokens("Bohot ACHAAAA!! https://x.co @user #Mast 42 😍") == [
            "bahut",
            "accha",
            "!",
            "!",
            "mast",
            "😍",
        ]

    def test_normalizer_can_be_switched_off(self) -> None:
        assert feature_tokens("bohot", hinglish=False) == ["bohot"]

    def test_feature_types(self) -> None:
        features = extract_features("mast phone")
        assert "w:mast" in features
        assert "b:mast phone" in features
        assert "c:<ma" in features
        assert "c:one>" in features

    def test_empty_text(self) -> None:
        assert extract_features("") == []

    def test_negation_marks_word_on_either_side(self) -> None:
        options = FeatureOptions(negation=True, char_sizes=(), word_ngrams=())
        assert "n:accha" in extract_features("accha nahi hai", options)
        assert "n:good" in extract_features("not good", options)

    def test_exclamation_ends_negation_scope(self) -> None:
        options = FeatureOptions(negation=True, char_sizes=(), word_ngrams=())
        assert "n:good" not in extract_features("not ! good", options)

    def test_spelling_features_match_across_variants(self) -> None:
        options = FeatureOptions(hinglish=False, skeleton=True, char_sizes=(), word_ngrams=())
        assert extract_features("bohot", options) == extract_features("bht", options)
        phonetic = FeatureOptions(phonetic=True, char_sizes=(), word_ngrams=())
        assert extract_features("khoobsoorat", phonetic) == extract_features("khubsurat", phonetic)

    def test_code_mix_features(self) -> None:
        options = FeatureOptions(code_mix=True, char_sizes=(), word_ngrams=())
        assert extract_features("bahut accha hai", options) == ["mix:lang_hi-en", "mix:hi_share_4"]
        assert extract_features("", options) == ["mix:lang_und", "mix:hi_share_0"]

    def test_lexicon_features(self) -> None:
        options = FeatureOptions(lexicon=True, char_sizes=(), word_ngrams=())
        assert "lex:positive" in extract_features("excellent", options)

    def test_raw_words_keep_original_spelling(self) -> None:
        options = FeatureOptions(raw_words=True, char_sizes=(), word_ngrams=())
        assert extract_features("bohot", options) == ["r:bohot"]

    def test_drop_noise_removes_retweet_mark_and_cut_off_word(self) -> None:
        options = FeatureOptions(drop_noise=True, char_sizes=(), word_ngrams=(1,))
        assert extract_features("RT accha film jan…", options) == ["w:accha", "w:film"]

    def test_word_trigrams(self) -> None:
        options = FeatureOptions(word_ngrams=(3,), char_sizes=())
        assert extract_features("a b c", options) == ["t:a b c"]

    def test_identity_words_are_masked_everywhere(self) -> None:
        options = FeatureOptions(mask_identity=True, phonetic=True, skeleton=True)
        features = extract_features("Muslim log accha", options)
        assert "w:<identity>" in features
        assert not any("muslim" in feature for feature in features)


class TestLinearModel:
    PAYLOAD: ClassVar[dict[str, Any]] = {
        "classes": ["negative", "positive"],
        "options": {"hinglish": False, "word_ngrams": [1, 2], "char_sizes": [3, 4]},
        "features": ["w:bad", "w:good"],
        "idf": [1.0, 1.0],
        "coefficients": [[2.0, -2.0], [-2.0, 2.0]],
        "intercepts": [0.0, 0.0],
    }

    def test_probabilities_sum_to_one_and_follow_weights(self) -> None:
        model = LinearModel(self.PAYLOAD)
        negative, positive = model.probabilities("good")
        assert positive > negative
        assert positive + negative == pytest.approx(1.0)

    def test_unknown_text_falls_back_to_intercepts(self) -> None:
        assert LinearModel(self.PAYLOAD).probabilities("zzz") == pytest.approx([0.5, 0.5])

    def test_repeated_features_use_sublinear_counts(self) -> None:
        payload = {
            **self.PAYLOAD,
            "features": ["w:bad", "w:good"],
            "coefficients": [[1.0, 0.0], [0.0, 1.0]],
        }
        model = LinearModel(payload)
        assert model.probabilities("bad good")[0] == pytest.approx(0.5)
        assert model.probabilities("bad bad good")[0] > 0.5


class TestBundledClassifier:
    def test_is_registered_as_preferred_for_hinglish(self) -> None:
        result = ti.sentiment("yeh movie bahut bakwas hai yaar")
        assert result.backend == "classical"
        assert result.language == "hi-en"

    def test_loads_lazily(self) -> None:
        assert not ClassicalSentiment().loaded

    def test_output_is_valid(self) -> None:
        result = ti.sentiment("kya bakwas service hai", backend="classical", language="hi-en")
        assert result.label in {"positive", "negative", "neutral"}
        assert -1.0 <= result.score <= 1.0

    def test_empty_text_does_not_crash(self) -> None:
        result = ti.sentiment("", backend="classical", language="hi-en")
        assert result.label in {"positive", "negative", "neutral"}


class TestHybrid:
    @pytest.fixture
    def backend(self, monkeypatch: pytest.MonkeyPatch) -> HybridSentiment:
        instance = HybridSentiment()
        instance._head = hybrid._load_head()
        instance._classical = load_bundled_model()
        size = len(instance._head["mean"])
        monkeypatch.setattr(instance, "_embed", lambda texts: [[0.0] * size for _ in texts])
        return instance

    def test_blends_into_a_valid_result(self, backend: HybridSentiment) -> None:
        result = backend.predict(["bahut accha product hai"])[0]
        assert result.backend == "hybrid"
        assert result.label in {"positive", "negative", "neutral"}
        assert -1.0 <= result.score <= 1.0

    def test_head_output_is_a_distribution(self) -> None:
        head = hybrid._load_head()
        probabilities = hybrid.head_probabilities(head, head["mean"])
        assert len(probabilities) == 3
        assert sum(probabilities) == pytest.approx(1.0)

    def test_auto_never_selects_it(self) -> None:
        by_name = {i.name: i for i in ti.list_backends("sentiment")}
        assert by_name["hybrid"].priority < by_name["classical"].priority

    def test_missing_packages_give_install_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(HybridSentiment, "requires", ("module_that_does_not_exist_xyz",))
        with pytest.raises(ti.BackendNotAvailableError, match=r"textinsight\[transformers\]"):
            ti.sentiment("hello there friend", backend="hybrid", language="en")

    def test_load_failure_is_reported_clearly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*args: object, **kwargs: object) -> None:
            raise OSError("not found")

        fake = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=fail),
            AutoModel=types.SimpleNamespace(from_pretrained=fail),
        )
        monkeypatch.setitem(sys.modules, "transformers", fake)
        ti.config.offline = True
        with pytest.raises(ti.ModelDownloadError, match="Offline mode"):
            HybridSentiment()._load_encoder()
        ti.config.offline = False
        with pytest.raises(ti.ModelDownloadError, match="Could not load"):
            HybridSentiment()._load_encoder()

    def test_load_uses_pinned_safe_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, object]] = []

        class Encoder:
            def eval(self) -> Encoder:
                return self

        def from_pretrained(name: str, **kwargs: object) -> Encoder:
            calls.append(kwargs)
            return Encoder()

        loader = types.SimpleNamespace(from_pretrained=from_pretrained)
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=loader, AutoModel=loader),
        )
        instance = HybridSentiment()
        instance.load()
        assert instance._classical is load_bundled_model()
        assert calls[1]["use_safetensors"] is True
        assert calls[0]["revision"] == hybrid.ENCODER_REVISION

    def test_embed_mean_pools_only_real_tokens(self) -> None:
        torch = pytest.importorskip("torch")

        class Tokenizer:
            def __call__(self, batch: list[str], **kwargs: object) -> dict[str, object]:
                return {
                    "input_ids": torch.zeros((1, 2), dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 0]]),
                }

        def encoder(**kwargs: object) -> types.SimpleNamespace:
            return types.SimpleNamespace(last_hidden_state=torch.tensor([[[2.0], [100.0]]]))

        instance = HybridSentiment()
        instance._tokenizer = Tokenizer()
        instance._encoder = encoder
        assert instance._embed(["x"]) == [[2.0]]


class TestRouting:
    def test_english_goes_to_classifier(self) -> None:
        result = ti.sentiment("The food was great and the staff were friendly")
        assert result.backend == "classical"
        assert result.language == "en"

    def test_emoji_only_text_goes_to_lexicon(self) -> None:
        result = ti.sentiment("😍😍")
        assert (result.backend, result.language, result.label) == ("lexicon", "und", "positive")

    def test_batch_keeps_input_order_across_backends(self) -> None:
        texts = ["😍", "yeh movie bahut bakwas hai yaar", "😡", "The staff were rude"]
        results = ti.sentiment(texts)
        assert [r.backend for r in results] == ["lexicon", "classical", "lexicon", "classical"]
        assert [r.language for r in results] == ["und", "hi-en", "und", "en"]
        assert results[0].label == "positive"
        assert results[2].label == "negative"

    def test_empty_list(self) -> None:
        assert ti.sentiment([]) == []

    def test_empty_string_is_neutral(self) -> None:
        result = ti.sentiment("")
        assert (result.label, result.language) == ("neutral", "und")

    def test_devanagari_is_rejected_clearly(self) -> None:
        with pytest.raises(ti.UnsupportedLanguageError, match="'hi'"):
            ti.sentiment("यह बहुत अच्छा है")

    def test_classical_refuses_undetermined_text(self) -> None:
        with pytest.raises(ti.UnsupportedLanguageError):
            ti.sentiment("😍", backend="classical")

    def test_invalid_input(self) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.sentiment(None)  # type: ignore[call-overload]

    def test_unknown_backend(self) -> None:
        with pytest.raises(ti.UnknownBackendError):
            ti.sentiment("hello there friend", backend="nope")

    def test_result_is_json_ready(self) -> None:
        assert set(ti.sentiment("good").to_dict()) == {"label", "score", "backend", "language"}


class TestLexiconBackoff:
    def test_clear_short_hinglish_is_no_longer_neutral(self):
        result = ti.sentiment("zabardast phone hai, maza aa gaya", backend="classical")
        assert result.label == "positive"
        assert result.score > 0

    def test_factual_sentence_stays_neutral(self):
        assert ti.sentiment("yeh phone kal launch hoga", backend="classical").label == "neutral"

    @pytest.mark.parametrize(
        "text",
        [
            "zabardast phone hai #launch",
            "zabardast phone hai \U0001f602",
            "zabardast phone hai " + "aur phir kal subah hum sab log wahan gaye the " * 2,
        ],
    )
    def test_tweets_and_long_text_keep_the_model_prediction(self, text):
        assert lexicon_backoff(text, "neutral", 0.01) == ("neutral", 0.01)

    def test_only_neutral_predictions_can_change(self):
        text = "zabardast phone hai"
        assert lexicon_backoff(text, "negative", -0.2) == ("negative", -0.2)

    def test_weak_lexicon_evidence_changes_nothing(self):
        assert lexicon_backoff("phone kal aayega", "neutral", 0.0) == ("neutral", 0.0)
