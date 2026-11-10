from __future__ import annotations

import pytest

import textinsight as ti
from textinsight.aspects import RuleAspects
from textinsight.aspects.extractor import split_clauses
from textinsight.pii import (
    is_valid_aadhaar,
    is_valid_card,
    is_valid_pan,
    luhn_valid,
    verhoeff_valid,
)
from textinsight.stats import count_syllables


def make_aadhaar(start: str = "23456789012") -> str:
    """Return a checksum-valid, made-up 12-digit number."""
    return next(start + str(d) for d in range(10) if verhoeff_valid(start + str(d)))


class TestAspects:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "camera mast hai but battery bekaar hai",
                {"camera": "positive", "battery": "negative"},
            ),
            ("room accha tha lekin staff rude tha", {"room": "positive", "staff": "negative"}),
            (
                "The camera is excellent but the battery is awful",
                {"camera": "positive", "battery": "negative"},
            ),
            ("Great display, terrible speakers", {"display": "positive", "speakers": "negative"}),
            ("keyboard accha nahi hai", {"keyboard": "negative"}),
            ("camera aur battery dono acche hain", {"camera": "positive", "battery": "positive"}),
        ],
    )
    def test_labels(self, text: str, expected: dict[str, str]) -> None:
        assert ti.aspects(text).labels == expected

    def test_the_owner_before_ka_ki_ke_is_not_the_aspect(self) -> None:
        assert ti.aspects("laptop ki speed mast hai", language="hi-en").labels == {
            "speed": "positive"
        }

    def test_ke_liye_keeps_the_word_before_it(self) -> None:
        assert "battery" in ti.aspects("battery ke liye best hai", language="hi-en").labels

    @pytest.mark.parametrize(
        "text", ["yeh phone kal launch hoga", "The train leaves at noon", "", "   "]
    )
    def test_no_opinion_means_no_aspects(self, text: str) -> None:
        assert len(ti.aspects(text)) == 0

    def test_opposite_opinions_are_mixed(self) -> None:
        text = "The camera is great. The camera is awful."
        result = ti.aspects(text, language="en")
        assert result.labels == {"camera": "mixed"}
        assert result.aspects[0].mentions == 2

    def test_most_mentioned_comes_first(self) -> None:
        text = "The screen is great. The battery is great. The battery is excellent."
        assert ti.aspects(text, language="en").aspects[0].name == "battery"

    def test_min_mentions_filters(self) -> None:
        text = "The screen is great. The battery is great. The battery is excellent."
        assert list(ti.aspects(text, min_mentions=2, language="en").labels) == ["battery"]

    def test_spelling_variants_merge(self) -> None:
        text = "khana mast tha. khaana bekaar tha."
        result = ti.aspects(text, language="hi-en")
        assert len(result) == 1
        assert result.aspects[0].mentions == 2

    def test_extra_opinion_words_count(self) -> None:
        assert ti.aspects("biryani swaadisht thi", language="hi-en").labels == {
            "biryani": "positive"
        }

    def test_list_input(self) -> None:
        results = ti.aspects(["camera mast hai", "battery bekaar hai"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_language_fallback_for_undetermined_text(self) -> None:
        assert ti.aspects("xyzzy").method == "rules"

    def test_devanagari_is_rejected(self) -> None:
        with pytest.raises(ti.UnsupportedLanguageError):
            ti.aspects("यह फोन बहुत अच्छा है और कैमरा भी अच्छा है")

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_invalid_min_mentions(self, value: object) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.aspects("some text", min_mentions=value)  # type: ignore[call-overload]

    def test_listed(self) -> None:
        assert {b.name for b in ti.list_backends("aspects")} == {"rules"}

    def test_backend_can_be_called_directly(self) -> None:
        assert RuleAspects()(["camera mast hai"])[0].labels == {"camera": "positive"}

    def test_result_type_rejects_bad_label(self) -> None:
        with pytest.raises(ValueError, match="label"):
            ti.Aspect(name="x", label="great", score=0.5, mentions=1)

    def test_result_iterates_and_serializes(self) -> None:
        result = ti.aspects("camera mast hai")
        assert [a.name for a in result] == ["camera"]
        assert result.to_dict()["aspects"][0]["name"] == "camera"


class TestClauses:
    def test_contrast_always_splits(self) -> None:
        assert split_clauses("camera mast but battery") == ["camera mast", "battery"]

    def test_and_splits_only_when_both_sides_have_opinions(self) -> None:
        assert split_clauses("camera mast aur battery bekaar") == ["camera mast", "battery bekaar"]
        assert split_clauses("camera aur battery mast") == ["camera aur battery mast"]

    def test_empty_pieces_are_dropped(self) -> None:
        assert split_clauses("but") == []


class TestStats:
    def test_counts(self) -> None:
        result = ti.stats("The cat sat. The dog ran away!", language="en")
        assert result.words == 7
        assert result.sentences == 2
        assert result.unique_words == 6
        assert result.characters == 30
        assert result.type_token_ratio == pytest.approx(6 / 7, abs=1e-4)
        assert result.average_sentence_length == 3.5

    def test_flesch_formula(self) -> None:
        # 5 words, 1 sentence and 5 syllables: 206.835 - 1.015 * 5 - 84.6 * 1
        assert ti.stats("The cat sat on mats.", language="en").flesch_reading_ease == 117.16

    def test_simple_text_reads_easier_than_complex_text(self) -> None:
        simple = ti.stats("The dog ran. The cat sat.", language="en")
        hard = ti.stats(
            "Institutional considerations necessitate comprehensive organizational restructuring.",
            language="en",
        )
        assert simple.flesch_reading_ease is not None
        assert hard.flesch_reading_ease is not None
        assert simple.flesch_reading_ease > hard.flesch_reading_ease

    def test_flesch_is_only_for_english(self) -> None:
        assert ti.stats("Yeh phone bahut accha hai", language="hi-en").flesch_reading_ease is None

    def test_hinglish_variants_count_once(self) -> None:
        assert ti.stats("acha accha achha", language="hi-en").unique_words == 1

    def test_english_words_are_not_merged(self) -> None:
        assert ti.stats("acha accha", language="en").unique_words == 2

    def test_reading_time(self) -> None:
        assert ti.stats(" ".join(["word"] * 238), language="en").reading_time_seconds == 60.0

    @pytest.mark.parametrize("text", ["", "   ", "!!! ???"])
    def test_text_without_words(self, text: str) -> None:
        result = ti.stats(text, language="en")
        assert result.words == 0
        assert result.sentences == 0
        assert result.type_token_ratio == 0.0
        assert result.flesch_reading_ease is None

    def test_links_and_emoji_are_not_words(self) -> None:
        assert ti.stats("hello https://a.co \U0001f602 @sam", language="en").words == 1

    def test_hashtags_and_numbers_are_words(self) -> None:
        assert ti.stats("#cricket 2024", language="en").words == 2

    def test_language_is_detected(self) -> None:
        assert ti.stats("yeh phone bahut accha hai aur battery bhi mast hai").language == "hi-en"

    def test_list_input(self) -> None:
        results = ti.stats(["one two", "three"], language="en")
        assert [r.words for r in results] == [2, 1]

    def test_invalid_input(self) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.stats(None)  # type: ignore[call-overload]

    @pytest.mark.parametrize(
        ("word", "count"),
        [
            ("cat", 1),
            ("make", 1),
            ("makes", 1),
            ("table", 2),
            ("boxes", 2),
            ("walked", 1),
            ("wanted", 2),
            ("happy", 2),
            ("banana", 3),
            ("the", 1),
            ("rhythm", 1),
            ("hmm", 1),
        ],
    )
    def test_syllables(self, word: str, count: int) -> None:
        assert count_syllables(word) == count


class TestValidators:
    @pytest.mark.parametrize("number", ["79927398713", "4111111111111111", "4111 1111 1111 1111"])
    def test_luhn_accepts(self, number: str) -> None:
        assert luhn_valid(number)

    @pytest.mark.parametrize("number", ["79927398710", "4111111111111112", ""])
    def test_luhn_rejects(self, number: str) -> None:
        assert not luhn_valid(number)

    def test_card_rules(self) -> None:
        assert is_valid_card("4111111111111111")
        assert not is_valid_card("0000000000000000")
        assert not is_valid_card("79927398713")  # too short for a card

    def test_verhoeff_published_example(self) -> None:
        assert verhoeff_valid("2363")
        assert not verhoeff_valid("2364")
        assert not verhoeff_valid("")

    def test_verhoeff_catches_a_changed_digit(self) -> None:
        number = make_aadhaar()
        changed = number[:5] + str((int(number[5]) + 1) % 10) + number[6:]
        assert not verhoeff_valid(changed)

    def test_aadhaar_rules(self) -> None:
        assert is_valid_aadhaar(make_aadhaar())
        assert not is_valid_aadhaar(make_aadhaar("12345678901"))  # cannot start with 1
        assert not is_valid_aadhaar("2363")

    @pytest.mark.parametrize("value", ["ABCPE1234F", "abcpe1234f"])
    def test_pan_accepts(self, value: str) -> None:
        assert is_valid_pan(value)

    @pytest.mark.parametrize("value", ["ABCZE1234F", "ABCPE12345", "ABC", "1BCPE1234F"])
    def test_pan_rejects(self, value: str) -> None:
        assert not is_valid_pan(value)


class TestRedact:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Mail a.b@example.com now", "Mail [EMAIL] now"),
            ("call 9876543210 kal", "call [PHONE] kal"),
            ("call +91 98765 43210", "call [PHONE]"),
            ("call +91-98765-43210.", "call [PHONE]."),
            ("call 098765 43210", "call [PHONE]"),
            ("call 987 654 3210", "call [PHONE]"),
            ("ring +1 415 555 2671", "ring [PHONE]"),
            ("pay rahul@okaxis.", "pay [UPI]."),
            ("PAN is ABCPE1234F", "PAN is [PAN]"),
            ("card 4111 1111 1111 1111", "card [CARD]"),
            ("card 4111-1111-1111-1111", "card [CARD]"),
            ("see https://example.com/a?b=1", "see [URL]"),
        ],
    )
    def test_finds(self, text: str, expected: str) -> None:
        assert ti.redact(text).text == expected

    def test_aadhaar(self) -> None:
        number = make_aadhaar()
        spaced = f"{number[:4]} {number[4:8]} {number[8:]}"
        assert ti.redact(f"aadhaar {spaced}").text == "aadhaar [AADHAAR]"

    @pytest.mark.parametrize(
        "text",
        [
            "card 4111 1111 1111 1112",
            "order 1234567890 shipped",
            "date 12-05-2024",
            "price Rs 45000",
            "ABCZE1234F is a code",
            "@sam said hi",
            "plain text with no secrets",
            "",
        ],
    )
    def test_leaves_alone(self, text: str) -> None:
        result = ti.redact(text)
        assert result.text == text
        assert len(result) == 0

    def test_invalid_aadhaar_is_left_alone(self) -> None:
        number = make_aadhaar()
        changed = number[:5] + str((int(number[5]) + 1) % 10) + number[6:]
        assert ti.redact(f"id {changed}").text == f"id {changed}"

    def test_entity_positions_match_the_original(self) -> None:
        text = "a@b.co and 9876543210"
        result = ti.redact(text)
        assert [text[e.start : e.end] for e in result] == ["a@b.co", "9876543210"]
        assert [e.kind for e in result] == ["EMAIL", "PHONE"]

    def test_repr_hides_the_value(self) -> None:
        assert "a@b.co" not in repr(ti.redact("a@b.co"))

    def test_kinds_limits_what_is_found(self) -> None:
        result = ti.redact("a@b.co 9876543210", kinds=["email"])
        assert result.text == "[EMAIL] 9876543210"

    def test_kinds_accepts_a_single_string(self) -> None:
        assert ti.redact("a@b.co", kinds="EMAIL").text == "[EMAIL]"

    def test_card_is_not_split_into_other_kinds(self) -> None:
        assert [e.kind for e in ti.redact("4111 1111 1111 1111")] == ["CARD"]

    def test_unknown_kind(self) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.redact("text", kinds=["SSN"])

    def test_list_input(self) -> None:
        results = ti.redact(["a@b.co", "none"])
        assert isinstance(results, list)
        assert results[0].text == "[EMAIL]"

    def test_invalid_input(self) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.redact(None)  # type: ignore[call-overload]
