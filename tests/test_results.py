from __future__ import annotations

import dataclasses
import json

import pytest

from textinsight.core import (
    Keyword,
    KeywordResult,
    LanguageResult,
    SentimentResult,
    SummaryResult,
)


def test_sentiment_result_round_trips_to_json() -> None:
    result = SentimentResult(label="positive", score=0.8, backend="lexicon", language="hi-en")
    data = result.to_dict()
    assert data == {"label": "positive", "score": 0.8, "backend": "lexicon", "language": "hi-en"}
    assert json.loads(json.dumps(data)) == data


@pytest.mark.parametrize(
    ("label", "score"), [("happy", 0.1), ("positive", 1.5), ("negative", -1.01)]
)
def test_sentiment_result_validates(label: str, score: float) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        SentimentResult(label=label, score=score, backend="x")


def test_results_are_immutable() -> None:
    result = SentimentResult(label="neutral", score=0.0, backend="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.label = "positive"  # type: ignore[misc]


def test_language_result() -> None:
    result = LanguageResult(language="hi-en", score=0.9, token_languages=("hi", "en"))
    assert result.to_dict()["token_languages"] == ("hi", "en")
    with pytest.raises(ValueError):  # noqa: PT011
        LanguageResult(language="en", score=1.2)


def test_keyword_result_helpers() -> None:
    result = KeywordResult(
        keywords=(Keyword("machine learning", 0.9), Keyword("health", 0.4)), method="textrank"
    )
    assert result.texts == ["machine learning", "health"]
    assert len(result) == 2
    assert [k.score for k in result] == [0.9, 0.4]
    assert result.to_dict()["keywords"][0] == {"text": "machine learning", "score": 0.9}


def test_summary_result_defaults() -> None:
    result = SummaryResult(summary="Short.", method="textrank")
    assert result.sentence_indices == ()
