"""Benchmark runners used by ``textinsight benchmark``."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Sequence
from typing import Any, cast

from ..core.registry import get_backend
from ..langid import HeuristicLanguageDetector, detect_language
from ..sentiment import sentiment
from .datasets import Example
from .metrics import classification_report

__all__ = ["DEFAULT_METHODS", "SENTIMENT_METHODS", "evaluate_language", "evaluate_sentiment"]

Predictor = Callable[[list[str]], list[str]]

_LABELS = ("negative", "neutral", "positive")
_VADER_THRESHOLD = 0.05
_GOLD_LANGUAGE = {"Eng": "en", "Hin": "hi"}
_OPTIONAL_METHODS = {"vader": "vaderSentiment", "hybrid": "transformers"}


def _three_way(label: str) -> str:
    return "neutral" if label == "mixed" else label  # the dataset has no "mixed" class


def _majority(texts: list[str]) -> list[str]:
    return ["neutral"] * len(texts)  # the largest class in the training split


def _vader(texts: list[str]) -> list[str]:
    module = importlib.import_module("vaderSentiment.vaderSentiment")
    analyzer = module.SentimentIntensityAnalyzer()
    labels = []
    for text in texts:
        compound = analyzer.polarity_scores(text)["compound"]
        if compound >= _VADER_THRESHOLD:
            labels.append("positive")
        elif compound <= -_VADER_THRESHOLD:
            labels.append("negative")
        else:
            labels.append("neutral")
    return labels


def _library(backend: str, language: str | None) -> Predictor:
    def predict(texts: list[str]) -> list[str]:
        return [
            _three_way(result.label)
            for result in sentiment(texts, backend=backend, language=language)
        ]

    return predict


def _auto(texts: list[str]) -> list[str]:
    """Run the full pipeline, sending Devanagari text (unsupported) to neutral."""
    detected = [result.language for result in detect_language(texts)]
    supported = [i for i, code in enumerate(detected) if code != "hi"]
    labels = ["neutral"] * len(texts)
    results = sentiment([texts[i] for i in supported])
    for index, result in zip(supported, results, strict=True):
        labels[index] = _three_way(result.label)
    return labels


SENTIMENT_METHODS: dict[str, Predictor] = {
    "majority": _majority,
    "vader": _vader,
    "lexicon": _library("lexicon", "hi-en"),
    "classical": _library("classical", "hi-en"),
    "hybrid": _library("hybrid", "hi-en"),
    "auto": _auto,
}


# "hybrid" is left out by default because running it can download a 438 MB model.
DEFAULT_METHODS = ("majority", "vader", "lexicon", "classical", "auto")


def evaluate_sentiment(
    examples: Sequence[Example], methods: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Score sentiment methods on labelled examples, skipping any not installed."""
    chosen = list(methods if methods is not None else DEFAULT_METHODS)
    missing = {
        method
        for method, module in _OPTIONAL_METHODS.items()
        if importlib.util.find_spec(module) is None
    }
    chosen = [method for method in chosen if method not in missing]
    texts = [example.text for example in examples]
    gold = [example.label for example in examples]
    return {
        name: classification_report(gold, SENTIMENT_METHODS[name](texts), _LABELS)
        for name in chosen
    }


def evaluate_language(examples: Sequence[Example]) -> dict[str, Any]:
    """Compare per-word language labels with the dataset's gold tags."""
    backend = cast(HeuristicLanguageDetector, get_backend("langid", "heuristic"))
    backend.ensure_loaded()
    gold: list[str] = []
    predicted: list[str] = []
    for example in examples:
        pairs = [
            (_GOLD_LANGUAGE[tag], backend._label_word(token)[0])
            for token, tag in zip(example.tokens, example.token_languages, strict=True)
            if tag in _GOLD_LANGUAGE
        ]
        guesses = [guess for _, guess in pairs]
        majority = "hi" if guesses.count("hi") >= guesses.count("en") else "en"
        for truth, guess in pairs:
            gold.append(truth)
            predicted.append(majority if guess in {"ambiguous", "other"} else guess)
    return classification_report(gold, predicted, ("en", "hi"))
