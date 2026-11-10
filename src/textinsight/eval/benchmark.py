"""Benchmark runners used by ``textinsight benchmark``."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Sequence
from statistics import fmean
from typing import Any, cast

from ..core.registry import get_backend
from ..keywords import keywords
from ..langid import HeuristicLanguageDetector, detect_language
from ..preprocess import sentences as split_sentences
from ..sentiment import sentiment
from ..summarize import summarize
from .datasets import Example, KeyphraseExample, SummaryExample
from .metrics import classification_report
from .text_metrics import keyphrase_scores, rouge_scores

__all__ = [
    "DEFAULT_METHODS",
    "KEYWORD_METHODS",
    "SENTIMENT_METHODS",
    "SUMMARY_METHODS",
    "evaluate_keywords",
    "evaluate_language",
    "evaluate_sentiment",
    "evaluate_summaries",
]

Predictor = Callable[[list[str]], list[str]]

KEYWORD_METHODS = ("tfidf", "rake", "textrank")
SUMMARY_METHODS = ("lead", "frequency", "textrank")

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


def evaluate_keywords(
    examples: Sequence[KeyphraseExample],
    methods: Sequence[str] = KEYWORD_METHODS,
    cutoffs: Sequence[int] = (5, 10),
) -> dict[str, dict[str, float]]:
    """Mean precision, recall and F1 at each cutoff; all documents share an IDF table."""
    texts = [example.text for example in examples]
    report: dict[str, dict[str, float]] = {}
    for method in methods:
        predicted = keywords(texts, method=method, top_n=max(cutoffs), language="en")
        scores: dict[str, float] = {}
        for k in cutoffs:
            rows = [
                keyphrase_scores(result.texts, example.keyphrases, k)
                for result, example in zip(predicted, examples, strict=True)
            ]
            for metric in ("precision", "recall", "f1"):
                scores[f"{metric}@{k}"] = fmean(row[metric] for row in rows)
        report[method] = scores
    return report


def _lead(article: str, count: int) -> str:
    return " ".join(split_sentences(article)[:count])


def evaluate_summaries(
    examples: Sequence[SummaryExample],
    methods: Sequence[str] = SUMMARY_METHODS,
    sentences: int = 3,
) -> dict[str, dict[str, float]]:
    """Mean ROUGE F1 of ``sentences``-sentence summaries; ``lead`` is the first sentences."""
    report: dict[str, dict[str, float]] = {}
    for method in methods:
        candidates = (
            [_lead(example.article, sentences) for example in examples]
            if method == "lead"
            else [
                result.summary
                for result in summarize(
                    [example.article for example in examples],
                    ratio=1.0,
                    max_sentences=sentences,
                    method=method,
                    language="en",
                )
            ]
        )
        rows = [
            rouge_scores(candidate, example.reference)
            for candidate, example in zip(candidates, examples, strict=True)
        ]
        report[method] = {
            name: fmean(row[name]["f1"] for row in rows)
            for name in ("rouge-1", "rouge-2", "rouge-l")
        }
    return report
