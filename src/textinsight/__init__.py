"""TextInsight: lightweight, offline-first NLP for English and Hinglish text.

Importing this package is cheap: it never downloads models or opens network
connections. Models are fetched lazily the first time a backend needs them.
"""

import logging

from ._version import __version__
from .aspects import aspects
from .core import (
    Analysis,
    Aspect,
    AspectResult,
    BackendInfo,
    BackendNotAvailableError,
    Config,
    ConfigError,
    InvalidInputError,
    Keyword,
    KeywordResult,
    LanguageResult,
    LLMError,
    ModelDownloadError,
    PIIEntity,
    PIIResult,
    SentimentResult,
    SummaryResult,
    TextInsightError,
    TextStats,
    UnknownBackendError,
    UnsupportedLanguageError,
    config,
    list_backends,
)
from .keywords import keywords
from .langid import detect_language
from .pii import redact
from .pipeline import Pipeline, analyze
from .preprocess import clean, normalize, normalize_hinglish, remove_stopwords, sentences, tokenize
from .report import Report
from .sentiment import sentiment
from .stats import stats
from .summarize import summarize

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Analysis",
    "Aspect",
    "AspectResult",
    "BackendInfo",
    "BackendNotAvailableError",
    "Config",
    "ConfigError",
    "InvalidInputError",
    "Keyword",
    "KeywordResult",
    "LLMError",
    "LanguageResult",
    "ModelDownloadError",
    "PIIEntity",
    "PIIResult",
    "Pipeline",
    "Report",
    "SentimentResult",
    "SummaryResult",
    "TextInsightError",
    "TextStats",
    "UnknownBackendError",
    "UnsupportedLanguageError",
    "__version__",
    "analyze",
    "aspects",
    "clean",
    "config",
    "detect_language",
    "keywords",
    "list_backends",
    "normalize",
    "normalize_hinglish",
    "redact",
    "remove_stopwords",
    "sentences",
    "sentiment",
    "stats",
    "summarize",
    "tokenize",
]
