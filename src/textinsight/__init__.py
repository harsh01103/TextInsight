"""TextInsight: lightweight, offline-first NLP for English and Hinglish text.

Importing this package is cheap: it never downloads models or opens network
connections. Models are fetched lazily the first time a backend needs them.
"""

import logging

from ._version import __version__
from .core import (
    BackendInfo,
    BackendNotAvailableError,
    Config,
    ConfigError,
    InvalidInputError,
    Keyword,
    KeywordResult,
    LanguageResult,
    ModelDownloadError,
    SentimentResult,
    SummaryResult,
    TextInsightError,
    UnknownBackendError,
    UnsupportedLanguageError,
    config,
    list_backends,
)
from .preprocess import clean

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BackendInfo",
    "BackendNotAvailableError",
    "Config",
    "ConfigError",
    "InvalidInputError",
    "Keyword",
    "KeywordResult",
    "LanguageResult",
    "ModelDownloadError",
    "SentimentResult",
    "SummaryResult",
    "TextInsightError",
    "UnknownBackendError",
    "UnsupportedLanguageError",
    "__version__",
    "clean",
    "config",
    "list_backends",
]
