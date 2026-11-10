"""TextInsight: lightweight, offline-first NLP for English and Hinglish text.

Quick start::

    import textinsight as ti

    ti.sentiment("yeh phone mast hai").label         # 'positive'
    ti.aspects("camera mast hai but battery bekaar hai").labels
    ti.keywords(text, top_n=5).texts
    ti.summarize(text, max_sentences=2).summary
    ti.redact("call me at 9876543210").text           # 'call me at [PHONE]'
    ti.analyze(texts).to_html("report.html")

Use ``help(ti.sentiment)`` (or any other function) for its options,
``ti.list_backends()`` to see which backends are installed, and
``ti.show_versions()`` for details to include in a bug report. The same tasks
are available from the ``textinsight`` command; run ``textinsight --help``.

Importing this package is cheap: it never downloads models or opens network
connections. Models are fetched lazily the first time a backend needs them.
"""

import logging

from ._version import __version__
from .about import info, show_versions
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
    "info",
    "keywords",
    "list_backends",
    "normalize",
    "normalize_hinglish",
    "redact",
    "remove_stopwords",
    "sentences",
    "sentiment",
    "show_versions",
    "stats",
    "summarize",
    "tokenize",
]
