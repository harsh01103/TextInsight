"""Core building blocks shared by every TextInsight module."""

from .base import ANY_LANGUAGE, Backend
from .config import Config, config
from .exceptions import (
    BackendNotAvailableError,
    ConfigError,
    InvalidInputError,
    ModelDownloadError,
    TextInsightError,
    UnknownBackendError,
    UnsupportedLanguageError,
)
from .models import ModelManager, ModelSpec, models
from .registry import (
    AUTO,
    BackendInfo,
    get_backend,
    list_backends,
    register_backend,
    unregister_backend,
)
from .results import (
    SENTIMENT_LABELS,
    Keyword,
    KeywordResult,
    LanguageResult,
    Result,
    SentimentResult,
    SummaryResult,
)

__all__ = [
    "ANY_LANGUAGE",
    "AUTO",
    "SENTIMENT_LABELS",
    "Backend",
    "BackendInfo",
    "BackendNotAvailableError",
    "Config",
    "ConfigError",
    "InvalidInputError",
    "Keyword",
    "KeywordResult",
    "LanguageResult",
    "ModelDownloadError",
    "ModelManager",
    "ModelSpec",
    "Result",
    "SentimentResult",
    "SummaryResult",
    "TextInsightError",
    "UnknownBackendError",
    "UnsupportedLanguageError",
    "config",
    "get_backend",
    "list_backends",
    "models",
    "register_backend",
    "unregister_backend",
]
