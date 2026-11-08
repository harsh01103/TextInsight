"""Exception hierarchy for TextInsight.

Every error raised on purpose by the library derives from :class:`TextInsightError`,
so users can catch all library errors with a single ``except`` clause. Each
exception also inherits from the closest built-in exception (``ValueError``,
``ImportError``, ``OSError``) so generic handlers keep working.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "BackendNotAvailableError",
    "ConfigError",
    "InvalidInputError",
    "ModelDownloadError",
    "TextInsightError",
    "UnknownBackendError",
    "UnsupportedLanguageError",
]


class TextInsightError(Exception):
    """Base class for all TextInsight errors."""


class ConfigError(TextInsightError, ValueError):
    """A configuration value (or environment variable) is invalid."""


class InvalidInputError(TextInsightError, ValueError):
    """Input text is missing, of the wrong type, or too large."""


class UnknownBackendError(TextInsightError, ValueError):
    """The requested task or backend name is not registered."""


class UnsupportedLanguageError(TextInsightError, ValueError):
    """No backend for the task supports the requested language."""


class BackendNotAvailableError(TextInsightError, ImportError):
    """A backend exists but its optional dependencies are not installed."""

    def __init__(
        self,
        task: str,
        backend: str,
        missing: Iterable[str],
        extra: str | None = None,
    ) -> None:
        self.task = task
        self.backend = backend
        self.missing = tuple(missing)
        self.extra = extra
        hint = (
            f'Install it with: pip install "textinsight[{extra}]"'
            if extra
            else f"Install the missing package(s): pip install {' '.join(self.missing)}"
        )
        message = (
            f"The '{backend}' backend for '{task}' needs {', '.join(self.missing)}, "
            f"which is not installed. {hint}"
        )
        super().__init__(message)


class ModelDownloadError(TextInsightError, OSError):
    """A model could not be downloaded, failed verification, or offline mode blocked it."""
