"""Base class for all backends.

A backend is one concrete way of performing a task, for example the
``"lexicon"`` backend for the ``"sentiment"`` task. Backends declare what they
need through class attributes and do their expensive setup in :meth:`load`,
which runs lazily the first time the backend is used.

Example::

    from textinsight.core import Backend, register_backend, SentimentResult

    @register_backend
    class MyBackend(Backend[SentimentResult]):
        task = "sentiment"
        name = "mine"
        languages = frozenset({"en"})

        def predict(self, texts):
            return [SentimentResult("neutral", 0.0, backend=self.name) for _ in texts]
"""

from __future__ import annotations

import importlib.util
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Generic, TypeVar

__all__ = ["ANY_LANGUAGE", "Backend"]

ResultT = TypeVar("ResultT")

ANY_LANGUAGE = "*"


def _module_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class Backend(ABC, Generic[ResultT]):
    """Abstract base class for task backends."""

    #: Task this backend implements, e.g. ``"sentiment"``.
    task: ClassVar[str] = ""
    #: Unique name within the task, e.g. ``"lexicon"``.
    name: ClassVar[str] = ""
    #: Supported language codes; include ``"*"`` for language-independent backends.
    languages: ClassVar[frozenset[str]] = frozenset({"en"})
    #: Importable module names this backend needs (checked without importing them).
    requires: ClassVar[tuple[str, ...]] = ()
    #: Name of the pip extra that installs ``requires``, e.g. ``"onnx"``.
    extra: ClassVar[str | None] = None
    #: Higher priority wins when the user asks for ``backend="auto"``.
    priority: ClassVar[int] = 0

    def __init__(self) -> None:
        self._loaded = False
        self._load_lock = threading.Lock()

    # -------------------------------------------------------------- metadata
    @classmethod
    def missing_dependencies(cls) -> tuple[str, ...]:
        """Modules from :attr:`requires` that are not installed."""
        return tuple(m for m in cls.requires if not _module_installed(m))

    @classmethod
    def is_available(cls) -> bool:
        """True when every required module is installed."""
        return not cls.missing_dependencies()

    @classmethod
    def supports(cls, language: str) -> bool:
        """True when this backend can handle ``language``."""
        return ANY_LANGUAGE in cls.languages or language in cls.languages

    # ------------------------------------------------------------- lifecycle
    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Do expensive setup (load models, lexicons). Called once, lazily."""

    def ensure_loaded(self) -> None:
        """Run :meth:`load` exactly once, even when called from several threads."""
        if self._loaded:
            return
        with self._load_lock:
            if not self._loaded:
                self.load()
                self._loaded = True

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> list[ResultT]:
        """Return one result per input text. Inputs are already validated."""

    def __call__(self, texts: Sequence[str]) -> list[ResultT]:
        self.ensure_loaded()
        return self.predict(texts)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} task={self.task!r} name={self.name!r}>"
