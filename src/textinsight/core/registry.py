"""Backend registry.

Backends register themselves with :func:`register_backend`; task functions look
them up with :func:`get_backend`. ``backend="auto"`` picks the highest-priority
installed backend that supports the requested language.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, TypeVar

from .base import Backend
from .exceptions import BackendNotAvailableError, UnknownBackendError, UnsupportedLanguageError

__all__ = [
    "AUTO",
    "BackendInfo",
    "get_backend",
    "list_backends",
    "register_backend",
    "unregister_backend",
]

AUTO = "auto"

BackendT = TypeVar("BackendT", bound=type[Backend[Any]])

_registry: dict[str, dict[str, type[Backend[Any]]]] = {}
_instances: dict[tuple[str, str], Backend[Any]] = {}
_lock = threading.RLock()


@dataclass(frozen=True)
class BackendInfo:
    """Description of a registered backend, as returned by :func:`list_backends`."""

    task: str
    name: str
    languages: tuple[str, ...]
    priority: int
    available: bool
    missing: tuple[str, ...]
    extra: str | None


def register_backend(cls: BackendT) -> BackendT:
    """Class decorator that adds a backend to the registry."""
    if not (isinstance(cls, type) and issubclass(cls, Backend)):
        raise TypeError("register_backend expects a Backend subclass")
    if not cls.task or not cls.name:
        raise ValueError(f"{cls.__name__} must define non-empty 'task' and 'name'")
    if cls.name == AUTO:
        raise ValueError(f"'{AUTO}' is reserved and cannot be used as a backend name")
    with _lock:
        task_backends = _registry.setdefault(cls.task, {})
        existing = task_backends.get(cls.name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"A backend named '{cls.name}' is already registered for '{cls.task}' "
                f"({existing.__module__}.{existing.__qualname__})"
            )
        task_backends[cls.name] = cls
    return cls


def unregister_backend(task: str, name: str) -> None:
    """Remove a backend (mainly useful in tests and plugins)."""
    with _lock:
        task_backends = _registry.get(task, {})
        task_backends.pop(name, None)
        if not task_backends:
            _registry.pop(task, None)
        _instances.pop((task, name), None)


def _backends_for(task: str) -> dict[str, type[Backend[Any]]]:
    try:
        return _registry[task]
    except KeyError:
        known = ", ".join(sorted(_registry)) or "none registered yet"
        raise UnknownBackendError(f"Unknown task '{task}'. Known tasks: {known}") from None


def _instance(cls: type[Backend[Any]]) -> Backend[Any]:
    key = (cls.task, cls.name)
    with _lock:
        inst = _instances.get(key)
        if inst is None or type(inst) is not cls:
            inst = cls()
            _instances[key] = inst
        return inst


def get_backend(task: str, name: str = AUTO, language: str | None = None) -> Backend[Any]:
    """Return a (shared) backend instance for ``task``.

    Args:
        task: Task name, e.g. ``"sentiment"``.
        name: Backend name, or ``"auto"`` to choose the best available one.
        language: If given, the backend must support this language code.

    Raises:
        UnknownBackendError: the task or backend name is not registered.
        UnsupportedLanguageError: no matching backend supports ``language``.
        BackendNotAvailableError: the backend's optional dependencies are missing.
    """
    with _lock:
        backends = dict(_backends_for(task))

    if name != AUTO:
        cls = backends.get(name)
        if cls is None:
            raise UnknownBackendError(
                f"Unknown backend '{name}' for '{task}'. "
                f"Available: {', '.join(sorted(backends))}, or '{AUTO}'"
            )
        if language is not None and not cls.supports(language):
            raise UnsupportedLanguageError(
                f"The '{name}' backend for '{task}' does not support language '{language}'. "
                f"It supports: {', '.join(sorted(cls.languages))}"
            )
        missing = cls.missing_dependencies()
        if missing:
            raise BackendNotAvailableError(task, name, missing, cls.extra)
        return _instance(cls)

    candidates = [cls for cls in backends.values() if language is None or cls.supports(language)]
    if not candidates:
        raise UnsupportedLanguageError(f"No '{task}' backend supports language '{language}'")
    candidates.sort(key=lambda c: (-c.priority, c.name))
    for cls in candidates:
        if cls.is_available():
            return _instance(cls)
    best = candidates[0]
    raise BackendNotAvailableError(task, best.name, best.missing_dependencies(), best.extra)


def list_backends(task: str | None = None) -> list[BackendInfo]:
    """Describe registered backends, optionally for a single task."""
    with _lock:
        if task is None:
            classes = [cls for group in _registry.values() for cls in group.values()]
        else:
            classes = list(_backends_for(task).values())
    infos = []
    for cls in classes:
        missing = cls.missing_dependencies()
        infos.append(
            BackendInfo(
                task=cls.task,
                name=cls.name,
                languages=tuple(sorted(cls.languages)),
                priority=cls.priority,
                available=not missing,
                missing=missing,
                extra=cls.extra,
            )
        )
    return sorted(infos, key=lambda i: (i.task, -i.priority, i.name))
