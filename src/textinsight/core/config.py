"""Global configuration.

Settings are read from environment variables when the package is imported and
can be changed at runtime::

    import textinsight as ti

    ti.config.offline = True
    ti.config.cache_dir = "D:/models"

    with ti.config.override(offline=False):
        ...  # temporarily changed, restored afterwards
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .exceptions import ConfigError

__all__ = ["Config", "config"]

ENV_CACHE_DIR = "TEXTINSIGHT_CACHE_DIR"
ENV_OFFLINE = "TEXTINSIGHT_OFFLINE"
ENV_DEVICE = "TEXTINSIGHT_DEVICE"
ENV_MAX_INPUT_CHARS = "TEXTINSIGHT_MAX_INPUT_CHARS"

DEFAULT_MAX_INPUT_CHARS = 1_000_000
_DEVICE_PATTERN = re.compile(r"^(cpu|mps|cuda(:\d+)?)$")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def default_cache_dir() -> Path:
    """Return the per-user cache folder for this operating system."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "textinsight"


def _parse_bool(value: str | bool, name: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(f"{name} must be a boolean (true/false/1/0), got {value!r}")


class Config:
    """Runtime settings shared by every TextInsight module."""

    __slots__ = ("_cache_dir", "_device", "_max_input_chars", "_offline")
    _FIELDS = ("cache_dir", "offline", "device", "max_input_chars")

    def __init__(self) -> None:
        self._cache_dir: Path = default_cache_dir()
        self._offline = False
        self._device = "cpu"
        self._max_input_chars = DEFAULT_MAX_INPUT_CHARS
        self.reset()

    # ------------------------------------------------------------------ fields
    @property
    def cache_dir(self) -> Path:
        """Folder where downloaded models are stored."""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: str | os.PathLike[str]) -> None:
        if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
            raise ConfigError(f"cache_dir must be a non-empty path, got {value!r}")
        self._cache_dir = Path(value).expanduser()

    @property
    def offline(self) -> bool:
        """When True, models are never downloaded; only cached models are used."""
        return self._offline

    @offline.setter
    def offline(self, value: bool | str) -> None:
        if not isinstance(value, (bool, str)):
            raise ConfigError(f"offline must be a bool, got {value!r}")
        self._offline = _parse_bool(value, "offline")

    @property
    def device(self) -> str:
        """Compute device for optional neural backends: 'cpu', 'cuda', 'cuda:N' or 'mps'."""
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        if not isinstance(value, str) or not _DEVICE_PATTERN.match(value.strip().lower()):
            raise ConfigError(f"device must be 'cpu', 'cuda', 'cuda:N' or 'mps', got {value!r}")
        self._device = value.strip().lower()

    @property
    def max_input_chars(self) -> int:
        """Maximum characters accepted per text, to protect against accidental huge inputs."""
        return self._max_input_chars

    @max_input_chars.setter
    def max_input_chars(self, value: int | str) -> None:
        if isinstance(value, bool):
            raise ConfigError("max_input_chars must be a positive integer, got a bool")
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ConfigError(
                f"max_input_chars must be a positive integer, got {value!r}"
            ) from None
        if number <= 0:
            raise ConfigError(f"max_input_chars must be a positive integer, got {value!r}")
        self._max_input_chars = number

    # ----------------------------------------------------------------- helpers
    def reset(self) -> None:
        """Restore defaults, then apply any TEXTINSIGHT_* environment variables."""
        env = os.environ
        self.cache_dir = env.get(ENV_CACHE_DIR) or default_cache_dir()
        self.offline = env.get(ENV_OFFLINE, "false")
        self.device = env.get(ENV_DEVICE, "cpu")
        self.max_input_chars = env.get(ENV_MAX_INPUT_CHARS, DEFAULT_MAX_INPUT_CHARS)

    @contextmanager
    def override(self, **changes: Any) -> Iterator[Config]:
        """Temporarily change settings inside a ``with`` block."""
        unknown = set(changes) - set(self._FIELDS)
        if unknown:
            raise ConfigError(
                f"Unknown setting(s): {', '.join(sorted(unknown))}. "
                f"Valid settings: {', '.join(self._FIELDS)}"
            )
        saved = self.to_dict()
        try:
            for key, value in changes.items():
                setattr(self, key, value)
            yield self
        finally:
            for key, value in saved.items():
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._FIELDS}

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"Config({fields})"


config = Config()
