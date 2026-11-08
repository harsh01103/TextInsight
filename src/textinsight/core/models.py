"""Model manager: download, verify, cache and lazily load model files.

Nothing here touches the network at import time. A model file is downloaded
only when a backend first asks for it, is verified with SHA-256, and is stored
under ``config.cache_dir/models/<name>/<version>/``. In offline mode only
already-cached files are used.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import unquote, urlparse

from .config import config
from .exceptions import ModelDownloadError

__all__ = ["ModelManager", "ModelSpec", "models"]

logger = logging.getLogger(__name__)

T = TypeVar("T")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ALLOWED_SCHEMES = {"https", "file"}
_CHUNK = 1 << 20
_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class ModelSpec:
    """Where to get a model file and how to verify it."""

    name: str
    version: str
    url: str
    sha256: str
    filename: str | None = None

    def __post_init__(self) -> None:
        for label, value in (("name", self.name), ("version", self.version)):
            if not _SAFE_PART.match(value):
                raise ValueError(f"ModelSpec.{label} must be a simple identifier, got {value!r}")
        if not _SHA256.match(self.sha256):
            raise ValueError("ModelSpec.sha256 must be 64 lowercase hex characters")
        scheme = urlparse(self.url).scheme
        if scheme not in _ALLOWED_SCHEMES:
            raise ValueError(f"ModelSpec.url must use https (or file for tests), got {scheme!r}")
        if not _SAFE_PART.match(self.file_name):
            raise ValueError(f"Unsafe model file name {self.file_name!r}")

    @property
    def file_name(self) -> str:
        if self.filename:
            return self.filename
        return Path(unquote(urlparse(self.url).path)).name or "model.bin"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelManager:
    """Handles model files for all backends."""

    def __init__(self, cache_dir: str | os.PathLike[str] | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._loaded: dict[tuple[str, str, str], Any] = {}
        self._verified: set[tuple[Path, int, int]] = set()
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        """Folder holding all models (follows ``config.cache_dir`` unless overridden)."""
        base = self._cache_dir if self._cache_dir is not None else config.cache_dir
        return base / "models"

    def path_for(self, spec: ModelSpec) -> Path:
        return self.root / spec.name / spec.version / spec.file_name

    # ------------------------------------------------------------ verification
    def _is_valid(self, path: Path, spec: ModelSpec) -> bool:
        if not path.is_file():
            return False
        stat = path.stat()
        key = (path, stat.st_size, stat.st_mtime_ns)
        if key in self._verified:
            return True
        if sha256_of(path) == spec.sha256:
            self._verified.add(key)
            return True
        return False

    def is_cached(self, spec: ModelSpec) -> bool:
        """True if a verified copy of the model is on disk."""
        with self._lock:
            return self._is_valid(self.path_for(spec), spec)

    # ---------------------------------------------------------------- fetching
    def fetch(self, spec: ModelSpec) -> Path:
        """Return the local path of a model, downloading it first if needed."""
        with self._lock:
            path = self.path_for(spec)
            if self._is_valid(path, spec):
                return path
            if path.exists():
                logger.warning("Cached model %s failed verification; re-downloading", path)
                path.unlink()
            if config.offline:
                raise ModelDownloadError(
                    f"Model '{spec.name}' ({spec.version}) is not cached and offline mode is on. "
                    "Set textinsight.config.offline = False to download it, or copy the file to "
                    f"{path}"
                )
            self._download(spec, path)
            return path

    def _download(self, spec: ModelSpec, dest: Path) -> None:
        from urllib.request import urlopen  # imported lazily: no network code at import

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        digest = hashlib.sha256()
        logger.info("Downloading model %s %s from %s", spec.name, spec.version, spec.url)
        try:
            with urlopen(spec.url, timeout=_TIMEOUT_SECONDS) as response, tmp.open("wb") as out:
                for chunk in iter(lambda: response.read(_CHUNK), b""):
                    digest.update(chunk)
                    out.write(chunk)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"Could not download model '{spec.name}' from {spec.url}: {exc}"
            ) from exc
        if digest.hexdigest() != spec.sha256:
            tmp.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"Checksum mismatch for model '{spec.name}': expected {spec.sha256}, "
                f"got {digest.hexdigest()}. The download may be corrupted or tampered with."
            )
        os.replace(tmp, dest)

    # ----------------------------------------------------------------- loading
    def load(self, spec: ModelSpec, loader: Callable[[Path], T]) -> T:
        """Fetch a model and load it with ``loader``; the loaded object is kept in memory."""
        key = (spec.name, spec.version, spec.file_name)
        with self._lock:
            if key not in self._loaded:
                self._loaded[key] = loader(self.fetch(spec))
            result: T = self._loaded[key]
            return result

    # ---------------------------------------------------------------- cleanup
    def list_cached(self) -> list[Path]:
        """All model files currently on disk."""
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix != ".part")

    def remove(self, spec: ModelSpec) -> None:
        """Delete one model from disk and memory."""
        with self._lock:
            self._loaded.pop((spec.name, spec.version, spec.file_name), None)
            self.path_for(spec).unlink(missing_ok=True)

    def clear(self) -> None:
        """Delete every cached model from disk and memory."""
        with self._lock:
            self._loaded.clear()
            self._verified.clear()
            if self.root.is_dir():
                shutil.rmtree(self.root)


models = ModelManager()
