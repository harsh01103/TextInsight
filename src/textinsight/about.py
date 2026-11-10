"""Version and environment information, for bug reports and support."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ._version import __version__
from .core.config import config
from .core.registry import list_backends

__all__ = ["info", "show_versions"]

_OPTIONAL_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "transformers",
    "onnxruntime",
    "tokenizers",
    "anthropic",
)


def _installed(package: str) -> str | None:
    # Imported here: importlib.metadata loads socket, which a plain import should not.
    import importlib.metadata

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def info() -> dict[str, Any]:
    """Return TextInsight's version, the environment, optional packages and backends."""
    import platform

    return {
        "textinsight": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "location": str(Path(__file__).parent),
        "offline": config.offline,
        "cache_dir": str(config.cache_dir),
        "optional_packages": {name: _installed(name) for name in _OPTIONAL_PACKAGES},
        "backends": [
            {
                "task": backend.task,
                "name": backend.name,
                "available": backend.available,
                "extra": backend.extra,
            }
            for backend in list_backends()
        ],
    }


def show_versions() -> None:
    """Print versions and installed backends. Include this output in bug reports."""
    details = info()
    print(f"textinsight-nlp {details['textinsight']}")
    for label, key in (
        ("Python", "python"),
        ("Platform", "platform"),
        ("Installed at", "location"),
        ("Cache", "cache_dir"),
        ("Offline mode", "offline"),
    ):
        print(f"{label + ':':<14}{details[key]}")

    print("\nOptional packages")
    for name, version in details["optional_packages"].items():
        print(f"  {name:<14}{version or 'not installed'}")

    print("\nBackends")
    for backend in details["backends"]:
        state = "ready" if backend["available"] else f"needs [{backend['extra']}]"
        print(f"  {backend['task']:<10}{backend['name']:<12}{state}")
