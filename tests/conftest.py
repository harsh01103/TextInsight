from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from textinsight.core import config, registry


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test gets default settings and a private cache folder."""
    for var in (
        "TEXTINSIGHT_CACHE_DIR",
        "TEXTINSIGHT_OFFLINE",
        "TEXTINSIGHT_DEVICE",
        "TEXTINSIGHT_MAX_INPUT_CHARS",
    ):
        monkeypatch.delenv(var, raising=False)
    config.reset()
    config.cache_dir = tmp_path / "cache"
    saved = config.to_dict()
    yield
    # Restore explicitly: tests may leave invalid TEXTINSIGHT_* variables set
    # until monkeypatch undoes them, so re-reading the environment is unsafe here.
    for key, value in saved.items():
        setattr(config, key, value)


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Snapshot the backend registry and restore it after the test."""
    saved = {task: dict(backends) for task, backends in registry._registry.items()}
    saved_instances = dict(registry._instances)
    registry._registry.clear()
    registry._instances.clear()
    yield
    registry._registry.clear()
    registry._registry.update(saved)
    registry._instances.clear()
    registry._instances.update(saved_instances)
