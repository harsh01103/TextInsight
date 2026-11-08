from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from textinsight.core import ModelDownloadError, ModelManager, ModelSpec, config

PAYLOAD = b"pretend model weights"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source" / "model.bin"
    path.parent.mkdir()
    path.write_bytes(PAYLOAD)
    return path


@pytest.fixture
def spec(source: Path) -> ModelSpec:
    return ModelSpec(name="demo", version="1.0", url=source.as_uri(), sha256=DIGEST)


@pytest.fixture
def manager(tmp_path: Path) -> ModelManager:
    return ModelManager(cache_dir=tmp_path / "cache")


def test_fetch_downloads_verifies_and_caches(manager: ModelManager, spec: ModelSpec) -> None:
    assert not manager.is_cached(spec)
    path = manager.fetch(spec)
    assert path == manager.root / "demo" / "1.0" / "model.bin"
    assert path.read_bytes() == PAYLOAD
    assert manager.is_cached(spec)
    assert manager.list_cached() == [path]


def test_second_fetch_uses_cache(manager: ModelManager, spec: ModelSpec, source: Path) -> None:
    first = manager.fetch(spec)
    source.unlink()  # the source is gone, so a re-download would fail
    assert manager.fetch(spec) == first


def test_checksum_mismatch_is_rejected(manager: ModelManager, source: Path) -> None:
    bad = ModelSpec(name="demo", version="1.0", url=source.as_uri(), sha256="0" * 64)
    with pytest.raises(ModelDownloadError, match="Checksum mismatch"):
        manager.fetch(bad)
    assert manager.list_cached() == []


def test_corrupted_cache_is_redownloaded(manager: ModelManager, spec: ModelSpec) -> None:
    path = manager.fetch(spec)
    path.write_bytes(b"corrupted")
    assert not manager.is_cached(spec)
    assert manager.fetch(spec).read_bytes() == PAYLOAD


def test_offline_mode_blocks_download(manager: ModelManager, spec: ModelSpec) -> None:
    config.offline = True
    with pytest.raises(ModelDownloadError, match="offline mode"):
        manager.fetch(spec)


def test_offline_mode_still_uses_cache(manager: ModelManager, spec: ModelSpec) -> None:
    manager.fetch(spec)
    config.offline = True
    assert manager.fetch(spec).is_file()


def test_download_failure_is_wrapped(manager: ModelManager, tmp_path: Path) -> None:
    missing = ModelSpec(
        name="demo", version="1.0", url=(tmp_path / "nope.bin").as_uri(), sha256=DIGEST
    )
    with pytest.raises(ModelDownloadError, match="Could not download"):
        manager.fetch(missing)
    assert not list((manager.root / "demo" / "1.0").glob("*.part"))


def test_load_keeps_object_in_memory(manager: ModelManager, spec: ModelSpec) -> None:
    calls: list[Path] = []

    def loader(path: Path) -> bytes:
        calls.append(path)
        return path.read_bytes()

    assert manager.load(spec, loader) == PAYLOAD
    assert manager.load(spec, loader) == PAYLOAD
    assert len(calls) == 1


def test_remove_and_clear(manager: ModelManager, spec: ModelSpec) -> None:
    manager.fetch(spec)
    manager.remove(spec)
    assert not manager.is_cached(spec)
    manager.fetch(spec)
    manager.clear()
    assert manager.list_cached() == []
    manager.clear()  # clearing an empty cache is fine


def test_default_root_follows_config(tmp_path: Path) -> None:
    manager = ModelManager()
    config.cache_dir = tmp_path / "elsewhere"
    assert manager.root == tmp_path / "elsewhere" / "models"


def test_explicit_filename(source: Path) -> None:
    spec = ModelSpec(
        name="demo", version="2", url=source.as_uri(), sha256=DIGEST, filename="weights.onnx"
    )
    assert spec.file_name == "weights.onnx"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "http://example.com/model.bin"},  # plain http is not allowed
        {"url": "ftp://example.com/model.bin"},
        {"sha256": "abc"},
        {"sha256": "G" * 64},
        {"name": "../escape"},
        {"version": "1.0/../../x"},
        {"filename": "../evil.bin"},
    ],
)
def test_spec_validation(kwargs: dict[str, str]) -> None:
    base = {
        "name": "demo",
        "version": "1.0",
        "url": "https://example.com/model.bin",
        "sha256": DIGEST,
    }
    with pytest.raises(ValueError):  # noqa: PT011
        ModelSpec(**{**base, **kwargs})
