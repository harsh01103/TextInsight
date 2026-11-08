from __future__ import annotations

from pathlib import Path

import pytest

from textinsight.core import Config, ConfigError, config
from textinsight.core.config import DEFAULT_MAX_INPUT_CHARS, default_cache_dir


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config()
    assert cfg.offline is False
    assert cfg.device == "cpu"
    assert cfg.max_input_chars == DEFAULT_MAX_INPUT_CHARS
    assert cfg.cache_dir == default_cache_dir()
    assert cfg.cache_dir.name == "textinsight"


def test_environment_variables_are_applied(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEXTINSIGHT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TEXTINSIGHT_OFFLINE", "yes")
    monkeypatch.setenv("TEXTINSIGHT_DEVICE", "CUDA:1")
    monkeypatch.setenv("TEXTINSIGHT_MAX_INPUT_CHARS", "500")
    cfg = Config()
    assert cfg.cache_dir == tmp_path
    assert cfg.offline is True
    assert cfg.device == "cuda:1"
    assert cfg.max_input_chars == 500


def test_invalid_environment_variable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXTINSIGHT_OFFLINE", "maybe")
    with pytest.raises(ConfigError, match="offline"):
        Config()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("offline", "perhaps"),
        ("offline", 1),
        ("device", "gpu"),
        ("device", 3),
        ("max_input_chars", 0),
        ("max_input_chars", -5),
        ("max_input_chars", "lots"),
        ("max_input_chars", True),
        ("cache_dir", ""),
        ("cache_dir", 42),
    ],
)
def test_invalid_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ConfigError):
        setattr(config, field, value)


def test_cache_dir_accepts_strings_and_expands_user() -> None:
    config.cache_dir = "~/ti-cache"
    assert config.cache_dir == Path("~/ti-cache").expanduser()


def test_unknown_attribute_cannot_be_set() -> None:
    with pytest.raises(AttributeError):
        config.typo = True  # type: ignore[attr-defined]


def test_override_restores_previous_values() -> None:
    config.offline = False
    with config.override(offline=True, device="mps") as cfg:
        assert cfg is config
        assert config.offline is True
        assert config.device == "mps"
    assert config.offline is False
    assert config.device == "cpu"


def test_override_restores_even_on_error() -> None:
    with pytest.raises(RuntimeError), config.override(offline=True):
        raise RuntimeError("boom")
    assert config.offline is False


def test_override_rejects_unknown_settings() -> None:
    with pytest.raises(ConfigError, match="Unknown setting"), config.override(oflline=True):
        pass


def test_to_dict_and_repr() -> None:
    data = config.to_dict()
    assert set(data) == {"cache_dir", "offline", "device", "max_input_chars"}
    assert "offline=False" in repr(config)
