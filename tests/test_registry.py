from __future__ import annotations

import threading
from collections.abc import Sequence

import pytest

from textinsight.core import (
    ANY_LANGUAGE,
    Backend,
    BackendNotAvailableError,
    SentimentResult,
    UnknownBackendError,
    UnsupportedLanguageError,
    get_backend,
    list_backends,
    register_backend,
    unregister_backend,
)

pytestmark = pytest.mark.usefixtures("clean_registry")


class _Fake(Backend[SentimentResult]):
    task = "sentiment"
    label = "neutral"

    def predict(self, texts: Sequence[str]) -> list[SentimentResult]:
        return [SentimentResult(self.label, 0.0, backend=self.name) for _ in texts]


def _make(name: str, **attrs: object) -> type[_Fake]:
    return type(f"Fake_{name}", (_Fake,), {"name": name, **attrs})


def test_register_and_get_by_name() -> None:
    cls = register_backend(_make("lexicon"))
    backend = get_backend("sentiment", "lexicon")
    assert isinstance(backend, cls)
    assert backend is get_backend("sentiment", "lexicon"), "instances are shared"


def test_backend_call_loads_once_then_predicts() -> None:
    calls: list[int] = []

    def load(self: Backend[SentimentResult]) -> None:
        calls.append(1)

    register_backend(_make("lazy", load=load))
    backend = get_backend("sentiment", "lazy")
    assert not backend.loaded
    assert backend(["a", "b"])[0].label == "neutral"
    backend(["c"])
    assert backend.loaded
    assert calls == [1]


def test_load_runs_once_across_threads() -> None:
    calls: list[int] = []
    gate = threading.Barrier(8)

    def load(self: Backend[SentimentResult]) -> None:
        calls.append(1)

    register_backend(_make("threaded", load=load))
    backend = get_backend("sentiment", "threaded")

    def worker() -> None:
        gate.wait()
        backend.ensure_loaded()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [1]


def test_auto_picks_highest_priority_supporting_language() -> None:
    register_backend(_make("low", priority=1, languages=frozenset({"en", "hi-en"})))
    register_backend(_make("high", priority=10, languages=frozenset({"en"})))
    assert get_backend("sentiment").name == "high"
    assert get_backend("sentiment", language="en").name == "high"
    assert get_backend("sentiment", language="hi-en").name == "low"


def test_auto_skips_unavailable_backends() -> None:
    register_backend(_make("fancy", priority=99, requires=("module_that_does_not_exist_xyz",)))
    register_backend(_make("basic", priority=1))
    assert get_backend("sentiment").name == "basic"


def test_auto_reports_best_missing_backend_when_none_available() -> None:
    register_backend(
        _make("onnx", priority=5, requires=("module_that_does_not_exist_xyz",), extra="onnx")
    )
    with pytest.raises(
        BackendNotAvailableError, match=r'pip install "textinsight-nlp\[onnx\]"'
    ) as info:
        get_backend("sentiment")
    assert info.value.missing == ("module_that_does_not_exist_xyz",)
    assert isinstance(info.value, ImportError)


def test_named_backend_missing_dependency_without_extra() -> None:
    register_backend(_make("needs", requires=("module_that_does_not_exist_xyz",)))
    with pytest.raises(
        BackendNotAvailableError, match="pip install module_that_does_not_exist_xyz"
    ):
        get_backend("sentiment", "needs")


def test_unknown_task_and_backend() -> None:
    with pytest.raises(UnknownBackendError, match="Unknown task"):
        get_backend("nope")
    register_backend(_make("lexicon"))
    with pytest.raises(UnknownBackendError, match="Available: lexicon"):
        get_backend("sentiment", "typo")


def test_language_checks() -> None:
    register_backend(_make("english", languages=frozenset({"en"})))
    with pytest.raises(UnsupportedLanguageError, match="supports: en"):
        get_backend("sentiment", "english", language="fr")
    with pytest.raises(UnsupportedLanguageError, match="No 'sentiment' backend"):
        get_backend("sentiment", language="fr")


def test_any_language_backend() -> None:
    register_backend(_make("universal", languages=frozenset({ANY_LANGUAGE})))
    assert get_backend("sentiment", language="ta").name == "universal"


def test_registration_errors() -> None:
    register_backend(_make("dup"))
    with pytest.raises(ValueError, match="already registered"):
        register_backend(_make("dup"))
    with pytest.raises(ValueError, match="reserved"):
        register_backend(_make("auto"))
    with pytest.raises(ValueError, match="non-empty"):
        register_backend(_make(""))
    with pytest.raises(TypeError):
        register_backend(object)  # type: ignore[type-var]


def test_registering_same_class_twice_is_harmless() -> None:
    cls = _make("same")
    register_backend(cls)
    register_backend(cls)
    assert [i.name for i in list_backends("sentiment")] == ["same"]


def test_unregister() -> None:
    register_backend(_make("temp"))
    get_backend("sentiment", "temp")
    unregister_backend("sentiment", "temp")
    with pytest.raises(UnknownBackendError):
        get_backend("sentiment", "temp")
    unregister_backend("sentiment", "never-existed")  # no error


def test_list_backends() -> None:
    assert list_backends() == []
    register_backend(_make("a", priority=1))
    register_backend(_make("b", priority=5, requires=("module_that_does_not_exist_xyz",)))
    infos = list_backends("sentiment")
    assert [i.name for i in infos] == ["b", "a"]
    assert infos[0].available is False
    assert infos[0].missing == ("module_that_does_not_exist_xyz",)
    assert infos[1].available is True
    assert list_backends() == infos
    with pytest.raises(UnknownBackendError):
        list_backends("missing-task")


def test_repr() -> None:
    register_backend(_make("shown"))
    assert "name='shown'" in repr(get_backend("sentiment", "shown"))
