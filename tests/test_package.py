from __future__ import annotations

import subprocess
import sys

import textinsight as ti
from textinsight._version import __version__


def test_version_is_exposed() -> None:
    assert ti.__version__ == __version__
    assert __version__.count(".") == 2


def test_public_api_names_exist() -> None:
    for name in ti.__all__:
        assert hasattr(ti, name), name


def test_import_is_lightweight() -> None:
    """Importing textinsight must not load network code or heavy libraries."""
    code = (
        "import sys, textinsight\n"
        "heavy = {'urllib.request', 'http.client', 'socket', 'numpy', 'sklearn', 'torch'}\n"
        "print(','.join(sorted(heavy & set(sys.modules))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == ""


def test_all_library_errors_share_a_base_class() -> None:
    for cls in (
        ti.ConfigError,
        ti.InvalidInputError,
        ti.UnknownBackendError,
        ti.UnsupportedLanguageError,
        ti.BackendNotAvailableError,
        ti.ModelDownloadError,
    ):
        assert issubclass(cls, ti.TextInsightError)
