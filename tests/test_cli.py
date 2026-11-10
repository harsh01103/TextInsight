from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

import textinsight as ti
from textinsight import about, cli


def run(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = cli.main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestVersionAndHelp:
    @pytest.mark.parametrize("flag", ["-V", "--version"])
    def test_version(self, flag: str, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as stop:
            cli.main([flag])
        assert stop.value.code == 0
        assert capsys.readouterr().out.strip() == f"textinsight-nlp {ti.__version__}"

    def test_help_lists_commands_and_examples(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            cli.main(["--help"])
        out = capsys.readouterr().out
        for command in ("sentiment", "aspects", "keywords", "redact", "analyze", "info"):
            assert command in out
        assert "examples:" in out

    def test_command_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            cli.main(["keywords", "--help"])
        assert "--top-n" in capsys.readouterr().out

    def test_python_dash_m(self) -> None:
        output = subprocess.run(
            [sys.executable, "-m", "textinsight", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert output.stdout.strip() == f"textinsight-nlp {ti.__version__}"


class TestTaskCommands:
    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            (["sentiment", "bahut", "accha", "hai"], "positive"),
            (["aspects", "camera mast hai but battery bekaar hai"], "camera: positive"),
            (
                ["keywords", "--top-n", "1", "battery life is great. battery life matters."],
                "battery life",
            ),
            (["summarize", "--max-sentences", "1", "One idea. Two ideas."], "One idea."),
            (["stats", "--language", "en", "The cat sat."], "3 words"),
            (["redact", "call 9876543210"], "call [PHONE]"),
            (["language", "yeh phone mast hai but battery bekaar"], "hi-en"),
            (["clean", "--lowercase", "Visit https://a.co NOW"], "visit now"),
        ],
    )
    def test_readable_output(
        self, args: list[str], expected: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = run(args, capsys)
        assert code == 0
        assert expected in out

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(["sentiment", "--json", "--backend", "lexicon", "accha hai"], capsys)
        assert code == 0
        assert json.loads(out)["backend"] == "lexicon"

    def test_json_clean(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out, _ = run(["clean", "--json", "hi"], capsys)
        assert json.loads(out) == {"text": "hi"}

    def test_json_redact_hides_values(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out, _ = run(["redact", "--json", "mail a.b@example.com"], capsys)
        data = json.loads(out)
        assert "a.b@example.com" not in out
        assert data["entities"][0]["kind"] == "EMAIL"

    def test_redact_kinds(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out, _ = run(["redact", "--kinds", "EMAIL", "a@b.co 9876543210"], capsys)
        assert out.strip() == "[EMAIL] 9876543210"

    def test_reads_lines_from_standard_input(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("bahut accha hai\n\nbilkul bekaar hai\n"))
        code, out, _ = run(["sentiment", "--language", "hi-en"], capsys)
        assert code == 0
        assert out.splitlines()[0].startswith("positive")
        assert out.splitlines()[1].startswith("negative")

    def test_no_text_and_no_pipe(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        terminal = io.StringIO()
        terminal.isatty = lambda: True  # type: ignore[method-assign]
        monkeypatch.setattr(sys, "stdin", terminal)
        code, _, err = run(["sentiment"], capsys)
        assert code == 2
        assert "give some text" in err

    def test_library_errors_become_messages(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _, err = run(["sentiment", "यह फोन बहुत अच्छा है और कैमरा भी अच्छा है"], capsys)
        assert code == 2
        assert err.startswith("error:")

    def test_aspects_without_opinions(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out, _ = run(["aspects", "The train leaves at noon"], capsys)
        assert "no aspects" in out

    def test_keywords_without_content(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out, _ = run(["keywords", "--language", "en", "the and of"], capsys)
        assert "no keywords" in out

    def test_stats_reading_ease(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, out, _ = run(["stats", "--language", "en", "The cat sat on the mat."], capsys)
        assert "reading ease" in out


class TestInfo:
    def test_info_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(["info"], capsys)
        assert code == 0
        assert f"textinsight-nlp {ti.__version__}" in out
        assert "classical" in out
        assert "Optional packages" in out

    def test_info_dictionary(self) -> None:
        details = ti.info()
        assert details["textinsight"] == ti.__version__
        assert {"task", "name", "available", "extra"} <= set(details["backends"][0])

    def test_missing_package_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(about, "_OPTIONAL_PACKAGES", ("package_that_does_not_exist_xyz",))
        assert ti.info()["optional_packages"] == {"package_that_does_not_exist_xyz": None}

    def test_unavailable_backend_shows_extra(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake = {"task": "sentiment", "name": "x", "available": False, "extra": "onnx"}
        monkeypatch.setattr(about, "info", lambda: {**ti.info(), "backends": [fake]})
        about.show_versions()
        assert "needs [onnx]" in capsys.readouterr().out


def test_describe_falls_back_to_str() -> None:
    assert cli._describe(42) == "42"
