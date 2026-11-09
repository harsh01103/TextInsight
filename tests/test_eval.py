from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from textinsight import cli
from textinsight.eval import classification_report, load_sentimix, read_conll
from textinsight.eval.benchmark import evaluate_language, evaluate_sentiment
from textinsight.eval.datasets import strip_handles_and_links

TRAIN = (
    "meta\t1\tpositive\r\n"
    "@\tO\r\nsome\tEng\r\n_\tO\r\nuser\tEng\r\n"
    "bahut\tHin\r\naccha\tHin\r\ngood\tEng\r\n"
    "https\tEng\r\n//\tO\r\nt\tEng\r\n.\tO\r\nco\tEng\r\n/\tO\r\nabc\tEng\r\n"
    "\r\n"
    "meta\t2\tnegative\r\n"
    "meta\tEng\r\nbekaar\tHin\r\nbad\tEng\r\n"
    "\r\n"
)
TEST = "meta\t3\r\nmast\tHin\r\nphone\tEng\r\n\r\n"
FILES = {
    "Hinglish_train_14k_split_conll.txt": TRAIN,
    "Hinglish_dev_3k_split_conll.txt": TRAIN,
    "Hinglish_test_unalbelled_conll_updated.txt": TEST,
    "Hinglish_test_labels.txt": "Uid,Sentiment\r\n3,positive\r\n",
}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    for name, content in FILES.items():
        (tmp_path / name).write_bytes(content.encode("utf-8"))
    return tmp_path


class TestLoader:
    def test_reads_labels_tokens_and_languages(self, data_dir: Path) -> None:
        first, second = load_sentimix(data_dir, "train")
        assert (first.uid, first.label) == ("1", "positive")
        assert first.tokens[:2] == ("@", "some")
        assert first.token_languages[-1] == "Eng"
        assert second.label == "negative"

    def test_usernames_and_links_are_removed_from_text(self, data_dir: Path) -> None:
        assert load_sentimix(data_dir, "train")[0].text == "bahut accha good"

    def test_word_meta_inside_a_tweet_is_not_a_new_tweet(self, data_dir: Path) -> None:
        examples = load_sentimix(data_dir, "train")
        assert len(examples) == 2
        assert examples[1].text == "meta bekaar bad"

    def test_test_split_takes_labels_from_separate_file(self, data_dir: Path) -> None:
        assert load_sentimix(data_dir, "test")[0].label == "positive"

    def test_read_conll_without_labels(self, data_dir: Path) -> None:
        path = data_dir / "Hinglish_test_unalbelled_conll_updated.txt"
        assert read_conll(path)[0].label == ""

    def test_bad_split(self, data_dir: Path) -> None:
        with pytest.raises(ValueError, match="split"):
            load_sentimix(data_dir, "validation")

    @pytest.mark.parametrize(
        ("tokens", "expected"),
        [
            (["a", "@", "x", "b"], ["a", "b"]),
            (["@", "x", "_", "y", "_", "z", "b"], ["b"]),
            (["https", "//", "tco", "/", "id", "b"], ["b"]),
            (["https", "//", "t", "co", "/", "id"], []),
            (["https", "//", "t"], []),
            (["https", "b"], ["https", "b"]),
            (["@"], []),
        ],
    )
    def test_strip_handles_and_links(self, tokens: list[str], expected: list[str]) -> None:
        assert strip_handles_and_links(tokens) == expected


class TestMetrics:
    def test_hand_computed_example(self) -> None:
        gold = ["a", "a", "a", "b"]
        predicted = ["a", "a", "b", "b"]
        report = classification_report(gold, predicted)
        assert report["accuracy"] == pytest.approx(0.75)
        assert report["per_label"]["a"]["precision"] == pytest.approx(1.0)
        assert report["per_label"]["a"]["recall"] == pytest.approx(2 / 3)
        assert report["per_label"]["a"]["f1"] == pytest.approx(0.8)
        assert report["per_label"]["b"]["f1"] == pytest.approx(2 / 3)
        assert report["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
        assert report["weighted_f1"] == pytest.approx((0.8 * 3 + 2 / 3) / 4)
        assert report["confusion"]["a"] == {"a": 2, "b": 1}

    def test_unpredicted_class_scores_zero(self) -> None:
        report = classification_report(["a", "b"], ["a", "a"], ("a", "b"))
        assert report["per_label"]["b"]["f1"] == 0.0

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            classification_report(["a"], [])

    def test_empty(self) -> None:
        assert classification_report([], [], ("a",))["accuracy"] == 0.0


class TestBenchmarks:
    def test_sentiment_methods_run(self, data_dir: Path) -> None:
        examples = load_sentimix(data_dir, "train")
        reports = evaluate_sentiment(examples, ["majority", "lexicon", "classical", "auto"])
        assert set(reports) == {"majority", "lexicon", "classical", "auto"}
        assert reports["majority"]["n"] == 2

    def test_auto_treats_devanagari_as_neutral(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_text("meta\t1\tneutral\nयह\tHin\nअच्छा\tHin\nहै\tHin\n\n", encoding="utf-8")
        assert evaluate_sentiment(read_conll(path), ["auto"])["auto"]["accuracy"] == 1.0

    def test_vader_is_skipped_when_not_installed(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        examples = load_sentimix(data_dir, "train")
        assert "vader" not in evaluate_sentiment(examples, ["vader", "majority"])

    def test_vader_labels_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from textinsight.eval import benchmark

        class FakeAnalyzer:
            def polarity_scores(self, text: str) -> dict[str, float]:
                return {"compound": {"up": 0.5, "down": -0.5}.get(text, 0.0)}

        class FakeModule:
            SentimentIntensityAnalyzer = FakeAnalyzer

        monkeypatch.setattr(importlib, "import_module", lambda name: FakeModule)
        assert benchmark._vader(["up", "down", "x"]) == ["positive", "negative", "neutral"]

    def test_language_compares_word_tags(self, data_dir: Path) -> None:
        report = evaluate_language(load_sentimix(data_dir, "test"))
        assert report["n"] == 2
        assert report["accuracy"] == 1.0


class TestCli:
    def test_sentiment_prints_table_and_writes_json(
        self, data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "reports" / "r.json"
        code = cli.main(
            [
                "benchmark",
                "sentiment",
                "--data-dir",
                str(data_dir),
                "--split",
                "test",
                "--out",
                str(out),
            ]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "majority" in printed
        report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        assert "lexicon" in report

    def test_langid(self, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert (
            cli.main(["benchmark", "langid", "--data-dir", str(data_dir), "--split", "test"]) == 0
        )
        assert "per_word" in capsys.readouterr().out

    def test_hybrid_only_runs_when_asked(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requested: list[list[str]] = []

        def fake(examples: object, methods: list[str]) -> dict[str, Any]:
            requested.append(methods)
            return {}

        monkeypatch.setattr("textinsight.eval.benchmark.evaluate_sentiment", fake)
        base = ["benchmark", "sentiment", "--data-dir", str(data_dir), "--split", "test"]
        cli.main(base)
        cli.main([*base, "--hybrid"])
        assert "hybrid" not in requested[0]
        assert "hybrid" in requested[1]

    def test_requires_a_command(self) -> None:
        with pytest.raises(SystemExit):
            cli.main([])

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            cli.main(["--version"])
        assert capsys.readouterr().out.strip().count(".") == 2
