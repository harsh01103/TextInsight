from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from textinsight import cli
from textinsight.eval import (
    KeyphraseExample,
    SummaryExample,
    classification_report,
    load_cnn_dailymail,
    load_inspec,
    load_sentimix,
    read_conll,
)
from textinsight.eval.benchmark import (
    evaluate_keywords,
    evaluate_language,
    evaluate_sentiment,
    evaluate_summaries,
)
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


class TestBenchmarkLoaders:
    def _write_inspec(self, folder: Path) -> None:
        record = {
            "document": ["Neural", "networks", "-LRB-", "NN", "-RRB-", "learn", "."],
            "extractive_keyphrases": ["neural networks"],
            "abstractive_keyphrases": ["deep learning"],
        }
        (folder / "valid.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    def test_inspec_restores_brackets_and_joins_references(self, tmp_path: Path) -> None:
        self._write_inspec(tmp_path)
        example = load_inspec(tmp_path, "dev")[0]
        assert example.text == "Neural networks ( NN ) learn ."
        assert example.keyphrases == ("neural networks", "deep learning")

    def test_unknown_split_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="split"):
            load_inspec(tmp_path, "nope")
        with pytest.raises(ValueError, match="split"):
            load_cnn_dailymail(tmp_path, "train")

    def test_cnn_dailymail(self, tmp_path: Path) -> None:
        pyarrow = pytest.importorskip("pyarrow")
        from pyarrow import parquet

        table = pyarrow.table({"article": ["A long article."], "highlights": ["Short."]})
        parquet.write_table(table, tmp_path / "test.parquet")
        assert load_cnn_dailymail(tmp_path, "test")[0].reference == "Short."

    def test_missing_pyarrow_gives_install_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        with pytest.raises(ImportError, match=r"textinsight-nlp\[training\]"):
            load_cnn_dailymail(tmp_path, "test")

    def test_keyword_report(self) -> None:
        examples = [KeyphraseExample("Solar panels make solar power.", ("solar power",))]
        report = evaluate_keywords(examples, methods=["rake"])
        assert set(report["rake"]) >= {"precision@5", "f1@10"}

    def test_summary_report_includes_lead_baseline(self) -> None:
        article = "The council approved solar power. The cat slept. Rain fell. Birds sang."
        examples = [SummaryExample(article, "The council approved solar power.")]
        report = evaluate_summaries(examples, sentences=1)
        assert report["lead"]["rouge-1"] == 1.0
        assert set(report) == {"lead", "frequency", "textrank"}

    def test_cli_keywords(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self._write_inspec(tmp_path)
        out = tmp_path / "report.json"
        assert (
            cli.main(["benchmark", "keywords", "--data-dir", str(tmp_path), "--out", str(out)]) == 0
        )
        assert "Inspec" in capsys.readouterr().out
        assert "textrank" in json.loads(out.read_text(encoding="utf-8"))

    def test_cli_summarize(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        pyarrow = pytest.importorskip("pyarrow")
        from pyarrow import parquet

        table = pyarrow.table(
            {"article": ["Solar power grew. Rain fell."], "highlights": ["Solar."]}
        )
        parquet.write_table(table, tmp_path / "validation.parquet")
        assert cli.main(["benchmark", "summarize", "--data-dir", str(tmp_path)]) == 0
        assert "CNN/DailyMail" in capsys.readouterr().out
