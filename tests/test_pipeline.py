from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import textinsight as ti
from textinsight import cli, pipeline
from textinsight.report.loading import read_texts

REVIEWS = [
    "camera mast hai but battery bekaar hai",
    "The food was excellent but the service was terrible",
    "camera mast hai but battery bekaar hai",
]
ALL_TASKS = ("sentiment", "aspects", "keywords", "summarize", "stats", "redact")
DEVANAGARI = "यह फोन बहुत अच्छा है और कैमरा भी अच्छा है"


class TestAnalyze:
    def test_fills_requested_tasks_only(self) -> None:
        row = ti.analyze(["bahut accha hai"], tasks=["sentiment"]).rows[0]
        assert row.sentiment is not None
        assert row.keywords is None
        assert row.pii is None

    def test_all_tasks(self) -> None:
        row = ti.analyze(REVIEWS[:1], tasks=ALL_TASKS).rows[0]
        assert row.sentiment is not None
        assert row.aspects is not None
        assert row.aspects.labels["camera"] == "positive"
        assert row.keywords is not None
        assert row.summary is not None
        assert row.stats is not None
        assert row.pii is not None

    def test_keeps_input_order_and_length(self) -> None:
        report = ti.analyze(REVIEWS)
        assert [row.text for row in report] == REVIEWS
        assert len(report) == 3

    def test_single_string(self) -> None:
        assert len(ti.analyze("bahut accha hai")) == 1

    def test_generator_input(self) -> None:
        assert len(ti.analyze(text for text in REVIEWS)) == 3

    def test_empty_list(self) -> None:
        report = ti.analyze([])
        assert len(report) == 0
        assert report.summary()["documents"] == 0

    def test_cache_analyses_each_distinct_text_once(self) -> None:
        seen: list[int] = []
        original = pipeline.run_sentiment

        def counting(batch: list[str], **options: str) -> object:
            seen.append(len(batch))
            return original(batch, **options)

        with patch.object(pipeline, "run_sentiment", counting):
            ti.analyze(REVIEWS, tasks=["sentiment"], cache=True)
            assert sum(seen) == 2
            seen.clear()
            ti.analyze(REVIEWS, tasks=["sentiment"], cache=False)
            assert sum(seen) == 3

    def test_cached_and_uncached_results_match(self) -> None:
        cached = ti.analyze(REVIEWS, cache=True).to_dict()
        uncached = ti.analyze(REVIEWS, cache=False).to_dict()
        assert cached == uncached

    def test_batches_are_limited(self) -> None:
        sizes: list[int] = []
        original = pipeline.run_sentiment

        def recording(batch: list[str], **options: str) -> object:
            sizes.append(len(batch))
            return original(batch, **options)

        texts = [f"product number {i} is good" for i in range(5)]
        with patch.object(pipeline, "run_sentiment", recording):
            ti.Pipeline(["sentiment"], batch_size=2).run(texts)
        assert max(sizes) <= 2

    def test_unsupported_language_is_skipped_not_fatal(self) -> None:
        report = ti.analyze([DEVANAGARI, REVIEWS[0]], tasks=["sentiment", "stats"])
        assert report.rows[0].sentiment is None
        assert report.rows[0].skipped == ("sentiment",)
        assert report.rows[0].stats is not None
        assert report.rows[1].sentiment is not None
        assert report.summary()["skipped"] == 1

    def test_undetermined_text_is_treated_as_english(self) -> None:
        assert ti.analyze(["???"], tasks=["keywords"]).rows[0].keywords is not None

    @pytest.mark.parametrize("tasks", [[], ["nope"]])
    def test_invalid_tasks(self, tasks: list[str]) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.Pipeline(tasks)

    def test_single_task_string_and_duplicates(self) -> None:
        assert ti.Pipeline("sentiment").tasks == ("sentiment",)
        assert ti.Pipeline(["stats", "stats"]).tasks == ("stats",)

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ti.InvalidInputError):
            ti.Pipeline(batch_size=0)


class TestReport:
    def test_summary_totals(self) -> None:
        summary = ti.analyze(REVIEWS, tasks=ALL_TASKS).summary()
        assert summary["documents"] == 3
        assert sum(summary["sentiment"].values()) == 3
        assert -1 <= summary["average_sentiment"] <= 1
        assert summary["stats"]["total_words"] > 0
        assert summary["pii"] == {}
        camera = next(item for item in summary["aspects"] if item["name"] == "camera")
        assert camera["positive"] == 2

    def test_summary_leaves_out_tasks_that_did_not_run(self) -> None:
        summary = ti.analyze(REVIEWS, tasks=["stats"]).summary()
        assert "sentiment" not in summary
        assert "keywords" not in summary

    def test_reading_ease_is_absent_without_english(self) -> None:
        summary = ti.analyze(["yeh phone bahut accha hai aur mast hai"], tasks=["stats"]).summary()
        assert summary["stats"]["average_reading_ease"] is None

    def test_json_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        text = ti.analyze(REVIEWS).to_json(path)
        assert json.loads(text) == json.loads(path.read_text(encoding="utf-8"))
        assert json.loads(text)["summary"]["documents"] == 3

    def test_html_is_written_and_has_sections(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        page = ti.analyze(REVIEWS, tasks=ALL_TASKS).to_html(path)
        assert path.read_text(encoding="utf-8") == page
        for heading in ("Sentiment", "Aspects", "Top keywords", "Languages", "Documents"):
            assert f"<h2>{heading}</h2>" in page

    def test_html_escapes_untrusted_text(self) -> None:
        attack = "<script>alert(1)</script> camera mast hai"
        page = ti.analyze([attack], tasks=["sentiment", "keywords", "aspects"]).to_html()
        assert "<script" not in page
        assert "&lt;script&gt;" in page

    def test_html_loads_nothing_from_elsewhere(self) -> None:
        page = ti.analyze(REVIEWS).to_html()
        assert "http://" not in page
        assert "https://" not in page

    def test_html_limits_the_table(self) -> None:
        page = ti.analyze([f"item {i} is good" for i in range(205)], tasks=["sentiment"]).to_html()
        assert "Showing the first 200 of 205." in page

    def test_html_shortens_long_text(self) -> None:
        assert "…" in ti.analyze(["word " * 100], tasks=["stats"]).to_html()

    def test_html_shows_skipped_and_empty_report(self) -> None:
        assert "skipped" in ti.analyze([DEVANAGARI], tasks=["sentiment"]).to_html()
        assert "<h2>Documents</h2>" not in ti.analyze([]).to_html()

    def test_exports_hide_personal_information(self) -> None:
        secret = "mail a.person@example.com about camera"
        report = ti.analyze([secret], tasks=["redact", "stats"])
        for output in (report.to_json(), report.to_html()):
            assert "a.person@example.com" not in output
            assert "[EMAIL]" in output
        assert report.summary()["pii"] == {"EMAIL": 1}


class TestReadTexts:
    def test_csv_with_named_column(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("id,body\n1,good\n2,\n", encoding="utf-8")
        assert read_texts(path, "body") == ["good", ""]

    @pytest.mark.parametrize("header", ["text", "review", "comment", "content"])
    def test_csv_finds_common_columns(self, tmp_path: Path, header: str) -> None:
        path = tmp_path / "data.csv"
        path.write_text(f"id,{header}\n1,good\n", encoding="utf-8")
        assert read_texts(path) == ["good"]

    def test_csv_single_column(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("anything\nhello\n", encoding="utf-8")
        assert read_texts(path) == ["hello"]

    def test_csv_ignores_excel_byte_order_mark(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_bytes("text\nhello\n".encode("utf-8-sig"))
        assert read_texts(path, "text") == ["hello"]

    def test_csv_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ti.InvalidInputError, match="--column"):
            read_texts(path)
        with pytest.raises(ti.InvalidInputError, match="no column"):
            read_texts(path, "c")
        path.write_text("", encoding="utf-8")
        with pytest.raises(ti.InvalidInputError, match="header"):
            read_texts(path)

    def test_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"text": "one"}\n\n{"text": "two"}\n', encoding="utf-8")
        assert read_texts(path) == ["one", "two"]

    @pytest.mark.parametrize(("line", "match"), [("{bad", "invalid JSON"), ("[1]", "object")])
    def test_jsonl_errors(self, tmp_path: Path, line: str, match: str) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(ti.InvalidInputError, match=match):
            read_texts(path)

    def test_plain_text_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.txt"
        path.write_text("one\n\n two \n", encoding="utf-8")
        assert read_texts(path) == ["one", "two"]


class TestCli:
    @pytest.fixture
    def reviews(self, tmp_path: Path) -> Path:
        path = tmp_path / "reviews.csv"
        path.write_text(
            'id,review\n1,camera mast hai but battery bekaar hai\n2,"Great food, slow service"\n',
            encoding="utf-8",
        )
        return path

    def test_html_report_end_to_end(self, reviews: Path, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        args = ["analyze", str(reviews), "--column", "review", "--out", str(out)]
        args += ["--tasks", "sentiment,aspects,keywords"]
        assert cli.main(args) == 0
        assert "camera" in out.read_text(encoding="utf-8")

    def test_json_report(self, reviews: Path, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        assert cli.main(["analyze", str(reviews), "--out", str(out)]) == 0
        assert json.loads(out.read_text(encoding="utf-8"))["summary"]["documents"] == 2

    def test_prints_summary_without_out(
        self, reviews: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["analyze", str(reviews)]) == 0
        assert '"documents": 2' in capsys.readouterr().out

    def test_bad_output_type(
        self, reviews: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["analyze", str(reviews), "--out", str(tmp_path / "x.pdf")]) == 2
        assert ".html or .json" in capsys.readouterr().err

    def test_missing_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main(["analyze", str(tmp_path / "none.csv")]) == 2
        assert "error" in capsys.readouterr().err

    def test_bad_task(self, reviews: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main(["analyze", str(reviews), "--tasks", "nope"]) == 2
        assert "tasks" in capsys.readouterr().err

    def test_models_list_and_clear(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main(["models", "list"]) == 0
        assert "No cached" in capsys.readouterr().out
        assert cli.main(["models", "clear"]) == 0
        assert "Cleared" in capsys.readouterr().out

    def test_models_list_shows_files(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        with patch("textinsight.core.models.models.list_cached", return_value=[tmp_path / "m"]):
            assert cli.main(["models", "list"]) == 0
        assert str(tmp_path / "m") in capsys.readouterr().out


class TestIntegrations:
    def test_sklearn_transformer(self) -> None:
        pytest.importorskip("sklearn")
        from textinsight.integrations.sklearn import FEATURE_NAMES, TextInsightFeatures

        features = TextInsightFeatures(language="hi-en").fit(REVIEWS)
        matrix = features.transform(REVIEWS)
        assert matrix.shape == (3, len(FEATURE_NAMES))
        assert list(features.get_feature_names_out()) == list(FEATURE_NAMES)

    def test_sklearn_handles_no_texts(self) -> None:
        pytest.importorskip("sklearn")
        from textinsight.integrations.sklearn import FEATURE_NAMES, TextInsightFeatures

        assert TextInsightFeatures(language="en").transform([]).shape == (0, len(FEATURE_NAMES))

    def test_sklearn_pipeline(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        from textinsight.integrations.sklearn import TextInsightFeatures

        texts = ["bahut accha hai", "bilkul bekaar hai"] * 5
        labels = [1, 0] * 5
        model = make_pipeline(TextInsightFeatures(language="hi-en"), LogisticRegression())
        assert model.fit(texts, labels).score(texts, labels) == 1.0

    def test_pandas_accessor(self) -> None:
        pd = pytest.importorskip("pandas")
        import textinsight.integrations.pandas_accessor  # noqa: F401

        series = pd.Series(["camera mast hai", None], index=[10, 20])
        assert series.ti.sentiment(language="hi-en").loc[10, "label"] == "positive"
        assert list(series.ti.sentiment(language="hi-en").index) == [10, 20]
        assert series.ti.keywords(language="hi-en").loc[10] == ["camera"]
        assert series.ti.aspects(language="hi-en").loc[10] == {"camera": "positive"}
        assert series.ti.stats(language="hi-en").loc[10, "words"] == 3
        assert pd.Series(["a@b.co"]).ti.redact().iloc[0] == "[EMAIL]"
        assert len(series.ti.analyze(["stats"])) == 2
