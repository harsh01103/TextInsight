"""Command-line interface for TextInsight.

Run ``textinsight --help`` (or ``textinsight help``) for every command, its options and the
Python functions, or ``textinsight help <command>`` for one command.
``python -m textinsight`` works the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ._version import __version__

__all__ = ["main"]

_EXAMPLES = """examples:
  textinsight sentiment "yeh phone mast hai"
  textinsight keywords --top-n 5 < article.txt
  textinsight redact "call me at 9876543210" --json
  textinsight analyze reviews.csv --column review --out report.html
  textinsight info

Texts can also be piped in, one per line. Use --json for machine-readable output.
"""


def _add_input(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "text", nargs="*", help="text to analyse; if left out, lines are read from standard input"
    )
    command.add_argument("--json", action="store_true", help="print one JSON object per text")


def _add_language(command: argparse.ArgumentParser) -> None:
    command.add_argument("--language", help="skip detection and use this code, such as hi-en")


def _add_task_commands(commands: Any) -> None:
    sentiment = commands.add_parser("sentiment", help="positive, negative or neutral")
    _add_input(sentiment)
    _add_language(sentiment)
    sentiment.add_argument("--backend", default="auto", help="lexicon, classical, hybrid, ...")

    aspects = commands.add_parser("aspects", help="what a review praises or criticizes")
    _add_input(aspects)
    _add_language(aspects)
    aspects.add_argument("--method", default="auto", help="rules or llm")
    aspects.add_argument(
        "--min-mentions", type=int, default=1, help="ignore aspects mentioned fewer times"
    )

    keywords = commands.add_parser("keywords", help="the most relevant words and phrases")
    _add_input(keywords)
    _add_language(keywords)
    keywords.add_argument("--method", default="auto", help="textrank, rake or tfidf")
    keywords.add_argument("--top-n", type=int, default=10, help="how many keywords (default 10)")

    summarize = commands.add_parser("summarize", help="the most representative sentences")
    _add_input(summarize)
    _add_language(summarize)
    summarize.add_argument("--method", default="auto", help="textrank, frequency or abstractive")
    summarize.add_argument(
        "--ratio", type=float, default=0.3, help="share of sentences to keep (default 0.3)"
    )
    summarize.add_argument("--max-sentences", type=int, help="most sentences to keep")

    stats = commands.add_parser("stats", help="word counts, readability and reading time")
    _add_input(stats)
    _add_language(stats)

    redact = commands.add_parser("redact", help="hide emails, phone numbers, PAN, Aadhaar, ...")
    _add_input(redact)
    redact.add_argument("--kinds", help="comma-separated, such as EMAIL,PHONE")

    language = commands.add_parser("language", help="detect English, Hindi or code-mixed text")
    _add_input(language)

    clean = commands.add_parser("clean", help="remove URLs and HTML, optionally lowercase")
    _add_input(clean)
    clean.add_argument("--lowercase", action="store_true", help="also lowercase the text")


class _MainParser(argparse.ArgumentParser):
    """Shows options and the full command listing, instead of argparse's short command list."""

    def format_help(self) -> str:
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        formatter.start_section("options")
        formatter.add_arguments([action for action in self._actions if action.option_strings])
        formatter.end_section()
        formatter.add_text(self.epilog)
        return formatter.format_help()


def _build_parser() -> argparse.ArgumentParser:
    parser = _MainParser(
        prog="textinsight",
        description="Lightweight NLP for English and Hinglish text.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"textinsight-nlp {__version__}"
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="command", parser_class=argparse.ArgumentParser
    )
    _add_task_commands(commands)

    analyze = commands.add_parser("analyze", help="analyse a CSV, JSON-lines or text file")
    analyze.add_argument("file", type=Path)
    analyze.add_argument("--column", help="the text column in a CSV or JSON-lines file")
    analyze.add_argument("--tasks", default="sentiment,keywords,aspects", help="comma-separated")
    analyze.add_argument("--out", type=Path, help="write a report, as .html or .json")

    commands.add_parser("info", help="show versions and which backends are installed")

    help_command = commands.add_parser("help", help="show this full listing, or one command's help")
    help_command.add_argument("topic", nargs="?", help="a command name, such as keywords")

    model_command = commands.add_parser("models", help="manage cached model files")
    model_command.add_argument("action", choices=["list", "clear"])

    benchmark = commands.add_parser("benchmark", help="measure quality on a labelled dataset")
    benchmark.add_argument("task", choices=["sentiment", "langid", "keywords", "summarize"])
    benchmark.add_argument("--data-dir", required=True, help="the folder holding the dataset")
    benchmark.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    benchmark.add_argument("--out", type=Path, help="also write the full report as JSON")
    benchmark.add_argument(
        "--hybrid",
        action="store_true",
        help="also run the hybrid backend (may download a 438 MB model)",
    )
    from .help_text import command_reference, function_reference

    parser.epilog = "\n\n".join([command_reference(parser), function_reference(), _EXAMPLES])
    return parser


# ------------------------------------------------------------------ tasks
def _read_input(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [" ".join(args.text)]
    if sys.stdin is None or sys.stdin.isatty():
        raise ValueError("give some text, or pipe text in")
    return [line.strip() for line in sys.stdin if line.strip()]


def _describe(result: Any) -> str:
    """One readable line for a result."""
    from .core.results import (
        AspectResult,
        KeywordResult,
        LanguageResult,
        PIIResult,
        SentimentResult,
        SummaryResult,
        TextStats,
    )

    if isinstance(result, SentimentResult):
        return f"{result.label} ({result.score:+.2f}, {result.backend})"
    if isinstance(result, AspectResult):
        found = ", ".join(f"{a.name}: {a.label}" for a in result)
        return found or "no aspects with an opinion"
    if isinstance(result, KeywordResult):
        return ", ".join(result.texts) or "no keywords"
    if isinstance(result, SummaryResult):
        return result.summary
    if isinstance(result, PIIResult):
        return result.text
    if isinstance(result, LanguageResult):
        return f"{result.language} ({result.score:.2f})"
    if isinstance(result, TextStats):
        ease = result.flesch_reading_ease
        return (
            f"{result.words} words, {result.sentences} sentences, "
            f"reading time {result.reading_time_seconds:.0f} s"
            + (f", reading ease {ease}" if ease is not None else "")
        )
    return str(result)


def _task_runner(args: argparse.Namespace) -> Callable[[list[str]], list[Any]]:
    import textinsight as ti

    language = getattr(args, "language", None)
    runners: dict[str, Callable[[list[str]], list[Any]]] = {
        "sentiment": lambda t: ti.sentiment(t, backend=args.backend, language=language),
        "aspects": lambda t: ti.aspects(
            t, method=args.method, min_mentions=args.min_mentions, language=language
        ),
        "keywords": lambda t: ti.keywords(
            t, method=args.method, top_n=args.top_n, language=language
        ),
        "summarize": lambda t: ti.summarize(
            t,
            method=args.method,
            ratio=args.ratio,
            max_sentences=args.max_sentences,
            language=language,
        ),
        "stats": lambda t: ti.stats(t, language=language),
        "redact": lambda t: ti.redact(t, kinds=args.kinds.split(",") if args.kinds else None),
        "language": lambda t: ti.detect_language(t),
    }
    return runners[args.command]


def _run_task(args: argparse.Namespace) -> int:
    from .core.exceptions import TextInsightError

    try:
        texts = _read_input(args)
        if args.command == "clean":
            from .preprocess import clean

            results: list[Any] = clean(texts, lowercase=args.lowercase)
        else:
            results = _task_runner(args)(texts)
    except (TextInsightError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for result in results:
        if isinstance(result, str):
            print(json.dumps({"text": result}, ensure_ascii=False) if args.json else result)
        elif args.json:
            print(json.dumps(_exportable(result), ensure_ascii=False))
        else:
            print(_describe(result))
    return 0


def _exportable(result: Any) -> dict[str, Any]:
    from .core.results import PIIResult

    data: dict[str, Any] = result.to_dict()
    if isinstance(result, PIIResult):
        # Leave the original values out of the output, which may end up in logs.
        data["entities"] = [
            {"kind": e.kind, "start": e.start, "end": e.end} for e in result.entities
        ]
    return data


# --------------------------------------------------------------- commands
def _info() -> int:
    from .about import show_versions

    show_versions()
    return 0


def _print_scores(report: dict[str, dict[str, float]]) -> None:
    columns = list(next(iter(report.values())))
    print(f"{'method':<10} " + " ".join(f"{column:>12}" for column in columns))
    for name, scores in report.items():
        print(f"{name:<10} " + " ".join(f"{scores[column]:>12.4f}" for column in columns))


def _row(name: str, report: dict[str, Any]) -> str:
    return (
        f"{name:<10} {report['accuracy']:>9.4f} {report['macro_f1']:>9.4f} "
        f"{report['weighted_f1']:>12.4f}"
    )


def _run_scores(args: argparse.Namespace) -> dict[str, Any]:
    from .eval import load_cnn_dailymail, load_inspec
    from .eval.benchmark import evaluate_keywords, evaluate_summaries

    report: dict[str, dict[str, float]]
    if args.task == "keywords":
        documents = load_inspec(args.data_dir, args.split)
        print(f"keywords on Inspec {args.split} ({len(documents)} documents)")
        report = evaluate_keywords(documents)
    else:
        articles = load_cnn_dailymail(args.data_dir, args.split)
        print(f"summarize on CNN/DailyMail {args.split} ({len(articles)} articles)")
        report = evaluate_summaries(articles)
    _print_scores(report)
    return report


def _run_labels(args: argparse.Namespace) -> dict[str, Any]:
    from .eval import load_sentimix
    from .eval.benchmark import DEFAULT_METHODS, evaluate_language, evaluate_sentiment

    examples = load_sentimix(args.data_dir, args.split)
    print(f"{args.task} on SentiMix {args.split} ({len(examples)} tweets)")
    print(f"{'method':<10} {'accuracy':>9} {'macro_f1':>9} {'weighted_f1':>12}")

    report: dict[str, Any]
    if args.task == "sentiment":
        methods = [*DEFAULT_METHODS, "hybrid"] if args.hybrid else list(DEFAULT_METHODS)
        report = evaluate_sentiment(examples, methods)
        for name, result in report.items():
            print(_row(name, result))
    else:
        report = {"per_word": evaluate_language(examples)}
        print(_row("per_word", report["per_word"]))
    return report


def _benchmark(args: argparse.Namespace) -> int:
    report = _run_scores(args) if args.task in {"keywords", "summarize"} else _run_labels(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    from .core.exceptions import InvalidInputError
    from .pipeline import analyze
    from .report.loading import read_texts

    suffix = args.out.suffix.lower() if args.out else ""
    if args.out and suffix not in {".html", ".json"}:
        print("error: --out must end in .html or .json", file=sys.stderr)
        return 2
    try:
        texts = read_texts(args.file, args.column)
        report = analyze(texts, [task.strip() for task in args.tasks.split(",") if task.strip()])
    except (OSError, InvalidInputError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    summary = report.summary()
    print(f"Analysed {summary['documents']} documents")
    if args.out is None:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif suffix == ".html":
        report.to_html(args.out)
    else:
        report.to_json(args.out)
    if args.out:
        print(f"Wrote {args.out}")
    return 0


def _models(args: argparse.Namespace) -> int:
    from .core.models import models

    if args.action == "clear":
        models.clear()
        print("Cleared the model cache.")
        return 0
    cached = models.list_cached()
    print("\n".join(str(path) for path in cached) if cached else "No cached model files.")
    return 0


def _help(parser: argparse.ArgumentParser, topic: str | None) -> int:
    if topic is None:
        parser.print_help()
        return 0
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    if topic not in subparsers.choices:
        print(f"error: unknown command {topic!r}. Run 'textinsight help'.", file=sys.stderr)
        return 2
    subparsers.choices[topic].print_help()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``textinsight`` command. Returns the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "analyze": _analyze,
        "models": _models,
        "benchmark": _benchmark,
        "info": lambda _args: _info(),
        "help": lambda arguments: _help(parser, arguments.topic),
    }
    return handlers.get(args.command, _run_task)(args)
