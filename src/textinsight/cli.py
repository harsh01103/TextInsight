"""Command-line interface: ``textinsight benchmark ...``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._version import __version__

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="textinsight", description="TextInsight tools.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

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
    return parser


def _row(name: str, report: dict[str, Any]) -> str:
    return (
        f"{name:<10} {report['accuracy']:>9.4f} {report['macro_f1']:>9.4f} "
        f"{report['weighted_f1']:>12.4f}"
    )


def _print_scores(report: dict[str, dict[str, float]]) -> None:
    columns = list(next(iter(report.values())))
    print(f"{'method':<10} " + " ".join(f"{column:>12}" for column in columns))
    for name, scores in report.items():
        print(f"{name:<10} " + " ".join(f"{scores[column]:>12.4f}" for column in columns))


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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``textinsight`` command."""
    args = _build_parser().parse_args(argv)
    report = _run_scores(args) if args.task in {"keywords", "summarize"} else _run_labels(args)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0
