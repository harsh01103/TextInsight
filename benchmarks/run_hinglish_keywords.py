"""Check keyword extraction on short Hinglish paragraphs.

Usage: python benchmarks/run_hinglish_keywords.py [--split tune|holdout] [--out FILE]

The paragraphs and keyphrases in data/hinglish_keywords.tsv were written by hand for
this check, so it shows behaviour on Hinglish and is not an independent benchmark.
Odd-numbered lines are for tuning and even-numbered lines are kept back.

Each method runs twice: with Hinglish handling, and treating the text as English.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import textinsight as ti
from textinsight.preprocess.content import content_runs

DATA = Path(__file__).parent / "data" / "hinglish_keywords.tsv"
METHODS = ("tfidf", "rake", "textrank")
CUTOFF = 5


def load(split: str) -> list[tuple[str, list[str]]]:
    rows = []
    for number, line in enumerate(DATA.read_text(encoding="utf-8").splitlines(), start=1):
        if (number % 2 == 1) == (split == "tune"):
            text, phrases = line.split("\t")
            rows.append((text, phrases.split("|")))
    return rows


def matching_form(phrase: str) -> tuple[str, ...]:
    """Compare phrases by sound, so that a variant spelling still counts as a match."""
    return tuple(term.key for run in content_runs(phrase, "hi-en") for term in run)


def score(rows: list[tuple[str, list[str]]], method: str, language: str) -> dict[str, float]:
    precision, recall = [], []
    results = ti.keywords(
        [text for text, _ in rows], method=method, top_n=CUTOFF, language=language
    )
    for (_, gold), result in zip(rows, results, strict=True):
        targets = {matching_form(phrase) for phrase in gold}
        found = {matching_form(text) for text in result.texts}
        hits = len(found & targets)
        precision.append(hits / len(found) if found else 0.0)
        recall.append(hits / len(targets))
    mean_p, mean_r = fmean(precision), fmean(recall)
    f1 = 2 * mean_p * mean_r / (mean_p + mean_r) if mean_p + mean_r else 0.0
    return {"precision": round(mean_p, 3), "recall": round(mean_r, 3), "f1": round(f1, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["tune", "holdout"], default="tune")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = load(args.split)
    report = {
        method: {"hinglish": score(rows, method, "hi-en"), "as_english": score(rows, method, "en")}
        for method in METHODS
    }
    print(args.split, len(rows), json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
