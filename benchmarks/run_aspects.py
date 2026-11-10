"""Check aspect sentiment on short English and Hinglish reviews.

Usage: python benchmarks/run_aspects.py [--split tune|holdout] [--out FILE] [--show]

The reviews and labels in data/aspect_reviews.tsv were written by hand for this
check, so it shows behaviour and is not an independent benchmark. Odd-numbered lines
are for tuning and even-numbered lines are kept back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import textinsight as ti
from textinsight.preprocess.content import content_runs

DATA = Path(__file__).parent / "data" / "aspect_reviews.tsv"


def load(split: str) -> list[tuple[str, dict[str, str]]]:
    rows = []
    for number, line in enumerate(DATA.read_text(encoding="utf-8").splitlines(), start=1):
        if (number % 2 == 1) == (split == "tune"):
            text, _, gold = line.partition("\t")
            rows.append((text, dict(pair.split(":") for pair in gold.split("|") if pair)))
    return rows


def form(name: str) -> tuple[str, ...]:
    return tuple(term.key for run in content_runs(name, "hi-en") for term in run) or (name,)


def f1(hits: int, predicted: int, gold: int) -> dict[str, float]:
    precision = hits / predicted if predicted else 0.0
    recall = hits / gold if gold else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(score, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["tune", "holdout"], default="tune")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--show", action="store_true", help="print each mistake")
    args = parser.parse_args()

    found_total = gold_total = aspect_hits = pair_hits = 0
    rows = load(args.split)
    for text, gold in rows:
        predicted = {form(name): label for name, label in ti.aspects(text).labels.items()}
        target = {form(name): label for name, label in gold.items()}
        found_total += len(predicted)
        gold_total += len(target)
        aspect_hits += len(predicted.keys() & target.keys())
        pair_hits += len(predicted.items() & target.items())
        if args.show and predicted != target:
            print(text, "=>", ti.aspects(text).labels, "| expected", gold)

    report = {
        "reviews": len(rows),
        "aspects": f1(aspect_hits, found_total, gold_total),
        "aspects_with_sentiment": f1(pair_hits, found_total, gold_total),
    }
    print(args.split, json.dumps(report))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
