"""Check sentiment on short, review-style Hinglish sentences.

Usage: python benchmarks/run_reviews.py [--hybrid] [--out FILE]

The sentences in data/hinglish_reviews.tsv were written by hand for this check, so
they show behaviour outside tweets and are not an independent benchmark. For each
class, odd-numbered lines were used to choose the lexicon fallback rule and
even-numbered lines were kept back to measure it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import textinsight as ti
from textinsight.eval import classification_report

DATA = Path(__file__).parent / "data" / "hinglish_reviews.tsv"
LABELS = ["negative", "neutral", "positive"]
FALLBACK_USERS = ("textinsight.sentiment.classical", "textinsight.sentiment.hybrid")


def load() -> dict[str, list[tuple[str, str]]]:
    splits: dict[str, list[tuple[str, str]]] = {"tune": [], "holdout": []}
    seen: dict[str, int] = {}
    for line in DATA.read_text(encoding="utf-8").splitlines():
        label, text = line.split("\t")
        seen[label] = seen.get(label, 0) + 1
        splits["tune" if seen[label] % 2 else "holdout"].append((label, text))
    return splits


def macro_f1(rows: list[tuple[str, str]], backend: str) -> float:
    results = ti.sentiment([text for _, text in rows], backend=backend, language="hi-en")
    report = classification_report([label for label, _ in rows], [r.label for r in results], LABELS)
    return round(report["macro_f1"], 3)


def without_fallback(rows: list[tuple[str, str]], backend: str) -> float:
    def unchanged(_text: str, label: str, score: float) -> tuple[str, float]:
        return label, score

    with (
        patch(f"{FALLBACK_USERS[0]}.lexicon_backoff", unchanged),
        patch(f"{FALLBACK_USERS[1]}.lexicon_backoff", unchanged),
    ):
        return macro_f1(rows, backend)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", action="store_true", help="also run the transformer backend")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    results: dict[str, dict[str, float]] = {}
    for split, rows in load().items():
        scores = {"lexicon": macro_f1(rows, "lexicon"), "classical": macro_f1(rows, "classical")}
        scores["classical_without_fallback"] = without_fallback(rows, "classical")
        if args.hybrid:
            scores["hybrid"] = macro_f1(rows, "hybrid")
            scores["hybrid_without_fallback"] = without_fallback(rows, "hybrid")
        results[split] = scores
        print(split, len(rows), scores)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
