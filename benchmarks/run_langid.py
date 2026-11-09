"""Measure the language detector on a labelled sample and record the result.

Usage: python benchmarks/run_langid.py
Writes benchmarks/results/langid.json.

The sample was written by the same author as the word lists, so treat the score
as a sanity check. For results on real data, run ``textinsight benchmark langid``.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import textinsight as ti

ROOT = Path(__file__).parent
SAMPLE = ROOT / "data" / "langid_sample.tsv"
OUTPUT = ROOT / "results" / "langid.json"
LABELS = ("en", "hi", "hi-en", "und")


def main() -> None:
    with SAMPLE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))

    predictions = ti.detect_language([text for text, _ in rows])
    pairs = [(gold, result.language) for (_, gold), result in zip(rows, predictions, strict=True)]
    confusion = Counter(pairs)
    correct = sum(n for (gold, predicted), n in confusion.items() if gold == predicted)

    per_label = {}
    for label in LABELS:
        support = sum(n for (gold, _), n in confusion.items() if gold == label)
        predicted = sum(n for (_, guess), n in confusion.items() if guess == label)
        if not support and not predicted:
            continue
        hits = confusion[(label, label)]
        precision = hits / predicted if predicted else 0.0
        recall = hits / support if support else 0.0
        denominator = precision + recall
        per_label[label] = {
            "support": support,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(2 * precision * recall / denominator if denominator else 0.0, 3),
        }

    report = {
        "backend": "heuristic",
        "version": ti.__version__,
        "sample_size": len(rows),
        "accuracy": round(correct / len(rows), 3),
        "per_label": per_label,
        "confusion": {f"{gold} -> {guess}": n for (gold, guess), n in sorted(confusion.items())},
        "errors": [
            {"text": text, "gold": gold, "predicted": result.language}
            for (text, gold), result in zip(rows, predictions, strict=True)
            if gold != result.language
        ],
        "caveat": (
            "Small hand-written sample by the same author as the word lists; "
            "not independent data. Treat as a sanity check."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sample_size", "accuracy", "per_label")}, indent=2))


if __name__ == "__main__":
    main()
