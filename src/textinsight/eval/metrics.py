"""Classification metrics, implemented without external dependencies."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = ["classification_report"]


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_report(
    gold: Sequence[str], predicted: Sequence[str], labels: Iterable[str] | None = None
) -> dict[str, Any]:
    """Compute accuracy, macro and weighted F1, per-label scores and a confusion matrix.

    Macro F1 treats every class equally. Weighted F1 weights by class size and is
    the ranking metric of SemEval-2020 Task 9, so it allows direct comparison.
    """
    if len(gold) != len(predicted):
        raise ValueError(f"Length mismatch: {len(gold)} gold and {len(predicted)} predicted")
    names = sorted(set(labels) if labels is not None else set(gold) | set(predicted))
    pairs = Counter(zip(gold, predicted, strict=True))
    per_label: dict[str, dict[str, float]] = {}
    for name in names:
        hits = pairs[(name, name)]
        support = sum(n for (g, _), n in pairs.items() if g == name)
        guessed = sum(n for (_, p), n in pairs.items() if p == name)
        precision = _divide(hits, guessed)
        recall = _divide(hits, support)
        per_label[name] = {
            "precision": precision,
            "recall": recall,
            "f1": _divide(2 * precision * recall, precision + recall),
            "support": support,
        }
    total = len(gold)
    return {
        "n": total,
        "accuracy": _divide(sum(n for (g, p), n in pairs.items() if g == p), total),
        "macro_f1": _divide(sum(s["f1"] for s in per_label.values()), len(names)),
        "weighted_f1": _divide(sum(s["f1"] * s["support"] for s in per_label.values()), total),
        "per_label": per_label,
        "confusion": {g: {p: pairs[(g, p)] for p in names} for g in names},
    }
