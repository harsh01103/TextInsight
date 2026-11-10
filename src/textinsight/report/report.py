"""The result of analysing many texts, with totals and JSON or HTML export."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from ..core.results import Analysis
from .html import render_html

__all__ = ["Report"]

_TOP_KEYWORDS = 15
_TOP_ASPECTS = 15


def _safe_row(row: Analysis) -> dict[str, Any]:
    """Return a row for export, with personal information removed if ``redact`` ran."""
    data = row.to_dict()
    if row.pii is not None:
        data["text"] = row.pii.text
        data["pii"] = {
            "entities": [{"kind": e.kind, "start": e.start, "end": e.end} for e in row.pii.entities]
        }
    return data


@dataclass(frozen=True)
class Report:
    """One :class:`~textinsight.core.results.Analysis` per input text, in input order."""

    rows: tuple[Analysis, ...]
    tasks: tuple[str, ...]

    def __iter__(self) -> Iterator[Analysis]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def summary(self) -> dict[str, Any]:
        """Return totals across all documents, ready for display or JSON."""
        totals: dict[str, Any] = {
            "documents": len(self.rows),
            "languages": dict(Counter(row.language for row in self.rows).most_common()),
            "skipped": sum(1 for row in self.rows if row.skipped),
        }
        sentiments = [row.sentiment for row in self.rows if row.sentiment is not None]
        if sentiments:
            totals["sentiment"] = dict(Counter(s.label for s in sentiments).most_common())
            totals["average_sentiment"] = round(fmean(s.score for s in sentiments), 4)
        phrases = Counter(
            text for row in self.rows if row.keywords is not None for text in row.keywords.texts
        )
        if "keywords" in self.tasks:
            totals["keywords"] = phrases.most_common(_TOP_KEYWORDS)
        if "aspects" in self.tasks:
            totals["aspects"] = self._aspect_totals()
        stats = [row.stats for row in self.rows if row.stats is not None]
        if stats:
            ease = [s.flesch_reading_ease for s in stats if s.flesch_reading_ease is not None]
            totals["stats"] = {
                "total_words": sum(s.words for s in stats),
                "average_words": round(fmean(s.words for s in stats), 2),
                "average_reading_ease": round(fmean(ease), 2) if ease else None,
                "total_reading_time_seconds": round(sum(s.reading_time_seconds for s in stats), 1),
            }
        if "redact" in self.tasks:
            totals["pii"] = dict(
                Counter(
                    entity.kind
                    for row in self.rows
                    if row.pii is not None
                    for entity in row.pii.entities
                ).most_common()
            )
        return totals

    def _aspect_totals(self) -> list[dict[str, Any]]:
        counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for row in self.rows:
            if row.aspects is not None:
                for aspect in row.aspects:
                    counts[aspect.name][aspect.label] += 1
        ranked = sorted(counts.items(), key=lambda item: -sum(item[1].values()))
        return [
            {
                "name": name,
                "mentions": sum(labels.values()),
                "positive": labels["positive"],
                "negative": labels["negative"],
                "mixed": labels["mixed"],
            }
            for name, labels in ranked[:_TOP_ASPECTS]
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return the summary and rows. Rows hold redacted text if ``redact`` ran."""
        return {
            "tasks": list(self.tasks),
            "summary": self.summary(),
            "rows": [_safe_row(row) for row in self.rows],
        }

    def to_json(self, path: str | os.PathLike[str] | None = None) -> str:
        """Return the report as JSON, also writing it to ``path`` if one is given."""
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def to_html(self, path: str | os.PathLike[str] | None = None) -> str:
        """Return a self-contained HTML report, also writing it to ``path`` if one is given."""
        page = render_html(self.summary(), [_safe_row(row) for row in self.rows], self.tasks)
        if path is not None:
            Path(path).write_text(page, encoding="utf-8")
        return page
