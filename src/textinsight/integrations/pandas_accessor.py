"""A pandas accessor: ``df["review"].ti.sentiment()``.

Importing this module registers the accessor. Install with
``pip install "textinsight-nlp[pandas]"``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..aspects import aspects as run_aspects
from ..keywords import keywords as run_keywords
from ..pii import redact as run_redact
from ..pipeline import DEFAULT_TASKS, analyze
from ..report import Report
from ..sentiment import sentiment as run_sentiment
from ..stats import stats as run_stats

__all__ = ["TextInsightAccessor"]


@pd.api.extensions.register_series_accessor("ti")
class TextInsightAccessor:
    """TextInsight tasks for a Series of strings. Missing values become empty strings."""

    def __init__(self, series: Any) -> None:
        self._series = series

    def _texts(self) -> list[str]:
        return ["" if pd.isna(value) else str(value) for value in self._series]

    def sentiment(self, **options: Any) -> Any:
        """Return a DataFrame with ``label``, ``score``, ``backend`` and ``language``."""
        rows = [result.to_dict() for result in run_sentiment(self._texts(), **options)]
        return pd.DataFrame(rows, index=self._series.index)

    def keywords(self, **options: Any) -> Any:
        """Return a Series with a list of keywords per text."""
        results = run_keywords(self._texts(), **options)
        return pd.Series([r.texts for r in results], index=self._series.index, name="keywords")

    def aspects(self, **options: Any) -> Any:
        """Return a Series with a dictionary of aspect names and labels per text."""
        results = run_aspects(self._texts(), **options)
        return pd.Series([r.labels for r in results], index=self._series.index, name="aspects")

    def stats(self, **options: Any) -> Any:
        """Return a DataFrame with one column per statistic."""
        rows = [result.to_dict() for result in run_stats(self._texts(), **options)]
        return pd.DataFrame(rows, index=self._series.index)

    def redact(self, **options: Any) -> Any:
        """Return a Series of texts with personal information replaced."""
        results = run_redact(self._texts(), **options)
        return pd.Series([r.text for r in results], index=self._series.index, name="redacted")

    def analyze(self, tasks: Any = DEFAULT_TASKS, **options: Any) -> Report:
        """Run several tasks and return a report."""
        return analyze(self._texts(), tasks, **options)
