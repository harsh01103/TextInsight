"""Run several tasks over many texts, sharing language detection and repeated work."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .aspects import aspects as run_aspects
from .core.exceptions import InvalidInputError, UnsupportedLanguageError
from .core.results import Analysis
from .core.validation import ensure_choice, ensure_positive_int, ensure_texts
from .keywords import keywords as run_keywords
from .langid import detect_language
from .pii import redact as run_redact
from .report import Report
from .sentiment import sentiment as run_sentiment
from .stats import stats as run_stats
from .summarize import summarize as run_summarize

__all__ = ["DEFAULT_TASKS", "TASKS", "Pipeline", "analyze"]

TASKS = ("sentiment", "aspects", "keywords", "summarize", "stats", "redact")
DEFAULT_TASKS = ("sentiment", "keywords", "aspects")

_TOP_KEYWORDS = 5
_SUMMARY_SENTENCES = 2
_DEFAULT_BATCH_SIZE = 256
_FIELD = {"summarize": "summary", "redact": "pii"}


def _batches(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Pipeline:
    """A reusable set of tasks.

    Args:
        tasks: Which tasks to run, from ``TASKS``.
        cache: Analyse each distinct text once, which saves time on repeated texts.
        batch_size: How many texts to send to a task at a time.
    """

    def __init__(
        self,
        tasks: Sequence[str] | str = DEFAULT_TASKS,
        *,
        cache: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        requested = [tasks] if isinstance(tasks, str) else list(tasks)
        if not requested:
            raise InvalidInputError("tasks must not be empty")
        self.tasks = tuple(
            dict.fromkeys(ensure_choice(task, TASKS, name="tasks") for task in requested)
        )
        self.cache = cache
        self.batch_size = ensure_positive_int(batch_size, name="batch_size")

    def _callers(self) -> dict[str, Callable[[list[str], str], Sequence[Any]]]:
        return {
            "sentiment": lambda batch, code: run_sentiment(batch, language=code),
            "aspects": lambda batch, code: run_aspects(batch, language=code),
            "keywords": lambda batch, code: run_keywords(batch, top_n=_TOP_KEYWORDS, language=code),
            "summarize": lambda batch, code: run_summarize(
                batch, max_sentences=_SUMMARY_SENTENCES, language=code
            ),
            "stats": lambda batch, code: run_stats(batch, language=code),
            "redact": lambda batch, code: run_redact(batch),  # patterns work in any language
        }

    def _run_task(
        self, task: str, texts: list[str], codes: list[str]
    ) -> tuple[list[Any], list[bool]]:
        """Return each text's result and whether its language made the task impossible."""
        call = self._callers()[task]
        values: list[Any] = [None] * len(texts)
        skipped = [False] * len(texts)
        by_language: dict[str, list[int]] = {}
        for index, code in enumerate(codes):
            by_language.setdefault(code, []).append(index)
        for code, indices in by_language.items():
            target = "en" if code == "und" else code
            for batch in _batches(indices, self.batch_size):
                try:
                    results = call([texts[i] for i in batch], target)
                except UnsupportedLanguageError:
                    for index in batch:
                        skipped[index] = True
                    continue
                for index, result in zip(batch, results, strict=True):
                    values[index] = result
        return values, skipped

    def run(self, texts: str | Iterable[str]) -> Report:
        """Analyse the texts and return a :class:`~textinsight.report.Report`."""
        items, _ = ensure_texts(texts if isinstance(texts, str) else list(texts))
        # dict.fromkeys keeps first-seen order and drops repeats.
        distinct = list(dict.fromkeys(items)) if self.cache else items
        codes = [result.language for result in detect_language(distinct)] if distinct else []

        outputs = {task: self._run_task(task, distinct, codes) for task in self.tasks}
        analyses: list[Analysis] = []
        for index, text in enumerate(distinct):
            fields = {_FIELD.get(t, t): outputs[t][0][index] for t in self.tasks}
            skipped = tuple(t for t in self.tasks if outputs[t][1][index])
            analyses.append(Analysis(text=text, language=codes[index], skipped=skipped, **fields))

        if self.cache:
            by_text = {analysis.text: analysis for analysis in analyses}
            rows = tuple(by_text[text] for text in items)
        else:
            rows = tuple(analyses)
        return Report(rows=rows, tasks=self.tasks)


def analyze(
    texts: str | Iterable[str],
    tasks: Sequence[str] | str = DEFAULT_TASKS,
    *,
    cache: bool = True,
) -> Report:
    """Run several tasks over one text or many, and collect the results in a report.

    Texts in an unsupported language, such as Devanagari Hindi, are kept. Tasks that
    cannot read them are listed in that row's ``skipped`` field instead of raising.

    Args:
        texts: One string or an iterable of strings.
        tasks: Any of ``"sentiment"``, ``"aspects"``, ``"keywords"``, ``"summarize"``,
            ``"stats"`` and ``"redact"``.
        cache: Analyse each distinct text once.

    >>> analyze(["bahut accha hai"], tasks=["sentiment"]).rows[0].sentiment.label
    'positive'
    """
    return Pipeline(tasks, cache=cache).run(texts)
