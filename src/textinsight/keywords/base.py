"""Shared logic for keyword backends: preparing text and choosing the final phrases."""

from __future__ import annotations

from abc import abstractmethod
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import ClassVar

from ..core.base import Backend
from ..core.results import Keyword, KeywordResult
from ..preprocess.content import Term, parse_document

__all__ = [
    "DEFAULT_TOP_N",
    "MAX_PHRASE_WORDS",
    "Document",
    "KeywordBackend",
    "Phrase",
    "candidate_phrases",
    "ngrams",
]

DEFAULT_TOP_N = 10
MAX_PHRASE_WORDS = 3

Phrase = tuple[str, ...]
Document = list[list[list[Term]]]  # sentences, then runs, then terms


def ngrams(run: Sequence[Term], longest: int = MAX_PHRASE_WORDS) -> Iterator[Sequence[Term]]:
    """Yield every contiguous group of up to ``longest`` terms in a run."""
    for start in range(len(run)):
        for end in range(start + 1, min(start + longest, len(run)) + 1):
            yield run[start:end]


def candidate_phrases(document: Document) -> Iterator[Phrase]:
    """Yield each run of content words, splitting very long runs, which are usually lists."""
    for sentence in document:
        for run in sentence:
            keys = tuple(term.key for term in run)
            for start in range(0, len(keys), MAX_PHRASE_WORDS):
                yield keys[start : start + MAX_PHRASE_WORDS]


def phrase_key(terms: Sequence[Term]) -> Phrase:
    return tuple(term.key for term in terms)


def _contains(longer: Phrase, shorter: Phrase) -> bool:
    size = len(shorter)
    return any(longer[i : i + size] == shorter for i in range(len(longer) - size + 1))


def _surface_forms(document: Document) -> dict[Phrase, Counter[str]]:
    """Count how each phrase was written, for every phrase a backend can propose."""
    forms: dict[Phrase, Counter[str]] = {}
    for sentence in document:
        for run in sentence:
            for terms in ngrams(run, longest=len(run)):
                text = " ".join(term.surface for term in terms)
                forms.setdefault(phrase_key(terms), Counter())[text] += 1
    return forms


class KeywordBackend(Backend[KeywordResult]):
    """Base class: subclasses only decide how candidate phrases are scored."""

    task: ClassVar[str] = "keywords"
    languages: ClassVar[frozenset[str]] = frozenset({"en", "hi-en"})

    @abstractmethod
    def score(self, documents: Sequence[Document]) -> list[dict[Phrase, float]]:
        """Return candidate phrases and their scores for each document."""

    def extract(self, texts: Sequence[str], *, top_n: int, language: str) -> list[KeywordResult]:
        """Return the ``top_n`` best phrases of each text."""
        self.ensure_loaded()
        documents = [parse_document(text, language) for text in texts]
        return [
            self._select(document, scores, top_n)
            for document, scores in zip(documents, self.score(documents), strict=True)
        ]

    def predict(self, texts: Sequence[str]) -> list[KeywordResult]:
        # "hi-en" applies both stopword lists, so it suits unknown input.
        return self.extract(texts, top_n=DEFAULT_TOP_N, language="hi-en")

    def _select(self, document: Document, scores: dict[Phrase, float], top_n: int) -> KeywordResult:
        forms = _surface_forms(document)
        chosen: list[Phrase] = []
        # sorted() is stable, so ties keep the order in which phrases first appeared.
        for phrase in sorted(scores, key=lambda candidate: -scores[candidate]):
            if len(chosen) == top_n:
                break
            if not any(_contains(phrase, c) or _contains(c, phrase) for c in chosen):
                chosen.append(phrase)
        top = scores[chosen[0]] if chosen else 0.0
        keywords = tuple(
            Keyword(
                text=forms[phrase].most_common(1)[0][0],
                score=round(scores[phrase] / top, 4) if top else 0.0,
            )
            for phrase in chosen
        )
        return KeywordResult(keywords=keywords, method=self.name)
