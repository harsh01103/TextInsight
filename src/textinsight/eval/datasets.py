"""Dataset loaders for the benchmarks. No dataset is distributed with TextInsight.

SentiMix (SemEval-2020 Task 9, Hinglish): download it from
https://doi.org/10.5281/zenodo.3974927 (CC-BY-4.0) and pass the path of its
``Hinglish`` folder.

Citation: Patwa et al., "SemEval-2020 Task 9: Overview of Sentiment Analysis of
Code-Mixed Tweets", Proceedings of SemEval-2020.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Example",
    "KeyphraseExample",
    "SummaryExample",
    "load_cnn_dailymail",
    "load_inspec",
    "load_sentimix",
    "read_conll",
    "strip_handles_and_links",
]

_FILES = {
    "train": "Hinglish_train_14k_split_conll.txt",
    "dev": "Hinglish_dev_3k_split_conll.txt",
    "test": "Hinglish_test_unalbelled_conll_updated.txt",
}
_TEST_LABELS = "Hinglish_test_labels.txt"
_MAX_LINK_TOKENS = 8


@dataclass(frozen=True)
class Example:
    """One labelled tweet."""

    uid: str
    text: str
    label: str
    tokens: tuple[str, ...]
    token_languages: tuple[str, ...]


def strip_handles_and_links(tokens: Sequence[str]) -> list[str]:
    """Remove usernames and t.co links, which the dataset splits into many tokens.

    In the raw files ``@some_user`` is ``@ some _ user`` and a link is
    ``https // t . co / abc``. Usernames also must not reach trained models.
    """
    kept: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        token = tokens[i]
        if token == "@":
            i += 2
            while i < n and tokens[i] == "_":
                i += 2
        elif token.lower() in {"http", "https"} and i + 1 < n and tokens[i + 1] == "//":
            j = i + 2
            while j < n and j < i + _MAX_LINK_TOKENS and tokens[j] != "/":
                j += 1
            i = j + 2 if j < n and tokens[j] == "/" else j
        else:
            kept.append(token)
            i += 1
    return kept


def read_conll(
    path: str | os.PathLike[str], labels: Mapping[str, str] | None = None
) -> list[Example]:
    """Read a SentiMix CoNLL file.

    A tweet starts with ``meta<TAB>uid[<TAB>label]``, followed by one
    ``token<TAB>language`` line per token. ``labels`` supplies labels by uid for
    the test split, whose file has none.
    """
    tweets: list[tuple[str, str, list[str], list[str]]] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            # A tweet can contain the word "meta", so also require a numeric id.
            if parts[0] == "meta" and len(parts) in (2, 3) and parts[1].isdigit():
                label = parts[2] if len(parts) == 3 else (labels or {}).get(parts[1], "")
                tweets.append((parts[1], label, [], []))
            elif len(parts) == 2 and parts[0] and tweets:
                tweets[-1][2].append(parts[0])
                tweets[-1][3].append(parts[1])
    return [
        Example(
            uid=uid,
            text=" ".join(strip_handles_and_links(tokens)),
            label=label,
            tokens=tuple(tokens),
            token_languages=tuple(languages),
        )
        for uid, label, tokens, languages in tweets
    ]


def load_sentimix(root: str | os.PathLike[str], split: str) -> list[Example]:
    """Load the ``train``, ``dev`` or ``test`` split from the dataset's Hinglish folder."""
    if split not in _FILES:
        raise ValueError(f"split must be one of {sorted(_FILES)}, got {split!r}")
    folder = Path(root)
    labels = None
    if split == "test":
        with (folder / _TEST_LABELS).open(encoding="utf-8", newline="") as handle:
            labels = {row["Uid"]: row["Sentiment"] for row in csv.DictReader(handle)}
    return read_conll(folder / _FILES[split], labels)


@dataclass(frozen=True)
class KeyphraseExample:
    """A document and its reference keyphrases."""

    text: str
    keyphrases: tuple[str, ...]


@dataclass(frozen=True)
class SummaryExample:
    """An article and its reference summary."""

    article: str
    reference: str


_BRACKETS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}
_INSPEC_SPLITS = {"train": "train", "dev": "valid", "test": "test"}
_CNN_SPLITS = {"dev": "validation", "test": "test"}


def load_inspec(data_dir: str | os.PathLike[str], split: str) -> list[KeyphraseExample]:
    """Load Inspec from the JSON-lines files of Hugging Face ``midas/inspec``.

    References combine the keyphrases that appear in the text with those that do not.
    """
    if split not in _INSPEC_SPLITS:
        raise ValueError(f"split must be one of {sorted(_INSPEC_SPLITS)}, got {split!r}")
    path = Path(data_dir) / f"{_INSPEC_SPLITS[split]}.jsonl"
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            words = (_BRACKETS.get(word, word) for word in record["document"])
            phrases = (*record["extractive_keyphrases"], *record["abstractive_keyphrases"])
            examples.append(KeyphraseExample(text=" ".join(words), keyphrases=phrases))
    return examples


def load_cnn_dailymail(data_dir: str | os.PathLike[str], split: str) -> list[SummaryExample]:
    """Load CNN/DailyMail from the parquet files of Hugging Face ``abisee/cnn_dailymail``.

    Needs ``pyarrow``, from the ``training`` extra.
    """
    if split not in _CNN_SPLITS:
        raise ValueError(f"split must be one of {sorted(_CNN_SPLITS)}, got {split!r}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            'Reading CNN/DailyMail needs pyarrow: pip install "textinsight-nlp[training]"'
        ) from error
    table = parquet.read_table(
        Path(data_dir) / f"{_CNN_SPLITS[split]}.parquet", columns=["article", "highlights"]
    )
    return [
        SummaryExample(article=row["article"], reference=row["highlights"])
        for row in table.to_pylist()
    ]
