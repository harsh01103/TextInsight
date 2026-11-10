"""Measure analyze() speed, with and without the repeated-text cache.

Usage: python benchmarks/run_performance.py --data-dir <SentiMix Hinglish folder> [--out FILE]

Uses 10,000 real tweets, then a version where half of the texts are repeats.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import textinsight as ti
from textinsight.eval import load_sentimix

SIZE = 10_000
SEED = 13
TASKS = ("sentiment", "keywords", "aspects")
_COLD_START = (
    "import time; s = time.perf_counter(); import textinsight; print(time.perf_counter() - s)"
)


def throughput(texts: list[str], cache: bool) -> float:
    started = time.perf_counter()
    ti.analyze(texts, TASKS, cache=cache)
    return round(len(texts) / (time.perf_counter() - started), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tweets = [example.text for example in load_sentimix(args.data_dir, "train")][:SIZE]
    rng = random.Random(SEED)
    repeated = tweets[: SIZE // 2] + rng.choices(tweets[: SIZE // 2], k=SIZE // 2)
    rng.shuffle(repeated)

    ti.analyze(tweets[:10], TASKS)  # loads models, so that they do not count against either run
    report: dict[str, object] = {
        "texts": SIZE,
        "cold_start_import_seconds": round(
            float(subprocess.check_output([sys.executable, "-c", _COLD_START], text=True)), 3
        ),
        "texts_per_second": {
            "no_repeats_cache_off": throughput(tweets, cache=False),
            "no_repeats_cache_on": throughput(tweets, cache=True),
            "half_repeats_cache_off": throughput(repeated, cache=False),
            "half_repeats_cache_on": throughput(repeated, cache=True),
        },
    }
    tracemalloc.start()
    ti.analyze(tweets, TASKS)
    report["peak_python_memory_mb"] = round(tracemalloc.get_traced_memory()[1] / 1e6, 1)
    tracemalloc.stop()

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
