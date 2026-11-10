"""Compare backends on quality, speed and memory.

Usage:
    python benchmarks/run_backends.py sentiment --data-dir <SentiMix Hinglish folder>
    python benchmarks/run_backends.py summarize --data-dir <CNN/DailyMail folder> [--limit 300]

Each backend runs in its own process, so load time and memory are not shared.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean

SPEED_TEXTS = 1000


def peak_memory_mb() -> float:
    """Return this process's peak memory use."""
    try:
        import resource
    except ImportError:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return round(counters.PeakWorkingSetSize / 1e6, 1)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(usage / (1e6 if sys.platform == "darwin" else 1e3), 1)


def run_sentiment(backend: str, data_dir: str, split: str) -> dict[str, float]:
    import textinsight as ti
    from textinsight.eval import classification_report, load_sentimix

    # ONNX is English only, so it reads every tweet as English.
    language = "en" if backend == "onnx" else "hi-en"
    sample = [e.text for e in load_sentimix(data_dir, "train")][:SPEED_TEXTS]
    started = time.perf_counter()
    ti.sentiment(sample[:1], backend=backend, language=language)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    ti.sentiment(sample, backend=backend, language=language)
    speed = len(sample) / (time.perf_counter() - started)

    scored = load_sentimix(data_dir, split)
    results = ti.sentiment([e.text for e in scored], backend=backend, language=language)
    predicted = ["neutral" if r.label == "mixed" else r.label for r in results]
    labels = ["negative", "neutral", "positive"]
    report = classification_report([e.label for e in scored], predicted, labels)
    return {
        "macro_f1": round(report["macro_f1"], 4),
        "weighted_f1": round(report["weighted_f1"], 4),
        "load_seconds": round(load_seconds, 2),
        "texts_per_second": round(speed, 1),
        "peak_memory_mb": peak_memory_mb(),
    }


def run_summarize(method: str, data_dir: str, limit: int) -> dict[str, float]:
    import textinsight as ti
    from textinsight.eval import load_cnn_dailymail
    from textinsight.eval.text_metrics import rouge_scores

    articles = load_cnn_dailymail(data_dir, "test")[:limit]
    started = time.perf_counter()
    results = ti.summarize(
        [a.article for a in articles], ratio=1.0, max_sentences=3, method=method, language="en"
    )
    speed = len(articles) / (time.perf_counter() - started)
    rows = [rouge_scores(r.summary, a.reference) for r, a in zip(results, articles, strict=True)]
    return {
        "rouge_1_f1": round(fmean(row["rouge-1"]["f1"] for row in rows), 4),
        "rouge_2_f1": round(fmean(row["rouge-2"]["f1"] for row in rows), 4),
        "rouge_l_f1": round(fmean(row["rouge-l"]["f1"] for row in rows), 4),
        "texts_per_second": round(speed, 2),
        "peak_memory_mb": peak_memory_mb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["sentiment", "summarize"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--split", default="test", help="SentiMix split to score")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.one:
        result = (
            run_sentiment(args.one, args.data_dir, args.split)
            if args.task == "sentiment"
            else run_summarize(args.one, args.data_dir, args.limit)
        )
        print(json.dumps(result))
        return

    defaults = (
        ["lexicon", "classical", "hybrid", "onnx"]
        if args.task == "sentiment"
        else ["frequency", "textrank", "abstractive"]
    )
    report = {}
    for backend in args.backends or defaults:
        command = [sys.executable, __file__, args.task, "--data-dir", args.data_dir]
        command += ["--one", backend, "--limit", str(args.limit), "--split", args.split]
        output = subprocess.check_output(command, text=True)
        report[backend] = json.loads(output.strip().splitlines()[-1])
        print(backend, report[backend], flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
