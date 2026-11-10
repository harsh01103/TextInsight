# Backends

Each task can run with different backends. `"auto"` always picks a bundled one, so nothing is
downloaded and no paid API is called unless you name a backend yourself.

## Sentiment

Measured on the SentiMix test split (3,000 Hinglish tweets). Speed is measured on 1,000 tweets
on a laptop CPU after the model has loaded. Memory is the peak for the whole Python process.

| Backend | Install | Download | Macro F1 | Texts per second | Load time | Peak memory |
|---|---|---|---|---|---|---|
| `lexicon` | core | none | 0.540 | 5,056 | instant | 77 MB |
| **`classical`** (default) | core | none | **0.707** | 572 | 0.2 s | 77 MB |
| `hybrid` | `[transformers]`, about 670 MB of packages | 438 MB | **0.728** | 9 | 26 s | 1,063 MB |
| `onnx` | `[onnx]`, about 56 MB of packages | 68 MB | 0.486 | 77 | 1.0 s | 343 MB |
| `llm` | `[llm]`, about 30 MB of packages | none | not measured | limited by the API | network | small |

- `onnx` is an English model with no neutral class, run here on Hinglish tweets as if they were
  English. Its low score shows why a Hinglish model matters, not how it does on English text.
  Its neutral threshold was tuned on the dev split, and this was its first run on the test split.
- `hybrid` is about 0.02 more accurate than `classical`, but about 60 times slower and needs a
  large download.
- `llm` sends each text to Anthropic and costs money per call, so it was not run on the
  benchmark. It needs your own Anthropic API access, and the model can be changed with the
  `TEXTINSIGHT_LLM_MODEL` environment variable.

## Summaries

Measured on the first 300 articles of the CNN/DailyMail test split, three-sentence summaries,
ROUGE F1.

| Method | Install | Download | ROUGE-1 | ROUGE-2 | ROUGE-L | Articles per second | Peak memory |
|---|---|---|---|---|---|---|---|
| Lead-3 baseline | | | 0.303 | 0.119 | 0.209 | | |
| `frequency` | core | none | 0.256 | 0.072 | 0.177 | 261 | 366 MB |
| **`textrank`** (default) | core | none | 0.282 | 0.102 | 0.195 | 220 | 369 MB |
| `abstractive` | `[transformers]` | 242 MB | **0.316** | 0.117 | **0.234** | 0.3 | 1,323 MB |

- These first 300 articles are harder than the full split: Lead-3 scores 0.402 ROUGE-1 on all
  11,490 articles but 0.303 here. Compare methods within this table, not with the full-split
  numbers in [Benchmarks](benchmarks.md).
- `abstractive` was the only method to beat Lead-3 on these articles, but it is about 700 times
  slower than `textrank`, English only, and can state things the article does not say.

## Aspects

| Method | Install | Notes |
|---|---|---|
| **`rules`** (default) | core | 0.877 F1 on 20 hand-written held-back reviews |
| `llm` | `[llm]` | Not measured, for the same reasons as `llm` sentiment |

## Keywords

All keyword methods are bundled: `textrank` (default), `rake` and `tfidf`. See
[Benchmarks](benchmarks.md).
