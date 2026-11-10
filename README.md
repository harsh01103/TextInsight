# TextInsight

**Lightweight, offline-first NLP for English and Hinglish text, with pluggable backends.**

> Status: alpha. Preprocessing, language detection and sentiment work; other tasks arrive in upcoming releases.
> See `TextInsight_Architecture_and_Roadmap.docx` for the full plan.

## Why TextInsight?

- **Built for Hinglish** – handles code-mixed text such as *"yeh phone mast hai but battery bekaar hai"*: spelling variants, Hindi word order for negation, and a model trained on real Hinglish tweets.
- **Lightweight by default** – small base install; heavy models are optional extras.
- **No surprises** – importing the library never downloads anything.
- **Pluggable backends** – swap lexicon, classical ML, ONNX, transformer or LLM implementations with one argument.
- **Honest benchmarks** – every backend ships with measured results.

## Installation

```bash
pip install textinsight-nlp
```

The package installs as `textinsight-nlp` and is imported as `textinsight`.

## Quick start

```python
import textinsight as ti

ti.clean("Visit https://example.com now!", lowercase=True)  # 'visit now!'
ti.tokenize("Don't miss it :) #sale")  # ["Don't", 'miss', 'it', ':)', '#sale']
ti.sentences("Dr. Rao aaye. yeh accha hai!")  # ['Dr. Rao aaye.', 'yeh accha hai!']
ti.normalize_hinglish("bohot acha hai")  # 'bahut accha hai'
ti.detect_language("yeh phone mast hai but battery bekaar").language  # 'hi-en'
ti.sentiment("bahut accha product hai")  # SentimentResult(label='positive', ...)
ti.sentiment("yeh accha nahi hai", backend="lexicon")  # handles Hindi negation

doc = (
    "Phone ki battery life acchi hai. Camera quality bhi mast hai. Battery life do din chalti hai."
)
ti.keywords(doc, top_n=2).texts  # ['battery life', 'camera quality']
ti.summarize(doc, max_sentences=1).summary  # the most representative sentence

ti.aspects("camera mast hai but battery bekaar hai").labels
# {'camera': 'positive', 'battery': 'negative'}
ti.stats(doc).words  # also readability, diversity and reading time
ti.redact("Mail a.b@example.com or call 9876543210").text
# 'Mail [EMAIL] or call [PHONE]'

ti.list_backends()  # registered backends
ti.config.offline = True  # never download models
```

Language labels are `en`, `hi` (mainly Devanagari), `hi-en` (code-mixed or romanized Hindi)
and `und` (undetermined).

## Benchmarks

Sentiment on the held-out test split of [SentiMix](https://doi.org/10.5281/zenodo.3974927)
(3,000 real Hinglish tweets):

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Majority class | 0.367 | 0.179 | 0.197 |
| VADER | 0.492 | 0.470 | 0.475 |
| TextInsight `lexicon` | 0.561 | 0.540 | 0.546 |
| **TextInsight `classical`** (default, 529 KB, no dependencies) | **0.702** | **0.707** | **0.703** |
| **TextInsight `hybrid`** (optional transformer blend) | **0.723** | **0.728** | **0.725** |

Cross-validated macro F1 on the other 17,000 tweets is lower and more conservative: 0.639 for
`classical` and 0.666 for `hybrid`. Word-level language detection on the test tweets: 72.7%.

Short, plain sentences such as reviews use a lexicon fallback, because a model trained on tweets
reads them as neutral too often. On 60 hand-written review-style Hinglish sentences kept back for
this check, macro F1 rose from 0.581 to 0.819 for `classical` and from 0.499 to 0.834 for `hybrid`.
That set is small and not independent, so treat it as a behaviour check.

The test-split scores above are for tweets only. See the
[model card](docs/model_cards/sentiment_classical.md) for method, limitations and bias.

To reproduce them, install the `training` extra, download the dataset and run:

```bash
textinsight benchmark sentiment --data-dir path/to/Hinglish --split test
```

### Keywords and summaries

Keywords on the [Inspec](https://aclanthology.org/W03-1028/) test split (500 abstracts):

| Method | Precision@10 | Recall@10 | F1@10 |
|---|---|---|---|
| TF-IDF | 0.249 | 0.297 | 0.255 |
| RAKE | 0.253 | 0.302 | 0.259 |
| **TextRank** (default) | **0.275** | **0.318** | **0.278** |

Summaries on the CNN/DailyMail test split (11,490 articles, three sentences each):

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---|---|---|
| Lead-3 (first three sentences) | 0.402 | 0.175 | 0.251 |
| **TextRank** (default) | 0.378 | 0.158 | 0.241 |
| Frequency | 0.320 | 0.113 | 0.212 |

TextRank does not beat Lead-3 on news, where the main facts come first. On 20 hand-written
Hinglish paragraphs, TextRank keywords scored 0.375 F1 at 5 keywords, against 0.279 without
Hinglish handling. That set is small and not independent. See the
[method card](docs/model_cards/keywords_summarize.md).

### Aspects and PII

On 20 hand-written reviews in English and Hinglish, `aspects()` found the right aspect and
sentiment with 0.877 F1. PII rules scored 1.0 precision and recall on generated values, which
only shows that they work on the formats they were written for. Neither is an independent
benchmark. See the [method card](docs/model_cards/aspects_stats_pii.md) for known limits:
for example, `redact()` cannot find names, and treats any 10-digit number starting with 6 to 9
as a phone number.

## Most accurate option

```bash
pip install "textinsight-nlp[transformers]"
```

```python
ti.sentiment("zabardast movie thi, maza aa gaya", backend="hybrid")
```

First use downloads the 438 MB HingBERT encoder, so `hybrid` runs only when you ask for it.

## Credits

The bundled sentiment model derives from SentiMix (CC-BY-4.0): Patwa et al., "SemEval-2020
Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets", SemEval-2020. The `hybrid` backend
uses HingBERT (CC-BY-4.0): Joshi, "L3Cube-HingCorpus and HingBERT", 2022. Benchmarks use Inspec
(Hulth, 2003) and CNN/DailyMail (Hermann et al., 2015), which are not redistributed here.

## Planned features

| Release | Features |
|---|---|
| 0.1.0 | Preprocessing, language ID, English + Hinglish sentiment, benchmarks |
| 0.2.0 | Keywords (TF-IDF, RAKE, TextRank), extractive summarization |
| 0.3.0 | Aspect sentiment, text stats, PII redaction, `analyze()`, CLI, pandas |
| 1.0.0 | Optional ONNX / transformer / LLM backends, docs site, live demo |

## Development

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -e ".[dev]"
pre-commit install

pytest                          # tests + coverage
ruff check . && ruff format --check .
mypy
```

## Configuration

| Setting | Environment variable | Default |
|---|---|---|
| `config.cache_dir` | `TEXTINSIGHT_CACHE_DIR` | OS user cache folder + `/textinsight` |
| `config.offline` | `TEXTINSIGHT_OFFLINE` | `False` |
| `config.device` | `TEXTINSIGHT_DEVICE` | `"cpu"` |
| `config.max_input_chars` | `TEXTINSIGHT_MAX_INPUT_CHARS` | `1_000_000` |

## License

MIT
