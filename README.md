# TextInsight

**Lightweight, offline-first NLP for English and Hinglish text, with pluggable backends.**

> Status: pre-alpha. Preprocessing and language detection work; sentiment and other tasks arrive in upcoming releases.
> See `TextInsight_Architecture_and_Roadmap.docx` for the full plan.

## Why TextInsight?

- **Hinglish support** – handles code-mixed text such as *"yeh phone mast hai but battery bekaar hai"*.
- **Lightweight by default** – small base install; heavy models are optional extras.
- **No surprises** – importing the library never downloads anything.
- **Pluggable backends** – swap lexicon, classical ML, ONNX, transformer or LLM implementations with one argument.
- **Honest benchmarks** – every backend ships with measured results.

## Installation

```bash
pip install textinsight
```

(Not yet published. For development, see below.)

## Quick start

```python
import textinsight as ti

ti.clean("Visit https://example.com now!", lowercase=True)  # 'visit now!'
ti.tokenize("Don't miss it :) #sale")  # ["Don't", 'miss', 'it', ':)', '#sale']
ti.sentences("Dr. Rao aaye. yeh accha hai!")  # ['Dr. Rao aaye.', 'yeh accha hai!']
ti.normalize_hinglish("bohot acha hai")  # 'bahut accha hai'
ti.detect_language("yeh phone mast hai but battery bekaar").language  # 'hi-en'

ti.list_backends()  # registered backends
ti.config.offline = True  # never download models
```

Language labels are `en`, `hi` (mainly Devanagari), `hi-en` (code-mixed or romanized Hindi)
and `und` (undetermined).

## Benchmarks

| Task | Data | Result | Notes |
|---|---|---|---|
| Language detection | 120 hand-written sentences | 99.2% accuracy | Sanity check only: the sample and the word lists have the same author. See `benchmarks/`. |

An independent measurement on real code-mixed data is planned for Phase 3.

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
