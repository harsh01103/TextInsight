# TextInsight

Lightweight NLP for English and code-mixed Hinglish, such as
*"yeh phone mast hai but battery bekaar hai"*.

```bash
pip install textinsight-nlp
```

```python
import textinsight as ti

ti.sentiment("bahut accha product hai").label  # 'positive'
ti.aspects("camera mast hai but battery bekaar hai").labels
# {'camera': 'positive', 'battery': 'negative'}
```

## What it does

| Task | Functions |
|---|---|
| Clean and split text, including Hinglish spelling variants | `clean`, `tokenize`, `sentences`, `normalize_hinglish` |
| Tell English, Hindi and code-mixed text apart | `detect_language` |
| Sentiment | `sentiment` |
| What a review praises or criticizes | `aspects` |
| Keywords | `keywords` |
| Summaries | `summarize` |
| Word counts and readability | `stats` |
| Hide emails, phone numbers, PAN, Aadhaar and cards | `redact` |
| Run several tasks over many texts and export a report | `analyze` |

The core install has no dependencies and works offline. Larger models and paid APIs are
optional, and run only when you ask for them by name. See [Backends](backends.md).

Start with the [Quickstart](quickstart.md). Accuracy and known limits are recorded in the
[benchmarks](benchmarks.md) and the model cards.
