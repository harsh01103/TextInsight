# Quickstart

You need Python 3.10 or newer.

```bash
pip install textinsight-nlp
```

The package installs as `textinsight-nlp` and is imported as `textinsight`.

## One text at a time

```python
import textinsight as ti

review = "Phone ki battery life acchi hai. Camera quality bhi mast hai."

ti.detect_language(review).language  # 'hi-en'
ti.sentiment(review).label  # 'positive'
ti.keywords(review, top_n=2).texts  # ['battery life', 'camera quality']
ti.summarize(review, max_sentences=1).summary
ti.aspects("camera mast hai but battery bekaar hai").labels
ti.stats(review).words
ti.redact("Mail a@example.com or call 9876543210").text
# 'Mail [EMAIL] or call [PHONE]'
```

## Many texts

Most functions accept a list and return a list:

```python
ti.sentiment(["bahut accha hai", "bilkul bekaar hai"])
```

To run several tasks and get one report, use [`analyze`](guides/analyze.md).

## Results

Results are typed objects with a `to_dict()` method, so they are easy to inspect and save.

## Errors

Every error is a subclass of `ti.TextInsightError`. Hindi written in Devanagari raises
`ti.UnsupportedLanguageError` for sentiment, keywords, aspects and summaries. A backend that
needs an extra package raises `ti.BackendNotAvailableError` with the install command.

## Settings

```python
ti.config.offline = True  # never download a model
ti.config.cache_dir = "D:/models"
```
