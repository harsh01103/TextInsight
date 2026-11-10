# Analysing many texts

`analyze` runs several tasks over your texts and collects the results in a `Report`.

```python
import textinsight as ti

report = ti.analyze(reviews, tasks=["sentiment", "aspects", "keywords"])

report.summary()  # totals across all documents
report.rows[0].sentiment  # the result for one text
report.to_json("report.json")
report.to_html("report.html")
```

Tasks are `sentiment`, `aspects`, `keywords`, `summarize`, `stats` and `redact`.

## How it works

- Language is detected once per text and shared between tasks.
- Each distinct text is analysed once. Pass `cache=False` to turn this off.
- Texts in an unsupported language stay in the report. Tasks that could not read them are
  listed in that row's `skipped` field instead of raising an error.

On 10,000 tweets with sentiment, keywords and aspects, `analyze` handled about 340 texts a
second on a laptop CPU. When half of the texts were repeats, the cache raised that from 354 to
715 texts a second.

## Personal information in reports

If `redact` is one of the tasks, JSON and HTML exports contain the redacted text and leave out
the original values. Without it, exports contain your original text, so treat those files as
carefully as the input.

The HTML report is one self-contained file. It runs no scripts and loads nothing from elsewhere.
