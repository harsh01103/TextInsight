# API reference

Each function takes a string and returns one result, or takes a list and returns a list.
`language` skips detection and treats all input as that language code.

## Text processing

| Function | Returns |
|---|---|
| `clean(text, *, lowercase=False, urls=True, html_tags=True, emails=False, ...)` | `str`, with the chosen parts removed (URLs and HTML by default) |
| `tokenize(text, *, lowercase=False, punctuation=True)` | `list[str]` |
| `sentences(text)` | `list[str]` |
| `normalize(text, *, hinglish=False, stopwords=False, language="en")` | `str` |
| `normalize_hinglish(text)` | `str`, with known spelling variants mapped to one form |
| `remove_stopwords(tokens, *, language="en")` | `list[str]` |
| `detect_language(text)` | `LanguageResult(language, score, token_languages)` |

## Tasks

| Function | Methods | Returns |
|---|---|---|
| `sentiment(text, *, backend="auto")` | `lexicon`, `classical`, `hybrid`, `onnx`, `llm` | `SentimentResult(label, score, backend, language)` |
| `aspects(text, *, min_mentions=1, method="auto")` | `rules`, `llm` | `AspectResult(aspects, method)`, with `.labels` |
| `keywords(text, *, top_n=10, method="auto")` | `textrank`, `rake`, `tfidf` | `KeywordResult(keywords, method)`, with `.texts` |
| `summarize(text, *, ratio=0.3, max_sentences=None, method="auto")` | `textrank`, `frequency`, `abstractive` | `SummaryResult(summary, method, sentence_indices)` |
| `stats(text)` | | `TextStats(words, sentences, unique_words, type_token_ratio, flesch_reading_ease, reading_time_seconds, ...)` |
| `redact(text, *, kinds=None)` | | `PIIResult(text, entities)` |
| `analyze(texts, tasks=("sentiment", "keywords", "aspects"), *, cache=True)` | | `Report` |

`score` in sentiment runs from -1 to +1 and is not a calibrated probability. `label` is one of
`positive`, `negative`, `neutral` or `mixed`.

## Report

| Member | Does |
|---|---|
| `rows` | One `Analysis` per input text, in input order |
| `summary()` | Totals across all documents |
| `to_dict()`, `to_json(path=None)`, `to_html(path=None)` | Exports |

## Backends and settings

| Name | Does |
|---|---|
| `list_backends(task=None)` | Lists backends and whether each is installed |
| `config.offline`, `config.cache_dir`, `config.max_input_chars` | Settings, also readable from `TEXTINSIGHT_*` environment variables |
| `Pipeline(tasks, *, cache=True, batch_size=256)` | A reusable `analyze` with fixed settings |

## Errors

All errors inherit from `TextInsightError`: `InvalidInputError`, `UnsupportedLanguageError`,
`UnknownBackendError`, `BackendNotAvailableError`, `ModelDownloadError`, `ConfigError` and
`LLMError`.
