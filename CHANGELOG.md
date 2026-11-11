# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- `textinsight --help` now lists every command with its options, grouped by purpose, and every
  public Python function. `textinsight help <command>` shows one command's details.

## [1.0.0] - 2026-11-11

### Added
- `aspects()`: sentiment per aspect for English and Hinglish, with `AspectResult`.
- `stats()`: counts, type-token ratio, reading time and Flesch reading ease, with `TextStats`.
- `redact()`: finds URLs, emails, UPI IDs, PAN, cards (Luhn), Aadhaar (Verhoeff) and phone
  numbers, with `PIIResult`.
- `analyze()` and `Pipeline`: run several tasks over many texts, sharing language detection and
  skipping repeated texts, with JSON and self-contained HTML reports.
- `textinsight analyze` and `textinsight models` commands.
- A pandas accessor and a scikit-learn feature transformer.
- Optional backends: `onnx` English sentiment, `abstractive` summaries, and `llm` sentiment and
  aspects using Claude.
- A side-by-side benchmark of every backend, a documentation site, a Streamlit demo, a
  contributing guide and issue templates.
- Command-line tasks: `textinsight sentiment`, `aspects`, `keywords`, `summarize`, `stats`,
  `redact`, `language` and `clean`, reading from arguments or standard input, with `--json`.
- `textinsight -V`, `textinsight info`, `python -m textinsight`, and `ti.info()` and
  `ti.show_versions()` for bug reports.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, a pull request template, Dependabot,
  examples, more PyPI links, and an `all` extra.

## [0.2.0] - 2026-11-10

### Added
- `keywords()` with `textrank` (written from scratch), `rake` and `tfidf` methods.
- `summarize()` with extractive `textrank` and `frequency` methods, and `ratio` and
  `max_sentences` options.
- Both treat Hinglish spellings such as "khana" and "khaana" as one word.
- `eval.text_metrics`: Porter stemmer, keyphrase scoring and ROUGE, with no dependencies.
- `textinsight benchmark keywords` (Inspec) and `textinsight benchmark summarize`
  (CNN/DailyMail), with a Lead-3 baseline.

## [0.1.0] - 2026-11-10

### Added
- Project skeleton: src-layout package, `pyproject.toml`, ruff, mypy, pytest, pre-commit, GitHub Actions CI.
- Core foundation: configuration, exception hierarchy, input validation, typed result classes,
  backend base class, backend registry with `"auto"` selection, and a model manager with
  on-disk caching, SHA-256 verification and offline mode.
- Configurable `clean()`: HTML, URLs, emails, mentions, hashtags, emojis, numbers, punctuation.
- `tokenize()` and `sentences()` for informal English and Hinglish, including Devanagari and the danda.
- Hinglish spelling normalization: `normalize_hinglish()` and `cluster_variants()` for mining new variants.
- `normalize()` and `remove_stopwords()` for English and Hinglish.
- `detect_language()` with a from-scratch character n-gram classifier.
- `sentiment()` with three backends: `lexicon` (English and Hinglish, with post-word negation),
  bundled `classical` (NB-LR with Hinglish spelling features, trained on SentiMix), and optional
  `hybrid` (blended with HingBERT, via the `transformers` extra). Short, plain text falls back to the lexicon when the model
  says neutral but the lexicon is clear.
- Evaluation tools: SentiMix loader, metrics, and the `textinsight benchmark` command.
- Benchmarks on the held-out SentiMix test split, and a model card.
