# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project skeleton: src-layout package, `pyproject.toml`, ruff, mypy, pytest, pre-commit, GitHub Actions CI.
- Core foundation: configuration, exception hierarchy, input validation, typed result classes,
  backend base class, backend registry with `"auto"` selection, and a model manager with
  on-disk caching, SHA-256 verification and offline mode.
- Configurable `clean()`: HTML, URLs, emails, mentions, hashtags, emojis, numbers, punctuation.
- `tokenize()` and `sentences()` for informal English and Hinglish, including Devanagari and the danda.
- Hinglish spelling normalization: `normalize_hinglish()` and `cluster_variants()` for mining new variants.
- `normalize()` and `remove_stopwords()` for English and Hinglish.
- `detect_language()` with a from-scratch character n-gram classifier, plus a first benchmark.
