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
- Placeholder `clean()` that normalizes whitespace (full cleaner arrives in Phase 2).
