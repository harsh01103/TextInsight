# Security policy

## Supported versions

Security fixes go into the latest release on PyPI.

## Reporting a problem

Please do not open a public issue for a security problem. Use GitHub's
[private vulnerability reporting](https://github.com/harsh01103/TextInsight/security/advisories/new)
instead, with steps to reproduce it. You should get a reply within a week.

## Good to know

- `redact()` finds personal information by pattern only. It cannot find names or addresses, so
  do not rely on it as the only safeguard for sensitive data.
- Model files are downloaded over HTTPS from pinned revisions, and checked against SHA-256
  hashes where TextInsight manages the download. Bundled models are stored as JSON, not pickle.
- The `llm` backends send your text to Anthropic. Every other backend runs on your machine.
