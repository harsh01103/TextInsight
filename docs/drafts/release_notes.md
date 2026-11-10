# Draft release notes

*Drafts for GitHub releases. Check the numbers against `benchmarks/results` before publishing.*

## v0.3.0: analyze, reports and integrations

- `analyze()` runs several tasks over many texts, shares language detection, and skips repeated
  texts. On 10,000 tweets with half of them repeated, throughput rose from 354 to 715 texts a
  second.
- JSON and self-contained HTML reports. Exports hide personal information when `redact` runs.
- `textinsight analyze reviews.csv --out report.html`, plus `textinsight models list` and
  `textinsight models clear`.
- A pandas accessor (`df["review"].ti.sentiment()`) and a scikit-learn feature transformer.
- `aspects()`, `stats()` and `redact()` from Phase 5.

## v1.0.0: optional backends, docs and demo

- Optional backends that run only when you name them: `onnx` (fast English sentiment),
  `abstractive` summaries, and `llm` sentiment and aspects using Claude.
- A side-by-side comparison of every backend's quality, speed and memory.
- A documentation site, a Streamlit demo, a contributing guide and issue templates.
- The public API is now considered stable.
