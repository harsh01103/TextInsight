# Contributing to TextInsight

Thank you for helping. Bug reports, Hinglish examples that go wrong, and pull requests are all
welcome.

## Set up

```bash
python -m venv .venv
```

```bash
.venv/Scripts/activate
```

On macOS or Linux, run `source .venv/bin/activate` instead.

```bash
pip install -e ".[dev]"
```

## Before you open a pull request

Run the same checks as CI:

```bash
ruff check .
```

```bash
ruff format --check .
```

```bash
mypy src
```

```bash
pytest
```

Coverage must stay at 90% or above.

## Guidelines

- Keep the core install free of dependencies. Heavy packages go in an optional extra, and their
  backends must have a lower priority than the bundled ones, so that `"auto"` never downloads
  anything.
- Every change to a model, word list or rule needs a measured result. Tune on train or
  validation data, and record any use of a test split in the model card.
- Do not commit datasets. Note the licence and source of any data you use.
- Add tests for new behaviour, including Hinglish examples where they apply.

## Reporting a wrong result

Open an issue with the exact text, what TextInsight returned, and what you expected. Remove any
personal information first.
