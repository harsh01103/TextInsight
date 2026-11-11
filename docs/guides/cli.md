# Command line

`textinsight --help` (or `textinsight help`) lists every command with its options, and every
Python function. `textinsight help keywords` shows one command in detail, as does
`textinsight keywords --help`. `python -m textinsight` works the same way as `textinsight`.

```bash
textinsight --help
```

```bash
textinsight --version
```

## One text at a time

```bash
textinsight sentiment "yeh phone mast hai"
```

```bash
textinsight aspects "camera mast hai but battery bekaar hai"
```

The task commands are `sentiment`, `aspects`, `keywords`, `summarize`, `stats`, `redact`,
`language` and `clean`. Leave out the text to read lines from standard input, one text per
line, and add `--json` for one JSON object per line:

```bash
textinsight sentiment --json < reviews.txt
```

## Versions and installed backends

```bash
textinsight info
```

In Python, `ti.show_versions()` prints the same details and `ti.info()` returns them as a
dictionary. Include this output in bug reports.

## Analyse a file

```bash
textinsight analyze reviews.csv --column review --tasks sentiment,aspects,keywords --out report.html
```

- Input can be `.csv`, `.jsonl`, or plain text with one document per line.
- `--column` can be left out if the file has one column, or a column named `text`, `review`,
  `comment` or `content`.
- `--out` must end in `.html` or `.json`. Without it, the totals are printed.

## Benchmarks

```bash
textinsight benchmark sentiment --data-dir path/to/Hinglish --split test
```

Tasks are `sentiment`, `langid`, `keywords` and `summarize`. Datasets are not included, and
`pip install "textinsight-nlp[training]"` adds what the benchmarks need.

## Model cache

```bash
textinsight models list
```

```bash
textinsight models clear
```
