# Command line

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
