# Benchmarks

Every number here comes from a script in `benchmarks/` or the `textinsight benchmark` command,
and the raw results are in `benchmarks/results/`. Design choices were made on training or
validation data. The model cards record each time a test split was used.

## Sentiment: SentiMix test split (3,000 Hinglish tweets)

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Majority class | 0.367 | 0.179 | 0.197 |
| VADER | 0.492 | 0.470 | 0.475 |
| `lexicon` | 0.561 | 0.540 | 0.546 |
| **`classical`** (default) | 0.702 | 0.707 | **0.703** |
| **`hybrid`** | 0.723 | 0.728 | **0.725** |

For context, the SemEval-2020 Task 9 overview reports 0.654 weighted F1 for its multilingual BERT
baseline and 0.750 for the best of 61 teams.

## Keywords: Inspec test split (500 abstracts)

| Method | Precision@10 | Recall@10 | F1@10 |
|---|---|---|---|
| TF-IDF | 0.249 | 0.297 | 0.255 |
| RAKE | 0.253 | 0.302 | 0.259 |
| **TextRank** | **0.275** | **0.318** | **0.278** |

## Summaries: CNN/DailyMail test split (11,490 articles)

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|---|---|---|
| **Lead-3 baseline** | **0.402** | **0.175** | **0.251** |
| TextRank | 0.378 | 0.158 | 0.241 |
| Frequency | 0.320 | 0.113 | 0.212 |

TextRank does not beat Lead-3 on news, where the main facts come first.

## Small behaviour checks

These sets were written by hand for this project, so they show behaviour and are not
independent benchmarks.

| Check | Result |
|---|---|
| Hinglish keywords, 20 paragraphs | 0.375 F1 at 5 keywords, against 0.279 without Hinglish handling |
| Aspects, 20 English and Hinglish reviews | 0.877 F1 for aspect and sentiment together |
| Review-style Hinglish sentiment, 60 sentences | 0.819 macro F1 for `classical`, against 0.581 without the lexicon fallback |
| PII, generated values | 1.0 precision and recall; only shows the rules work on their own formats |

## Speed

| Measure | Result |
|---|---|
| `import textinsight` | 0.54 s |
| `analyze()` with sentiment, keywords and aspects, 10,000 tweets | about 340 texts a second |
| The same, with half the texts repeated | 715 texts a second with the cache, against 354 without |
| Peak Python memory for that run | 17 MB |

Backend speed and memory are compared in [Backends](backends.md).
