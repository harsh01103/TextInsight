# Model card: TextInsight sentiment models

Three-way sentiment (negative, neutral, positive) for informal Hinglish and English.

| Backend | How to get it | Size | What it is |
|---|---|---|---|
| `classical` (default) | `pip install textinsight-nlp` | 529 KB, in the package | NB-LR over word, character and Hinglish spelling features, run in plain Python |
| `hybrid` | `pip install "textinsight-nlp[transformers]"`, then `backend="hybrid"` | 12 KB in the package, plus a 438 MB encoder downloaded on first use | 50/50 blend of `classical` and a classifier on HingBERT embeddings |
| `lexicon` | `pip install textinsight-nlp` | Word lists | Rules, with Hindi post-word negation ("accha nahi") |

## Results on the held-out test split (3,000 Hinglish tweets)

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Majority class | 0.367 | 0.179 | 0.197 |
| VADER (English lexicon) | 0.492 | 0.470 | 0.475 |
| TextInsight `lexicon` | 0.561 | 0.540 | 0.546 |
| TextInsight `classical` | 0.702 | 0.707 | **0.703** |
| TextInsight `hybrid` | 0.723 | 0.728 | **0.725** |

Per-class F1: `classical` negative 0.722, neutral 0.630, positive 0.767; `hybrid`
negative 0.750, neutral 0.657, positive 0.778.

For context, the SemEval-2020 Task 9 overview reports weighted F1 of 0.654 for its
multilingual BERT baseline and 0.750 for the best of 61 teams. This comparison is
indicative only: those systems worked under competition conditions.

**How the test split was used.** It has been evaluated three times: for the first
version (`classical` 0.684 weighted F1), for the models before the lexicon fallback
(`classical` 0.708, `hybrid` 0.729), and for the final models above. The fallback
lowered both scores slightly, as cross-validation predicted. All design choices
between those runs were made with 5-fold cross-validation on the
train and dev tweets only (`training/experiment_sentiment.py`), so the test split did
not guide them. Cross-validation is also the more conservative estimate: macro F1 of
0.639 for `classical` and 0.666 for `hybrid`. The test split appears to be somewhat
easier than the rest of the data.

## What improved the model (5-fold cross-validated macro F1)

| Change | Macro F1 |
|---|---|
| First version: TF-IDF + logistic regression | 0.627 |
| NB-LR instead of plain logistic regression | 0.640 |
| Adding Hinglish spelling, negation, lexicon and code-mix features | 0.642 |
| With identity words masked (shipped) | 0.641 |
| HingBERT embeddings alone | 0.659 |
| **Blend of both (hybrid)** | **0.669** |

Each individual feature family changed the score by less than its standard deviation
(about 0.008), so those small steps may be noise. The change of model type and the
transformer blend are the improvements that clearly mattered.

## Short, plain text

The training tweets have a median of 20 words and noisy labels for many clear opinion
words: "zabardast" appears 11 times, 6 of them labelled neutral. So `classical` and
`hybrid` called short reviews neutral far too often. If either predicts neutral, the
text has at most 12 words and no hashtag or emoji, and the lexicon score is at least
0.3 in size, the lexicon's label and score are used instead.

| Macro F1 | Without fallback | With fallback |
|---|---|---|
| `classical`, 60 held-back review sentences | 0.581 | **0.819** |
| `hybrid`, 60 held-back review sentences | 0.499 | **0.834** |
| `classical`, SentiMix 5-fold CV | 0.641 | 0.639 |
| `hybrid`, SentiMix 5-fold CV | 0.669 | 0.666 |

The review sentences (`benchmarks/data/hinglish_reviews.tsv`, run with
`benchmarks/run_reviews.py`) were written by hand for this check, 40 per class, so the
set is small and not independent. The rule was chosen on the other 60 sentences and on
SentiMix train + dev. The lexicon alone scored 0.840 on the held-back sentences, so the
fallback mainly brings the models up to the lexicon on this kind of text. Opinions
that use words missing from the lexicon can still come out neutral.

## What did not help

These were tried and rejected. None of them changed the shipped models.

| Experiment | Result | Compared with |
|---|---|---|
| Ensemble of NB-LR, NB-SVM, LR and SVC | CV macro F1 0.640 | 0.641 for `classical` |
| Fine-tuning HingBERT's top 4 layers (3 epochs, CPU, one seed) | dev macro F1 0.656 | 0.658 for the frozen encoder |
| The same fine-tuned model blended with `classical` | dev macro F1 0.664 | 0.667 for the frozen blend |
| Learned stacking instead of a fixed blend | CV macro F1 0.670 | 0.669 for `hybrid` |
| Last-4-layer pooling with a re-tuned blend weight | CV macro F1 0.670 to 0.671, better in 3 of 5 folds | 0.669 for `hybrid` |
| Naming emojis, as HingBERT reads nearly all of them as `[UNK]` | CV macro F1 0.6635 for the embeddings alone | 0.6635 without names |

Word features and HingBERT embeddings appear to have reached a plateau of about 0.67
on this dataset. The fine-tuning result covers only one configuration on a CPU, so it
does not rule out gains from a longer search on a GPU.

## Training data and credits

- SentiMix (SemEval-2020 Task 9, Hinglish), CC-BY-4.0. Patwa et al., "SemEval-2020
  Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets", SemEval-2020.
  DOI: 10.5281/zenodo.3974927. The dataset is not redistributed with this package.
  Usernames and t.co links were removed before training.
- HingBERT encoder: `l3cube-pune/hing-bert`, CC-BY-4.0. Joshi, "L3Cube-HingCorpus and
  HingBERT: A Code Mixed Hindi-English Dataset and BERT Language Models", 2022. It is
  used unchanged at a pinned revision, in safetensors format.

## Method

- `classical`: word 1-2 grams, character 3-4 grams, phonetic keys, vowel-dropped
  skeletons, negation scope, lexicon score and code-mix ratio; sublinear TF-IDF;
  50,000 features; one-vs-rest NB-LR with C = 1.0, exported as linear weights.
- Both models use the lexicon fallback described above on short, plain text.
- `hybrid` adds logistic regression (C = 0.001) on mean-pooled HingBERT embeddings.
  The encoder is not fine-tuned, so only the small classifier needs to ship.
- The final models were refit on train + dev (17,000 tweets).
- The plain-Python model matched scikit-learn on all 3,000 dev tweets.

## Limitations

- Measured only on tweets. Accuracy on product reviews or formal text is unknown.
- English use is measured only on English-looking tweets from this dataset.
- The neutral class is the weakest, and the dataset's labels are themselves noisy.
- Scores are not calibrated probabilities.
- Hindi in Devanagari script is not supported.

## Bias

The data is heavy in Indian political tweets, and religion, nationality and party
words lean negative in it (for example "muslim" appears in 29 positive and 117
negative training tweets, and "hindu" in 33 and 142). `classical` masks these words
so it cannot learn that a group of people is negative, at a cost of about 0.001 macro
F1. The masking list is short and not exhaustive. The HingBERT part of `hybrid` is
**not** masked. Do not use these models to make decisions about people.
