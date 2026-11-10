# Method card: keywords and summarization

Both tasks use algorithms with no trained model, written in plain Python. They run
offline and need no extra packages.

| Function | Methods | What it does |
|---|---|---|
| `keywords()` | `textrank` (default), `rake`, `tfidf` | Ranks phrases of up to 3 words |
| `summarize()` | `textrank` (default), `frequency` | Picks existing sentences; it does not write new ones |

## Results

Each test split below was evaluated once, after every design choice was made.

**Keywords on Inspec** (500 test abstracts, words matched after Porter stemming):

| Method | Precision@5 | F1@5 | Precision@10 | Recall@10 | F1@10 |
|---|---|---|---|---|---|
| TF-IDF | 0.255 | 0.185 | 0.249 | 0.297 | 0.255 |
| RAKE | 0.245 | 0.178 | 0.253 | 0.302 | 0.259 |
| **TextRank** | **0.319** | **0.227** | **0.275** | **0.318** | **0.278** |

References include keyphrases that never appear in the abstract, which no extractive
method can find, so perfect scores are impossible.

**Summaries on CNN/DailyMail** (11,490 test articles, three-sentence summaries, ROUGE F1):

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| **Lead-3 baseline** (first three sentences) | **0.402** | **0.175** | **0.251** |
| TextRank | 0.378 | 0.158 | 0.241 |
| Frequency | 0.320 | 0.113 | 0.212 |

TextRank does not beat Lead-3. News articles state the main facts first, which makes
Lead-3 hard to beat with methods that only pick sentences.

**Keywords on Hinglish** (20 held-back paragraphs, F1 at 5 keywords):

| Method | With Hinglish handling | Treated as English |
|---|---|---|
| TF-IDF | 0.275 | 0.221 |
| RAKE | 0.364 | 0.335 |
| **TextRank** | **0.375** | 0.279 |

These paragraphs and their keyphrases were written by hand for this check
(`benchmarks/data/hinglish_keywords.tsv`). The set is small and not independent, so it
is a behaviour check. No public Hinglish keyphrase benchmark was used. Summaries were
not measured on Hinglish.

ROUGE and Porter stemming here are this project's own implementations, so scores can
be compared with published ones only roughly.

## How the settings were chosen

Keyword settings were chosen on the 1,500 Inspec train and validation documents. On
those documents, TextRank's F1@10 rose from 0.189 to 0.260 after these changes:

- ignoring adverbs and a short list of generic words such as "using" and "proposed";
- scoring every run of content words instead of merging only the top third of words;
- using a co-occurrence window of 10;
- dividing a phrase's score by the square root of its length.

Summary settings were chosen on 2,000 CNN/DailyMail validation articles. Letting
PageRank's random jumps favour earlier sentences raised ROUGE-1 from 0.360 to 0.383.

The generic-word list includes research-writing terms chosen with Inspec in mind, so
it may help less on other kinds of text.

## Hinglish handling

- Words are matched by sound, so "khana" and "khaana" or "kitab" and "kitaab" count as one.
- Hinglish filler words, common verbs and opinion words are not treated as topics.
- Negations such as "nahi" end a phrase.

## Limitations

- There is no part-of-speech tagging, so verbs and adjectives can appear as keywords.
- Hinglish verbs are filtered from a fixed list, so unlisted verb forms can still appear.
- Sentences are found from punctuation. Hinglish typed with no full stops counts as one
  sentence and is returned unchanged by `summarize()`.
- Short texts give less evidence, so their keywords are less reliable.
- Hindi in Devanagari script is not supported.
- `tfidf` compares documents only when given a list. A single text uses its own
  sentences, which is weaker.

## Data and credits

Neither dataset is redistributed with this package.

- Inspec: Hulth, "Improved Automatic Keyword Extraction Given More Linguistic
  Knowledge", EMNLP 2003. Loaded from Hugging Face `midas/inspec`.
- CNN/DailyMail 3.0.0: Hermann et al., 2015; See et al., 2017. Loaded from Hugging Face
  `abisee/cnn_dailymail`.
- TextRank: Mihalcea and Tarau, 2004. RAKE: Rose et al., 2010. Porter stemmer: Porter, 1980.
  ROUGE: Lin, 2004.
