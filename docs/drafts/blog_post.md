# Draft: Why I built an NLP library for Hinglish

*A draft for a blog post or LinkedIn article. Rewrite it in your own words before posting.*

Millions of people in India write like this: *"yeh phone mast hai but battery bekaar hai."*
That's Hinglish, Hindi and English mixed together in Latin letters. Most NLP tools assume
clean English, so they miss most of what this sentence says.

I built TextInsight, an open-source Python library, to handle text like this. It's on PyPI as
`textinsight-nlp`.

## The problem

Hinglish has no fixed spelling: "accha", "acha" and "achha" are the same word. Hindi grammar also
puts negation after the word it changes, so "accha nahi" means "not good". A tool that reads word
by word gets that backwards.

## What I built

- A normalizer that maps spelling variants to one form, and matches unlisted variants by sound.
- A sentiment model trained on 17,000 real Hinglish tweets, which runs in plain Python with no
  dependencies.
- Keyword extraction and summaries using TextRank, which I implemented from scratch along with
  PageRank.
- Aspect sentiment ("camera: positive, battery: negative") and PII redaction with Aadhaar and
  card checksums.

## What the numbers say

On the SentiMix test set of 3,000 Hinglish tweets, the bundled model scored 0.703 weighted F1,
against 0.475 for VADER, a popular English tool. The best of 61 teams in the original shared
task scored 0.750, so it's competitive without beating the state of the art.

Some things didn't work. Fine-tuning a transformer on my laptop's CPU gave no gain, and on news
articles my TextRank summaries lost to simply taking the first three sentences. I kept those
results in the docs, because a benchmark only means something if you report the losses too.

## What I learned

*Add two or three things you learned yourself here.*

## Try it

```bash
pip install textinsight-nlp
```

*Add the GitHub and demo links.*
