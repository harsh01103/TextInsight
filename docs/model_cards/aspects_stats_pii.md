# Method card: aspects, text statistics and PII redaction

All three run offline in plain Python, with no trained model and no extra packages.

| Function | Result type | What it does |
|---|---|---|
| `aspects()` | `AspectResult` | Finds what a review praises or criticizes |
| `stats()` | `TextStats` | Counts, lexical diversity, reading time and Flesch reading ease |
| `redact()` | `PIIResult` | Replaces personal information with placeholders such as `[EMAIL]` |

## Aspects

**How it works.** The text is split into clauses at contrast words such as "but" and
"lekin", and at "and", "aur" or commas when each side has its own opinion. Each
clause's lexicon score goes to the content words in it. An aspect is reported only if
some clause expresses an opinion about it. In Hinglish, the owner before ka, ki or ke
is removed, so "phone ki battery" is about the battery.

**Results** on 20 held-back reviews, in English and Hinglish, scored once:

| Measure | Precision | Recall | F1 |
|---|---|---|---|
| Aspect found | 0.929 | 0.897 | 0.912 |
| Aspect found and sentiment correct | 0.893 | 0.862 | 0.877 |

The reviews (`benchmarks/data/aspect_reviews.tsv`, run with `benchmarks/run_aspects.py`)
were written by hand and are short and direct. This is a small behaviour check and not
an independent benchmark, so expect lower accuracy on real reviews. No public aspect
dataset was used.

**Limitations.**

- There is no part-of-speech tagging, so a verb or adjective can join an aspect, as in
  "battery drains".
- Opinions come from a word list, so unlisted words such as "saaf" or "drains fast" are
  missed, and opinions that depend on context, such as "too high", are not understood.
- "par" is not treated as "but", because it also means "on", so clauses joined by it can
  share one score.
- Sarcasm and comparisons between products are not handled.
- Hindi in Devanagari script is not supported.

## Text statistics

- Words include numbers and hashtags, but not links, mentions or emojis.
- In Hinglish, spelling variants count once in `unique_words` and `type_token_ratio`.
- Reading time assumes 238 words a minute, an average for silent reading in English
  (Brysbaert, 2019), so it is only a rough guide for Hinglish.

**Limitations.** Flesch reading ease is calculated only for English, and is `None`
otherwise. Syllables are estimated from spelling, so the score is approximate. Sentence
counts depend on punctuation.

## PII redaction

| Kind | How it is checked |
|---|---|
| `URL`, `EMAIL` | Pattern |
| `UPI` | Pattern: a name, then `@`, then a provider with no dot |
| `PAN` | Format, including a valid holder-type letter |
| `CARD` | 13 to 19 digits and the Luhn checksum |
| `AADHAAR` | 12 digits, not starting with 0 or 1, and the Verhoeff checksum |
| `PHONE` | Indian mobile formats, and international numbers starting with `+` |

**Results** on a generated set (2,000 fake values of each kind, plus look-alikes that
must not be flagged, from `benchmarks/run_pii.py`): precision and recall were 1.0 for
every kind.

Treat those numbers as a check that the rules work as designed. The test values use
formats the rules were written for, so they say nothing about accuracy on real text.
The look-alike order numbers start with 1 to 5, so they do not test the known
false positive below.

**Limitations.**

- Names, addresses and other information without a fixed pattern are not found.
- Any 10-digit number starting with 6 to 9 looks like a phone number, so some order
  numbers and IDs will be redacted by mistake.
- About 1 in 10 random 12-digit numbers starting with 2 to 9 passes Verhoeff, so an
  unrelated number can be flagged as Aadhaar. Luhn lets a similar share through for cards.
- A valid checksum shows only that a number is well formed, not that it was issued.
- `PIIEntity.value` and `to_dict()` contain the original text, although `repr()` hides it.
  Do not log those values.
- Do not rely on this as the only safeguard for sensitive data.
