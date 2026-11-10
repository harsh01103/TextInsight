# Hinglish

Hinglish mixes Hindi and English in Latin letters, with no fixed spelling. TextInsight handles
that in each task.

## Spelling

`normalize_hinglish` maps known variants to one form:

```python
ti.normalize_hinglish("bohot acha hai")  # 'bahut accha hai'
```

Keywords, aspects and summaries also match words by sound, so "khana" and "khaana", or "kitab"
and "kitaab", count as one word even when the dictionary does not list them.

## Word order

Hindi puts negation after the word it changes, so "accha nahi" is negative:

```python
ti.sentiment("yeh accha nahi hai", backend="lexicon").label  # 'negative'
```

In aspects, the owner before *ka*, *ki* or *ke* is removed, so "phone ki battery" is about the
battery.

## Language labels

| Label | Meaning |
|---|---|
| `en` | English |
| `hi` | Mainly Devanagari Hindi |
| `hi-en` | Code-mixed or romanized Hindi |
| `und` | Could not be determined |

## Limits

- Devanagari Hindi is not supported for sentiment, keywords, aspects or summaries.
- The sentiment model learned from tweets. A lexicon fallback helps with short reviews, but
  accuracy on other kinds of text is less well measured.
- Word-level language detection scored 72.7% on real tweets, so labels can be wrong.
