"""Run every TextInsight task on a few Hinglish and English texts.

Usage: python examples/quickstart.py
"""

import textinsight as ti

review = (
    "Is phone ki battery life bahut acchi hai. Camera quality bhi mast hai "
    "lekin display thoda dim hai."
)

print("Language: ", ti.detect_language(review).language)
print("Sentiment:", ti.sentiment("bahut accha product hai").label)
print("Aspects:  ", ti.aspects("camera mast hai but battery bekaar hai").labels)
print("Keywords: ", ti.keywords(review, top_n=3).texts)
print("Summary:  ", ti.summarize(review, max_sentences=1).summary)
print("Redacted: ", ti.redact("Mail a.person@example.com or call 9876543210").text)
print("Words:    ", ti.stats(review).words)
