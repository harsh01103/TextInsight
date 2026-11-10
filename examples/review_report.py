"""Analyse a batch of reviews and save an HTML report.

Usage: python examples/review_report.py
"""

import textinsight as ti

reviews = [
    "camera mast hai but battery bekaar hai",
    "The food was excellent but the service was terrible",
    "delivery late aayi aur packing kharab thi",
    "screen bahut badhiya hai aur sound bhi zabardast hai",
]

report = ti.analyze(reviews, tasks=["sentiment", "aspects", "keywords"])
print(report.summary()["sentiment"])
report.to_html("review_report.html")
print("Wrote review_report.html")
