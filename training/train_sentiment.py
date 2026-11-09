"""Train the bundled sentiment model.

Usage:
    pip install -e ".[training]"
    python training/train_sentiment.py --data-dir data/sentimix/Semeval_2020_task9_data/Hinglish

The design below was chosen by 5-fold cross-validation on train + dev (see
experiment_sentiment.py). This script never reads the test split.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from textinsight.eval import classification_report, load_sentimix
from textinsight.sentiment.classical import LinearModel
from textinsight.sentiment.features import FeatureOptions, extract_features

OUTPUT = (
    Path(__file__).parent.parent / "src" / "textinsight" / "_data" / "sentiment_hinglish.json.gz"
)
OPTIONS = FeatureOptions(
    phonetic=True, skeleton=True, code_mix=True, negation=True, lexicon=True, mask_identity=True
)
C = 1.0
MIN_DF = 2
MAX_FEATURES = 50_000
NB_ALPHA = 1.0


def nb_ratios(matrix, target):
    """Naive Bayes log-count ratios for one class against the rest."""
    positive = np.asarray(matrix[target].sum(axis=0)).ravel() + NB_ALPHA
    negative = np.asarray(matrix[~target].sum(axis=0)).ravel() + NB_ALPHA
    return np.log((positive / positive.sum()) / (negative / negative.sum()))


def fit(texts, labels, vocabulary=None):
    """Fit one-vs-rest NB-LR. Because the ratios only rescale features, the result is linear."""
    vectorizer = CountVectorizer(
        analyzer=partial(extract_features, options=OPTIONS),
        min_df=1 if vocabulary is not None else MIN_DF,
        vocabulary=vocabulary,
    )
    counts = vectorizer.fit_transform(texts)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
    matrix = tfidf.fit_transform(counts)
    binary = (counts > 0).astype(float)
    y = np.array(labels)
    classes = sorted(set(labels))
    coefficients, intercepts = [], []
    for name in classes:
        target = y == name
        ratios = nb_ratios(binary, target)
        model = LogisticRegression(C=C, max_iter=3000)
        model.fit(matrix.multiply(ratios).tocsr(), target)
        coefficients.append(model.coef_[0] * ratios)
        intercepts.append(float(model.intercept_[0]))
    return vectorizer, tfidf, classes, np.vstack(coefficients), np.array(intercepts)


def predict(model, texts):
    vectorizer, tfidf, classes, coefficients, intercepts = model
    matrix = tfidf.transform(vectorizer.transform(texts))
    scores = matrix @ coefficients.T + intercepts
    return [classes[i] for i in np.argmax(scores, axis=1)]


def strongest_features(model, limit):
    vectorizer, _, _, coefficients, _ = model
    strength = np.abs(coefficients).max(axis=0)
    names = vectorizer.get_feature_names_out()
    return sorted(names[np.argsort(-strength)[:limit]].tolist())


def export(model):
    vectorizer, tfidf, classes, coefficients, intercepts = model
    return {
        "classes": classes,
        "options": asdict(OPTIONS),
        "features": vectorizer.get_feature_names_out().tolist(),
        "idf": [round(float(v), 4) for v in tfidf.idf_],
        "coefficients": [[round(float(v), 4) for v in row] for row in coefficients],
        "intercepts": [round(float(v), 4) for v in intercepts],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    train = load_sentimix(args.data_dir, "train")
    dev = load_sentimix(args.data_dir, "dev")
    train_texts, train_labels = [e.text for e in train], [e.label for e in train]
    dev_texts, dev_labels = [e.text for e in dev], [e.label for e in dev]

    full = fit(train_texts, train_labels)
    report = classification_report(dev_labels, predict(full, dev_texts))
    print(f"{len(full[0].vocabulary_)} features: dev macro-F1 {report['macro_f1']:.4f}")

    kept = strongest_features(full, MAX_FEATURES)
    pruned = fit(train_texts, train_labels, vocabulary=kept)
    pruned_predictions = predict(pruned, dev_texts)
    report = classification_report(dev_labels, pruned_predictions)
    print(f"pruned to {len(kept)}: dev macro-F1 {report['macro_f1']:.4f}")

    # Check that the plain-Python model reproduces scikit-learn before shipping it.
    python_model = LinearModel(export(pruned))
    agree = 0
    for text, expected in zip(dev_texts, pruned_predictions, strict=True):
        probabilities = python_model.probabilities(text)
        agree += python_model.classes[probabilities.index(max(probabilities))] == expected
    print(f"plain-Python agreement with scikit-learn on dev: {agree}/{len(dev_texts)}")
    if agree / len(dev_texts) < 0.995:
        raise SystemExit("Exported model disagrees with scikit-learn; not writing it.")

    final = fit(train_texts + dev_texts, train_labels + dev_labels, vocabulary=kept)
    with gzip.GzipFile(args.output, "wb", mtime=0) as raw:
        raw.write(json.dumps(export(final), ensure_ascii=False, separators=(",", ":")).encode())
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
