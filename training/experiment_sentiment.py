"""Compare sentiment model designs with 5-fold cross-validation.

Usage:
    python training/experiment_sentiment.py --data-dir <Hinglish folder> --stage 1

Only the train and dev splits are used. The test split is never read here, so
these comparisons cannot leak information from it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from textinsight.eval import classification_report, load_sentimix
from textinsight.sentiment.features import FeatureOptions, extract_features

RESULTS = Path(__file__).parent.parent / "benchmarks" / "results" / "sentiment_experiments.json"
FOLDS = 5
SEED = 13


@dataclass(frozen=True)
class Spec:
    name: str
    options: FeatureOptions = field(default_factory=FeatureOptions)
    model: str = "lr"
    c: float = 1.0
    min_df: int = 2
    alpha: float = 1.0
    beta: float = 1.0
    balanced: bool = False


def identity(document):
    return document


def nb_ratios(matrix, target, alpha=1.0):
    """Naive Bayes log-count ratios for one class against the rest."""
    positive = np.asarray(matrix[target].sum(axis=0)).ravel() + alpha
    negative = np.asarray(matrix[~target].sum(axis=0)).ravel() + alpha
    return np.log((positive / positive.sum()) / (negative / negative.sum()))


def ovr_scores(train, test, binary, y_train, classes, spec, kind):
    """One-vs-rest decision scores, plus the training-score spread used to average models."""
    test_columns, train_columns = [], []
    for name in classes:
        target = y_train == name
        weight = "balanced" if spec.balanced else None
        if kind in {"nblr", "nbsvm"}:
            ratios = nb_ratios(binary, target, spec.alpha)
            x_train, x_test = train.multiply(ratios).tocsr(), test.multiply(ratios).tocsr()
        else:
            x_train, x_test = train, test
        if kind in {"nblr", "lr"}:
            model = LogisticRegression(C=spec.c, max_iter=3000, class_weight=weight)
        else:
            model = LinearSVC(C=spec.c, max_iter=5000, class_weight=weight)
        model.fit(x_train, target)
        coefficient = model.coef_[0]
        if spec.beta < 1.0:  # Wang & Manning's interpolation towards the mean weight
            coefficient = (1 - spec.beta) * np.abs(coefficient).mean() + spec.beta * coefficient
        test_columns.append(x_test @ coefficient + model.intercept_[0])
        train_columns.append(x_train @ coefficient + model.intercept_[0])
    return np.vstack(test_columns).T, float(np.vstack(train_columns).std())


def run_fold(documents, labels, train_index, test_index, spec):
    vectorizer = CountVectorizer(analyzer=identity, min_df=spec.min_df)
    counts = vectorizer.fit_transform(documents[i] for i in train_index)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
    train = tfidf.fit_transform(counts)
    test = tfidf.transform(vectorizer.transform(documents[i] for i in test_index))
    y_train = np.array([labels[i] for i in train_index])
    gold = [labels[i] for i in test_index]

    if spec.model == "lr":
        classifier = LogisticRegression(C=spec.c, max_iter=3000)
        predicted = classifier.fit(train, y_train).predict(test)
    elif spec.model == "svc":
        classifier = LinearSVC(C=spec.c, max_iter=5000)
        predicted = classifier.fit(train, y_train).predict(test)
    elif spec.model in {"nblr", "nbsvm"}:
        binary = (counts > 0).astype(float)
        classes = sorted(set(y_train))
        scores, _ = ovr_scores(train, test, binary, y_train, classes, spec, spec.model)
        predicted = np.array(classes)[np.argmax(scores, axis=1)]
    elif spec.model == "ensemble":
        binary = (counts > 0).astype(float)
        classes = sorted(set(y_train))
        parts = [
            ovr_scores(train, test, binary, y_train, classes, spec, "nblr"),
            ovr_scores(train, test, binary, y_train, classes, replace(spec, c=0.03), "nbsvm"),
            ovr_scores(train, test, binary, y_train, classes, replace(spec, c=0.5), "lr"),
            ovr_scores(train, test, binary, y_train, classes, replace(spec, c=0.1), "svc"),
        ]
        total = sum(test_scores / scale for test_scores, scale in parts)
        predicted = np.array(classes)[np.argmax(total, axis=1)]
    else:
        raise ValueError(spec.model)
    report = classification_report(gold, list(predicted))
    return report["macro_f1"], report["weighted_f1"]


def evaluate(spec, texts, labels):
    documents = [extract_features(text, spec.options) for text in texts]
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    outcomes = Parallel(n_jobs=FOLDS)(
        delayed(run_fold)(documents, labels, train, test, spec)
        for train, test in splitter.split(texts, labels)
    )
    macro = [m for m, _ in outcomes]
    return statistics.mean(macro), statistics.stdev(macro), statistics.mean(w for _, w in outcomes)


def stage_one():
    return [
        Spec("baseline"),
        Spec("+negation", FeatureOptions(negation=True)),
        Spec("+lexicon", FeatureOptions(lexicon=True)),
        Spec("+raw_words", FeatureOptions(raw_words=True)),
        Spec("+drop_noise", FeatureOptions(drop_noise=True)),
        Spec("no_normalizer", FeatureOptions(hinglish=False)),
        Spec("char_2-5", FeatureOptions(char_sizes=(2, 3, 4, 5))),
        Spec("char_3-5", FeatureOptions(char_sizes=(3, 4, 5))),
        Spec("no_char", FeatureOptions(char_sizes=())),
        Spec("word_1-3", FeatureOptions(word_ngrams=(1, 2, 3))),
        Spec("word_1", FeatureOptions(word_ngrams=(1,))),
        Spec("min_df_1", min_df=1),
        Spec("min_df_3", min_df=3),
        Spec("lr_c0.5", c=0.5),
        Spec("lr_c2", c=2.0),
        Spec("nblr_c0.3", model="nblr", c=0.3),
        Spec("nblr_c1", model="nblr", c=1.0),
        Spec("svc_c0.05", model="svc", c=0.05),
        Spec("svc_c0.1", model="svc", c=0.1),
        Spec("svc_c0.3", model="svc", c=0.3),
    ]


def stage_two():
    nblr = {"model": "nblr", "c": 1.0}
    return [
        Spec("nblr_base", **nblr),
        Spec("nblr_c2", model="nblr", c=2.0),
        Spec("nblr_c4", model="nblr", c=4.0),
        Spec("nblr_phonetic", FeatureOptions(phonetic=True), **nblr),
        Spec("nblr_skeleton", FeatureOptions(skeleton=True), **nblr),
        Spec("nblr_code_mix", FeatureOptions(code_mix=True), **nblr),
        Spec("nblr_no_normalizer", FeatureOptions(hinglish=False), **nblr),
        Spec("nblr_negation", FeatureOptions(negation=True), **nblr),
        Spec("nblr_lexicon", FeatureOptions(lexicon=True), **nblr),
        Spec("nblr_char_3-5", FeatureOptions(char_sizes=(3, 4, 5)), **nblr),
        Spec("nblr_no_char", FeatureOptions(char_sizes=()), **nblr),
        Spec("nblr_word_1", FeatureOptions(word_ngrams=(1,)), **nblr),
        Spec(
            "nblr_all_hinglish",
            FeatureOptions(
                phonetic=True, skeleton=True, code_mix=True, negation=True, lexicon=True
            ),
            **nblr,
        ),
    ]


def stage_three():
    full = FeatureOptions(
        phonetic=True, skeleton=True, code_mix=True, negation=True, lexicon=True, mask_identity=True
    )
    rich = replace(full, char_sizes=(3, 4, 5), word_ngrams=(1, 2, 3))
    return [
        Spec("shipped_nblr", full, "nblr"),
        Spec("nblr_alpha_0.1", full, "nblr", alpha=0.1),
        Spec("nblr_alpha_0.5", full, "nblr", alpha=0.5),
        Spec("nblr_balanced", full, "nblr", balanced=True),
        Spec("nblr_beta_0.75", full, "nblr", beta=0.75),
        Spec("nblr_rich", rich, "nblr"),
        Spec("nbsvm_c0.01", full, "nbsvm", c=0.01),
        Spec("nbsvm_c0.03", full, "nbsvm", c=0.03),
        Spec("nbsvm_c0.1", full, "nbsvm", c=0.1),
        Spec("nbsvm_beta_0.75", full, "nbsvm", c=0.03, beta=0.75),
        Spec("ensemble_4", full, "ensemble"),
        Spec("ensemble_4_rich", rich, "ensemble"),
    ]


STAGES = {1: stage_one, 2: stage_two, 3: stage_three}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    examples = load_sentimix(args.data_dir, "train") + load_sentimix(args.data_dir, "dev")
    texts, labels = [e.text for e in examples], [e.label for e in examples]

    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    rows = []
    print(f"{'experiment':<26} {'macro_f1':>9} {'std':>7} {'weighted':>9} {'secs':>6}", flush=True)
    for spec in STAGES[args.stage]():
        started = time.perf_counter()
        macro, deviation, weighted = evaluate(spec, texts, labels)
        print(
            f"{spec.name:<26} {macro:>9.4f} {deviation:>7.4f} {weighted:>9.4f} "
            f"{time.perf_counter() - started:>6.0f}",
            flush=True,
        )
        rows.append(
            {"macro_f1": macro, "std": deviation, "weighted_f1": weighted, "spec": asdict(spec)}
        )
    results[f"stage_{args.stage}"] = rows
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
