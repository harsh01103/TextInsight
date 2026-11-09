"""Fine-tune HingBERT on the SentiMix train split, checking results on dev.

Usage:
    python training/finetune_transformer.py --data-dir <Hinglish folder>

The lower layers stay frozen to keep CPU training practical. Saves the best
checkpoint to data/finetuned/ and its dev probabilities to data/embeddings/.
The test split is never read here.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from textinsight.eval import classification_report, load_sentimix

MODEL = "l3cube-pune/hing-bert"
LABELS = ["negative", "neutral", "positive"]
MAX_LENGTH = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
FROZEN_LAYERS = 8
ROOT = Path(__file__).parent.parent / "data"
SEED = 13


def batches(indices, size):
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def probabilities(model, tokenizer, texts):
    model.eval()
    order = np.argsort([len(text) for text in texts])
    columns = []
    with torch.inference_mode():
        for group in batches(order, EVAL_BATCH_SIZE):
            encoded = tokenizer(
                [texts[i] for i in group],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            columns.append(torch.softmax(model(**encoded).logits, dim=-1).numpy())
    return np.vstack(columns)[np.argsort(order)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    train = load_sentimix(args.data_dir, "train")
    dev = load_sentimix(args.data_dir, "dev")
    train_texts = [e.text for e in train]
    train_labels = torch.tensor([LABELS.index(e.label) for e in train])
    dev_texts, dev_labels = [e.text for e in dev], [e.label for e in dev]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))
    for parameter in model.bert.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.bert.encoder.layer[:FROZEN_LAYERS]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE, weight_decay=0.01
    )

    best = 0.0
    (ROOT / "embeddings").mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = list(range(len(train_texts)))
        random.shuffle(indices)
        started = time.perf_counter()
        for step, group in enumerate(batches(indices, BATCH_SIZE)):
            encoded = tokenizer(
                [train_texts[i] for i in group],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            loss = model(**encoded, labels=train_labels[group]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.3f} {elapsed:.0f}s", flush=True
                )

        dev_probabilities = probabilities(model, tokenizer, dev_texts)
        predicted = [LABELS[i] for i in dev_probabilities.argmax(1)]
        report = classification_report(dev_labels, predicted, LABELS)
        print(
            f"epoch {epoch}: dev macro-F1 {report['macro_f1']:.4f} "
            f"weighted {report['weighted_f1']:.4f}",
            flush=True,
        )
        if report["macro_f1"] > best:
            best = report["macro_f1"]
            np.save(ROOT / "embeddings" / "finetuned_dev_probabilities.npy", dev_probabilities)
            model.save_pretrained(ROOT / "finetuned")
            tokenizer.save_pretrained(ROOT / "finetuned")
    print(f"best dev macro-F1 {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
