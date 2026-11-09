"""Compute transformer embeddings for SentiMix tweets.

Usage:
    pip install -e ".[transformers,training]"
    python training/embed_transformer.py --data-dir <Hinglish folder> --splits train dev

Saves CLS and mean-pooled vectors to data/embeddings/<model>_<split>.npz.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from textinsight.eval import load_sentimix
from textinsight.preprocess import normalize_hinglish

MODEL = "l3cube-pune/hing-bert"
MAX_LENGTH = 64
BATCH_SIZE = 32
OUTPUT = Path(__file__).parent.parent / "data" / "embeddings"


def embed(texts, tokenizer, model):
    # Similar lengths in a batch mean less padding, so sort first and undo it afterwards.
    order = np.argsort([len(text) for text in texts])
    cls_vectors, mean_vectors, last4_vectors = [], [], []
    started = time.perf_counter()
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [texts[i] for i in order[start : start + BATCH_SIZE]]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        with torch.inference_mode():
            layers = model(**encoded, output_hidden_states=True).hidden_states
        hidden = layers[-1]
        last4 = torch.stack(layers[-4:]).mean(0)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        cls_vectors.append(hidden[:, 0].numpy())
        mean_vectors.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
        last4_vectors.append(((last4 * mask).sum(1) / mask.sum(1)).numpy())
        if start % (BATCH_SIZE * 40) == 0:
            print(f"  {start}/{len(texts)} in {time.perf_counter() - started:.0f}s", flush=True)
    inverse = np.argsort(order)
    return (
        np.vstack(cls_vectors)[inverse],
        np.vstack(mean_vectors)[inverse],
        np.vstack(last4_vectors)[inverse],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--view", choices=["raw", "normalized"], default="raw")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = args.model.split("/")[-1]
    for split in args.splits:
        examples = load_sentimix(args.data_dir, split)
        print(f"{split}: {len(examples)} tweets", flush=True)
        texts = [e.text for e in examples]
        if args.view == "normalized":
            texts = normalize_hinglish(texts)
        cls, mean, last4 = embed(texts, tokenizer, model)
        np.savez(
            OUTPUT / f"{name}_{args.view}_{split}.npz",
            cls=cls,
            mean=mean,
            last4=last4,
            labels=np.array([e.label for e in examples]),
        )
        print(f"saved {split}", flush=True)


if __name__ == "__main__":
    main()
