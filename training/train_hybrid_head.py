"""Train the small classifier that sits on top of the transformer embeddings.

Usage: first run embed_transformer.py for train and dev, then
    python training/train_hybrid_head.py

Only this classifier ships in the package. The encoder itself is downloaded
from Hugging Face the first time the hybrid backend is used.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
EMBEDDINGS = ROOT / "data" / "embeddings"
OUTPUT = ROOT / "src" / "textinsight" / "_data" / "sentiment_hybrid_head.json.gz"
C = 0.001  # chosen by 5-fold cross-validation on train + dev


def main() -> None:
    parts = [np.load(EMBEDDINGS / f"hing-bert_{split}.npz") for split in ("train", "dev")]
    vectors = np.vstack([part["mean"] for part in parts])
    labels = np.concatenate([part["labels"] for part in parts])

    scaler = StandardScaler().fit(vectors)
    classifier = LogisticRegression(C=C, max_iter=3000).fit(scaler.transform(vectors), labels)
    payload = {
        "classes": classifier.classes_.tolist(),
        "mean": [round(float(v), 6) for v in scaler.mean_],
        "scale": [round(float(v), 6) for v in scaler.scale_],
        "coefficients": [[round(float(v), 6) for v in row] for row in classifier.coef_],
        "intercepts": [round(float(v), 6) for v in classifier.intercept_],
    }
    with gzip.GzipFile(OUTPUT, "wb", mtime=0) as raw:
        raw.write(json.dumps(payload, separators=(",", ":")).encode())
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
