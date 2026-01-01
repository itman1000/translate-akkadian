"""Inference using TF-IDF retrieval baseline."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from .utils import clean_text, get_artifacts_dir, get_data_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer translations with TF-IDF retrieval baseline.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--ckpt", required=True, help="Path to model.pkl.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    parser.add_argument("--test", default=None, help="Optional test.csv path.")
    parser.add_argument("--out", default=None, help="Output predictions CSV.")
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--id-col", default=None, help="ID column name.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, args.data_dir)
    src_col = args.src_col or cfg.get("src_col", "transliteration")
    id_col = args.id_col or cfg.get("id_col", "id")

    test_path = Path(args.test) if args.test else data_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found at: {test_path}")

    model_path = Path(args.ckpt)
    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found at: {model_path}")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    vectorizer = model["vectorizer"]
    x_train = model["x_train"]
    train_tgt = model["train_tgt"]

    from sklearn.neighbors import NearestNeighbors  # type: ignore

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(x_train)

    test_df = pd.read_csv(test_path)
    if id_col not in test_df.columns or src_col not in test_df.columns:
        raise ValueError(f"test.csv must contain {id_col} and {src_col}")

    src = test_df[src_col].astype(str).map(clean_text).tolist()
    x_test = vectorizer.transform(src)
    _, indices = nn.kneighbors(x_test, n_neighbors=1)
    preds = [train_tgt[i[0]] for i in indices]

    out_df = pd.DataFrame({"id": test_df[id_col], "translation": preds})

    out_path = Path(args.out) if args.out else get_artifacts_dir(cfg) / "predictions_real.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
