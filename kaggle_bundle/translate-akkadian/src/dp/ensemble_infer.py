"""Ensemble inference by averaging retrieval scores across models."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .rerank_utils import get_length_stats, rerank_score
from .utils import clean_text, get_data_dir, load_config


def load_model(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"model.pkl not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble inference with TF-IDF retrieval baselines.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--ckpts", required=True, help="Comma-separated model.pkl paths.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    parser.add_argument("--test", default=None, help="Optional test.csv path.")
    parser.add_argument("--out", required=True, help="Output predictions CSV path.")
    parser.add_argument("--k", type=int, default=3, help="Top-k per model to pool.")
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--id-col", default=None, help="ID column name.")
    parser.add_argument("--rerank", action="store_true", help="Apply heuristic reranking.")
    parser.add_argument("--length-weight", type=float, default=None, help="Length penalty weight.")
    parser.add_argument("--sentence-penalty", type=float, default=None, help="Multi-sentence penalty.")
    parser.add_argument("--digit-weight", type=float, default=None, help="Digit ratio penalty weight.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, args.data_dir)
    src_col = args.src_col or cfg.get("src_col", "transliteration")
    id_col = args.id_col or cfg.get("id_col", "id")

    length_weight = (
        args.length_weight
        if args.length_weight is not None
        else float(cfg.get("rerank_length_weight", 0.05))
    )
    sentence_penalty = (
        args.sentence_penalty
        if args.sentence_penalty is not None
        else float(cfg.get("rerank_sentence_penalty", 0.1))
    )
    digit_weight = (
        args.digit_weight
        if args.digit_weight is not None
        else float(cfg.get("rerank_digit_weight", 0.2))
    )

    test_path = Path(args.test) if args.test else data_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found at: {test_path}")

    test_df = pd.read_csv(test_path)
    if id_col not in test_df.columns or src_col not in test_df.columns:
        raise ValueError(f"test.csv must contain {id_col} and {src_col}")

    ckpt_paths = [Path(p.strip()) for p in args.ckpts.split(",") if p.strip()]
    if not ckpt_paths:
        raise ValueError("No ckpt paths provided")

    models = [load_model(path) for path in ckpt_paths]

    from sklearn.neighbors import NearestNeighbors  # type: ignore

    src = test_df[src_col].astype(str).map(clean_text).tolist()

    model_knn = []
    for model in models:
        vectorizer = model["vectorizer"]
        x_train = model["x_train"]
        train_tgt = model["train_tgt"]
        x_test = vectorizer.transform(src)
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(x_train)
        distances, indices = nn.kneighbors(x_test, n_neighbors=args.k)
        model_knn.append((distances, indices, train_tgt, model))

    predictions = []
    for row_idx, sample_id in enumerate(test_df[id_col].tolist()):
        pooled: Dict[str, List[float]] = {}
        stats = None
        for distances, indices, train_tgt, model in model_knn:
            if stats is None:
                stats = get_length_stats(model)
            cand_idxs = indices[row_idx]
            cand_dists = distances[row_idx]
            for idx, dist in zip(cand_idxs, cand_dists):
                text = train_tgt[int(idx)]
                sim = float(1.0 - dist)
                pooled.setdefault(text, []).append(sim)

        best_text = ""
        best_score = None
        stats = stats or {"tgt_len_mean": 0.0, "tgt_len_std": 1.0}
        for text, scores in pooled.items():
            avg_sim = sum(scores) / len(scores)
            final = avg_sim
            if args.rerank:
                final = rerank_score(
                    text,
                    avg_sim,
                    stats=stats,
                    length_weight=length_weight,
                    sentence_penalty=sentence_penalty,
                    digit_weight=digit_weight,
                )
            if best_score is None or final > best_score:
                best_score = final
                best_text = text

        predictions.append({"id": sample_id, "translation": best_text})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
