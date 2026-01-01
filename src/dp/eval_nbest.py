"""Evaluate n-best reranking on validation data."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .rerank_utils import get_length_stats, rerank_score
from .train_real import compute_metrics
from .utils import clean_text, load_config


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate n-best reranking on validation set.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--ckpt", required=True, help="Path to model.pkl.")
    parser.add_argument("--val", required=True, help="Validation data path.")
    parser.add_argument("--k", type=int, default=5, help="Number of candidates per row.")
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--tgt-col", default=None, help="Target column name.")
    parser.add_argument("--out", default=None, help="Output directory.")
    parser.add_argument("--variant", default=None, help="Variant label.")
    parser.add_argument("--fold", default=None, help="Fold id.")
    parser.add_argument("--length-weight", type=float, default=None, help="Length penalty weight.")
    parser.add_argument("--sentence-penalty", type=float, default=None, help="Multi-sentence penalty.")
    parser.add_argument("--digit-weight", type=float, default=None, help="Digit ratio penalty weight.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    src_col = args.src_col or cfg.get("src_col", "src_sent")
    tgt_col = args.tgt_col or cfg.get("tgt_col", "tgt_sent")

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

    val_path = Path(args.val)
    val_df = read_table(val_path)
    if src_col not in val_df.columns or tgt_col not in val_df.columns:
        raise ValueError(f"val data must include {src_col} and {tgt_col}")

    model_path = Path(args.ckpt)
    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found: {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)

    vectorizer = model["vectorizer"]
    x_train = model["x_train"]
    train_tgt = model["train_tgt"]

    from sklearn.neighbors import NearestNeighbors  # type: ignore

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(x_train)

    val_src = val_df[src_col].astype(str).map(clean_text).tolist()
    val_tgt = val_df[tgt_col].astype(str).map(clean_text).tolist()

    x_val = vectorizer.transform(val_src)
    distances, indices = nn.kneighbors(x_val, n_neighbors=args.k)

    base_preds: List[str] = []
    rerank_preds: List[str] = []
    stats = get_length_stats(model)

    for cand_idxs, cand_dists in zip(indices, distances):
        cand_texts = [train_tgt[int(idx)] for idx in cand_idxs]
        cand_scores = [float(1.0 - dist) for dist in cand_dists]
        base_preds.append(cand_texts[0])

        best_idx = 0
        best_score = None
        for i, (text, sim) in enumerate(zip(cand_texts, cand_scores)):
            final = rerank_score(
                text,
                sim,
                stats=stats,
                length_weight=length_weight,
                sentence_penalty=sentence_penalty,
                digit_weight=digit_weight,
            )
            if best_score is None or final > best_score:
                best_idx = i
                best_score = final
        rerank_preds.append(cand_texts[best_idx])

    base_metrics = compute_metrics(base_preds, val_tgt)
    rerank_metrics = compute_metrics(rerank_preds, val_tgt)

    metrics: Dict[str, Any] = {}
    metrics.update({f"{key}_base": value for key, value in base_metrics.items()})
    metrics.update(rerank_metrics)
    metrics["variant"] = args.variant or ""
    metrics["fold"] = args.fold or ""

    out_dir = Path(args.out) if args.out else Path("artifacts") / "rerank_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    preds_path = out_dir / "val_predictions.csv"
    pd.DataFrame(
        {
            "pred_base": base_preds,
            "pred_rerank": rerank_preds,
            "gold": val_tgt,
        }
    ).to_csv(preds_path, index=False)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {preds_path}")


if __name__ == "__main__":
    main()
