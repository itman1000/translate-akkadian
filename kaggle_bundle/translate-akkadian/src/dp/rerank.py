"""Rerank n-best predictions with simple heuristics."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from .rerank_utils import get_length_stats, rerank_score
from .utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank n-best predictions.")
    parser.add_argument("--nbest", required=True, help="Input nbest CSV path.")
    parser.add_argument("--out", required=True, help="Output predictions CSV path.")
    parser.add_argument("--ckpt", default=None, help="Optional model.pkl path for stats.")
    parser.add_argument("--config", default=None, help="Optional config path for weights.")
    parser.add_argument("--length-weight", type=float, default=None, help="Length penalty weight.")
    parser.add_argument("--sentence-penalty", type=float, default=None, help="Multi-sentence penalty.")
    parser.add_argument("--digit-weight", type=float, default=None, help="Digit ratio penalty weight.")
    parser.add_argument("--out-nbest", default=None, help="Optional output CSV with final scores.")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

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

    nbest_path = Path(args.nbest)
    if not nbest_path.exists():
        raise FileNotFoundError(f"nbest not found: {nbest_path}")
    nbest_df = pd.read_csv(nbest_path)
    for col in ["id", "score", "translation"]:
        if col not in nbest_df.columns:
            raise ValueError(f"nbest must include '{col}' column")

    stats = {"tgt_len_mean": 0.0, "tgt_len_std": 1.0}
    if args.ckpt:
        model_path = Path(args.ckpt)
        if not model_path.exists():
            raise FileNotFoundError(f"model.pkl not found: {model_path}")
        with model_path.open("rb") as f:
            model = pickle.load(f)
        stats = get_length_stats(model)

    predictions = []
    scored_rows = []
    for sample_id, group in nbest_df.groupby("id", sort=False):
        candidates = group.to_dict(orient="records")
        best = None
        best_score = None
        for cand in candidates:
            final = rerank_score(
                str(cand.get("translation", "")),
                float(cand.get("score", 0.0)),
                stats=stats,
                length_weight=length_weight,
                sentence_penalty=sentence_penalty,
                digit_weight=digit_weight,
            )
            cand["final_score"] = final
            if best_score is None or final > best_score:
                best = cand
                best_score = final
        if best:
            predictions.append({"id": sample_id, "translation": best["translation"]})
        scored_rows.extend(candidates)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")

    if args.out_nbest:
        out_nbest_path = Path(args.out_nbest)
        out_nbest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scored_rows).to_csv(out_nbest_path, index=False)
        print(f"Saved reranked nbest: {out_nbest_path}")


if __name__ == "__main__":
    main()
