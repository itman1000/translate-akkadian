"""Retrieve top-K translation-memory (TM++) pairs for each query.

This module is mostly for debugging / inspection. For candidate generation,
use `dp.tm_generate_candidates`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("joblib is required (it should come with scikit-learn)") from exc

from .tm_slotify import SlotifyConfig, slotify_src
from .utils import ensure_dir, find_repo_root


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retrieve top-K TM++ pairs for each query.")
    p.add_argument("--index", required=True, help="TM++ index joblib path.")
    p.add_argument("--queries", required=True, help="Queries file (csv/parquet).")
    p.add_argument("--id-col", required=True, help="Query id column.")
    p.add_argument("--src-col", required=True, help="Query source column.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--topk", type=int, default=50, help="Top-K per query (default: 50).")
    p.add_argument("--min-sim", type=float, default=0.0, help="Drop retrieved items with sim < min_sim.")
    p.add_argument("--exclude-exact-src", action="store_true", help="Exclude exact slotified self-matches.")

    # Optional overrides (useful when moving the index between environments)
    p.add_argument("--lexicon", default=None, help="Override lexicon path for PN slotification.")
    p.add_argument("--strip-lex-from-src", action="store_true", help="Strip <LEX>...</LEX> from query src.")
    return p


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= len(scores):
        idx = np.argsort(-scores)
        return idx
    part = np.argpartition(scores, -k)[-k:]
    part = part[np.argsort(-scores[part])]
    return part


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = find_repo_root() / index_path
    payload: Dict[str, Any] = joblib.load(index_path)

    vectorizer = payload["vectorizer"]
    X = payload["X"]
    meta: pd.DataFrame = payload["meta"]
    slot_cfg = SlotifyConfig.from_dict(payload.get("slotify", {}))

    # Overrides
    if args.lexicon:
        slot_cfg.lexicon_path = Path(args.lexicon)
    if args.strip_lex_from_src:
        slot_cfg.strip_lex_from_src = True

    # Load queries
    q_path = Path(args.queries)
    if not q_path.is_absolute():
        q_path = find_repo_root() / q_path
    q_df = _read_table(q_path)

    missing = [c for c in (args.id_col, args.src_col) if c not in q_df.columns]
    if missing:
        raise ValueError(f"Missing query columns: {missing} (columns={list(q_df.columns)[:50]}...)")

    ids = q_df[args.id_col].astype(str).tolist()
    srcs = q_df[args.src_col].fillna("").astype(str).tolist()

    topk = int(args.topk)
    min_sim = float(args.min_sim)

    rows: List[Dict[str, Any]] = []
    for qi, (qid, src) in enumerate(zip(ids, srcs)):
        slot_q, _map = slotify_src(src, slot_cfg)
        q_vec = vectorizer.transform([slot_q])  # (1, n_features)

        # Since TF-IDF vectors are L2-normalized by default, dot product = cosine similarity.
        sims = (X @ q_vec.T).toarray().ravel()

        idxs = _topk_indices(sims, topk)
        rank = 0
        for idx in idxs:
            sim = float(sims[idx])
            if sim < min_sim:
                break
            if args.exclude_exact_src:
                try:
                    if str(meta["src_slot"].iat[int(idx)]) == slot_q:
                        continue
                except Exception:
                    pass
            rank += 1
            rows.append(
                {
                    "id": qid,
                    "rank": rank,
                    "tm_sim": sim,
                    "tm_row": int(idx),
                    "tm_src": str(meta["src"].iat[int(idx)]),
                    "tm_tgt": str(meta["tgt"].iat[int(idx)]),
                }
            )
            if rank >= topk:
                break

        if (qi + 1) % 200 == 0:
            print(f"[tm_retrieve] processed {qi+1}/{len(ids)}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = find_repo_root() / out_path
    ensure_dir(out_path.parent)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[tm_retrieve] saved: {out_path} rows={len(rows)}")


if __name__ == "__main__":  # pragma: no cover
    main()
