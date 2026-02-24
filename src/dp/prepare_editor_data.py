"""Prototype Editing (編集モデル) 用の学習データを作る。

改善策B（Prototype Editing）の Step 1。

狙い
----
TM / 近傍検索で得られる「近いが微妙に違う」翻訳（プロトタイプ）を入力に付与し、
gold 翻訳へ編集する seq2seq モデル（editor）を学習できる形に変換する。

このスクリプトは以下を行う:

1) (src, tgt) の並列データ（train）を読み込む
2) src の TF-IDF (char n-gram) 近傍検索でプロトタイプ (tm_src, tm_tgt) を取得
3) 入力テキストを
     <SRC> {src} <TM_SRC> {tm_src} <TM_TGT> {tm_tgt}
   の形式にして (editor_input, tgt) の学習例を生成

リーク注意
----------
CV で評価する場合、index（検索対象）は必ず "その fold の train" のみで構築し、
val/test を混ぜないこと。

本実装は A:TM++ の joblib index 互換を *ある程度* 目指しており、
--index に dp.tm_index の出力（vectorizer/X/meta を含む dict）を渡せる。
ただし、このリポジトリ単体でも動くよう、--index が無い場合は
--pairs から index を自動構築できる。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .align_train import normalize_transliteration
from .utils import clean_text


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def parse_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_src(text: str, norm_variant: Optional[str]) -> str:
    s = str(text) if text is not None else ""
    if norm_variant:
        return normalize_transliteration(s, norm_variant)
    return clean_text(s)


def build_editor_input(src: str, tm_src: str, tm_tgt: str) -> str:
    # 明示的にスペースで区切る（ByT5 は byte-level なので壊れにくいが、特殊トークンを見せたい）
    return f"<SRC> {src} <TM_SRC> {tm_src} <TM_TGT> {tm_tgt}".strip()


@dataclass
class TmIndex:
    vectorizer: Any
    X: Any
    meta: pd.DataFrame
    norm_variant: Optional[str] = None


def load_index(path: Path) -> TmIndex:
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise ImportError("joblib is required. Install scikit-learn.") from exc

    obj = joblib.load(path)
    if not isinstance(obj, dict):
        raise ValueError("Index must be a dict with keys like vectorizer/X/meta")

    vectorizer = obj.get("vectorizer")
    X = obj.get("X")

    # DataFrame を `or` で評価すると ValueError になるため、None 判定で順に拾う。
    meta = obj.get("meta")
    if meta is None:
        meta = obj.get("meta_df")
    if meta is None:
        meta = obj.get("df")

    norm_variant = obj.get("norm_variant")
    if norm_variant is None:
        preprocess = obj.get("preprocess")
        if isinstance(preprocess, dict):
            norm_variant = preprocess.get("norm_variant")

    if vectorizer is None or X is None or meta is None:
        raise ValueError("Index dict must include 'vectorizer', 'X', and 'meta' (or 'meta_df').")

    if not isinstance(meta, pd.DataFrame):
        # Try to coerce
        meta = pd.DataFrame(meta)

    # Normalize column names to src/tgt
    col_map: Dict[str, str] = {}
    if "src" not in meta.columns:
        # Prefer non-slotified columns if present.
        for cand in [
            "src_sent",
            "source",
            "transliteration",
            "src_text",
            "src_raw",
            "text",
            "query",
            "src_slot",  # TM++ index may store slotified text
        ]:
            if cand in meta.columns:
                col_map[cand] = "src"
                break
    if "tgt" not in meta.columns:
        for cand in ["tgt_sent", "target", "tgt_text", "translation", "ref"]:
            if cand in meta.columns:
                col_map[cand] = "tgt"
                break
    if col_map:
        meta = meta.rename(columns=col_map)

    if "src" not in meta.columns or "tgt" not in meta.columns:
        raise ValueError(
            "Index meta must include src/tgt columns (or src_sent/tgt_sent etc). "
            f"columns={list(meta.columns)}"
        )

    return TmIndex(vectorizer=vectorizer, X=X, meta=meta.reset_index(drop=True), norm_variant=norm_variant)


def build_index_from_pairs(
    pairs_df: pd.DataFrame,
    *,
    src_col: str,
    tgt_col: str,
    norm_variant: Optional[str],
    ngram_min: int,
    ngram_max: int,
) -> TmIndex:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    except ImportError as exc:
        raise ImportError("scikit-learn is required for TF-IDF index.") from exc

    src_texts = pairs_df[src_col].fillna("").astype(str).tolist()
    src_texts = [_norm_src(s, norm_variant) for s in src_texts]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(int(ngram_min), int(ngram_max)),
        lowercase=False,
    )
    X = vectorizer.fit_transform(src_texts)

    meta = pairs_df[[src_col, tgt_col]].copy()
    meta = meta.rename(columns={src_col: "src", tgt_col: "tgt"}).reset_index(drop=True)
    # 保存・デバッグ用に正規化済み src も残す
    meta["src_norm"] = src_texts
    return TmIndex(vectorizer=vectorizer, X=X, meta=meta, norm_variant=norm_variant)


def save_index(index: TmIndex, path: Path) -> None:
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise ImportError("joblib is required. Install scikit-learn.") from exc

    payload = {
        "vectorizer": index.vectorizer,
        "X": index.X,
        "meta": index.meta,
        "norm_variant": index.norm_variant,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def _len_ratio(a: str, b: str) -> float:
    la = max(1, len(a.strip()))
    lb = max(1, len(b.strip()))
    return la / lb


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training data for the Prototype Editing model.")
    parser.add_argument("--pairs", required=True, help="Train parallel pairs (csv/parquet).")
    parser.add_argument("--src-col", default="src_sent", help="Source column name in pairs.")
    parser.add_argument("--tgt-col", default="tgt_sent", help="Target column name in pairs.")
    parser.add_argument("--id-col", default=None, help="Optional ID column name (for logging only).")
    parser.add_argument("--out", required=True, help="Output editor training data (csv/parquet).")

    parser.add_argument(
        "--index",
        default=None,
        help="Optional: prebuilt TM/TF-IDF index joblib (dict with vectorizer/X/meta).",
    )
    parser.add_argument(
        "--index-out",
        default=None,
        help="Optional: save built index to this path (joblib).",
    )

    parser.add_argument("--norm-variant", default=None, help="Normalize source with A/B/C (same as NMT).")
    parser.add_argument("--ngram-min", type=int, default=3, help="TF-IDF char n-gram min.")
    parser.add_argument("--ngram-max", type=int, default=5, help="TF-IDF char n-gram max.")

    parser.add_argument("--topk", type=int, default=10, help="Retrieve topK prototypes per src.")
    parser.add_argument("--prototypes-per-src", type=int, default=2, help="How many prototypes to keep.")
    parser.add_argument("--min-sim", type=float, default=0.25, help="Min cosine sim to accept prototype.")
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Exclude exact self match when pairs are also the index source (recommended for training).",
    )
    parser.add_argument(
        "--min-len-ratio",
        type=float,
        default=0.5,
        help="Min len(tm_tgt)/len(gold_tgt) ratio (character length).",
    )
    parser.add_argument(
        "--max-len-ratio",
        type=float,
        default=2.0,
        help="Max len(tm_tgt)/len(gold_tgt) ratio (character length).",
    )

    parser.add_argument("--max-rows", type=int, default=None, help="Use first N rows from pairs.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Randomly sample N rows from pairs before retrieval (to limit runtime).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    norm_variant = args.norm_variant
    if norm_variant:
        norm_variant = str(norm_variant).upper()

    pairs_path = Path(args.pairs)
    df_pairs = read_table(pairs_path)

    src_col = str(args.src_col)
    tgt_col = str(args.tgt_col)
    if src_col not in df_pairs.columns or tgt_col not in df_pairs.columns:
        raise ValueError(
            f"pairs must include src/tgt columns: src={src_col} tgt={tgt_col} columns={list(df_pairs.columns)}"
        )

    if args.max_rows is not None and int(args.max_rows) > 0:
        df_pairs = df_pairs.head(int(args.max_rows)).reset_index(drop=True)

    if args.sample_rows is not None and int(args.sample_rows) > 0 and len(df_pairs) > int(args.sample_rows):
        rnd = random.Random(int(args.seed))
        idx = list(range(len(df_pairs)))
        rnd.shuffle(idx)
        take = idx[: int(args.sample_rows)]
        df_pairs = df_pairs.iloc[take].reset_index(drop=True)

    # Load or build index
    index: TmIndex
    if args.index:
        index = load_index(Path(args.index))
        if index.norm_variant is None:
            index.norm_variant = norm_variant
    else:
        index = build_index_from_pairs(
            df_pairs,
            src_col=src_col,
            tgt_col=tgt_col,
            norm_variant=norm_variant,
            ngram_min=int(args.ngram_min),
            ngram_max=int(args.ngram_max),
        )
        if args.index_out:
            save_index(index, Path(args.index_out))
            print(f"Saved index: {args.index_out}")

    # Retrieval
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
    except ImportError as exc:
        raise ImportError("scikit-learn is required for nearest neighbor retrieval") from exc

    X = index.X
    topk = max(1, int(args.topk))
    # +1 to allow removing self
    n_neighbors = topk + 1 if args.exclude_self else topk
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)
    # Query with the same set (training use case)
    dists, nbrs = nn.kneighbors(X, return_distance=True)
    sims = 1.0 - dists

    min_sim = float(args.min_sim)
    max_keep = max(1, int(args.prototypes_per_src))
    min_lr = float(args.min_len_ratio) if args.min_len_ratio is not None else 0.0
    max_lr = float(args.max_len_ratio) if args.max_len_ratio is not None else 10.0

    rows_out: List[Dict[str, Any]] = []

    # Precompute normalized source for queries
    src_raw = df_pairs[src_col].fillna("").astype(str).tolist()
    tgt_raw = df_pairs[tgt_col].fillna("").astype(str).tolist()
    src_norm = [_norm_src(s, index.norm_variant) for s in src_raw]
    df_pairs = df_pairs.reset_index(drop=True)

    id_vals: Optional[List[str]] = None
    if args.id_col and str(args.id_col) in df_pairs.columns:
        id_vals = df_pairs[str(args.id_col)].astype(str).tolist()
    else:
        id_vals = [str(i) for i in range(len(df_pairs))]

    for i in range(len(df_pairs)):
        gold_src = src_raw[i]
        gold_tgt = tgt_raw[i]
        if not str(gold_src).strip() or not str(gold_tgt).strip():
            continue

        kept = 0
        for j, nb_idx in enumerate(nbrs[i].tolist()):
            if args.exclude_self and int(nb_idx) == int(i):
                continue
            sim = float(sims[i, j])
            if sim < min_sim:
                continue

            tm_src = str(index.meta.loc[int(nb_idx), "src"]) if int(nb_idx) < len(index.meta) else ""
            tm_tgt = str(index.meta.loc[int(nb_idx), "tgt"]) if int(nb_idx) < len(index.meta) else ""
            tm_src = clean_text(tm_src)
            tm_tgt = clean_text(tm_tgt)
            if not tm_src or not tm_tgt:
                continue

            lr = _len_ratio(tm_tgt, gold_tgt)
            if lr < min_lr or lr > max_lr:
                continue

            editor_in = build_editor_input(clean_text(gold_src), tm_src, tm_tgt)

            rows_out.append(
                {
                    "id": id_vals[i],
                    "src": clean_text(gold_src),
                    "tgt": clean_text(gold_tgt),
                    "tm_src": tm_src,
                    "tm_tgt": tm_tgt,
                    "tm_sim": sim,
                    "editor_input": editor_in,
                    "src_norm": src_norm[i],
                }
            )
            kept += 1
            if kept >= max_keep:
                break

    out_df = pd.DataFrame(rows_out)
    out_path = Path(args.out)
    write_table(out_df, out_path)

    uniq = out_df["id"].nunique() if not out_df.empty else 0
    avg = len(out_df) / max(1, uniq)
    print(
        "=== editor training data ===\n"
        f"pairs_rows={len(df_pairs)} examples={len(out_df)} uniq_ids={uniq} avg_protos_per_id={avg:.2f} "
        f"topk={topk} keep={max_keep} min_sim={min_sim} len_ratio=[{min_lr},{max_lr}]"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
