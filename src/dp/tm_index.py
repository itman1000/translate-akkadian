"""Build a TM++ TF-IDF char n-gram index over slotified sources.

Usage (example)
---------------
python -m dp.tm_index \
  --pairs artifacts/ablation/variant=C/fold=0/train.parquet \
  --src-col src_sent --tgt-col tgt_sent \
  --lexicon OA_Lexicon_eBL_PN_GN_no_word_overlap.csv \
  --strip-lex-from-src \
  --out artifacts/tm/variant=C/fold=0/index.joblib

Notes
-----
- For CV leak safety, build the index from *train split only* (per-fold).
- The saved index contains the vectorizer, sparse TF-IDF matrix, and a small meta table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
    p = argparse.ArgumentParser(description="Build TM++ (slotified) TF-IDF index.")
    p.add_argument("--pairs", nargs="+", required=True, help="Pairs file(s): parquet/csv with src/tgt columns.")
    p.add_argument("--src-col", required=True, help="Source column name.")
    p.add_argument("--tgt-col", required=True, help="Target column name.")
    p.add_argument("--id-col", default=None, help="Optional id column to keep in metadata.")
    p.add_argument("--out", required=True, help="Output joblib path (e.g., artifacts/tm/index.joblib).")

    # Slotify options
    p.add_argument("--lexicon", default=None, help="Optional lexicon CSV path for PN slotification.")
    p.add_argument("--lexicon-types", default="PN,GN", help="Comma-separated lexicon types (default: PN,GN).")
    p.add_argument("--strip-lex-from-src", action="store_true", help="Strip <LEX>...</LEX> blocks from src.")
    p.add_argument("--no-slot-numbers", action="store_true", help="Disable numeric slotification.")
    p.add_argument("--no-slot-units", action="store_true", help="Disable unit slotification.")

    # TF-IDF options
    p.add_argument("--ngram-range", default="3,5", help="Char n-gram range like '3,5' (default: 3,5).")
    p.add_argument("--max-features", type=int, default=None, help="Optional max_features for vectorizer.")
    p.add_argument("--min-df", type=int, default=1, help="min_df for vectorizer (default: 1).")
    p.add_argument("--max-df", type=float, default=1.0, help="max_df for vectorizer (default: 1.0).")

    # Data gates
    p.add_argument("--min-src-chars", type=int, default=1, help="Drop pairs with src shorter than this.")
    p.add_argument("--min-tgt-chars", type=int, default=1, help="Drop pairs with tgt shorter than this.")
    p.add_argument("--limit", type=int, default=None, help="Optional row limit (debug).")
    return p


def _parse_ngram(spec: str) -> Tuple[int, int]:
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--ngram-range must be like '3,5' but got: {spec}")
    return int(parts[0]), int(parts[1])


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = find_repo_root() / out_path
    ensure_dir(out_path.parent)

    # Slotify config (stored inside the index for consistent retrieval).
    lexicon_path = Path(args.lexicon) if args.lexicon else None
    lexicon_types = tuple(t.strip().upper() for t in str(args.lexicon_types).split(",") if t.strip()) or ("PN", "GN")
    slot_cfg = SlotifyConfig(
        lexicon_path=lexicon_path,
        lexicon_types=lexicon_types,
        strip_lex_from_src=bool(args.strip_lex_from_src),
        slot_numbers=not bool(args.no_slot_numbers),
        slot_units=not bool(args.no_slot_units),
    )

    # Load pairs
    dfs: List[pd.DataFrame] = []
    for pth in args.pairs:
        df = _read_table(Path(pth))
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    # Keep required columns
    need = [args.src_col, args.tgt_col]
    if args.id_col:
        need.append(args.id_col)
    missing = [c for c in need if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns in pairs: {missing} (columns={list(df_all.columns)[:50]}...)")

    df = df_all[need].copy()
    df[args.src_col] = df[args.src_col].fillna("").astype(str)
    df[args.tgt_col] = df[args.tgt_col].fillna("").astype(str)

    # gates
    df = df[df[args.src_col].str.len() >= int(args.min_src_chars)]
    df = df[df[args.tgt_col].str.len() >= int(args.min_tgt_chars)]
    df = df.reset_index(drop=True)

    if args.limit is not None and int(args.limit) > 0:
        df = df.head(int(args.limit)).copy()

    if len(df) == 0:
        raise ValueError("No valid pairs after filtering (check --src-col/--tgt-col and gates).")

    # Slotify sources
    slot_texts: List[str] = []
    # Keep slot maps out of the index to reduce size (they can be recomputed on-demand for topK hits).
    for i, s in enumerate(df[args.src_col].astype(str).tolist()):
        slot_s, _map = slotify_src(s, slot_cfg)
        slot_texts.append(slot_s)
        if (i + 1) % 20000 == 0:
            print(f"[tm_index] slotified {i+1}/{len(df)}")

    df_meta = pd.DataFrame(
        {
            "src": df[args.src_col].astype(str).tolist(),
            "tgt": df[args.tgt_col].astype(str).tolist(),
            "src_slot": slot_texts,
        }
    )
    if args.id_col:
        df_meta["pair_id"] = df[args.id_col].astype(str).tolist()

    # Vectorize
    ngram_range = _parse_ngram(args.ngram_range)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        lowercase=False,
        max_features=args.max_features,
        min_df=int(args.min_df),
        max_df=float(args.max_df),
    )
    X = vectorizer.fit_transform(df_meta["src_slot"].astype(str).tolist())

    payload: Dict[str, Any] = {
        "version": 1,
        "vectorizer": vectorizer,
        "X": X,
        "meta": df_meta,
        "slotify": slot_cfg.to_dict(),
        "stats": {
            "n_pairs": int(len(df_meta)),
            "ngram_range": list(ngram_range),
            "max_features": int(args.max_features) if args.max_features else None,
            "min_df": int(args.min_df),
            "max_df": float(args.max_df),
        },
    }

    joblib.dump(payload, out_path)
    print(f"[tm_index] saved: {out_path}")
    print(f"[tm_index] n_pairs={len(df_meta)} n_features={len(vectorizer.get_feature_names_out())}")

    # Small sidecar JSON for quick inspection (optional convenience).
    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    try:
        sidecar.write_text(json.dumps(payload["stats"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[tm_index] wrote: {sidecar}")
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
