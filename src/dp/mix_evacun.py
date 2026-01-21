"""aligned_train と EvaCun データを混合する。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from .utils import count_sentence_endings, get_artifacts_dir, load_config


_GAP_RE = re.compile(r"<\s*(gap|big_gap)\s*>", re.IGNORECASE)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_gap_rate(texts: Iterable[str]) -> float:
    total = 0
    hits = 0
    for text in texts:
        total += 1
        if _GAP_RE.search(text or ""):
            hits += 1
    return hits / max(total, 1)


def compute_caps_token_rate(texts: Iterable[str]) -> float:
    total = 0
    caps = 0
    for text in texts:
        for token in (text or "").split():
            total += 1
            if len(token) >= 2 and token.isupper():
                caps += 1
    return caps / max(total, 1)


def compute_avg_len(texts: Iterable[str]) -> float:
    total = 0
    count = 0
    for text in texts:
        count += 1
        total += len(text or "")
    return total / max(count, 1)


def _get_float(cfg: Dict[str, object], key: str, default: float) -> float:
    value = cfg.get(key, default)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _get_int(cfg: Dict[str, object], key: str, default: int) -> int:
    value = cfg.get(key, default)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def filter_sentence_endings(
    df: pd.DataFrame,
    tgt_col: str,
    max_sentence_endings: Optional[int],
) -> Tuple[pd.DataFrame, int]:
    if max_sentence_endings is None:
        return df, 0
    if tgt_col not in df.columns:
        raise ValueError(f"tgt_col not found in evacun data: {tgt_col}")
    counts = df[tgt_col].fillna("").astype(str).map(count_sentence_endings)
    mask = counts <= max_sentence_endings
    dropped = int((~mask).sum())
    return df.loc[mask].reset_index(drop=True), dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix aligned_train with EvaCun data.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--aligned", default=None, help="Aligned train path.")
    parser.add_argument("--evacun", default=None, help="EvaCun train path.")
    parser.add_argument("--out", default=None, help="Output path.")
    parser.add_argument("--ratio", type=float, default=None, help="EvaCun ratio vs aligned.")
    parser.add_argument("--variants", default=None, help="Variants to keep (A,B,C).")
    parser.add_argument(
        "--max-sentence-endings",
        type=int,
        default=None,
        help="Drop EvaCun rows with more than N sentence endings in target.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--src-col", default="src_sent", help="Source column for stats.")
    parser.add_argument("--tgt-col", default="tgt_sent", help="Target column for filtering/stats.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = get_artifacts_dir(cfg)

    aligned_path = (
        Path(args.aligned)
        if args.aligned
        else Path(cfg.get("evacun_mix_aligned_path", artifacts_dir / "aligned" / "aligned_train.parquet"))
    )
    evacun_path = (
        Path(args.evacun)
        if args.evacun
        else Path(cfg.get("evacun_mix_evacun_path", artifacts_dir / "evacun" / "evacun_transcription_train.parquet"))
    )
    out_path = (
        Path(args.out)
        if args.out
        else Path(cfg.get("evacun_mix_out", artifacts_dir / "evacun" / "mixed_train.parquet"))
    )

    ratio = args.ratio if args.ratio is not None else _get_float(cfg, "evacun_mix_ratio", 0.3)
    seed = args.seed if args.seed is not None else _get_int(cfg, "seed", 42)
    max_sentence_endings = args.max_sentence_endings
    if max_sentence_endings is None:
        max_sentence_endings = cfg.get("evacun_mix_max_sentence_endings")
        if max_sentence_endings is not None:
            try:
                max_sentence_endings = int(max_sentence_endings)
            except (TypeError, ValueError):
                max_sentence_endings = None

    variant_set = set()
    if args.variants:
        variant_set = {v.strip().upper() for v in args.variants.split(",") if v.strip()}

    aligned_df = read_table(aligned_path)
    if variant_set and "src_norm_variant" in aligned_df.columns:
        aligned_df = aligned_df[aligned_df["src_norm_variant"].isin(variant_set)].reset_index(drop=True)

    aligned_df = aligned_df.copy()
    aligned_df["source"] = "aligned"

    if ratio <= 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".parquet":
            aligned_df.to_parquet(out_path, index=False)
        else:
            aligned_df.to_csv(out_path, index=False)
        print("EvaCun mixing disabled. Saved baseline only.")
        print(f"Output: {out_path}")
        return

    evacun_df = read_table(evacun_path)
    if variant_set and "src_norm_variant" in evacun_df.columns:
        evacun_df = evacun_df[evacun_df["src_norm_variant"].isin(variant_set)].reset_index(drop=True)

    evacun_df, dropped_sentence = filter_sentence_endings(
        evacun_df,
        tgt_col=args.tgt_col,
        max_sentence_endings=max_sentence_endings,
    )

    target_n = int(round(len(aligned_df) * ratio))
    target_n = max(target_n, 0)

    if target_n > 0 and len(evacun_df) > target_n:
        evacun_df = evacun_df.sample(n=target_n, random_state=seed).reset_index(drop=True)

    if not evacun_df.empty:
        evacun_df = evacun_df.copy()
        evacun_df["source"] = "evacun"

    all_cols = sorted(set(aligned_df.columns).union(set(evacun_df.columns)))
    for col in all_cols:
        if col not in aligned_df.columns:
            aligned_df[col] = None
        if col not in evacun_df.columns:
            evacun_df[col] = None

    mixed_df = pd.concat([aligned_df[all_cols], evacun_df[all_cols]], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        mixed_df.to_parquet(out_path, index=False)
    else:
        mixed_df.to_csv(out_path, index=False)

    stats = {
        "aligned_rows": int(len(aligned_df)),
        "evacun_rows": int(len(evacun_df)),
        "mix_rows": int(len(mixed_df)),
        "evacun_ratio": float(ratio),
        "evacun_dropped_sentence": int(dropped_sentence),
        "aligned_gap_rate": compute_gap_rate(aligned_df.get(args.src_col, [])),
        "evacun_gap_rate": compute_gap_rate(evacun_df.get(args.src_col, [])),
        "aligned_caps_rate": compute_caps_token_rate(aligned_df.get(args.src_col, [])),
        "evacun_caps_rate": compute_caps_token_rate(evacun_df.get(args.src_col, [])),
        "aligned_src_avg_len": compute_avg_len(aligned_df.get(args.src_col, [])),
        "evacun_src_avg_len": compute_avg_len(evacun_df.get(args.src_col, [])),
        "aligned_tgt_avg_len": compute_avg_len(aligned_df.get(args.tgt_col, [])),
        "evacun_tgt_avg_len": compute_avg_len(evacun_df.get(args.tgt_col, [])),
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"Aligned rows: {len(aligned_df)}")
    print(f"EvaCun rows: {len(evacun_df)}")
    print(f"Output: {out_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
