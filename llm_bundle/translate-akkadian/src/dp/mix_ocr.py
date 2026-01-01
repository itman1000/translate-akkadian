"""aligned_train と OCR high tier を混合する。"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .utils import get_artifacts_dir, load_config


_GAP_RE = re.compile(r"<\s*(gap|big_gap)\s*>", re.IGNORECASE)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def list_ocr_parts(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("part-*.parquet"))
    if path.exists():
        return [path]
    pattern = f"{path.stem}-part-*.parquet"
    return sorted(path.parent.glob(pattern))


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix aligned_train with OCR high tier data.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--aligned", default=None, help="Aligned train path.")
    parser.add_argument("--ocr", default=None, help="OCR parts directory or prefix.")
    parser.add_argument("--out", default=None, help="Output path.")
    parser.add_argument("--tiers", default=None, help="Comma separated tiers (default: high).")
    parser.add_argument("--ratio", type=float, default=None, help="OCR ratio vs baseline.")
    parser.add_argument("--variants", default=None, help="Variants to keep (A,B,C).")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR mixing.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = get_artifacts_dir(cfg)

    aligned_path = (
        Path(args.aligned)
        if args.aligned
        else Path(cfg.get("ocr_mix_aligned_path", artifacts_dir / "aligned" / "aligned_train.parquet"))
    )
    ocr_path = (
        Path(args.ocr)
        if args.ocr
        else Path(cfg.get("ocr_mix_ocr_glob", artifacts_dir / "ocr_pairs" / "part-0000.parquet"))
    )
    out_path = (
        Path(args.out)
        if args.out
        else Path(cfg.get("ocr_mix_out", artifacts_dir / "ocr_pairs" / "mixed_train.parquet"))
    )

    enable_ocr = bool(cfg.get("ocr_mix_enable", True))
    if args.disable_ocr:
        enable_ocr = False
    ratio = args.ratio if args.ratio is not None else _get_float(cfg, "ocr_mix_ratio", 0.2)
    if ratio <= 0:
        enable_ocr = False

    tiers = args.tiers or str(cfg.get("ocr_mix_quality_tiers", "high"))
    tier_set = {t.strip().lower() for t in tiers.split(",") if t.strip()}
    variants = args.variants or str(cfg.get("ocr_mix_variants", ""))
    variant_set = {v.strip().upper() for v in variants.split(",") if v.strip()}

    seed = args.seed if args.seed is not None else _get_int(cfg, "seed", 42)
    rng = random.Random(seed)

    aligned_df = read_table(aligned_path)
    if variant_set and "src_norm_variant" in aligned_df.columns:
        aligned_df = aligned_df[aligned_df["src_norm_variant"].isin(variant_set)].reset_index(drop=True)

    aligned_df = aligned_df.copy()
    aligned_df["source"] = "aligned"

    if not enable_ocr:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".parquet":
            aligned_df.to_parquet(out_path, index=False)
        else:
            aligned_df.to_csv(out_path, index=False)
        print("OCR mixing disabled. Saved baseline only.")
        print(f"Output: {out_path}")
        return

    ocr_parts = list_ocr_parts(ocr_path)
    if not ocr_parts:
        raise FileNotFoundError(f"OCR parts not found at: {ocr_path}")

    target_n = int(round(len(aligned_df) * ratio))
    target_n = max(target_n, 0)

    sample_rows: List[Dict[str, object]] = []
    ocr_columns: Optional[List[str]] = None
    seen = 0
    tier_counts: Counter[str] = Counter()

    for part_path in ocr_parts:
        df = pd.read_parquet(part_path)
        if "quality_tier" in df.columns and tier_set:
            df = df[df["quality_tier"].str.lower().isin(tier_set)]
        if variant_set and "src_norm_variant" in df.columns:
            df = df[df["src_norm_variant"].isin(variant_set)]

        if df.empty:
            continue

        if "quality_tier" in df.columns:
            tier_counts.update(df["quality_tier"].fillna("").astype(str).str.lower().tolist())

        if ocr_columns is None:
            ocr_columns = list(df.columns)
        else:
            for col in ocr_columns:
                if col not in df.columns:
                    df[col] = None
            df = df[ocr_columns]

        for row in df.itertuples(index=False):
            seen += 1
            row_dict = row._asdict()
            if len(sample_rows) < target_n:
                sample_rows.append(row_dict)
            else:
                if target_n == 0:
                    break
                idx = rng.randrange(seen)
                if idx < target_n:
                    sample_rows[idx] = row_dict

    if target_n == 0:
        ocr_sample = pd.DataFrame(columns=ocr_columns or [])
    else:
        ocr_sample = pd.DataFrame(sample_rows)

    if not ocr_sample.empty:
        ocr_sample = ocr_sample.copy()
        ocr_sample["source"] = "ocr"

    # 列を揃える
    all_cols = sorted(set(aligned_df.columns).union(set(ocr_sample.columns)))
    for col in all_cols:
        if col not in aligned_df.columns:
            aligned_df[col] = None
        if col not in ocr_sample.columns:
            ocr_sample[col] = None

    mixed_df = pd.concat([aligned_df[all_cols], ocr_sample[all_cols]], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        mixed_df.to_parquet(out_path, index=False)
    else:
        mixed_df.to_csv(out_path, index=False)

    # ガードレール用の簡易統計
    stats = {
        "aligned_rows": int(len(aligned_df)),
        "ocr_rows": int(len(ocr_sample)),
        "mix_rows": int(len(mixed_df)),
        "ocr_ratio": ratio,
        "tier_counts": dict(tier_counts),
        "aligned_gap_rate": compute_gap_rate(aligned_df.get("src_sent", [])),
        "ocr_gap_rate": compute_gap_rate(ocr_sample.get("src_sent", [])),
        "aligned_caps_rate": compute_caps_token_rate(aligned_df.get("src_sent", [])),
        "ocr_caps_rate": compute_caps_token_rate(ocr_sample.get("src_sent", [])),
        "aligned_src_avg_len": compute_avg_len(aligned_df.get("src_sent", [])),
        "ocr_src_avg_len": compute_avg_len(ocr_sample.get("src_sent", [])),
        "aligned_tgt_avg_len": compute_avg_len(aligned_df.get("tgt_sent", [])),
        "ocr_tgt_avg_len": compute_avg_len(ocr_sample.get("tgt_sent", [])),
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"Aligned rows: {len(aligned_df)}")
    print(f"OCR rows: {len(ocr_sample)}")
    print(f"Output: {out_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
