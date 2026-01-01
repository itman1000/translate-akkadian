from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import get_artifacts_dir, load_config


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def summarize_series(series: pd.Series, label: str, fmt: str = ".4f") -> None:
    if series.empty:
        print(f"[{label}] no data")
        return
    q = series.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    fmt_str = "{:" + fmt + "}"
    print(
        f"[{label}] n={len(series)} min={fmt_str.format(series.min())} "
        f"max={fmt_str.format(series.max())} "
        f"p50={fmt_str.format(q[0.5])} p90={fmt_str.format(q[0.9])} "
        f"p95={fmt_str.format(q[0.95])} p99={fmt_str.format(q[0.99])}"
    )


def count_flags(flags_series: pd.Series) -> pd.Series:
    counts = {}
    for value in flags_series.fillna(""):
        for flag in str(value).split(";"):
            flag = flag.strip()
            if not flag:
                continue
            counts[flag] = counts.get(flag, 0) + 1
    if not counts:
        return pd.Series(dtype=int)
    return pd.Series(counts).sort_values(ascending=False)


def resolve_aligned_path(config_path: Optional[str], aligned_path: Optional[str]) -> Path:
    if aligned_path:
        return Path(aligned_path)
    if config_path:
        cfg = load_config(config_path)
        artifacts_dir = get_artifacts_dir(cfg)
        aligned_format = str(cfg.get("aligned_format", "parquet")).lower()
        name = f"aligned_train.{aligned_format}"
        return artifacts_dir / "aligned" / name
    return Path("artifacts/aligned/aligned_train.parquet")


def print_samples(df: pd.DataFrame, sample_size: int, seed: int) -> None:
    if sample_size <= 0 or df.empty:
        return
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    print("=== samples ===")
    for idx, row in sample.reset_index(drop=True).iterrows():
        oare_id = row.get("oare_id", "")
        align_score = row.get("align_score", "")
        len_ratio = row.get("len_ratio", "")
        flags = row.get("flags", "")
        print(f"[{idx}] oare_id={oare_id} align_score={align_score} len_ratio={len_ratio} flags={flags}")
        print(f"src: {row.get('src_sent', '')}")
        print(f"tgt: {row.get('tgt_sent', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect aligned_train quality.")
    parser.add_argument("--config", default=None, help="Config file path.")
    parser.add_argument("--aligned", default=None, help="aligned_train path.")
    parser.add_argument("--sample", type=int, default=20, help="Sample size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--only-flagged", action="store_true", help="Only flagged rows.")
    parser.add_argument("--only-clean", action="store_true", help="Only clean rows.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use first N rows.")
    args = parser.parse_args()

    if args.only_flagged and args.only_clean:
        raise ValueError("only one of --only-flagged or --only-clean can be set")

    aligned_path = resolve_aligned_path(args.config, args.aligned)
    df = read_table(aligned_path)
    if args.max_rows:
        df = df.head(args.max_rows)

    if "flags" in df.columns:
        flags = df["flags"].fillna("").astype(str)
        if args.only_flagged:
            df = df[flags != ""]
        elif args.only_clean:
            df = df[flags == ""]

    print(f"rows={len(df)} path={aligned_path}")

    if "align_score" in df.columns:
        summarize_series(df["align_score"], "align_score", ".4f")
    if "len_ratio" in df.columns:
        summarize_series(df["len_ratio"], "len_ratio", ".4f")
    if "token_align_score" in df.columns:
        summarize_series(df["token_align_score"], "token_align_score", ".4f")

    if "src_sent" in df.columns:
        src_empty = (df["src_sent"].fillna("").astype(str).str.strip() == "").mean()
        print(f"src_empty_rate={src_empty:.1%}")
    if "tgt_sent" in df.columns:
        tgt_empty = (df["tgt_sent"].fillna("").astype(str).str.strip() == "").mean()
        print(f"tgt_empty_rate={tgt_empty:.1%}")
    if "tgt_sentence_ends" in df.columns:
        multi_sentence_rate = (df["tgt_sentence_ends"] >= 2).mean()
        summarize_series(df["tgt_sentence_ends"], "tgt_sentence_ends", ".0f")
        print(f"multi_sentence_rate={multi_sentence_rate:.1%}")

    if "flags" in df.columns:
        counts = count_flags(df["flags"])
        if not counts.empty:
            print("flags_top10:")
            for name, count in counts.head(10).items():
                ratio = count / len(df) if len(df) else 0.0
                print(f"- {name}: {count} ({ratio:.1%})")
        src_multi = df["flags"].fillna("").str.contains("src_multi_tgt").mean()
        tgt_multi = df["flags"].fillna("").str.contains("tgt_multi_src").mean()
        print(f"src_multi_tgt_rate={src_multi:.1%} tgt_multi_src_rate={tgt_multi:.1%}")

    if "oare_id" in df.columns:
        counts = df.groupby("oare_id").size()
        summarize_series(counts, "sentences_per_oare_id", ".0f")

    print_samples(df, args.sample, args.seed)


if __name__ == "__main__":
    main()
