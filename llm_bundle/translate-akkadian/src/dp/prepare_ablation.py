"""fold 単位で A/B/C アブレーション用データを準備する。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .utils import get_artifacts_dir, get_data_dir, load_config


def read_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_folds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Folds file not found: {path}")
    return pd.read_csv(path)


def write_split(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ablation datasets per fold.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--aligned", default=None, help="Aligned data path.")
    parser.add_argument("--folds", default=None, help="Folds CSV path.")
    parser.add_argument("--format", default=None, choices=["parquet", "csv"], help="Output format.")
    parser.add_argument("--variants", default=None, help="Variants to export, e.g., A,B,C.")
    parser.add_argument("--drop-flagged", action="store_true", help="Drop rows with any flags.")
    parser.add_argument("--out-dir", default=None, help="Output base directory for splits.")
    parser.add_argument("--extra", default=None, help="Extra dataset to append (parquet/csv).")
    parser.add_argument(
        "--extra-mode",
        default="train",
        choices=["train", "all"],
        help="Append extra rows to train only or both train/val.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = get_artifacts_dir(cfg)

    aligned_path = Path(args.aligned) if args.aligned else Path(cfg.get("aligned_path", ""))
    if not aligned_path or str(aligned_path) == "":
        aligned_path = artifacts_dir / "aligned" / "aligned_train.parquet"

    if args.folds:
        folds_path = Path(args.folds)
    elif cfg.get("folds_path"):
        folds_path = Path(cfg.get("folds_path"))
    else:
        k_folds = int(cfg.get("k_folds", 5))
        folds_path = artifacts_dir / "splits" / f"cv_folds_k{k_folds}.csv"

    fmt = args.format or str(cfg.get("ablation_format", "parquet")).lower()
    variants_spec = args.variants or cfg.get("variants", "A,B,C")
    variants = [v.strip().upper() for v in str(variants_spec).split(",") if v.strip()]

    aligned_df = read_table(aligned_path, "Aligned file")
    folds_df = read_folds(folds_path)

    extra_df = None
    if args.extra:
        extra_path = Path(args.extra)
        extra_df = read_table(extra_path, "Extra file")
        if args.drop_flagged and "flags" in extra_df.columns:
            extra_df = extra_df[extra_df["flags"].fillna("") == ""].reset_index(drop=True)

    if args.drop_flagged and "flags" in aligned_df.columns:
        aligned_df = aligned_df[aligned_df["flags"].fillna("") == ""].reset_index(drop=True)

    merged = aligned_df.merge(folds_df, on="oare_id", how="inner")

    out_dir = artifacts_dir / "ablation"
    if cfg.get("ablation_out_dir"):
        out_dir = Path(cfg["ablation_out_dir"])
    if args.out_dir:
        out_dir = Path(args.out_dir)
    summary_rows = []

    for variant in variants:
        vdf = merged[merged["src_norm_variant"] == variant]
        for fold in sorted(folds_df["fold"].unique()):
            val_df = vdf[vdf["fold"] == fold].drop(columns=["fold"]).reset_index(drop=True)
            train_df = vdf[vdf["fold"] != fold].drop(columns=["fold"]).reset_index(drop=True)

            if extra_df is not None:
                train_df = pd.concat([train_df, extra_df], ignore_index=True, sort=False)
                if args.extra_mode == "all":
                    val_df = pd.concat([val_df, extra_df], ignore_index=True, sort=False)

            base = out_dir / f"variant={variant}" / f"fold={fold}"
            train_path = base / f"train.{fmt}"
            val_path = base / f"val.{fmt}"

            write_split(train_df, train_path, fmt)
            write_split(val_df, val_path, fmt)

            summary_rows.append(
                {
                    "variant": variant,
                    "fold": int(fold),
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                }
            )

    summary_path = out_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Saved ablation splits to: {out_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
