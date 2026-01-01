"""oare_id 単位の CV 分割を作成する。"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List

import pandas as pd

from .utils import get_artifacts_dir, get_data_dir, load_config


def load_ids_from_aligned(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Aligned file not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=["oare_id"])
    else:
        df = pd.read_csv(path, usecols=["oare_id"])
    return sorted(df["oare_id"].astype(str).unique().tolist())


def load_ids_from_train(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"train.csv not found: {path}")
    df = pd.read_csv(path, usecols=["oare_id"])
    return sorted(df["oare_id"].astype(str).unique().tolist())


def assign_folds(ids: List[str], k_folds: int, seed: int) -> List[tuple[str, int]]:
    rng = random.Random(seed)
    ids_copy = ids[:]
    rng.shuffle(ids_copy)
    return [(oid, idx % k_folds) for idx, oid in enumerate(ids_copy)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CV folds by oare_id.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--k", type=int, default=None, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--source", choices=["aligned", "train"], default=None, help="Source for IDs.")
    parser.add_argument("--aligned", default=None, help="Aligned data path.")
    parser.add_argument("--out", default=None, help="Output CSV path.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    k_folds = int(args.k or cfg.get("k_folds", 5))
    seed = int(args.seed or cfg.get("seed", 42))

    data_dir = get_data_dir(cfg, args.data_dir)
    artifacts_dir = get_artifacts_dir(cfg)

    aligned_path = Path(args.aligned) if args.aligned else Path(cfg.get("aligned_path", ""))
    if not aligned_path or str(aligned_path) == "":
        aligned_path = artifacts_dir / "aligned" / "aligned_train.parquet"

    source = args.source
    if not source:
        source = "aligned" if aligned_path.exists() else "train"

    if source == "aligned":
        ids = load_ids_from_aligned(aligned_path)
    else:
        ids = load_ids_from_train(data_dir / "train.csv")

    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")
    if len(ids) < k_folds:
        raise ValueError("Not enough documents for requested folds")

    assignments = assign_folds(ids, k_folds, seed)

    out_path = Path(args.out) if args.out else artifacts_dir / "splits" / f"cv_folds_k{k_folds}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    unique_ids = {oid for oid, _ in assignments}
    if len(unique_ids) != len(assignments):
        raise RuntimeError("Fold assignment has duplicate ids")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["oare_id", "fold"])
        writer.writerows(assignments)

    fold_counts = {}
    for _, fold in assignments:
        fold_counts[fold] = fold_counts.get(fold, 0) + 1

    print(f"Saved folds: {out_path}")
    print(f"Documents: {len(ids)}, folds: {k_folds}, seed: {seed}")
    print(
        "Fold counts: "
        + ", ".join(f"{fold}:{count}" for fold, count in sorted(fold_counts.items()))
    )


if __name__ == "__main__":
    main()
