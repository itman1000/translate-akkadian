#!/usr/bin/env bash
set -euo pipefail

# A/B/C ablation runner using TF-IDF retrieval baseline.

export PYTHONPATH=src

ROOT="artifacts/ablation"
RESULTS="artifacts/ablation_runs"
CONFIG="configs/train_real.yaml"

variants=(A B C)
folds=(0 1 2 3 4)

for v in "${variants[@]}"; do
  for f in "${folds[@]}"; do
    train_path="$ROOT/variant=$v/fold=$f/train.parquet"
    val_path="$ROOT/variant=$v/fold=$f/val.parquet"
    out_dir="$RESULTS/variant=$v/fold=$f"
    mkdir -p "$out_dir"

    python -m dp.train_real \
      --config "$CONFIG" \
      --train "$train_path" \
      --val "$val_path" \
      --out "$out_dir" \
      --variant "$v" \
      --fold "$f"
  done
done

python -m dp.collect_ablation --root "$RESULTS"
