#!/usr/bin/env bash
set -euo pipefail

# Placeholder CV runner for variant C (pattern strategy).

export PYTHONPATH=src

ROOT="artifacts/ablation_placeholders"
RESULTS="artifacts/ablation_placeholder_runs"
CONFIG="configs/train_real.yaml"

variants=(C)
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
      --fold "$f" \
      --src-col src_placeholders \
      --tgt-col tgt_placeholders \
      --restore \
      --map-col placeholder_map \
      --restore-target-col tgt_sent
  done
done

python -m dp.collect_ablation --root "$RESULTS"
