#!/usr/bin/env bash
set -euo pipefail

# n-best rerank CV runner for variant C.

export PYTHONPATH=src

ROOT="artifacts/ablation"
RESULTS="artifacts/ablation_rerank_runs"
CONFIG="configs/train_real.yaml"

variants=(C)
folds=(0 1 2 3 4)

for v in "${variants[@]}"; do
  for f in "${folds[@]}"; do
    val_path="$ROOT/variant=$v/fold=$f/val.parquet"
    ckpt="artifacts/ablation_runs/variant=$v/fold=$f/model.pkl"
    out_dir="$RESULTS/variant=$v/fold=$f"
    mkdir -p "$out_dir"

    python -m dp.eval_nbest \
      --config "$CONFIG" \
      --ckpt "$ckpt" \
      --val "$val_path" \
      --k 5 \
      --variant "$v" \
      --fold "$f" \
      --out "$out_dir"
  done
done

python -m dp.collect_ablation --root "$RESULTS"
