#!/usr/bin/env bash
set -euo pipefail

# Ensemble CV runner for variant C using multiple TF-IDF configs.

export PYTHONPATH=src

ROOT="artifacts/ablation"
MODEL_DIR="artifacts/ablation_ensemble_models"
RESULTS="artifacts/ablation_ensemble_runs"

configs=("configs/train_real.yaml" "configs/train_real_char_2_4.yaml" "configs/train_real_word_1_2.yaml")
names=("char_3_5" "char_2_4" "word_1_2")

folds=(0 1 2 3 4)

for f in "${folds[@]}"; do
  train_path="$ROOT/variant=C/fold=$f/train.parquet"
  val_path="$ROOT/variant=C/fold=$f/val.parquet"
  ckpts=()

  for i in "${!configs[@]}"; do
    cfg="${configs[$i]}"
    name="${names[$i]}"
    out_dir="$MODEL_DIR/model=$name/variant=C/fold=$f"
    mkdir -p "$out_dir"

    python -m dp.train_real \
      --config "$cfg" \
      --train "$train_path" \
      --out "$out_dir" \
      --variant C \
      --fold "$f"

    ckpts+=("$out_dir/model.pkl")
  done

  ckpt_list=$(IFS=, ; echo "${ckpts[*]}")
  out_eval="$RESULTS/variant=C/fold=$f"
  mkdir -p "$out_eval"

  python -m dp.eval_ensemble \
    --config "configs/train_real.yaml" \
    --ckpts "$ckpt_list" \
    --val "$val_path" \
    --k 3 \
    --variant C \
    --fold "$f" \
    --out "$out_eval"
done

python -m dp.collect_ablation --root "$RESULTS"
