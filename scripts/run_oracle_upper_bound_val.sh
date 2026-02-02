#!/usr/bin/env bash
set -euo pipefail

# Oracle Upper Bound runner (beam n-best + sampling) for a single val file.
#
# Usage:
#   bash scripts/run_oracle_upper_bound_val.sh \
#     configs/train_real.yaml \
#     artifacts/nmt/byt5_large \
#     artifacts/ablation/variant=C/fold=0/val.parquet \
#     src_sent id tgt_sent \
#     artifacts/oracle_run

CFG=${1:-configs/train_real.yaml}
CKPT_DIR=${2:-artifacts/nmt/byt5_large}
VAL_PATH=${3:-artifacts/ablation/variant=C/fold=0/val.parquet}
SRC_COL=${4:-src_sent}
ID_COL=${5:-id}
GOLD_COL=${6:-tgt_sent}
OUT_DIR=${7:-artifacts/oracle_run}

export PYTHONPATH=src

mkdir -p "${OUT_DIR}/cands"

echo "[1/3] beam n-best"
python -m dp.infer_nmt_nbest \
  --config "${CFG}" \
  --ckpt "${CKPT_DIR}" \
  --test "${VAL_PATH}" \
  --src-col "${SRC_COL}" --id-col "${ID_COL}" \
  --out "${OUT_DIR}/cands/beam64.csv" \
  --k 64 --num-beams 64 --tag beam64

echo "[2/3] sampling (32 x 4 runs)"
python -m dp.infer_nmt_nbest \
  --config "${CFG}" \
  --ckpt "${CKPT_DIR}" \
  --test "${VAL_PATH}" \
  --src-col "${SRC_COL}" --id-col "${ID_COL}" \
  --out "${OUT_DIR}/cands/sample_t0.9_p0.95.csv" \
  --do-sample --temperature 0.9 --top-p 0.95 \
  --k 32 --runs 4 --seed 42 --tag sample_t0.9_p0.95

echo "[3/3] oracle eval"
python -m dp.oracle_eval \
  --config "${CFG}" \
  --gold "${VAL_PATH}" \
  --id-col "${ID_COL}" --gold-text-col "${GOLD_COL}" \
  --cands "${OUT_DIR}/cands/beam64.csv" "${OUT_DIR}/cands/sample_t0.9_p0.95.csv" \
  --out "${OUT_DIR}/eval"

echo "Done. See: ${OUT_DIR}/eval/metrics.json"
