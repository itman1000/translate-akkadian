#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT}/kaggle_bundle"
OUT_DIR="${OUT_ROOT}/translate-akkadian"

if [ -e "$OUT_DIR" ]; then
  echo "Already exists: $OUT_DIR" >&2
  echo "Remove it before running this script again." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

cp "$ROOT/submit_kaggle.ipynb" "$OUT_DIR/submit_kaggle.ipynb"
cp "$ROOT/requirements.txt" "$OUT_DIR/requirements.txt"

cp -R "$ROOT/src" "$OUT_DIR/src"
cp -R "$ROOT/configs" "$OUT_DIR/configs"
cp -R "$ROOT/scripts" "$OUT_DIR/scripts"
cp -R "$ROOT/docs" "$OUT_DIR/docs"

mkdir -p "$OUT_DIR/data" "$OUT_DIR/artifacts"
touch "$OUT_DIR/data/.keep" "$OUT_DIR/artifacts/.keep"
