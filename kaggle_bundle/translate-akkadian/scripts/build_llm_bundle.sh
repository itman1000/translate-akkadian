#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT}/llm_bundle"
OUT_DIR="${OUT_ROOT}/translate-akkadian"
ZIP_PATH="${OUT_ROOT}/translate-akkadian.zip"

if [ -e "$OUT_DIR" ] || [ -e "$ZIP_PATH" ]; then
  echo "Already exists: $OUT_DIR or $ZIP_PATH" >&2
  echo "Remove them before running this script again." >&2
  exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "zip command not found." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

cp "$ROOT/README.md" "$OUT_DIR/README.md"
cp "$ROOT/README_KAGGLE.md" "$OUT_DIR/README_KAGGLE.md"
cp "$ROOT/README_COLAB.md" "$OUT_DIR/README_COLAB.md"
cp "$ROOT/requirements.txt" "$OUT_DIR/requirements.txt"
cp "$ROOT/submit_kaggle.ipynb" "$OUT_DIR/submit_kaggle.ipynb"

cp -R "$ROOT/src" "$OUT_DIR/src"
cp -R "$ROOT/configs" "$OUT_DIR/configs"
cp -R "$ROOT/docs" "$OUT_DIR/docs"
cp -R "$ROOT/scripts" "$OUT_DIR/scripts"
cp -R "$ROOT/tests" "$OUT_DIR/tests"

(cd "$OUT_ROOT" && zip -r "translate-akkadian.zip" "translate-akkadian")
echo "Created: $ZIP_PATH"
