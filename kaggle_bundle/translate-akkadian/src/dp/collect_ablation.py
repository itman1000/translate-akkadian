"""アブレーション指標を集計してサマリー表を作る。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


def infer_variant_fold(path: Path) -> Dict[str, str]:
    parts = path.parts
    info: Dict[str, str] = {}
    for part in parts:
        if part.startswith("variant="):
            info["variant"] = part.split("=", 1)[1]
        if part.startswith("fold="):
            info["fold"] = part.split("=", 1)[1]
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ablation metrics.")
    parser.add_argument("--root", required=True, help="Root directory containing metrics.json files.")
    parser.add_argument("--out", default=None, help="Output summary CSV path.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    metrics_files = list(root.rglob("metrics.json"))
    rows: List[Dict[str, str]] = []

    for path in metrics_files:
        data = json.loads(path.read_text())
        info = infer_variant_fold(path)
        variant = str(data.get("variant", info.get("variant", "")))
        fold = str(data.get("fold", info.get("fold", "")))

        row = {"variant": variant, "fold": fold}
        for key in ["bleu", "chrf", "score", "geometric_mean"]:
            if key in data:
                row[key] = data[key]
        rows.append(row)

    if not rows:
        raise RuntimeError("No metrics.json found under root")

    df = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else root / "summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
