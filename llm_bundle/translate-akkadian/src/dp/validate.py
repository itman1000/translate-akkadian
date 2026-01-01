"""提出形式と基本的な制約を検証する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .utils import get_data_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission.csv.")
    parser.add_argument("--submission", required=True, help="Submission CSV path.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    parser.add_argument("--test", default=None, help="Optional test.csv path.")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    sub_path = Path(args.submission)
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission not found: {sub_path}")

    sub_df = pd.read_csv(sub_path)
    errors = []
    warnings = []

    if "id" not in sub_df.columns or "translation" not in sub_df.columns:
        errors.append("Submission must contain 'id' and 'translation' columns")

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}")
        sys.exit(1)

    # ID の網羅性
    if args.test:
        test_path = Path(args.test)
    else:
        test_path = get_data_dir(cfg, args.data_dir) / "test.csv"

    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_ids = set(test_df["id"].tolist())
        sub_ids = sub_df["id"].tolist()
        sub_id_set = set(sub_ids)
        missing = test_ids - sub_id_set
        extra = sub_id_set - test_ids
        if missing:
            errors.append(f"Missing ids: {len(missing)}")
        if extra:
            errors.append(f"Extra ids: {len(extra)}")
        if len(sub_ids) != len(sub_id_set):
            errors.append("Duplicate ids found")
    else:
        warnings.append(f"test.csv not found at {test_path}; id coverage not checked")

    # 翻訳内容の検査
    translations = sub_df["translation"].astype(str)
    if (translations.str.len() == 0).any():
        errors.append("Empty translations found")
    if translations.str.contains(r"[\r\n]").any():
        errors.append("Translations contain newline characters")

    # 複数文の警告
    punct_counts = translations.str.count(r"[.!?]")
    if (punct_counts > 1).any():
        warnings.append("Some translations look like multiple sentences")

    for msg in warnings:
        print(f"WARN: {msg}")

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}")
        sys.exit(1)

    print("OK: submission looks valid")


if __name__ == "__main__":
    main()
