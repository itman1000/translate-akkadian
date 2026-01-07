"""CLI to generate `val_todo.csv` from `val_audit.csv`.

Usage:
  python -m dp.todo_csv --input artifacts/nmt/.../val_audit.csv \
    --output artifacts/nmt/.../val_todo.csv

You can run this even if you don't modify `train_nmt.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dp.todo_from_audit import ValTodoConfig, build_val_todo_df


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        required=True,
        help="Path to val_audit.csv",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to write val_todo.csv",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="Maximum number of TODO rows to output (default: 500)",
    )
    p.add_argument(
        "--max-rows-per-doc",
        type=int,
        default=8,
        help="Maximum TODO rows per oare_id/doc (default: 8)",
    )
    p.add_argument(
        "--min-priority",
        type=float,
        default=0.01,
        help="Filter items below this priority (default: 0.01)",
    )
    p.add_argument(
        "--include-near-match",
        action="store_true",
        help="Include near-match items (default: off unless flag provided)",
    )
    p.add_argument(
        "--include-style",
        action="store_true",
        help="Include style-only items (default: off)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audit_df = pd.read_csv(in_path)

    cfg = ValTodoConfig(
        max_rows=args.max_rows,
        max_rows_per_doc=args.max_rows_per_doc,
        min_priority=args.min_priority,
        include_near_match=bool(args.include_near_match),
        include_style=bool(args.include_style),
    )

    todo_df = build_val_todo_df(audit_df, cfg)
    todo_df.to_csv(out_path, index=False)

    # Tiny human-readable summary.
    print(f"[val_todo] saved: {out_path} rows={len(todo_df)}")
    if len(todo_df):
        top = todo_df["todo_type"].value_counts().head(10).to_dict()
        print(f"[val_todo] top_types={top}")


if __name__ == "__main__":
    main()
