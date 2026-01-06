"""Create mismatch-audit CSV from an existing src/ref/pred table.

This is useful when you already have model outputs (e.g., from a separate
inference pipeline) and want to classify mismatches without re-running training.

Example:
  python -m dp.audit_csv --input val_predictions.csv --out val_audit.csv \
    --src-col src_no_gloss --ref-col ref --pred-col pred --group-col oare_id

Tip:
  If your src includes appended glossary hints like "<LEX> ... </LEX>",
  prefer using a gloss-stripped column (e.g. src_no_gloss) for readability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .audit_mismatch import AuditConfig, build_audit_df


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mismatch-audit CSV from src/ref/pred table.")
    parser.add_argument("--input", required=True, help="Input CSV/Parquet path.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--src-col", default="src", help="Source column name.")
    parser.add_argument("--ref-col", default="ref", help="Reference/answer column name.")
    parser.add_argument("--pred-col", default="pred", help="Prediction column name.")
    parser.add_argument("--group-col", default=None, help="Optional group column for shift detection.")

    # shift / alignment hints
    parser.add_argument(
        "--max-shift",
        type=int,
        default=3,
        help="Look +/-k offsets for shift detection (default: 3).",
    )

    # similarity knobs
    parser.add_argument("--sim-th", type=float, default=0.85, help="Similarity threshold (default: 0.85).")
    parser.add_argument("--margin", type=float, default=0.08, help="Shift margin (default: 0.08).")
    parser.add_argument("--ngram-n", type=int, default=4, help="Char n-gram size (default: 4).")

    parser.add_argument(
        "--near-match-th",
        type=float,
        default=0.90,
        help="Classify as near-match (T04) when sim_self >= this (default: 0.90).",
    )

    # template-collapse tagging
    parser.add_argument(
        "--template-min-count",
        type=int,
        default=5,
        help="Min frequency to tag a prediction as a template (default: 5).",
    )
    parser.add_argument(
        "--template-sim-max",
        type=float,
        default=0.55,
        help="Only tag template collapse when sim_self < this (default: 0.55).",
    )

    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = read_table(inp)
    if args.src_col not in df.columns or args.ref_col not in df.columns or args.pred_col not in df.columns:
        raise ValueError(
            f"input must include columns: {args.src_col!r}, {args.ref_col!r}, {args.pred_col!r}"
        )

    cfg = AuditConfig(
        sim_th=float(args.sim_th),
        margin=float(args.margin),
        ngram_n=int(args.ngram_n),
        max_shift=int(args.max_shift),
        template_min_count=int(args.template_min_count),
        template_sim_max=float(args.template_sim_max),
        near_match_th=float(args.near_match_th),
    )
    group = args.group_col if args.group_col and args.group_col in df.columns else None

    audit = build_audit_df(
        df,
        src_col=args.src_col,
        ref_col=args.ref_col,
        pred_col=args.pred_col,
        group_col=group,
        cfg=cfg,
    )
    audit.to_csv(out, index=False)
    print(f"wrote: {out} rows={len(audit)}")


if __name__ == "__main__":
    main()
