"""Prepare placeholder-applied datasets for PN/GN ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .placeholders import (
    apply_placeholders,
    build_token_map,
    dump_mapping,
    load_lexicon_forms,
    replace_target_with_mapping,
)
from .utils import get_artifacts_dir, get_data_dir, load_config


def read_aligned(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Aligned file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_output(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply PN/GN placeholders to aligned data.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--aligned", default=None, help="Aligned data path.")
    parser.add_argument("--lexicon", default=None, help="Lexicon CSV path.")
    parser.add_argument("--out", default=None, help="Output file path.")
    parser.add_argument("--format", default=None, choices=["parquet", "csv"], help="Output format.")
    parser.add_argument("--variants", default=None, help="Variants to export, e.g., A,B,C.")
    parser.add_argument("--types", default=None, help="Lexicon types, e.g., PN,GN.")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["all", "strict", "pattern", "gn_only"],
        help="Placeholder strategy.",
    )
    parser.add_argument("--replace-target", action="store_true", help="Replace PN/GN in target if matched.")
    parser.add_argument("--sample", type=int, default=None, help="Use first N rows for quick runs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = get_artifacts_dir(cfg)
    data_dir = get_data_dir(cfg, None)

    aligned_path = Path(args.aligned) if args.aligned else Path(cfg.get("aligned_path", ""))
    if not aligned_path or str(aligned_path) == "":
        aligned_path = artifacts_dir / "aligned" / "aligned_train.parquet"

    lexicon_path = Path(args.lexicon) if args.lexicon else Path(cfg.get("lexicon_path", ""))
    if not lexicon_path or str(lexicon_path) == "":
        lexicon_path = data_dir / "OA_Lexicon_eBL.csv"

    fmt = args.format or str(cfg.get("placeholders_format", "parquet")).lower()
    variant_spec = args.variants or cfg.get("variants", "A,B,C")
    variants = {v.strip().upper() for v in str(variant_spec).split(",") if v.strip()}

    strategy = args.strategy or cfg.get("placeholder_strategy", "pattern")

    types_spec = args.types or cfg.get("placeholder_types", "PN,GN")
    types = [t.strip().upper() for t in str(types_spec).split(",") if t.strip()]
    filter_mode = "all"
    if strategy == "gn_only":
        types = ["GN"]
        filter_mode = "all"
    elif strategy == "strict":
        filter_mode = "strict"
    elif strategy == "pattern":
        filter_mode = "pattern"

    aligned_df = read_aligned(aligned_path)
    if args.sample:
        aligned_df = aligned_df.head(args.sample)

    if variants:
        aligned_df = aligned_df[aligned_df["src_norm_variant"].isin(variants)].reset_index(drop=True)

    forms_by_type = load_lexicon_forms(lexicon_path, types)
    token_map, max_len = build_token_map(forms_by_type)

    out_rows = []
    for idx, row in aligned_df.iterrows():
        src = str(row.get("src_sent", ""))
        tgt = str(row.get("tgt_sent", ""))
        src_ph, mapping = apply_placeholders(src, token_map, max_len, filter_mode=filter_mode)
        tgt_ph = tgt
        if args.replace_target:
            tgt_ph = replace_target_with_mapping(tgt, mapping)

        out_row = row.to_dict()
        out_row["pair_id"] = int(idx)
        out_row["src_placeholders"] = src_ph
        out_row["tgt_placeholders"] = tgt_ph
        out_row["placeholder_map"] = dump_mapping(mapping)
        out_rows.append(out_row)

    out_df = pd.DataFrame(out_rows)

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = artifacts_dir / "placeholders"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"aligned_train_placeholders_{strategy}.{fmt}"

    write_output(out_df, out_path, fmt)
    print(f"Saved placeholder data: {out_path}")


if __name__ == "__main__":
    main()
