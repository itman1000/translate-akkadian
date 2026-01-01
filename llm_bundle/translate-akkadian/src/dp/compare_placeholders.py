"""Compare placeholder strategies with proxy metrics."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .placeholders import apply_placeholders, build_token_map, load_lexicon_forms
from .utils import get_artifacts_dir, get_data_dir, load_config


def tokenize(text: str) -> List[str]:
    return text.split()


def compute_common_tokens(df: pd.DataFrame, top_n: int) -> set[str]:
    counter = Counter()
    for text in df["src_sent"].astype(str):
        counter.update(tokenize(text))
    return set([tok for tok, _ in counter.most_common(top_n)])


def run_strategy(
    df: pd.DataFrame,
    token_map: Dict[Tuple[str, ...], str],
    max_len: int,
    filter_mode: str,
    common_tokens: set[str],
) -> Dict[str, float]:
    total_rows = len(df)
    rows_with = 0
    total_placeholders = 0
    common_hits = 0
    short_hits = 0

    for _, row in df.iterrows():
        src = str(row.get("src_sent", ""))
        _, mapping = apply_placeholders(src, token_map, max_len, filter_mode=filter_mode)
        if mapping:
            rows_with += 1
        total_placeholders += len(mapping)
        for m in mapping:
            text_val = str(m.get("text", ""))
            tokens = text_val.split()
            if tokens and all(t in common_tokens for t in tokens):
                common_hits += 1
            if len(tokens) == 1 and len(tokens[0]) <= 3:
                short_hits += 1

    coverage = rows_with / total_rows if total_rows else 0.0
    avg_per_row = total_placeholders / total_rows if total_rows else 0.0
    common_rate = common_hits / total_placeholders if total_placeholders else 0.0
    short_rate = short_hits / total_placeholders if total_placeholders else 0.0
    score = coverage - 0.5 * common_rate - 0.5 * short_rate

    return {
        "rows": total_rows,
        "coverage": coverage,
        "avg_placeholders_per_row": avg_per_row,
        "common_token_rate": common_rate,
        "short_token_rate": short_rate,
        "score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare placeholder strategies.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--aligned", default=None, help="Aligned data path.")
    parser.add_argument("--lexicon", default=None, help="Lexicon CSV path.")
    parser.add_argument("--variants", default=None, help="Variants to include, e.g., A,B,C.")
    parser.add_argument("--strategies", default="strict,pattern,gn_only", help="Strategies to compare.")
    parser.add_argument("--sample", type=int, default=None, help="Use first N rows for quick runs.")
    parser.add_argument("--top-n", type=int, default=50, help="Top-N tokens for common-word heuristic.")
    parser.add_argument("--out", default=None, help="Output summary CSV path.")
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

    df = pd.read_parquet(aligned_path) if aligned_path.suffix == ".parquet" else pd.read_csv(aligned_path)
    if args.sample:
        df = df.head(args.sample)

    variant_spec = args.variants or cfg.get("variants", "A,B,C")
    variants = {v.strip().upper() for v in str(variant_spec).split(",") if v.strip()}
    if variants:
        df = df[df["src_norm_variant"].isin(variants)].reset_index(drop=True)

    common_tokens = compute_common_tokens(df, args.top_n)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    summary_rows = []

    for strat in strategies:
        if strat == "gn_only":
            types = ["GN"]
            filter_mode = "all"
        elif strat == "strict":
            types = ["PN", "GN"]
            filter_mode = "strict"
        else:
            types = ["PN", "GN"]
            filter_mode = "pattern"

        forms_by_type = load_lexicon_forms(lexicon_path, types)
        token_map, max_len = build_token_map(forms_by_type)

        metrics = run_strategy(df, token_map, max_len, filter_mode, common_tokens)
        metrics["strategy"] = strat
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("score", ascending=False)

    out_path = Path(args.out) if args.out else artifacts_dir / "placeholders" / "strategy_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print(summary)
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
