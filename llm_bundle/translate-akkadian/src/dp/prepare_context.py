"""Build context-conditioned inputs for train/test."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .utils import clean_text, get_artifacts_dir, get_data_dir, load_config


_LINE_RE = re.compile(r"^(?P<num>\d+)(?P<suffix>.*)$")


def read_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_line_ref(value: Any) -> Tuple[int, int, str]:
    if value is None:
        return (10**9, 0, "")
    text = str(value).strip()
    if not text:
        return (10**9, 0, "")
    match = _LINE_RE.match(text)
    if not match:
        return (10**9, 0, text)
    num = int(match.group("num"))
    suffix = match.group("suffix") or ""
    prime_count = suffix.count("'")
    rest = suffix.replace("'", "")
    return (num, prime_count, rest)


def build_context_strings(
    sentences: Sequence[str], k_prev: int, k_next: int
) -> List[str]:
    contexts: List[str] = []
    total = len(sentences)
    for idx in range(total):
        parts = ["<CTX>"]
        for offset in range(k_prev, 0, -1):
            prev_idx = idx - offset
            if prev_idx >= 0:
                parts.append(f"<P{offset}> {sentences[prev_idx]} </P{offset}>")
        parts.append(f"<T> {sentences[idx]} </T>")
        for offset in range(1, k_next + 1):
            next_idx = idx + offset
            if next_idx < total:
                parts.append(f"<N{offset}> {sentences[next_idx]} </N{offset}>")
        parts.append("</CTX>")
        contexts.append(" ".join(parts))
    return contexts


def resolve_group_cols(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [col for col in candidates if col in df.columns]


def add_context_column(
    df: pd.DataFrame,
    src_col: str,
    group_cols: List[str],
    order_cols: List[str],
    k_prev: int,
    k_next: int,
    context_col: str,
) -> pd.DataFrame:
    if src_col not in df.columns:
        raise ValueError(f"source column not found: {src_col}")
    if not group_cols:
        df = df.copy()
        df["_context_group"] = 0
        group_cols = ["_context_group"]
    if order_cols:
        df_sorted = df.sort_values(group_cols + order_cols, kind="mergesort")
    else:
        df_sorted = df.sort_values(group_cols, kind="mergesort")

    context_map: Dict[Any, str] = {}
    for _, group in df_sorted.groupby(group_cols, sort=False):
        sentences = [clean_text(str(val)) for val in group[src_col].fillna("")]
        contexts = build_context_strings(sentences, k_prev, k_next)
        for idx, context in zip(group.index, contexts):
            context_map[idx] = context

    df = df.copy()
    df[context_col] = df.index.map(lambda i: context_map.get(i, clean_text(str(df.at[i, src_col]))))
    if "_context_group" in df.columns:
        df = df.drop(columns=["_context_group"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare context-conditioned datasets.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--aligned", default=None, help="Aligned train path.")
    parser.add_argument("--test", default=None, help="Test CSV path.")
    parser.add_argument("--out", default=None, help="Output path for train context data.")
    parser.add_argument("--test-out", default=None, help="Output path for test context data.")
    parser.add_argument("--format", default=None, choices=["parquet", "csv"], help="Train output format.")
    parser.add_argument("--variants", default=None, help="Variants to keep, e.g., A,B,C.")
    parser.add_argument("--k", type=int, default=None, help="Use same window size for prev/next.")
    parser.add_argument("--k-prev", type=int, default=None, help="Previous context size.")
    parser.add_argument("--k-next", type=int, default=None, help="Next context size.")
    parser.add_argument("--src-col", default=None, help="Source column name for train data.")
    parser.add_argument("--test-src-col", default=None, help="Source column name for test data.")
    parser.add_argument("--context-col", default=None, help="Context column name.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, None)
    artifacts_dir = get_artifacts_dir(cfg)

    k_prev = int(cfg.get("k_prev", 2))
    k_next = int(cfg.get("k_next", 2))
    if args.k is not None:
        k_prev = k_next = int(args.k)
    if args.k_prev is not None:
        k_prev = int(args.k_prev)
    if args.k_next is not None:
        k_next = int(args.k_next)

    src_col = args.src_col or cfg.get("src_col", "src_sent")
    test_src_col = args.test_src_col or cfg.get("test_src_col", "transliteration")
    context_col = args.context_col or cfg.get("context_col", "src_context")

    variants_spec = args.variants or cfg.get("variants", "")
    variants = [v.strip().upper() for v in str(variants_spec).split(",") if v.strip()]

    aligned_path = Path(args.aligned) if args.aligned else Path(cfg.get("aligned_path", ""))
    if not aligned_path or str(aligned_path) == "":
        aligned_path = artifacts_dir / "aligned" / "aligned_train.parquet"

    fmt = args.format or str(cfg.get("context_format", "parquet")).lower()
    out_path = Path(args.out) if args.out else Path(cfg.get("context_out", ""))
    if not out_path or str(out_path) == "":
        out_path = artifacts_dir / "context" / f"aligned_train_context.{fmt}"

    test_path = Path(args.test) if args.test else Path(cfg.get("test_path", ""))
    if not test_path or str(test_path) == "":
        test_path = data_dir / "test.csv"

    test_out_path = Path(args.test_out) if args.test_out else Path(cfg.get("test_out", ""))
    if not test_out_path or str(test_out_path) == "":
        test_out_path = artifacts_dir / "context" / "test_context.csv"

    aligned_df = read_table(aligned_path, "Aligned train")
    if variants and "src_norm_variant" in aligned_df.columns:
        aligned_df = aligned_df[aligned_df["src_norm_variant"].isin(variants)].reset_index(drop=True)

    train_group_cols = resolve_group_cols(aligned_df, ["oare_id", "src_norm_variant"])
    train_order_cols = resolve_group_cols(aligned_df, ["pair_id"])
    aligned_ctx = add_context_column(
        aligned_df,
        src_col=src_col,
        group_cols=train_group_cols,
        order_cols=train_order_cols,
        k_prev=k_prev,
        k_next=k_next,
        context_col=context_col,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        aligned_ctx.to_parquet(out_path, index=False)
    else:
        aligned_ctx.to_csv(out_path, index=False)
    print(f"Saved train context: {out_path}")

    if test_path.exists():
        test_df = read_table(test_path, "Test")
        order_cols: List[str] = []
        if "line_start" in test_df.columns:
            line_keys = test_df["line_start"].apply(parse_line_ref)
        elif "line_end" in test_df.columns:
            line_keys = test_df["line_end"].apply(parse_line_ref)
        else:
            line_keys = None

        if line_keys is not None:
            test_df = test_df.copy()
            test_df["_line_num"] = [key[0] for key in line_keys]
            test_df["_line_prime"] = [key[1] for key in line_keys]
            test_df["_line_suffix"] = [key[2] for key in line_keys]
            order_cols = ["_line_num", "_line_prime", "_line_suffix"]

        test_group_cols = resolve_group_cols(test_df, ["text_id"])
        test_ctx = add_context_column(
            test_df,
            src_col=test_src_col,
            group_cols=test_group_cols,
            order_cols=order_cols,
            k_prev=k_prev,
            k_next=k_next,
            context_col=context_col,
        )
        if order_cols:
            test_ctx = test_ctx.drop(columns=order_cols)

        test_out_path.parent.mkdir(parents=True, exist_ok=True)
        test_ctx.to_csv(test_out_path, index=False)
        print(f"Saved test context: {test_out_path}")
    else:
        print(f"Test file not found, skipped: {test_path}")


if __name__ == "__main__":
    main()
