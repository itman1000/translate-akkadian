"""ターゲット長の分布と打ち切り率を確認するユーティリティ。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd

from .utils import clean_text, get_data_dir, load_config


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def choose_tgt_col(df: pd.DataFrame, cfg: dict[str, Any], arg: Optional[str]) -> str:
    if arg:
        return arg
    for cand in [cfg.get("tgt_col"), "tgt_sent", "translation"]:
        if cand and cand in df.columns:
            return str(cand)
    raise ValueError("tgt_col not found in input data")


def filter_df(df: pd.DataFrame, variant: Optional[str], drop_flagged: bool) -> pd.DataFrame:
    work = df.copy()
    if variant and "src_norm_variant" in work.columns:
        work = work[work["src_norm_variant"].astype(str) == variant]
    if drop_flagged and "flags" in work.columns:
        work = work[work["flags"].fillna("").astype(str).str.strip() == ""]
    return work


def summarize_lengths(lengths: List[int], label: str, limit: Optional[int]) -> None:
    if not lengths:
        print(f"[{label}] no data")
        return
    series = pd.Series(lengths)
    q = series.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    over_ratio = None
    if limit is not None and limit > 0:
        over_ratio = (series > limit).mean()
    base = (
        f"[{label}] n={len(series)} max={series.max():.0f} "
        f"p50={q[0.5]:.0f} p90={q[0.9]:.0f} p95={q[0.95]:.0f} p99={q[0.99]:.0f}"
    )
    if over_ratio is None:
        print(base)
    else:
        print(f"{base} over_{limit}={over_ratio:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect target length stats.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--train", default=None, help="Train CSV/Parquet path.")
    parser.add_argument("--tgt-col", default=None, help="Target column name.")
    parser.add_argument("--variant", default=None, help="Normalization variant filter (A/B/C).")
    parser.add_argument("--drop-flagged", action="store_true", help="Drop flagged rows.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use first N rows.")
    parser.add_argument("--sample", type=int, default=None, help="Sample N rows randomly.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path/name for token length.")
    parser.add_argument(
        "--use-tokenizer",
        action="store_true",
        help="設定の model_name_or_path を使ってトークン長を計測する。",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Tokenizer batch size.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, None)

    train_path = Path(args.train) if args.train else data_dir / "train.csv"
    df = read_table(train_path)

    variant = args.variant.upper() if args.variant else None
    work = filter_df(df, variant=variant, drop_flagged=bool(args.drop_flagged))

    if args.sample:
        work = work.sample(n=min(len(work), args.sample), random_state=args.seed)
    if args.max_rows:
        work = work.head(args.max_rows)

    tgt_col = choose_tgt_col(work, cfg, args.tgt_col)
    texts = work[tgt_col].fillna("").astype(str).map(clean_text)
    texts = texts[texts != ""].tolist()

    max_target_length = parse_optional_int(cfg.get("max_target_length"))
    gen_limit = parse_optional_int(cfg.get("generation_max_new_tokens"))
    if gen_limit is None:
        gen_limit = parse_optional_int(cfg.get("generation_max_length"))

    char_lens = [len(text) for text in texts]
    print("=== 文字数分布 ===")
    summarize_lengths(char_lens, "tgt_chars", max_target_length)

    tokenizer_name = args.tokenizer
    if tokenizer_name is None and args.use_tokenizer:
        tokenizer_name = cfg.get("model_name_or_path")
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise ImportError("transformers が見つかりません。") from exc

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name))
        token_lens: List[int] = []
        for batch in batch_iter(texts, args.batch_size):
            enc = tokenizer(batch, add_special_tokens=True, padding=False, truncation=False)
            token_lens.extend(len(ids) for ids in enc["input_ids"])
        print("=== トークン数分布 ===")
        summarize_lengths(token_lens, "tgt_tokens(train)", max_target_length)
        summarize_lengths(token_lens, "tgt_tokens(infer)", gen_limit)
    else:
        print("tokenizer not set: skip token length stats")


if __name__ == "__main__":
    main()
