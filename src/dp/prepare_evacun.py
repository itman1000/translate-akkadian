"""EvaCun 並列コーパスを NMT 学習用に整形する。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from .align_train import normalize_transliteration, normalize_translation
from .utils import clean_text, count_sentence_endings


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def compute_avg_len(texts: Iterable[str]) -> float:
    total = 0
    count = 0
    for text in texts:
        count += 1
        total += len(text or "")
    return total / max(count, 1)


def _resolve_split_paths(
    data_dir: Path,
    src_lang: str,
    *,
    split: str,
    src_override: Optional[str],
    tgt_override: Optional[str],
) -> Tuple[Path, Path]:
    src_lang = src_lang.strip().lower()
    if src_lang not in {"transcription", "akkadian"}:
        raise ValueError("src_lang must be 'transcription' or 'akkadian'")

    src_name = f"{src_lang}_{split}.txt"
    tgt_name = f"english_{split}.txt"

    src_path = Path(src_override) if src_override else data_dir / src_name
    tgt_path = Path(tgt_override) if tgt_override else data_dir / tgt_name
    return src_path, tgt_path


def build_df(
    src_lines: List[str],
    tgt_lines: List[str],
    *,
    normalize: bool,
    variant: str,
    source_label: str,
    split_label: str,
    max_rows: Optional[int],
    max_src_chars: Optional[int],
    max_tgt_chars: Optional[int],
    max_sentence_endings: Optional[int],
) -> Tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    dropped_empty = 0
    dropped_len = 0
    dropped_sentence = 0

    limit = max_rows if max_rows and max_rows > 0 else None
    for idx, (src_raw, tgt_raw) in enumerate(zip(src_lines, tgt_lines)):
        if limit is not None and idx >= limit:
            break

        src = src_raw.strip()
        tgt = tgt_raw.strip()

        if normalize:
            src = normalize_transliteration(src, variant)
            tgt = normalize_translation(tgt)
        else:
            src = clean_text(src)
            tgt = clean_text(tgt)

        if not src or not tgt:
            dropped_empty += 1
            continue

        if max_src_chars is not None and len(src) > max_src_chars:
            dropped_len += 1
            continue
        if max_tgt_chars is not None and len(tgt) > max_tgt_chars:
            dropped_len += 1
            continue

        if max_sentence_endings is not None:
            if count_sentence_endings(tgt) > max_sentence_endings:
                dropped_sentence += 1
                continue

        rows.append(
            {
                "oare_id": f"evacun_{split_label}_{idx:07d}",
                "src_sent": src,
                "tgt_sent": tgt,
                "source": source_label,
                "src_norm_variant": variant if normalize else "",
            }
        )

    stats = {
        "dropped_empty": dropped_empty,
        "dropped_len": dropped_len,
        "dropped_sentence": dropped_sentence,
    }
    return pd.DataFrame(rows), stats


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def log_stats(label: str, df: pd.DataFrame, stats: dict[str, int], total: int) -> None:
    kept = len(df)
    avg_src = compute_avg_len(df.get("src_sent", []))
    avg_tgt = compute_avg_len(df.get("tgt_sent", []))
    print(
        f"[{label}] total={total} kept={kept} "
        f"drop_empty={stats.get('dropped_empty', 0)} "
        f"drop_len={stats.get('dropped_len', 0)} "
        f"drop_sentence={stats.get('dropped_sentence', 0)} "
        f"avg_src_len={avg_src:.1f} avg_tgt_len={avg_tgt:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EvaCun parallel data for NMT training.")
    parser.add_argument(
        "--data-dir",
        default="data/evacun_oracc_parallel_v0.1",
        help="EvaCun data directory.",
    )
    parser.add_argument(
        "--src-lang",
        default="transcription",
        choices=["transcription", "akkadian"],
        help="Source language file prefix to use.",
    )
    parser.add_argument("--train-src", default=None, help="Optional train source path.")
    parser.add_argument("--train-tgt", default=None, help="Optional train target path.")
    parser.add_argument("--val-src", default=None, help="Optional validation source path.")
    parser.add_argument("--val-tgt", default=None, help="Optional validation target path.")
    parser.add_argument(
        "--out-dir",
        default="artifacts/evacun",
        help="Output directory for prepared files.",
    )
    parser.add_argument("--out-train", default=None, help="Optional output path for train file.")
    parser.add_argument("--out-val", default=None, help="Optional output path for val file.")
    parser.add_argument("--variant", default="C", help="Normalization variant (A/B/C).")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable normalization (keep raw text).",
    )
    parser.add_argument("--source-label", default="evacun", help="Value for source column.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Use first N train rows.")
    parser.add_argument("--max-val-rows", type=int, default=None, help="Use first N val rows.")
    parser.add_argument("--max-src-chars", type=int, default=None, help="Max source chars.")
    parser.add_argument("--max-tgt-chars", type=int, default=None, help="Max target chars.")
    parser.add_argument(
        "--max-sentence-endings",
        type=int,
        default=None,
        help="Drop rows with more than N sentence endings in target.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    variant = str(args.variant or "C").strip().upper()
    normalize = not bool(args.no_normalize)
    source_label = str(args.source_label).strip() or "evacun"
    if normalize and variant not in {"A", "B", "C"}:
        raise ValueError("variant must be A/B/C when normalization is enabled")

    train_src_path, train_tgt_path = _resolve_split_paths(
        data_dir,
        args.src_lang,
        split="train",
        src_override=args.train_src,
        tgt_override=args.train_tgt,
    )
    val_src_path, val_tgt_path = _resolve_split_paths(
        data_dir,
        args.src_lang,
        split="validation",
        src_override=args.val_src,
        tgt_override=args.val_tgt,
    )

    train_src = read_lines(train_src_path)
    train_tgt = read_lines(train_tgt_path)
    if len(train_src) != len(train_tgt):
        raise ValueError(
            f"Train line count mismatch: {train_src_path}={len(train_src)} "
            f"{train_tgt_path}={len(train_tgt)}"
        )

    val_src = read_lines(val_src_path)
    val_tgt = read_lines(val_tgt_path)
    if len(val_src) != len(val_tgt):
        raise ValueError(
            f"Val line count mismatch: {val_src_path}={len(val_src)} "
            f"{val_tgt_path}={len(val_tgt)}"
        )

    train_df, train_stats = build_df(
        train_src,
        train_tgt,
        normalize=normalize,
        variant=variant,
        source_label=source_label,
        split_label="train",
        max_rows=args.max_train_rows,
        max_src_chars=args.max_src_chars,
        max_tgt_chars=args.max_tgt_chars,
        max_sentence_endings=args.max_sentence_endings,
    )

    val_df, val_stats = build_df(
        val_src,
        val_tgt,
        normalize=normalize,
        variant=variant,
        source_label=source_label,
        split_label="val",
        max_rows=args.max_val_rows,
        max_src_chars=args.max_src_chars,
        max_tgt_chars=args.max_tgt_chars,
        max_sentence_endings=args.max_sentence_endings,
    )

    out_dir = Path(args.out_dir)
    out_train_path = (
        Path(args.out_train)
        if args.out_train
        else out_dir / f"evacun_{args.src_lang}_train.parquet"
    )
    out_val_path = (
        Path(args.out_val) if args.out_val else out_dir / f"evacun_{args.src_lang}_val.parquet"
    )

    write_table(train_df, out_train_path)
    write_table(val_df, out_val_path)

    log_stats("train", train_df, train_stats, len(train_src))
    log_stats("val", val_df, val_stats, len(val_src))
    print(f"Saved train: {out_train_path}")
    print(f"Saved val: {out_val_path}")


if __name__ == "__main__":
    main()
