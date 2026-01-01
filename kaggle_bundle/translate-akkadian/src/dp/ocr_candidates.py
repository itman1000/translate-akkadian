"""publications.csv から OCR 候補行を抽出する。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .ocr_utils import (
    EnglishThresholds,
    TranslitThresholds,
    flatten_block,
    is_english_like,
    is_translit_like,
    split_blocks,
)
from .utils import get_artifacts_dir, get_data_dir, load_config


def _iter_chunks(path: Path, usecols: List[str], chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(path, usecols=usecols, chunksize=chunksize, engine="python")


def _get_int(cfg: Dict[str, object], key: str, default: int) -> int:
    value = cfg.get(key, default)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _get_float(cfg: Dict[str, object], key: str, default: float) -> float:
    value = cfg.get(key, default)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _resolve_out(out_path: Path) -> Tuple[Path, str]:
    if out_path.suffix:
        return out_path.parent, out_path.stem
    return out_path, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR candidate rows from publications.csv.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--input", default=None, help="Input publications.csv path.")
    parser.add_argument("--out", default=None, help="Output path or directory.")
    parser.add_argument("--text-col", default=None, help="Text column name (default: page_text).")
    parser.add_argument("--has-akkadian-col", default=None, help="has_akkadian column name.")
    parser.add_argument("--chunksize", type=int, default=None, help="Chunk size.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use first N rows for quick runs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parts.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, None)
    artifacts_dir = get_artifacts_dir(cfg)

    input_path = Path(args.input) if args.input else data_dir / "publications.csv"
    out_path = Path(args.out) if args.out else artifacts_dir / "ocr" / "publications_candidates.parquet"
    out_dir, prefix = _resolve_out(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    text_col = args.text_col or str(cfg.get("ocr_text_col", "page_text"))
    has_col = args.has_akkadian_col or str(cfg.get("ocr_has_akkadian_col", "has_akkadian"))
    chunksize = args.chunksize or _get_int(cfg, "ocr_candidates_chunksize", 2000)

    min_eng = _get_int(cfg, "ocr_candidates_min_english_paragraphs", 1)
    min_tr = _get_int(cfg, "ocr_candidates_min_translit_paragraphs", 1)

    eng_thr = EnglishThresholds(
        min_chars=_get_int(cfg, "ocr_english_min_chars", 40),
        min_words=_get_int(cfg, "ocr_english_min_words", 5),
        min_alpha_ratio=_get_float(cfg, "ocr_english_min_alpha_ratio", 0.6),
        max_symbol_ratio=_get_float(cfg, "ocr_english_max_symbol_ratio", 0.25),
        max_digit_ratio=_get_float(cfg, "ocr_english_max_digit_ratio", 0.2),
        min_stopword_hits=_get_int(cfg, "ocr_english_min_stopword_hits", 2),
        max_upper_ratio=_get_float(cfg, "ocr_english_max_upper_ratio", 0.6),
    )
    tr_thr = TranslitThresholds(
        min_chars=_get_int(cfg, "ocr_translit_min_chars", 30),
        min_tokens=_get_int(cfg, "ocr_translit_min_tokens", 3),
        min_marker_ratio=_get_float(cfg, "ocr_translit_min_marker_ratio", 0.25),
        min_upper_ratio=_get_float(cfg, "ocr_translit_min_upper_ratio", 0.25),
        max_symbol_ratio=_get_float(cfg, "ocr_translit_max_symbol_ratio", 0.6),
        max_stopword_hits=_get_int(cfg, "ocr_translit_max_stopword_hits", 1),
    )

    base_cols = ["pdf_name", "page", text_col]
    if has_col:
        base_cols.append(has_col)

    stats: Counter[str] = Counter()
    part_idx = 0
    total_rows = 0
    kept_rows = 0

    for chunk in _iter_chunks(input_path, usecols=base_cols, chunksize=chunksize):
        if args.max_rows is not None and total_rows >= args.max_rows:
            break
        if args.max_rows is not None:
            chunk = chunk.head(max(0, args.max_rows - total_rows))

        total_rows += len(chunk)
        if has_col in chunk.columns:
            chunk = chunk[chunk[has_col] == True].reset_index(drop=True)  # noqa: E712
            stats["rows_after_has_akkadian"] += len(chunk)
        else:
            stats["rows_missing_has_akkadian"] += len(chunk)

        out_rows: List[Dict[str, object]] = []
        for row in chunk.itertuples(index=False):
            row_dict = row._asdict()
            raw_text = str(row_dict.get(text_col, "") or "")
            if not raw_text.strip():
                stats["rows_empty_text"] += 1
                continue

            blocks = split_blocks(raw_text)
            if not blocks:
                stats["rows_empty_blocks"] += 1
                continue

            eng_count = 0
            tr_count = 0
            eng_score_max = 0.0
            tr_score_max = 0.0
            for block in blocks:
                flat = flatten_block(block)
                eng_ok, eng_metrics = is_english_like(flat, eng_thr)
                tr_ok, tr_metrics = is_translit_like(flat, tr_thr)
                if eng_ok and not tr_ok:
                    eng_count += 1
                    eng_score_max = max(eng_score_max, float(eng_metrics.get("score", 0.0)))
                elif tr_ok and not eng_ok:
                    tr_count += 1
                    tr_score_max = max(tr_score_max, float(tr_metrics.get("score", 0.0)))

            if eng_count < min_eng:
                stats["rows_drop_no_english"] += 1
                continue
            if tr_count < min_tr:
                stats["rows_drop_no_translit"] += 1
                continue

            out_row: Dict[str, object] = {
                "pdf_name": row_dict.get("pdf_name", ""),
                "page": row_dict.get("page", ""),
                "has_akkadian": row_dict.get(has_col, ""),
                "page_text": raw_text,
                "paragraphs_total": len(blocks),
                "english_paragraphs": eng_count,
                "translit_paragraphs": tr_count,
                "english_score_max": round(eng_score_max, 6),
                "translit_score_max": round(tr_score_max, 6),
            }
            out_rows.append(out_row)

        part_name = f"part-{part_idx:04d}.parquet" if not prefix else f"{prefix}-part-{part_idx:04d}.parquet"
        part_path = out_dir / part_name
        if part_path.exists() and not args.overwrite:
            stats["parts_skipped"] += 1
            part_idx += 1
            continue

        if out_rows:
            pd.DataFrame(out_rows).to_parquet(part_path, index=False)
            kept_rows += len(out_rows)
            stats["parts_written"] += 1
        else:
            stats["parts_empty"] += 1
        part_idx += 1

    summary = {
        "input_rows": total_rows,
        "output_rows": kept_rows,
        "stats": dict(stats),
        "english_thresholds": eng_thr.__dict__,
        "translit_thresholds": tr_thr.__dict__,
        "min_english_paragraphs": min_eng,
        "min_translit_paragraphs": min_tr,
    }
    summary_name = f"{prefix}_stats.json" if prefix else "stats.json"
    summary_path = out_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Input rows: {total_rows}")
    print(f"Output rows: {kept_rows}")
    print(f"Saved parts: {out_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
