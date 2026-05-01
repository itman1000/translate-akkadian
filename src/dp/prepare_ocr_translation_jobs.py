"""OCR の非英語訳ブロックを抽出し、英語化ジョブを作る。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .ocr_pairs import _get_float, _get_int, _list_candidate_parts
from .ocr_utils import (
    EnglishThresholds,
    TranslationThresholds,
    TranslitThresholds,
    flatten_block,
    is_english_like,
    is_translation_like,
    is_translit_like,
    split_blocks,
    stable_block_hash,
)
from .utils import get_artifacts_dir, load_config


def _prompt(lang_name: str, text: str) -> str:
    return (
        f"Translate the following {lang_name} scholarly translation passage into English. "
        "Preserve names, numbers, brackets, quotes, and punctuation. "
        "Do not add explanations or commentary.\n\n"
        f"Text:\n{text}\n\nEnglish:"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare non-English OCR translation jobs.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--candidates", default=None, help="Candidates path or directory.")
    parser.add_argument("--out", default=None, help="Output csv/parquet path.")
    parser.add_argument("--langs", default=None, help="Comma separated language filter, e.g. fr,de,tr,other.")
    parser.add_argument("--max-parts", type=int, default=None, help="Limit number of candidate parts.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of candidate rows.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = get_artifacts_dir(cfg)
    candidates_path = (
        Path(args.candidates)
        if args.candidates
        else Path(cfg.get("ocr_candidates_out", artifacts_dir / "ocr" / "publications_candidates.parquet"))
    )
    out_path = (
        Path(args.out)
        if args.out
        else Path(cfg.get("ocr_translation_jobs_out", artifacts_dir / "ocr" / "translation_jobs.parquet"))
    )

    lang_filter = {part.strip() for part in str(args.langs or cfg.get("ocr_translation_langs", "fr,de,tr,other,unknown")).split(",") if part.strip()}

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
    translation_thr = TranslationThresholds(
        min_chars=_get_int(cfg, "ocr_translation_min_chars", 40),
        min_words=_get_int(cfg, "ocr_translation_min_words", 5),
        min_alpha_ratio=_get_float(cfg, "ocr_translation_min_alpha_ratio", 0.6),
        max_symbol_ratio=_get_float(cfg, "ocr_translation_max_symbol_ratio", 0.25),
        max_digit_ratio=_get_float(cfg, "ocr_translation_max_digit_ratio", 0.2),
        min_best_stopword_hits=_get_int(cfg, "ocr_translation_min_best_stopword_hits", 1),
        max_upper_ratio=_get_float(cfg, "ocr_translation_max_upper_ratio", 0.6),
    )

    parts = _list_candidate_parts(candidates_path)
    if args.max_parts is not None:
        parts = parts[: args.max_parts]
    if not parts:
        raise FileNotFoundError(f"No candidate parts found at: {candidates_path}")

    rows_seen = 0
    stats: Counter[str] = Counter()
    jobs: Dict[str, Dict[str, object]] = {}

    for part_path in parts:
        df = pd.read_parquet(part_path)
        if args.max_rows is not None and rows_seen >= args.max_rows:
            break
        if args.max_rows is not None:
            df = df.head(max(0, args.max_rows - rows_seen))

        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            rows_seen += 1
            page_text = str(row_dict.get("page_text_clean", row_dict.get("page_text", "")) or "")
            if not page_text.strip():
                continue
            pdf_name = str(row_dict.get("pdf_name", "") or "")
            page = str(row_dict.get("page", "") or "")
            candidate_reason = str(row_dict.get("candidate_reason", "") or "")
            metadata_top_score = float(row_dict.get("metadata_top_score", 0.0) or 0.0)
            for block_idx, block in enumerate(split_blocks(page_text)):
                flat = flatten_block(block)
                eng_ok, _ = is_english_like(flat, eng_thr)
                trans_ok, trans_metrics = is_translation_like(flat, translation_thr)
                tr_ok, tr_metrics = is_translit_like(flat, tr_thr)
                treat_as_translit = False
                if tr_ok and not eng_ok and not trans_ok:
                    treat_as_translit = True
                elif tr_ok and trans_ok and float(tr_metrics.get("marker_ratio", 0.0)) >= 0.35:
                    treat_as_translit = True
                if not trans_ok or treat_as_translit:
                    continue
                lang = str(trans_metrics.get("lang", "unknown") or "unknown")
                if lang == "en":
                    continue
                if lang_filter and lang not in lang_filter:
                    continue
                block_hash = stable_block_hash(flat)
                priority = float(trans_metrics.get("score", 0.0)) + 0.15 * metadata_top_score
                job = jobs.get(block_hash)
                if job is None:
                    jobs[block_hash] = {
                        "block_hash": block_hash,
                        "lang": lang,
                        "lang_name": str(trans_metrics.get("lang_name", "Unknown") or "Unknown"),
                        "text": flat,
                        "translation_en": "",
                        "pdf_name": pdf_name,
                        "page": page,
                        "occurrences": 1,
                        "metadata_top_score_max": metadata_top_score,
                        "priority_score": round(priority, 6),
                        "candidate_reasons": candidate_reason,
                        "sample_block_idx": block_idx,
                        "translation_prompt": _prompt(str(trans_metrics.get("lang_name", "Unknown") or "Unknown"), flat),
                    }
                else:
                    job["occurrences"] = int(job.get("occurrences", 0)) + 1
                    job["metadata_top_score_max"] = max(float(job.get("metadata_top_score_max", 0.0)), metadata_top_score)
                    job["priority_score"] = round(max(float(job.get("priority_score", 0.0)), priority), 6)
                    reasons = [part.strip() for part in str(job.get("candidate_reasons", "")).split("|") if part.strip()]
                    if candidate_reason and candidate_reason not in reasons:
                        reasons.append(candidate_reason)
                    job["candidate_reasons"] = "|".join(reasons)
                stats[f"lang_{lang}"] += 1

    out_df = pd.DataFrame(jobs.values())
    if not out_df.empty:
        out_df = out_df.sort_values(["priority_score", "occurrences", "metadata_top_score_max"], ascending=[False, False, False])
        out_df = out_df.reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        out_df.to_csv(out_path, index=False)
    else:
        out_df.to_parquet(out_path, index=False)

    summary_path = out_path.with_suffix(".stats.json")
    summary = {
        "rows_seen": rows_seen,
        "jobs": int(len(out_df)),
        "lang_filter": sorted(lang_filter),
        "stats": dict(stats),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Rows seen: {rows_seen}")
    print(f"Jobs: {len(out_df)}")
    print(f"Output: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
