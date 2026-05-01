"""publications.csv から metadata-first で OCR 候補行を抽出する。"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import pandas as pd

from .ocr_utils import (
    EnglishThresholds,
    TranslationThresholds,
    TranslitThresholds,
    dedupe_preserve_order,
    flatten_block,
    is_english_like,
    is_translation_like,
    is_translit_like,
    lookup_tokens,
    split_blocks,
    split_metadata_field,
)
from .utils import get_artifacts_dir, get_data_dir, load_config


@dataclass(frozen=True)
class MetadataKey:
    oare_id: str
    field: str
    raw_text: str
    normalized: str
    tokens: Tuple[str, ...]
    signature: str


_ALNUM_RE = re.compile(r"(?=.*[A-Z])(?=.*\d)")


def _iter_chunks(path: Path, usecols: List[str], chunksize: int) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]
            if not members:
                raise FileNotFoundError(f"No CSV found in ZIP: {path}")
            with zf.open(members[0]) as handle:
                yield from pd.read_csv(handle, usecols=usecols, chunksize=chunksize, engine="python")
        return
    yield from pd.read_csv(path, usecols=usecols, chunksize=chunksize, engine="python")


def _read_columns(path: Path) -> List[str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]
            if not members:
                raise FileNotFoundError(f"No CSV found in ZIP: {path}")
            with zf.open(members[0]) as handle:
                return pd.read_csv(handle, nrows=0, engine="python").columns.tolist()
    return pd.read_csv(path, nrows=0, engine="python").columns.tolist()


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


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _parse_fields(cfg: Dict[str, object], key: str, default: str) -> List[str]:
    raw = str(cfg.get(key, default) or default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _pick_column(preferred: List[str], available: List[str], fallback: str) -> str:
    available_set = set(available)
    for name in preferred:
        if name and name in available_set:
            return name
    return fallback


def _is_identifier_phrase(text: str) -> bool:
    tokens = lookup_tokens(text)
    if not tokens:
        return False
    if len(tokens) > 6:
        return False
    if len(tokens) == 1 and tokens[0].isdigit():
        return False
    has_digit = any(any(ch.isdigit() for ch in token) for token in tokens)
    if not has_digit:
        return False
    if len(tokens) <= 2 and not any(_ALNUM_RE.search(token) for token in tokens):
        return False
    return True


def _build_metadata_index(published_df: pd.DataFrame, fields: List[str]) -> Tuple[List[MetadataKey], DefaultDict[str, List[int]]]:
    keys: List[MetadataKey] = []
    token_freq: Counter[str] = Counter()

    for row in published_df.itertuples(index=False):
        row_dict = row._asdict()
        oare_id = str(row_dict.get("oare_id", "") or "").strip()
        if not oare_id:
            continue

        per_doc_seen: set[Tuple[str, str]] = set()
        per_doc_keys: List[Tuple[str, str, Tuple[str, ...]]] = []
        for field in fields:
            value = str(row_dict.get(field, "") or "").strip()
            if not value:
                continue
            candidates = split_metadata_field(value) if field in {"aliases", "publication_catalog", "inventory_position", "note"} else [value]
            for raw in candidates:
                if not raw:
                    continue
                if not _is_identifier_phrase(raw):
                    continue
                tokens = tuple(lookup_tokens(raw))
                if not tokens:
                    continue
                key = (field, " ".join(tokens))
                if key in per_doc_seen:
                    continue
                per_doc_seen.add(key)
                per_doc_keys.append((field, raw, tokens))
                token_freq.update(set(tokens))

        for field, raw, tokens in per_doc_keys:
            non_numeric = [token for token in tokens if not token.isdigit()]
            signature = min(non_numeric or list(tokens), key=lambda tok: (token_freq[tok], len(tok), tok))
            keys.append(
                MetadataKey(
                    oare_id=oare_id,
                    field=field,
                    raw_text=raw,
                    normalized=" ".join(tokens),
                    tokens=tokens,
                    signature=signature,
                )
            )

    by_signature: DefaultDict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(keys):
        by_signature[item.signature].append(idx)
    return keys, by_signature


def _match_metadata_hits(
    page_text: str,
    pdf_name: str,
    keys: List[MetadataKey],
    by_signature: DefaultDict[str, List[int]],
    top_k: int,
) -> Dict[str, object]:
    page_tokens = set(lookup_tokens(page_text))
    pdf_tokens = set(lookup_tokens(pdf_name))
    all_tokens = page_tokens.union(pdf_tokens)

    candidate_indices: set[int] = set()
    for token in all_tokens:
        candidate_indices.update(by_signature.get(token, []))

    best_by_oare: Dict[str, Dict[str, object]] = {}
    for idx in candidate_indices:
        key = keys[idx]
        key_set = set(key.tokens)
        page_hit = key_set.issubset(page_tokens)
        pdf_hit = key_set.issubset(pdf_tokens)
        if not (page_hit or pdf_hit):
            continue
        if len(key.tokens) <= 2 and not pdf_hit and key.field != "cdli_id":
            continue

        score = 0.0
        if pdf_hit:
            score += 1.0
        if page_hit:
            score += 0.75
        if key.field == "cdli_id":
            score += 0.75
        if len(key.tokens) >= 3:
            score += 0.15
        if any(_ALNUM_RE.search(token) for token in key.tokens):
            score += 0.1

        previous = best_by_oare.get(key.oare_id)
        if previous is None or score > float(previous["score"]):
            best_by_oare[key.oare_id] = {
                "oare_id": key.oare_id,
                "score": score,
                "field": key.field,
                "key": key.raw_text,
                "pdf_hit": pdf_hit,
                "page_hit": page_hit,
            }

    ranked = sorted(best_by_oare.values(), key=lambda item: (-float(item["score"]), str(item["oare_id"])))[:top_k]
    top_score = float(ranked[0]["score"]) if ranked else 0.0
    second_score = float(ranked[1]["score"]) if len(ranked) >= 2 else 0.0

    return {
        "matched_oare_ids": "|".join(str(item["oare_id"]) for item in ranked),
        "matched_oare_scores": "|".join(f"{float(item['score']):.6f}" for item in ranked),
        "matched_oare_fields": "|".join(str(item["field"]) for item in ranked),
        "matched_oare_keys": "|".join(str(item["key"]) for item in ranked),
        "metadata_hit_count": len(ranked),
        "metadata_top_score": round(top_score, 6),
        "metadata_second_score": round(second_score, 6),
        "metadata_score_gap": round(top_score - second_score, 6),
        "metadata_pdf_hits": int(sum(1 for item in ranked if item["pdf_hit"])),
        "metadata_page_hits": int(sum(1 for item in ranked if item["page_hit"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR candidate rows from publications.csv.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--input", default=None, help="Input publications.csv path.")
    parser.add_argument("--out", default=None, help="Output path or directory.")
    parser.add_argument("--text-col", default=None, help="Text column name (default: page_text).")
    parser.add_argument("--translation-text-col", default=None, help="Translation-side text column name.")
    parser.add_argument("--translit-text-col", default=None, help="Transliteration-side text column name.")
    parser.add_argument("--has-akkadian-col", default=None, help="has_akkadian column name.")
    parser.add_argument("--published-texts", default=None, help="published_texts.csv path.")
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

    available_cols = _read_columns(input_path)

    text_col = args.text_col or str(cfg.get("ocr_text_col", ""))
    text_col = _pick_column([text_col, "page_text_clean", "page_text"], available_cols, "page_text")
    translation_text_col = args.translation_text_col or str(cfg.get("ocr_translation_text_col", ""))
    translation_text_col = _pick_column([translation_text_col, "page_text_clean", text_col], available_cols, text_col)
    translit_text_col = args.translit_text_col or str(cfg.get("ocr_translit_text_col", ""))
    translit_text_col = _pick_column([translit_text_col, "page_text_translit", text_col], available_cols, text_col)
    has_col = args.has_akkadian_col or str(cfg.get("ocr_has_akkadian_col", "has_akkadian"))
    chunksize = args.chunksize or _get_int(cfg, "ocr_candidates_chunksize", 2000)
    published_texts_path = Path(args.published_texts) if args.published_texts else data_dir / "published_texts.csv"

    metadata_fields = _parse_fields(
        cfg,
        "ocr_metadata_fields",
        "aliases,publication_catalog,inventory_position,label,cdli_id,note",
    )
    metadata_top_k = _get_int(cfg, "ocr_metadata_top_k", 8)
    metadata_min_score = _get_float(cfg, "ocr_candidates_metadata_min_score", 0.95)
    metadata_only_min_score = _get_float(cfg, "ocr_candidates_metadata_only_min_score", 1.25)
    metadata_only_min_gap = _get_float(cfg, "ocr_candidates_metadata_only_min_gap", 0.10)
    allow_single_side_metadata = bool(cfg.get("ocr_candidates_allow_single_side_metadata", True))

    min_translation = _get_int(
        cfg,
        "ocr_candidates_min_translation_paragraphs",
        _get_int(cfg, "ocr_candidates_min_english_paragraphs", 1),
    )
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
    translation_thr = TranslationThresholds(
        min_chars=_get_int(cfg, "ocr_translation_min_chars", 40),
        min_words=_get_int(cfg, "ocr_translation_min_words", 5),
        min_alpha_ratio=_get_float(cfg, "ocr_translation_min_alpha_ratio", 0.6),
        max_symbol_ratio=_get_float(cfg, "ocr_translation_max_symbol_ratio", 0.25),
        max_digit_ratio=_get_float(cfg, "ocr_translation_max_digit_ratio", 0.2),
        min_best_stopword_hits=_get_int(cfg, "ocr_translation_min_best_stopword_hits", 1),
        max_upper_ratio=_get_float(cfg, "ocr_translation_max_upper_ratio", 0.6),
    )

    published_df = pd.read_csv(published_texts_path, dtype=str).fillna("")
    metadata_keys, by_signature = _build_metadata_index(published_df, metadata_fields)

    base_cols = dedupe_preserve_order(["pdf_name", "page", text_col, translation_text_col, translit_text_col])
    if has_col and has_col in available_cols:
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
        out_rows: List[Dict[str, object]] = []
        for row in chunk.itertuples(index=False):
            row_dict = row._asdict()
            raw_text = str(row_dict.get(text_col, "") or "")
            translation_text = str(row_dict.get(translation_text_col, raw_text) or "")
            translit_text = str(row_dict.get(translit_text_col, raw_text) or "")
            if not raw_text.strip():
                stats["rows_empty_text"] += 1
                continue

            has_akkadian = _safe_bool(row_dict.get(has_col, False)) if has_col else False
            translation_blocks = split_blocks(translation_text)
            translit_blocks = split_blocks(translit_text)
            if not translation_blocks and not translit_blocks:
                stats["rows_empty_blocks"] += 1
                continue

            eng_count = 0
            translation_count = 0
            translation_non_english_count = 0
            tr_count = 0
            eng_score_max = 0.0
            translation_score_max = 0.0
            non_english_score_max = 0.0
            tr_score_max = 0.0
            lang_counter: Counter[str] = Counter()

            for block in translation_blocks:
                flat = flatten_block(block)
                eng_ok, eng_metrics = is_english_like(flat, eng_thr)
                trans_ok, trans_metrics = is_translation_like(flat, translation_thr)
                tr_ok, tr_metrics = is_translit_like(flat, tr_thr)
                if eng_ok:
                    eng_count += 1
                    eng_score_max = max(eng_score_max, float(eng_metrics.get("score", 0.0)))
                if trans_ok and not (tr_ok and float(tr_metrics.get("marker_ratio", 0.0)) >= 0.35):
                    translation_count += 1
                    translation_score_max = max(translation_score_max, float(trans_metrics.get("score", 0.0)))
                    lang = str(trans_metrics.get("lang", "unknown") or "unknown")
                    lang_counter[lang] += 1
                    if lang != "en":
                        translation_non_english_count += 1
                        non_english_score_max = max(non_english_score_max, float(trans_metrics.get("score", 0.0)))

            for block in translit_blocks:
                flat = flatten_block(block)
                eng_ok, _ = is_english_like(flat, eng_thr)
                trans_ok, trans_metrics = is_translation_like(flat, translation_thr)
                tr_ok, tr_metrics = is_translit_like(flat, tr_thr)
                if tr_ok and not eng_ok and not trans_ok:
                    tr_count += 1
                    tr_score_max = max(tr_score_max, float(tr_metrics.get("score", 0.0)))
                elif tr_ok and trans_ok and float(tr_metrics.get("marker_ratio", 0.0)) >= 0.35:
                    tr_count += 1
                    tr_score_max = max(tr_score_max, float(tr_metrics.get("score", 0.0)))

            meta = _match_metadata_hits(raw_text, str(row_dict.get("pdf_name", "") or ""), metadata_keys, by_signature, metadata_top_k)
            has_blocks = translation_count >= min_translation and tr_count >= min_tr
            has_strong_metadata = float(meta["metadata_top_score"]) >= metadata_min_score
            allow_metadata_only = (
                allow_single_side_metadata
                and float(meta["metadata_top_score"]) >= metadata_only_min_score
                and float(meta["metadata_score_gap"]) >= metadata_only_min_gap
                and translation_count >= min_translation
                and max(translation_count, tr_count) >= 1
            )

            keep = False
            reason = ""
            if has_blocks and (has_akkadian or has_strong_metadata):
                keep = True
                reason = "metadata_and_blocks" if has_strong_metadata else "blocks_only"
            elif has_blocks:
                keep = True
                reason = "blocks_only"
            elif allow_metadata_only and (translation_count >= min_translation or tr_count >= min_tr):
                keep = True
                reason = "metadata_only"

            if not keep:
                if translation_count < min_translation:
                    stats["rows_drop_no_translation"] += 1
                elif tr_count < min_tr:
                    stats["rows_drop_no_translit"] += 1
                else:
                    stats["rows_drop_gate"] += 1
                continue

            if translation_count > 0 and eng_count == 0:
                stats["rows_translation_non_english_only"] += 1
            if translation_non_english_count > 0:
                stats["rows_with_non_english_translation"] += 1
            for lang, count in lang_counter.items():
                stats[f"translation_lang_{lang}"] += count

            lang_items = sorted(lang_counter.items(), key=lambda item: (-item[1], item[0]))
            lang_top = lang_items[0][0] if lang_items else "unknown"
            lang_counts = "|".join(f"{lang}:{count}" for lang, count in lang_items)

            out_row: Dict[str, object] = {
                "pdf_name": row_dict.get("pdf_name", ""),
                "page": row_dict.get("page", ""),
                "has_akkadian": has_akkadian,
                "page_text": raw_text,
                "page_text_clean": translation_text,
                "page_text_translit": translit_text,
                "paragraphs_total": max(len(translation_blocks), len(translit_blocks)),
                "english_paragraphs": eng_count,
                "translation_like_paragraphs": translation_count,
                "translation_non_english_paragraphs": translation_non_english_count,
                "translit_paragraphs": tr_count,
                "english_score_max": round(eng_score_max, 6),
                "translation_like_score_max": round(translation_score_max, 6),
                "non_english_translation_score_max": round(non_english_score_max, 6),
                "translit_score_max": round(tr_score_max, 6),
                "translation_lang_top": lang_top,
                "translation_lang_counts": lang_counts,
                "translation_text_col": translation_text_col,
                "translit_text_col": translit_text_col,
                "candidate_reason": reason,
                **meta,
            }
            out_rows.append(out_row)
            kept_rows += 1
            stats[f"reason_{reason}"] += 1
            if float(meta["metadata_top_score"]) >= metadata_min_score:
                stats["rows_with_metadata"] += 1

        part_name = f"part-{part_idx:04d}.parquet" if not prefix else f"{prefix}-part-{part_idx:04d}.parquet"
        part_path = out_dir / part_name
        if part_path.exists() and not args.overwrite:
            stats["parts_skipped"] += 1
            part_idx += 1
            continue

        if out_rows:
            pd.DataFrame(out_rows).to_parquet(part_path, index=False)
            stats["parts_written"] += 1
        else:
            stats["parts_empty"] += 1
        part_idx += 1

    summary = {
        "input_rows": total_rows,
        "output_rows": kept_rows,
        "stats": dict(stats),
        "english_thresholds": eng_thr.__dict__,
        "translation_thresholds": translation_thr.__dict__,
        "translit_thresholds": tr_thr.__dict__,
        "min_translation_paragraphs": min_translation,
        "min_translit_paragraphs": min_tr,
        "metadata_fields": metadata_fields,
        "metadata_keys": len(metadata_keys),
        "metadata_top_k": metadata_top_k,
        "metadata_min_score": metadata_min_score,
        "metadata_only_min_score": metadata_only_min_score,
        "metadata_only_min_gap": metadata_only_min_gap,
        "allow_single_side_metadata": allow_single_side_metadata,
        "text_col": text_col,
        "translation_text_col": translation_text_col,
        "translit_text_col": translit_text_col,
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
