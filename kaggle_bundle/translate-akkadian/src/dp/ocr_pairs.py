"""OCR candidates から文ペアを抽出する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .align_train import normalize_transliteration, normalize_translation, segment_source_tokens, split_english
from .ocr_utils import (
    EnglishThresholds,
    TranslitThresholds,
    anchor_overlap,
    count_sentence_endings,
    count_stopwords,
    flatten_block,
    is_english_like,
    is_translit_like,
    split_blocks,
    text_stats,
)
from .utils import clean_text, get_artifacts_dir, load_config


_LINE_NUM_RE = re.compile(r"^\s*\d{1,4}(?:['’]{1,2})?(?:[.)])?\s*")
_GAP_RE = re.compile(r"<\s*(gap|big_gap)\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class QualityGateConfig:
    min_len_ratio: float
    max_len_ratio: float
    min_src_tokens: int
    min_tgt_chars: int
    min_tgt_tokens: int
    max_tgt_tokens: int
    min_tgt_alpha_ratio: float
    max_tgt_symbol_ratio: float
    max_tgt_digit_ratio: float
    max_src_symbol_ratio: float
    max_tgt_sentence_endings: int
    min_stopword_hits: int


@dataclass(frozen=True)
class TierRule:
    name: str
    min_align_score: float
    min_len_ratio: float
    max_len_ratio: float
    min_tgt_alpha_ratio: float
    max_tgt_symbol_ratio: float
    min_stopword_hits: int


class RunningStats:
    def __init__(self, sample_size: int, rng: random.Random) -> None:
        self.sample_size = sample_size
        self.rng = rng
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self.sample: List[float] = []

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        if self.min_val is None or value < self.min_val:
            self.min_val = value
        if self.max_val is None or value > self.max_val:
            self.max_val = value
        if self.sample_size <= 0:
            return
        if len(self.sample) < self.sample_size:
            self.sample.append(value)
        else:
            idx = self.rng.randrange(self.count)
            if idx < self.sample_size:
                self.sample[idx] = value

    def summary(self) -> Dict[str, float | int]:
        if self.count == 0:
            return {"count": 0}
        mean = self.total / self.count
        var = self.total_sq / self.count - mean * mean
        std = (var if var > 0 else 0.0) ** 0.5
        summary: Dict[str, float | int] = {
            "count": self.count,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(self.min_val or 0.0, 6),
            "max": round(self.max_val or 0.0, 6),
        }
        if self.sample:
            sample_sorted = sorted(self.sample)
            summary.update(
                {
                    "p50": round(_quantile(sample_sorted, 0.5), 6),
                    "p90": round(_quantile(sample_sorted, 0.9), 6),
                    "p95": round(_quantile(sample_sorted, 0.95), 6),
                }
            )
        return summary


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    idx = int(round((len(values) - 1) * q))
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]


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


def _list_candidate_parts(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("part-*.parquet"))
    if path.exists():
        return [path]
    pattern = f"{path.stem}-part-*.parquet"
    return sorted(path.parent.glob(pattern))


def _extract_part_index(path: Path) -> Optional[int]:
    match = re.search(r"part-(\d+)\.parquet$", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"-part-(\d+)\.parquet$", path.name)
    if match:
        return int(match.group(1))
    return None


def _strip_line_numbers(block: str) -> str:
    lines: List[str] = []
    for line in block.splitlines():
        cleaned = _LINE_NUM_RE.sub("", line).strip()
        if cleaned:
            lines.append(cleaned)
    return " ".join(lines)


def _has_control_chars(text: str) -> bool:
    return any(ord(ch) < 32 and not ch.isspace() for ch in text)


def _has_broken_placeholder(text: str) -> bool:
    if "<" not in text and ">" not in text:
        return False
    stripped = _GAP_RE.sub("", text)
    return "<" in stripped or ">" in stripped


def _pair_hash(src: str, tgt: str) -> str:
    return hashlib.md5(f"{src}\t{tgt}".encode("utf-8")).hexdigest()


def _pair_score(src_text: str, tgt_text: str, w_len: float, w_anchor: float) -> float:
    src_len = len(src_text)
    tgt_len = len(tgt_text)
    len_score = min(src_len, tgt_len) / max(src_len, tgt_len, 1)
    overlap = anchor_overlap(src_text, tgt_text)
    return w_len * len_score + w_anchor * overlap


def _collect_blocks(
    page_text: str,
    eng_thr: EnglishThresholds,
    tr_thr: TranslitThresholds,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    english_blocks: List[Tuple[int, str]] = []
    translit_blocks: List[Tuple[int, str]] = []
    blocks = split_blocks(page_text)
    for idx, block in enumerate(blocks):
        flat = flatten_block(block)
        eng_ok, _ = is_english_like(flat, eng_thr)
        tr_ok, _ = is_translit_like(flat, tr_thr)
        if eng_ok and not tr_ok:
            english_blocks.append((idx, clean_text(flat)))
        elif tr_ok and not eng_ok:
            src_text = _strip_line_numbers(block)
            translit_blocks.append((idx, clean_text(src_text)))
    return english_blocks, translit_blocks


def _pair_blocks(
    english_blocks: List[Tuple[int, str]],
    translit_blocks: List[Tuple[int, str]],
    max_gap: int,
    pair_order: str,
    min_pair_score: float,
    w_len: float,
    w_anchor: float,
) -> List[Tuple[int, str, int, str, float]]:
    pairs: List[Tuple[int, str, int, str, float]] = []
    used_english: set[int] = set()
    for src_idx, src_text in translit_blocks:
        candidates: List[Tuple[float, int, str]] = []
        for eng_idx, eng_text in english_blocks:
            if eng_idx in used_english:
                continue
            gap = eng_idx - src_idx
            if pair_order == "src_then_tgt" and gap <= 0:
                continue
            if pair_order == "tgt_then_src" and gap >= 0:
                continue
            if abs(gap) > max_gap:
                continue
            score = _pair_score(src_text, eng_text, w_len, w_anchor)
            candidates.append((score, eng_idx, eng_text))
        if not candidates:
            continue
        best = max(candidates, key=lambda x: x[0])
        if best[0] < min_pair_score:
            continue
        used_english.add(best[1])
        pairs.append((src_idx, src_text, best[1], best[2], best[0]))
    return pairs


def _compute_flags(
    src_stats: Dict[str, float | int],
    tgt_stats: Dict[str, float | int],
    len_ratio: float,
    tgt_sentence_ends: int,
    stopword_hits: int,
    gate_cfg: QualityGateConfig,
    src_text: str,
    tgt_text: str,
) -> List[str]:
    flags: List[str] = []
    if int(src_stats["len"]) <= 0:
        flags.append("empty_src")
    if int(tgt_stats["len"]) <= 0:
        flags.append("empty_tgt")
    if len_ratio < gate_cfg.min_len_ratio or len_ratio > gate_cfg.max_len_ratio:
        flags.append("len_ratio")
    if int(src_stats["tokens"]) < gate_cfg.min_src_tokens:
        flags.append("short_src")
    if int(tgt_stats["len"]) < gate_cfg.min_tgt_chars:
        flags.append("short_tgt")
    if int(tgt_stats["tokens"]) < gate_cfg.min_tgt_tokens:
        flags.append("short_tgt_tokens")
    if int(tgt_stats["tokens"]) > gate_cfg.max_tgt_tokens:
        flags.append("long_tgt_tokens")
    if float(tgt_stats["alpha_ratio"]) < gate_cfg.min_tgt_alpha_ratio:
        flags.append("low_alpha_tgt")
    if float(tgt_stats["symbol_ratio"]) > gate_cfg.max_tgt_symbol_ratio:
        flags.append("symbol_heavy_tgt")
    if float(tgt_stats["digit_ratio"]) > gate_cfg.max_tgt_digit_ratio:
        flags.append("digit_heavy_tgt")
    if float(src_stats["symbol_ratio"]) > gate_cfg.max_src_symbol_ratio:
        flags.append("symbol_heavy_src")
    if tgt_sentence_ends > gate_cfg.max_tgt_sentence_endings:
        flags.append("multi_sentence_tgt")
    if stopword_hits < gate_cfg.min_stopword_hits:
        flags.append("low_stopword_tgt")
    if _has_broken_placeholder(tgt_text):
        flags.append("broken_placeholder")
    if _has_control_chars(src_text):
        flags.append("control_src")
    if _has_control_chars(tgt_text):
        flags.append("control_tgt")
    return flags


def _assign_tier(
    align_score: float,
    len_ratio: float,
    tgt_alpha_ratio: float,
    tgt_symbol_ratio: float,
    stopword_hits: int,
    high_rule: TierRule,
    med_rule: TierRule,
) -> str:
    if (
        align_score >= high_rule.min_align_score
        and high_rule.min_len_ratio <= len_ratio <= high_rule.max_len_ratio
        and tgt_alpha_ratio >= high_rule.min_tgt_alpha_ratio
        and tgt_symbol_ratio <= high_rule.max_tgt_symbol_ratio
        and stopword_hits >= high_rule.min_stopword_hits
    ):
        return high_rule.name
    if (
        align_score >= med_rule.min_align_score
        and med_rule.min_len_ratio <= len_ratio <= med_rule.max_len_ratio
        and tgt_alpha_ratio >= med_rule.min_tgt_alpha_ratio
        and tgt_symbol_ratio <= med_rule.max_tgt_symbol_ratio
        and stopword_hits >= med_rule.min_stopword_hits
    ):
        return med_rule.name
    return "low"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR sentence pairs from candidates.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--candidates", default=None, help="Candidates path or directory.")
    parser.add_argument("--out", default=None, help="Output directory for parts.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use first N rows for quick runs.")
    parser.add_argument("--max-parts", type=int, default=None, help="Limit number of parts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parts.")
    parser.add_argument("--no-dedupe-global", action="store_true", help="Disable global dedupe.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling stats.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = get_artifacts_dir(cfg)

    candidates_path = (
        Path(args.candidates)
        if args.candidates
        else Path(cfg.get("ocr_candidates_out", artifacts_dir / "ocr" / "publications_candidates.parquet"))
    )
    out_dir = (
        Path(args.out)
        if args.out
        else Path(cfg.get("ocr_pairs_out", artifacts_dir / "ocr_pairs"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

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

    gate_cfg = QualityGateConfig(
        min_len_ratio=_get_float(cfg, "ocr_gate_min_len_ratio", 0.25),
        max_len_ratio=_get_float(cfg, "ocr_gate_max_len_ratio", 4.0),
        min_src_tokens=_get_int(cfg, "ocr_gate_min_src_tokens", 3),
        min_tgt_chars=_get_int(cfg, "ocr_gate_min_tgt_chars", 20),
        min_tgt_tokens=_get_int(cfg, "ocr_gate_min_tgt_tokens", 3),
        max_tgt_tokens=_get_int(cfg, "ocr_gate_max_tgt_tokens", 80),
        min_tgt_alpha_ratio=_get_float(cfg, "ocr_gate_min_tgt_alpha_ratio", 0.3),
        max_tgt_symbol_ratio=_get_float(cfg, "ocr_gate_max_tgt_symbol_ratio", 0.35),
        max_tgt_digit_ratio=_get_float(cfg, "ocr_gate_max_tgt_digit_ratio", 0.3),
        max_src_symbol_ratio=_get_float(cfg, "ocr_gate_max_src_symbol_ratio", 0.6),
        max_tgt_sentence_endings=_get_int(cfg, "ocr_gate_max_tgt_sentence_endings", 1),
        min_stopword_hits=_get_int(cfg, "ocr_gate_min_stopword_hits", 1),
    )

    high_rule = TierRule(
        name="high",
        min_align_score=_get_float(cfg, "ocr_tier_high_min_align_score", 0.55),
        min_len_ratio=_get_float(cfg, "ocr_tier_high_min_len_ratio", 0.5),
        max_len_ratio=_get_float(cfg, "ocr_tier_high_max_len_ratio", 2.2),
        min_tgt_alpha_ratio=_get_float(cfg, "ocr_tier_high_min_tgt_alpha_ratio", 0.6),
        max_tgt_symbol_ratio=_get_float(cfg, "ocr_tier_high_max_tgt_symbol_ratio", 0.2),
        min_stopword_hits=_get_int(cfg, "ocr_tier_high_min_stopword_hits", 2),
    )
    med_rule = TierRule(
        name="med",
        min_align_score=_get_float(cfg, "ocr_tier_med_min_align_score", 0.4),
        min_len_ratio=_get_float(cfg, "ocr_tier_med_min_len_ratio", 0.35),
        max_len_ratio=_get_float(cfg, "ocr_tier_med_max_len_ratio", 3.0),
        min_tgt_alpha_ratio=_get_float(cfg, "ocr_tier_med_min_tgt_alpha_ratio", 0.5),
        max_tgt_symbol_ratio=_get_float(cfg, "ocr_tier_med_max_tgt_symbol_ratio", 0.25),
        min_stopword_hits=_get_int(cfg, "ocr_tier_med_min_stopword_hits", 1),
    )

    pair_order = str(cfg.get("ocr_pairs_pair_order", "src_then_tgt"))
    max_gap = _get_int(cfg, "ocr_pairs_max_gap", 1)
    min_pair_score = _get_float(cfg, "ocr_pairs_min_pair_score", 0.2)
    w_len = _get_float(cfg, "ocr_align_weight_len", 0.7)
    w_anchor = _get_float(cfg, "ocr_align_weight_anchor", 0.3)
    variant = str(cfg.get("ocr_pairs_variant", "C")).upper()
    drop_flagged = bool(cfg.get("ocr_pairs_drop_flagged", True))

    seed = args.seed if args.seed is not None else _get_int(cfg, "seed", 42)
    rng = random.Random(seed)

    stats = Counter()
    flag_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    len_ratio_stats = RunningStats(sample_size=200000, rng=rng)
    align_score_stats = RunningStats(sample_size=200000, rng=rng)
    anchor_stats = RunningStats(sample_size=200000, rng=rng)

    seen_pairs: set[str] = set()
    dedupe_global = not args.no_dedupe_global

    candidate_parts = _list_candidate_parts(candidates_path)
    if args.max_parts is not None:
        candidate_parts = candidate_parts[: args.max_parts]
    if not candidate_parts:
        raise FileNotFoundError(f"No candidate parts found at: {candidates_path}")

    total_rows = 0
    kept_rows = 0
    pair_seq = 0

    for part_idx, part_path in enumerate(candidate_parts):
        part_id = _extract_part_index(part_path)
        out_name = f"part-{part_id:04d}.parquet" if part_id is not None else f"part-{part_idx:04d}.parquet"
        out_path = out_dir / out_name
        if out_path.exists() and not args.overwrite:
            stats["parts_skipped"] += 1
            continue

        cand_df = pd.read_parquet(part_path)
        if args.max_rows is not None and total_rows >= args.max_rows:
            break
        if args.max_rows is not None:
            cand_df = cand_df.head(max(0, args.max_rows - total_rows))

        part_rows: List[Dict[str, object]] = []
        for row in cand_df.itertuples(index=False):
            row_dict = row._asdict()
            page_text = str(row_dict.get("page_text", "") or "")
            if not page_text.strip():
                stats["rows_empty_text"] += 1
                continue

            total_rows += 1
            english_blocks, translit_blocks = _collect_blocks(page_text, eng_thr, tr_thr)
            if not english_blocks or not translit_blocks:
                stats["rows_drop_blocks"] += 1
                continue

            stats["rows_with_blocks"] += 1
            stats["blocks_english"] += len(english_blocks)
            stats["blocks_translit"] += len(translit_blocks)

            pairs = _pair_blocks(
                english_blocks,
                translit_blocks,
                max_gap=max_gap,
                pair_order=pair_order,
                min_pair_score=min_pair_score,
                w_len=w_len,
                w_anchor=w_anchor,
            )
            if not pairs:
                stats["rows_drop_no_pairs"] += 1
                continue

            pdf_name = str(row_dict.get("pdf_name", ""))
            page = row_dict.get("page", "")
            ocr_doc_id = f"ocr:{pdf_name}:{page}"

            for src_idx, src_text, tgt_idx, tgt_text, para_score in pairs:
                tgt_sents = split_english(tgt_text)
                if not tgt_sents:
                    stats["drop_no_tgt_sents"] += 1
                    continue

                src_norm = normalize_transliteration(src_text, variant)
                src_tokens = src_norm.split()
                if len(src_tokens) < len(tgt_sents):
                    stats["drop_src_too_short"] += 1
                    continue

                target_lengths = [len(s) for s in tgt_sents]
                segments = segment_source_tokens(src_tokens, target_lengths)

                for sent_idx, tgt_sent in enumerate(tgt_sents):
                    src_seg = " ".join(segments[sent_idx]) if sent_idx < len(segments) else ""
                    src_seg = clean_text(src_seg)
                    tgt_sent = clean_text(normalize_translation(tgt_sent))

                    src_stats = text_stats(src_seg)
                    tgt_stats = text_stats(tgt_sent)
                    if int(src_stats["len"]) <= 0 or int(tgt_stats["len"]) <= 0:
                        stats["drop_empty_pair"] += 1
                        continue

                    src_len = int(src_stats["len"])
                    tgt_len = int(tgt_stats["len"])
                    len_ratio = src_len / max(tgt_len, 1)
                    token_align_score = 0.0
                    if int(src_stats["tokens"]) > 0 and int(tgt_stats["tokens"]) > 0:
                        token_align_score = min(int(src_stats["tokens"]), int(tgt_stats["tokens"])) / max(
                            int(src_stats["tokens"]), int(tgt_stats["tokens"])
                        )
                    overlap = anchor_overlap(src_seg, tgt_sent)
                    align_score = w_len * min(src_len, tgt_len) / max(src_len, tgt_len, 1) + w_anchor * overlap
                    tgt_sentence_ends = count_sentence_endings(tgt_sent)
                    stopword_hits = count_stopwords(tgt_sent)

                    flags = _compute_flags(
                        src_stats,
                        tgt_stats,
                        len_ratio,
                        tgt_sentence_ends,
                        stopword_hits,
                        gate_cfg,
                        src_seg,
                        tgt_sent,
                    )
                    if flags:
                        stats["pairs_flagged"] += 1
                        for flag in flags:
                            flag_counts[flag] += 1
                        if drop_flagged:
                            continue

                    tier = _assign_tier(
                        align_score,
                        len_ratio,
                        float(tgt_stats["alpha_ratio"]),
                        float(tgt_stats["symbol_ratio"]),
                        stopword_hits,
                        high_rule,
                        med_rule,
                    )

                    pair_hash = _pair_hash(src_seg, tgt_sent)
                    if dedupe_global and pair_hash in seen_pairs:
                        stats["pairs_deduped"] += 1
                        continue
                    if dedupe_global:
                        seen_pairs.add(pair_hash)

                    row_out: Dict[str, object] = {
                        "source_id": f"{ocr_doc_id}:{pair_seq}",
                        "oare_id": ocr_doc_id,
                        "pdf_name": pdf_name,
                        "page": page,
                        "src_sent": src_seg,
                        "tgt_sent": tgt_sent,
                        "src_norm_variant": variant,
                        "align_score": round(align_score, 6),
                        "len_ratio": round(len_ratio, 6),
                        "token_align_score": round(token_align_score, 6),
                        "anchor_overlap": round(overlap, 6),
                        "paragraph_score": round(para_score, 6),
                        "quality_tier": tier,
                        "flags": ";".join(flags),
                        "tgt_sentence_ends": tgt_sentence_ends,
                        "src_tokens": int(src_stats["tokens"]),
                        "tgt_tokens": int(tgt_stats["tokens"]),
                        "src_alpha_ratio": round(float(src_stats["alpha_ratio"]), 6),
                        "src_digit_ratio": round(float(src_stats["digit_ratio"]), 6),
                        "src_symbol_ratio": round(float(src_stats["symbol_ratio"]), 6),
                        "tgt_alpha_ratio": round(float(tgt_stats["alpha_ratio"]), 6),
                        "tgt_digit_ratio": round(float(tgt_stats["digit_ratio"]), 6),
                        "tgt_symbol_ratio": round(float(tgt_stats["symbol_ratio"]), 6),
                        "tgt_stopword_hits": stopword_hits,
                        "src_block_idx": src_idx,
                        "tgt_block_idx": tgt_idx,
                    }
                    part_rows.append(row_out)
                    kept_rows += 1
                    pair_seq += 1
                    tier_counts[tier] += 1
                    len_ratio_stats.add(len_ratio)
                    align_score_stats.add(align_score)
                    anchor_stats.add(overlap)

        if part_rows:
            pd.DataFrame(part_rows).to_parquet(out_path, index=False)
            stats["parts_written"] += 1
        else:
            stats["parts_empty"] += 1

    summary = {
        "input_rows": total_rows,
        "output_rows": kept_rows,
        "stats": dict(stats),
        "tier_counts": dict(tier_counts),
        "flag_counts": dict(flag_counts),
        "len_ratio": len_ratio_stats.summary(),
        "align_score": align_score_stats.summary(),
        "anchor_overlap": anchor_stats.summary(),
        "gate_config": gate_cfg.__dict__,
        "tier_high": high_rule.__dict__,
        "tier_med": med_rule.__dict__,
        "pair_order": pair_order,
        "max_gap": max_gap,
        "min_pair_score": min_pair_score,
        "variant": variant,
        "drop_flagged": drop_flagged,
        "dedupe_global": dedupe_global,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Input rows: {total_rows}")
    print(f"Output rows: {kept_rows}")
    print(f"Saved parts: {out_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
