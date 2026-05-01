"""publications.csv を OCR 用に安全に前処理する。"""

from __future__ import annotations

import argparse
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from .utils import get_artifacts_dir, get_data_dir, load_config


_DASH_MAP = str.maketrans({"–": "-", "—": "-", "−": "-"})
_APOSTROPHE_RE = re.compile(r"[’‘ʾʼ]")
_LINE_START_RE = re.compile(
    r"^\s*(?P<num>\d{1,4})(?P<prime>['’‘]{1,2})?(?P<sep>[\)\.:])?\s*(?P<rest>.*)$"
)
_DECIMAL_START_RE = re.compile(r"^\s*\d+\.\d+")
_SPLIT_RE = re.compile(r"[;/,]")
_STRIP_CHARS = ".,;:!?\"'()[]{}<>"
_MULTI_SPACE_RE = re.compile(r"\s+")
_DROP_LSTRIP = " -–—•*"
_PURE_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_SOFT_HYPHEN_RE = re.compile("\u00ad")

_STRICT_WHITELIST_PATTERNS = [
    re.compile(r"\{[^}]+\}"),
    re.compile(r"\(d\)", re.IGNORECASE),
    re.compile(r"<gap>|<big_gap>", re.IGNORECASE),
    re.compile(r"\bDUMU\b", re.IGNORECASE),
    re.compile(r"\bKIŠIB\b|\bKISIB\b", re.IGNORECASE),
    re.compile(r"\bIGI\b", re.IGNORECASE),
    re.compile(r"\bDINGIR\b", re.IGNORECASE),
    re.compile(r"\bKÙ\.BABBAR\b|\bKU\.BABBAR\b|\bKUG\.BABBAR\b", re.IGNORECASE),
    re.compile(r"\bAN\.NA\b", re.IGNORECASE),
    re.compile(r"\bGÍN\b|\bGIN\b", re.IGNORECASE),
    re.compile(r"\bMA\.NA\b|\bMA-NA\b", re.IGNORECASE),
    re.compile(r"\bTÚG\b|\bTUG\b", re.IGNORECASE),
    re.compile(r"\bANŠE\b|\bANSHE\b", re.IGNORECASE),
    re.compile(r"\bURUDU\b", re.IGNORECASE),
    re.compile(r"\bITU\b", re.IGNORECASE),
    re.compile(r"[₀-₉ₓ]"),
]


def _decode_newlines(text: str) -> str:
    if not text:
        return ""
    out = str(text).replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _SOFT_HYPHEN_RE.sub("", out)
    out = out.translate(_DASH_MAP)
    out = _APOSTROPHE_RE.sub("'", out)
    out = re.sub(r"(?<=\w)-\n(?=\w)", "", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def _is_strict_akkadian_line(text: str) -> bool:
    for pattern in _STRICT_WHITELIST_PATTERNS:
        if pattern.search(text):
            return True
    return False


def load_noise_patterns(path: Path) -> Tuple[List[re.Pattern[str]], List[re.Pattern[str]]]:
    drop_line: List[re.Pattern[str]] = []
    remove_inline: List[re.Pattern[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split("\t")
        if len(parts) < 2:
            continue
        mode = parts[0].strip().upper()
        pattern = parts[1].strip()
        if not pattern:
            continue
        compiled = re.compile(pattern, re.IGNORECASE)
        if mode == "DROP_LINE":
            drop_line.append(compiled)
        elif mode == "REMOVE_INLINE":
            remove_inline.append(compiled)
    return drop_line, remove_inline


def load_unit_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith(("目的", "使い方", "注意", "カテゴリ")):
            continue
        parts = text.split("\t")
        if len(parts) < 2:
            continue
        token_field = parts[1]
        var_field = parts[2] if len(parts) >= 3 else ""
        for raw in _split_token_field(token_field) + _split_token_field(var_field):
            norm = _normalize_unit_token(raw)
            if norm:
                tokens.add(norm)
    return tokens


def _split_token_field(text: str) -> List[str]:
    tokens: List[str] = []
    for part in _SPLIT_RE.split(text):
        part = part.strip()
        if not part:
            continue
        part = part.strip("()[]{}")
        if not part:
            continue
        if " " in part:
            part = part.split()[0]
        if part:
            tokens.append(part)
    return tokens


def _normalize_unit_token(token: str) -> str:
    token = token.translate(_DASH_MAP)
    token = _APOSTROPHE_RE.sub("'", token)
    token = token.strip(_STRIP_CHARS)
    return token.upper()


def _normalize_candidate_token(token: str) -> str:
    token = token.translate(_DASH_MAP)
    token = _APOSTROPHE_RE.sub("'", token)
    token = token.strip(_STRIP_CHARS)
    return token.upper()


def _is_quantity_token(token: str, unit_tokens: set[str]) -> bool:
    norm = _normalize_candidate_token(token)
    if not norm:
        return False
    if norm in unit_tokens:
        return True
    for sep in ("-", "."):
        if sep in norm:
            base = norm.split(sep)[0]
            if len(base) >= 2 and base in unit_tokens:
                return True
    return False


def _apply_noise_filters(
    candidate: str,
    drop_line_patterns: List[re.Pattern[str]],
    inline_patterns: List[re.Pattern[str]],
    stats: Counter[str],
    *,
    strict_whitelist: bool = False,
    stat_prefix: str = "",
) -> str | None:
    if not candidate:
        stats[f"{stat_prefix}lines_dropped_empty"] += 1
        return None
    match_target = candidate.lstrip(_DROP_LSTRIP)
    if _PURE_PAGE_NUM_RE.match(match_target):
        stats[f"{stat_prefix}page_number_dropped"] += 1
        return None
    if drop_line_patterns and any(p.search(match_target) for p in drop_line_patterns):
        stats[f"{stat_prefix}line_noise_dropped"] += 1
        return None
    removed = 0
    for pattern in inline_patterns:
        candidate, count = pattern.subn("", candidate)
        removed += count
    if removed:
        stats[f"{stat_prefix}inline_noise_removed"] += removed
        candidate = _MULTI_SPACE_RE.sub(" ", candidate).strip()
    if not candidate:
        stats[f"{stat_prefix}lines_dropped_empty"] += 1
        return None
    if strict_whitelist and not _is_strict_akkadian_line(candidate):
        stats[f"{stat_prefix}strict_whitelist_drop"] += 1
        return None
    return candidate


def clean_translation_page_text(
    text: str,
    drop_line_patterns: List[re.Pattern[str]],
    inline_patterns: List[re.Pattern[str]],
) -> Tuple[str, Counter[str]]:
    """翻訳段落を壊さない穏当なページクリーニング。"""
    stats: Counter[str] = Counter()
    if not text:
        return "", stats

    normalized = _decode_newlines(text)
    output_lines: List[str] = []
    prev_blank = False
    for raw in normalized.splitlines():
        stripped = raw.strip()
        if not stripped:
            if output_lines and not prev_blank:
                output_lines.append("")
            prev_blank = True
            continue
        prev_blank = False
        stats["translation_lines_total"] += 1
        candidate = _apply_noise_filters(
            stripped,
            drop_line_patterns,
            inline_patterns,
            stats,
            strict_whitelist=False,
            stat_prefix="translation_",
        )
        if candidate is None:
            continue
        output_lines.append(candidate)
    while output_lines and output_lines[-1] == "":
        output_lines.pop()
    return "\n".join(output_lines), stats


def clean_translit_page_text(
    text: str,
    unit_tokens: set[str],
    drop_ambiguous: bool,
    drop_line_patterns: List[re.Pattern[str]],
    inline_patterns: List[re.Pattern[str]],
    strict_whitelist: bool,
) -> Tuple[str, Counter[str]]:
    """行頭行番号を外しつつ、転写らしい行だけを残す。"""
    stats: Counter[str] = Counter()
    if not text:
        return "", stats

    lines = _decode_newlines(text).splitlines()
    cleaned_lines: List[str] = []
    for line in lines:
        raw_line = line.strip()
        if not raw_line:
            continue
        stats["lines_total"] += 1
        raw_line = _APOSTROPHE_RE.sub("'", raw_line)

        if _DECIMAL_START_RE.match(raw_line):
            candidate = _apply_noise_filters(
                raw_line,
                drop_line_patterns,
                inline_patterns,
                stats,
                strict_whitelist=strict_whitelist,
            )
            if candidate is None:
                continue
            cleaned_lines.append(candidate)
            stats["line_start_decimal_keep"] += 1
            continue

        match = _LINE_START_RE.match(raw_line)
        if not match:
            candidate = _apply_noise_filters(
                raw_line,
                drop_line_patterns,
                inline_patterns,
                stats,
                strict_whitelist=strict_whitelist,
            )
            if candidate is None:
                continue
            cleaned_lines.append(candidate)
            continue

        stats["line_start_total"] += 1
        rest = (match.group("rest") or "").lstrip()
        prime = match.group("prime") or ""
        if prime:
            stats["line_start_prime"] += 1
            stats["line_number_removed"] += 1
            candidate = _apply_noise_filters(
                rest,
                drop_line_patterns,
                inline_patterns,
                stats,
                strict_whitelist=strict_whitelist,
            )
            if candidate is None:
                continue
            cleaned_lines.append(candidate)
            continue

        if rest:
            next_token = rest.split()[0]
            if _is_quantity_token(next_token, unit_tokens):
                stats["line_start_quantity_keep"] += 1
                candidate = _apply_noise_filters(
                    raw_line,
                    drop_line_patterns,
                    inline_patterns,
                    stats,
                    strict_whitelist=strict_whitelist,
                )
                if candidate is None:
                    continue
                cleaned_lines.append(candidate)
                continue

        stats["line_start_ambiguous"] += 1
        if drop_ambiguous:
            stats["line_start_ambiguous_dropped"] += 1
            continue

        stats["line_number_removed"] += 1
        candidate = _apply_noise_filters(
            rest,
            drop_line_patterns,
            inline_patterns,
            stats,
            strict_whitelist=strict_whitelist,
        )
        if candidate is None:
            continue
        cleaned_lines.append(candidate)

    return "\n".join(cleaned_lines), stats


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean publications.csv for OCR extraction.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--input", default=None, help="Input publications.csv path.")
    parser.add_argument("--out", default=None, help="Output path (csv or parquet prefix).")
    parser.add_argument("--format", default="csv", choices=["csv", "parquet"], help="Output format.")
    parser.add_argument("--unit-list", default="docs/akkadian_after_number_tokens.txt", help="Unit list file path.")
    parser.add_argument("--noise-list", default="docs/publications_noise_patterns.txt", help="Noise pattern file.")
    parser.add_argument("--drop-ambiguous", action="store_true", help="Drop ambiguous line-start numbers.")
    parser.add_argument("--keep-ambiguous", action="store_true", help="Keep ambiguous line-start numbers.")
    parser.add_argument("--drop-empty", action="store_true", help="Drop rows whose cleaned page text is empty.")
    parser.add_argument("--strict-whitelist", action="store_true", help="Enable strict transliteration whitelist.")
    parser.add_argument("--no-strict-whitelist", action="store_true", help="Disable strict transliteration whitelist.")
    parser.add_argument("--filter-akkadian", action="store_true", help="Keep only has_akkadian == True rows.")
    parser.add_argument("--no-filter-akkadian", action="store_true", help="Disable has_akkadian filter.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use first N rows for quick runs.")
    parser.add_argument("--chunksize", type=int, default=2000, help="Chunk size.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, None)
    artifacts_dir = get_artifacts_dir(cfg)

    input_path = Path(args.input) if args.input else data_dir / "publications.csv"
    out_path = Path(args.out) if args.out else artifacts_dir / "ocr" / "publications_clean.csv"

    unit_list_path = Path(args.unit_list)
    unit_tokens = load_unit_tokens(unit_list_path)
    noise_list_path = Path(args.noise_list)
    drop_line_patterns, inline_patterns = load_noise_patterns(noise_list_path)

    drop_ambiguous = False if args.keep_ambiguous else True
    drop_empty = bool(args.drop_empty)
    if args.no_filter_akkadian:
        filter_akkadian = False
    elif args.filter_akkadian:
        filter_akkadian = True
    else:
        filter_akkadian = True
    if args.no_strict_whitelist:
        strict_whitelist = False
    elif args.strict_whitelist:
        strict_whitelist = True
    else:
        strict_whitelist = True

    usecols = ["pdf_name", "page", "has_akkadian", "page_text"]
    totals: Counter[str] = Counter()
    kept_rows = 0
    total_rows = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        out_dir = out_path if out_path.suffix == "" else out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = out_path.stem if out_path.suffix else "publications_clean"
    else:
        out_dir = None
        prefix = ""

    first_csv = True
    part_idx = 0
    for chunk in _iter_chunks(input_path, usecols=usecols, chunksize=args.chunksize):
        if args.max_rows is not None and total_rows >= args.max_rows:
            break
        if args.max_rows is not None:
            chunk = chunk.head(max(0, args.max_rows - total_rows))

        if filter_akkadian and "has_akkadian" in chunk.columns:
            chunk = chunk[chunk["has_akkadian"] == True].reset_index(drop=True)  # noqa: E712

        clean_rows: List[str] = []
        translit_rows: List[str] = []
        stats_rows: List[Counter[str]] = []
        for text in chunk["page_text"].fillna("").astype(str):
            clean_text, clean_stats = clean_translation_page_text(text, drop_line_patterns, inline_patterns)
            translit_text, translit_stats = clean_translit_page_text(
                text,
                unit_tokens,
                drop_ambiguous,
                drop_line_patterns,
                inline_patterns,
                strict_whitelist,
            )
            merged_stats = clean_stats + translit_stats
            clean_rows.append(clean_text)
            translit_rows.append(translit_text)
            stats_rows.append(merged_stats)
            totals.update(merged_stats)

        chunk_out = chunk.copy()
        chunk_out["page_text_clean"] = clean_rows
        chunk_out["page_text_translit"] = translit_rows

        stat_keys = [
            "translation_lines_total",
            "translation_line_noise_dropped",
            "translation_inline_noise_removed",
            "translation_lines_dropped_empty",
            "translation_page_number_dropped",
            "lines_total",
            "line_start_total",
            "line_start_prime",
            "line_number_removed",
            "line_start_quantity_keep",
            "line_start_decimal_keep",
            "line_start_ambiguous",
            "line_start_ambiguous_dropped",
            "strict_whitelist_drop",
            "line_noise_dropped",
            "inline_noise_removed",
            "lines_dropped_empty",
            "page_number_dropped",
        ]
        for key in stat_keys:
            chunk_out[key] = [row.get(key, 0) for row in stats_rows]

        if drop_empty:
            mask = chunk_out["page_text_clean"].fillna("").astype(str).str.strip() != ""
            chunk_out = chunk_out[mask]

        total_rows += len(chunk)
        kept_rows += len(chunk_out)

        if args.format == "csv":
            chunk_out.to_csv(out_path, mode="w" if first_csv else "a", header=first_csv, index=False)
            first_csv = False
        else:
            assert out_dir is not None
            part_path = out_dir / f"{prefix}-part-{part_idx:04d}.parquet"
            chunk_out.to_parquet(part_path, index=False)
            part_idx += 1

    summary_keys = [
        "translation_lines_total",
        "translation_line_noise_dropped",
        "translation_inline_noise_removed",
        "translation_lines_dropped_empty",
        "translation_page_number_dropped",
        "lines_total",
        "line_start_total",
        "line_start_prime",
        "line_number_removed",
        "line_start_quantity_keep",
        "line_start_decimal_keep",
        "line_start_ambiguous",
        "line_start_ambiguous_dropped",
        "strict_whitelist_drop",
        "line_noise_dropped",
        "inline_noise_removed",
        "lines_dropped_empty",
        "page_number_dropped",
    ]
    summary = {key: totals.get(key, 0) for key in summary_keys}

    print(f"Input rows: {total_rows}")
    print(f"Output rows: {kept_rows}")
    print(
        "drop_ambiguous={} drop_empty={} filter_akkadian={} strict_whitelist={}".format(
            drop_ambiguous,
            drop_empty,
            filter_akkadian,
            strict_whitelist,
        )
    )
    print("Summary:", summary)
    if args.format == "csv":
        print(f"Saved cleaned CSV: {out_path}")
    else:
        print(f"Saved cleaned parts: {out_dir}")


if __name__ == "__main__":
    main()
