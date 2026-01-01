"""publications.csv の行頭行番号を除去し、曖昧行を落とす。"""

from __future__ import annotations

import argparse
import re
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


def _is_strict_akkadian_line(text: str) -> bool:
    """超厳格ホワイトリストに合致するか判定する。"""
    for pattern in _STRICT_WHITELIST_PATTERNS:
        if pattern.search(text):
            return True
    return False


def load_noise_patterns(path: Path) -> Tuple[List[re.Pattern[str]], List[re.Pattern[str]]]:
    """ノイズ除去用のパターンを読み込む。"""
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
    """単位リストを読み込み、照合用のトークン集合を作る。"""
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
    strict_whitelist: bool,
    stats: Counter[str],
) -> str | None:
    """ノイズパターンの除去を適用して、空になれば None を返す。"""
    if not candidate:
        stats["lines_dropped_empty"] += 1
        return None
    match_target = candidate.lstrip(_DROP_LSTRIP)
    if drop_line_patterns and any(p.search(match_target) for p in drop_line_patterns):
        stats["line_noise_dropped"] += 1
        return None
    removed = 0
    for pattern in inline_patterns:
        candidate, count = pattern.subn("", candidate)
        removed += count
    if removed:
        stats["inline_noise_removed"] += removed
        candidate = _MULTI_SPACE_RE.sub(" ", candidate).strip()
    if not candidate:
        stats["lines_dropped_empty"] += 1
        return None
    if strict_whitelist and not _is_strict_akkadian_line(candidate):
        stats["strict_whitelist_drop"] += 1
        return None
    return candidate


def clean_page_text(
    text: str,
    unit_tokens: set[str],
    drop_ambiguous: bool,
    drop_line_patterns: List[re.Pattern[str]],
    inline_patterns: List[re.Pattern[str]],
    strict_whitelist: bool,
) -> Tuple[str, Counter[str]]:
    """行頭の行番号を除去し、判別不能な行を捨てる。"""
    stats: Counter[str] = Counter()
    if not text:
        return "", stats

    # CSV 内で "\\n" として保存された改行を実改行に戻す
    if "\\n" in text or "\\r" in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")

    lines = text.splitlines()
    cleaned_lines: List[str] = []
    for line in lines:
        raw_line = line.strip()
        if not raw_line:
            continue
        stats["lines_total"] += 1

        # アポストロフィを統一して判定しやすくする
        raw_line = _APOSTROPHE_RE.sub("'", raw_line)

        if _DECIMAL_START_RE.match(raw_line):
            candidate = _apply_noise_filters(
                raw_line,
                drop_line_patterns,
                inline_patterns,
                strict_whitelist,
                stats,
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
                strict_whitelist,
                stats,
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
            if rest:
                candidate = _apply_noise_filters(
                    rest,
                    drop_line_patterns,
                    inline_patterns,
                    strict_whitelist,
                    stats,
                )
                if candidate is None:
                    continue
                cleaned_lines.append(candidate)
            else:
                stats["lines_dropped_empty"] += 1
            continue

        if rest:
            next_token = rest.split()[0]
            if _is_quantity_token(next_token, unit_tokens):
                stats["line_start_quantity_keep"] += 1
                candidate = _apply_noise_filters(
                    raw_line,
                    drop_line_patterns,
                    inline_patterns,
                    strict_whitelist,
                    stats,
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
        if rest:
            candidate = _apply_noise_filters(
                rest,
                drop_line_patterns,
                inline_patterns,
                strict_whitelist,
                stats,
            )
            if candidate is None:
                continue
            cleaned_lines.append(candidate)
        else:
            stats["lines_dropped_empty"] += 1

    return "\n".join(cleaned_lines), stats


def _iter_chunks(path: Path, usecols: List[str], chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(path, usecols=usecols, chunksize=chunksize, engine="python")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean publications.csv line numbers.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--input", default=None, help="Input publications.csv path.")
    parser.add_argument("--out", default=None, help="Output path (csv or parquet prefix).")
    parser.add_argument("--format", default="csv", choices=["csv", "parquet"], help="Output format.")
    parser.add_argument(
        "--unit-list",
        default="docs/akkadian_after_number_tokens.txt",
        help="Unit list file path.",
    )
    parser.add_argument(
        "--noise-list",
        default="docs/publications_noise_patterns.txt",
        help="ノイズ除去パターンの一覧ファイル。",
    )
    parser.add_argument(
        "--drop-ambiguous",
        action="store_true",
        help="単位判定できない行頭数字行は削除する。",
    )
    parser.add_argument(
        "--keep-ambiguous",
        action="store_true",
        help="単位判定できない行頭数字行も残す。",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="クリーニング後に空になった行を除去する。",
    )
    parser.add_argument(
        "--strict-whitelist",
        action="store_true",
        help="超厳格ホワイトリストに合致しない行は削除する（デフォルトは有効）。",
    )
    parser.add_argument(
        "--no-strict-whitelist",
        action="store_true",
        help="超厳格ホワイトリストの削除を無効化する。",
    )
    parser.add_argument(
        "--filter-akkadian",
        action="store_true",
        help="has_akkadian == True の行だけを残す（デフォルトは有効）。",
    )
    parser.add_argument(
        "--no-filter-akkadian",
        action="store_true",
        help="has_akkadian フィルタを無効化する。",
    )
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

    # keep_ambiguous が明示されている場合はそちらを優先する
    if args.keep_ambiguous:
        drop_ambiguous = False
    else:
        drop_ambiguous = True
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

        cleaned_texts: List[str] = []
        stats_rows: List[Counter[str]] = []
        for text in chunk["page_text"].fillna("").astype(str):
            cleaned, stats = clean_page_text(
                text,
                unit_tokens,
                drop_ambiguous,
                drop_line_patterns,
                inline_patterns,
                strict_whitelist,
            )
            cleaned_texts.append(cleaned)
            stats_rows.append(stats)
            totals.update(stats)

        chunk_out = chunk.copy()
        chunk_out["page_text_clean"] = cleaned_texts

        stat_keys = [
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
            part_path = out_dir / f"{prefix}-part-{part_idx:04d}.parquet"
            chunk_out.to_parquet(part_path, index=False)
            part_idx += 1

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
    summary_keys = [
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
    ]
    summary = {key: totals.get(key, 0) for key in summary_keys}
    print(
        "lines_total={lines_total} line_start_total={line_start_total} "
        "line_start_prime={line_start_prime} line_number_removed={line_number_removed} "
        "line_start_quantity_keep={line_start_quantity_keep} line_start_decimal_keep={line_start_decimal_keep} "
        "line_start_ambiguous={line_start_ambiguous} line_start_ambiguous_dropped={line_start_ambiguous_dropped} "
        "strict_whitelist_drop={strict_whitelist_drop} line_noise_dropped={line_noise_dropped} "
        "inline_noise_removed={inline_noise_removed} "
        "lines_dropped_empty={lines_dropped_empty}".format(**summary)
    )
    if args.format == "csv":
        print(f"Saved cleaned CSV: {out_path}")
    else:
        print(f"Saved cleaned parts: {out_dir}")


if __name__ == "__main__":
    main()
