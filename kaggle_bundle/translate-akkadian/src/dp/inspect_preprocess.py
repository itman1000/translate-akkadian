"""正規化A/B/Cの前処理が想定通りかをログで点検する。"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import pandas as pd

from .align_train import normalize_transliteration
from .utils import get_data_dir, load_config


_GAP_TAG_RE = re.compile(r"<\s*gap\s*>", re.IGNORECASE)
_BIG_GAP_TAG_RE = re.compile(r"<\s*big_gap\s*>", re.IGNORECASE)
_BRACKET_MISSING_RE = re.compile(r"\[(?P<body>[xX\.\s\?…]+)\]")
_ELLIPSIS_RE = re.compile(r"…")
_DOTS_RUN_RE = re.compile(r"\.{3,}")
_X_RUN_RE = re.compile(r"(?<![A-Za-z0-9])x{3,}(?![A-Za-z0-9])", re.IGNORECASE)
_X_TOKEN_RUN_RE = re.compile(r"(?<![A-Za-z0-9])x(?:[\s-]+x){2,}(?![A-Za-z0-9])", re.IGNORECASE)
_PAREN_DET_RE = re.compile(r"\([^)]+\)")
_BRACE_DET_RE = re.compile(r"\{[^}]+\}")
_DOT_TOKEN_RE = re.compile(r"\w+\.\w+")
_ANGLE_RE = re.compile(r"<[^>]+>")
_CDLI_SZ_RE = re.compile(r"sz", re.IGNORECASE)
_CDLI_S_COMMA_RE = re.compile(r"s,", re.IGNORECASE)
_CDLI_T_COMMA_RE = re.compile(r"t,", re.IGNORECASE)
_CDLI_VOWEL_DIGIT_RE = re.compile(r"[aAeEiIuU][23₂₃]")
_CDLI_SUBSCRIPT_X_RE = re.compile(r"Xx")
_CDLI_H_RE = re.compile(r"[Hh]")
_BARE_GAP_RE = re.compile(r"(?<!<)\bgap\b(?!>)", re.IGNORECASE)
_BARE_BIG_GAP_RE = re.compile(r"(?<!<)\bbig_gap\b(?!>)", re.IGNORECASE)
_LINE_NUM_RE = re.compile(r"(?<!\w)\d+''?(?!\w)")
_LINE_START_NUM_RE = re.compile(r"(?m)^\s*\d+\b")
_TITLE_STRIP_RE = re.compile(r"^(?:\{[^}]+\}|\([^)]*\))+")


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def count_regex(texts: Sequence[str], pattern: re.Pattern[str]) -> int:
    return sum(len(pattern.findall(text)) for text in texts)


def count_x_runs(texts: Sequence[str]) -> Tuple[int, int]:
    gap = 0
    big = 0
    for text in texts:
        for match in _X_RUN_RE.finditer(text):
            if len(match.group(0)) >= 4:
                big += 1
            else:
                gap += 1
    return gap, big


def count_x_token_runs(texts: Sequence[str]) -> Tuple[int, int]:
    gap = 0
    big = 0
    for text in texts:
        for match in _X_TOKEN_RUN_RE.finditer(text):
            x_count = len(re.findall(r"[xX]", match.group(0)))
            if x_count >= 4:
                big += 1
            else:
                gap += 1
    return gap, big


def count_bracket_missing(texts: Sequence[str]) -> Tuple[int, int]:
    gap = 0
    big = 0
    for text in texts:
        for match in _BRACKET_MISSING_RE.finditer(text):
            body = match.group("body")
            if "…" in body or re.search(r"\.{3,}", body):
                big += 1
                continue
            x_count = len(re.findall(r"[xX]", body))
            if x_count >= 4:
                big += 1
            else:
                gap += 1
    return gap, big


def has_all_caps(text: str) -> bool:
    for token in text.split():
        if len(token) >= 2 and token.isupper():
            return True
    return False


def count_all_caps_tokens(texts: Sequence[str]) -> int:
    count = 0
    for text in texts:
        for token in text.split():
            if len(token) >= 2 and token.isupper():
                count += 1
    return count


def _is_title_token(token: str) -> bool:
    token = _TITLE_STRIP_RE.sub("", token)
    if not token:
        return False
    first_cased = None
    has_upper = False
    has_lower = False
    for ch in token:
        if not ch.isalpha():
            continue
        if first_cased is None:
            first_cased = ch
        if ch.isupper():
            has_upper = True
        elif ch.islower():
            has_lower = True
    if first_cased is None:
        return False
    if not (has_upper and has_lower):
        return False
    return first_cased.isupper()


def has_title_case(text: str) -> bool:
    return any(_is_title_token(token) for token in text.split())


def count_title_tokens(texts: Sequence[str]) -> int:
    count = 0
    for text in texts:
        for token in text.split():
            if _is_title_token(token):
                count += 1
    return count


def count_unexpected_angles(texts: Sequence[str]) -> int:
    count = 0
    for text in texts:
        for match in _ANGLE_RE.finditer(text):
            tag = match.group(0)
            if _GAP_TAG_RE.fullmatch(tag) or _BIG_GAP_TAG_RE.fullmatch(tag):
                continue
            count += 1
    return count


def indices_where(texts: Sequence[str], pred: Callable[[str], bool]) -> List[int]:
    return [idx for idx, text in enumerate(texts) if pred(text)]


def select_examples(indices: List[int], max_examples: int, rng: random.Random) -> List[int]:
    if max_examples <= 0:
        return []
    if len(indices) <= max_examples:
        return indices
    return rng.sample(indices, max_examples)


def print_examples(
    label: str,
    indices: List[int],
    raw_texts: Sequence[str],
    norm_texts: Sequence[str],
    ids: Sequence[str] | None,
    max_examples: int,
    rng: random.Random,
    limit: int,
) -> None:
    chosen = select_examples(indices, max_examples, rng)
    if not chosen:
        return
    print(f"[{label}] {len(indices)}件中{len(chosen)}件表示")
    for idx in chosen:
        id_info = f"oare_id={ids[idx]}" if ids else f"idx={idx}"
        print(f"- {id_info}")
        print(f"  raw : {truncate(raw_texts[idx], limit)}")
        print(f"  norm: {truncate(norm_texts[idx], limit)}")


def summarize_texts(texts: Sequence[str], label: str) -> None:
    series = pd.Series(texts, dtype="string")
    if len(series) == 0:
        print(f"[{label}] rows=0")
        return
    empty_ratio = (series.fillna("").str.strip() == "").mean()
    lengths = series.fillna("").str.len()
    token_counts = series.fillna("").str.split().map(len)
    q_len = lengths.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    q_tok = token_counts.quantile([0.5, 0.9]).to_dict()
    unique_ratio = series.nunique() / max(len(series), 1)
    print(
        f"[{label}] rows={len(series)} empty={empty_ratio:.1%} unique={unique_ratio:.2f} "
        f"len_med={q_len[0.5]:.0f} p90={q_len[0.9]:.0f} p95={q_len[0.95]:.0f} p99={q_len[0.99]:.0f} "
        f"tok_med={q_tok[0.5]:.0f} p90={q_tok[0.9]:.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect normalization invariants.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--train", default=None, help="Train CSV/Parquet path.")
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--id-col", default=None, help="ID column for logging.")
    parser.add_argument("--variants", default=None, help="Variants to inspect, e.g., A,B,C.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use first N rows.")
    parser.add_argument("--sample", type=int, default=None, help="Sample N rows randomly.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-examples", type=int, default=3, help="Examples per category.")
    parser.add_argument("--example-limit", type=int, default=200, help="Max chars per example.")
    parser.add_argument(
        "--use-aligned",
        action="store_true",
        help="入力が正規化済みの場合は再正規化せずに検査する。",
    )
    parser.add_argument("--variant-col", default=None, help="正規化済みデータのバリアント列名。")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, None)

    train_path = Path(args.train) if args.train else data_dir / "train.csv"
    df = read_table(train_path)

    src_col = args.src_col or cfg.get("src_col", "transliteration")
    id_col = args.id_col or "oare_id"

    if src_col not in df.columns:
        raise ValueError(f"source column not found: {src_col}")

    if args.sample is not None:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed).reset_index(drop=True)
    elif args.max_rows is not None:
        df = df.head(args.max_rows).reset_index(drop=True)

    rng = random.Random(args.seed)

    def inspect_variant(
        label: str,
        raw_texts: Sequence[str],
        norm_texts: Sequence[str],
        ids: List[str] | None,
    ) -> None:
        print(f"\n=== 正規化ログ: variant={label} ===")
        if args.use_aligned:
            print("mode=aligned (no re-normalize)")
        else:
            changed = sum(1 for raw, norm in zip(raw_texts, norm_texts) if raw != norm)
            print(f"changed_rows={changed} ({changed / max(len(raw_texts), 1):.1%})")
        summarize_texts(raw_texts, "raw")
        summarize_texts(norm_texts, "norm")

        raw_bracket_gap, raw_bracket_big = count_bracket_missing(raw_texts)
        raw_ellipsis = count_regex(raw_texts, _ELLIPSIS_RE)
        raw_dots = count_regex(raw_texts, _DOTS_RUN_RE)
        raw_x_gap, raw_x_big = count_x_runs(raw_texts)
        raw_xs_gap, raw_xs_big = count_x_token_runs(raw_texts)
        raw_gap_tag = count_regex(raw_texts, _GAP_TAG_RE)
        raw_big_gap_tag = count_regex(raw_texts, _BIG_GAP_TAG_RE)
        raw_cdli_sz = count_regex(raw_texts, _CDLI_SZ_RE)
        raw_cdli_s_comma = count_regex(raw_texts, _CDLI_S_COMMA_RE)
        raw_cdli_t_comma = count_regex(raw_texts, _CDLI_T_COMMA_RE)
        raw_cdli_vowel = count_regex(raw_texts, _CDLI_VOWEL_DIGIT_RE)
        raw_cdli_xx = count_regex(raw_texts, _CDLI_SUBSCRIPT_X_RE)
        raw_cdli_h = count_regex(raw_texts, _CDLI_H_RE)
        raw_line_num = count_regex(raw_texts, _LINE_NUM_RE)
        raw_line_start = count_regex(raw_texts, _LINE_START_NUM_RE)

        norm_gap = count_regex(norm_texts, _GAP_TAG_RE)
        norm_big_gap = count_regex(norm_texts, _BIG_GAP_TAG_RE)
        norm_cdli_sz = count_regex(norm_texts, _CDLI_SZ_RE)
        norm_cdli_s_comma = count_regex(norm_texts, _CDLI_S_COMMA_RE)
        norm_cdli_t_comma = count_regex(norm_texts, _CDLI_T_COMMA_RE)
        norm_cdli_vowel = count_regex(norm_texts, _CDLI_VOWEL_DIGIT_RE)
        norm_cdli_xx = count_regex(norm_texts, _CDLI_SUBSCRIPT_X_RE)
        norm_cdli_h = count_regex(norm_texts, _CDLI_H_RE)
        norm_bare_gap = count_regex(norm_texts, _BARE_GAP_RE)
        norm_bare_big_gap = count_regex(norm_texts, _BARE_BIG_GAP_RE)
        norm_line_num = count_regex(norm_texts, _LINE_NUM_RE)
        norm_line_start = count_regex(norm_texts, _LINE_START_NUM_RE)

        raw_paren = count_regex(raw_texts, _PAREN_DET_RE)
        raw_brace = count_regex(raw_texts, _BRACE_DET_RE)
        norm_brace = count_regex(norm_texts, _BRACE_DET_RE)

        raw_dot = count_regex(raw_texts, _DOT_TOKEN_RE)
        norm_dot = count_regex(norm_texts, _DOT_TOKEN_RE)

        raw_caps = count_all_caps_tokens(raw_texts)
        norm_caps = count_all_caps_tokens(norm_texts)
        raw_title = count_title_tokens(raw_texts)
        norm_title = count_title_tokens(norm_texts)

        raw_angle_other = count_unexpected_angles(raw_texts)
        norm_angle_other = count_unexpected_angles(norm_texts)

        print(
            "gap_src=[bracket_gap]={} gap_src=[bracket_big]={} gap_src=…={} "
            "gap_src=...={} gap_src=<gap>={} gap_src=<big_gap>={} xrun_gap={} xrun_big={} "
            "xspace_gap={} xspace_big={} -> norm_gap={} norm_big_gap={}".format(
                raw_bracket_gap,
                raw_bracket_big,
                raw_ellipsis,
                raw_dots,
                raw_gap_tag,
                raw_big_gap_tag,
                raw_x_gap,
                raw_x_big,
                raw_xs_gap,
                raw_xs_big,
                norm_gap,
                norm_big_gap,
            )
        )
        print(
            "cdli_raw=sz:{} s,:{} t,:{} vowel:{} Xx:{} h:{} -> cdli_norm=sz:{} s,:{} t,:{} vowel:{} Xx:{} h:{} "
            "bare_gap_norm={} bare_big_gap_norm={}".format(
                raw_cdli_sz,
                raw_cdli_s_comma,
                raw_cdli_t_comma,
                raw_cdli_vowel,
                raw_cdli_xx,
                raw_cdli_h,
                norm_cdli_sz,
                norm_cdli_s_comma,
                norm_cdli_t_comma,
                norm_cdli_vowel,
                norm_cdli_xx,
                norm_cdli_h,
                norm_bare_gap,
                norm_bare_big_gap,
            )
        )
        print(
            "line_num_raw={} line_num_norm={} line_start_raw={} line_start_norm={}".format(
                raw_line_num,
                norm_line_num,
                raw_line_start,
                norm_line_start,
            )
        )
        print(
            f"det_paren_raw={raw_paren} det_brace_raw={raw_brace} det_brace_norm={norm_brace}"
        )
        print(f"dot_tokens_raw={raw_dot} dot_tokens_norm={norm_dot}")
        print(f"all_caps_tokens_raw={raw_caps} all_caps_tokens_norm={norm_caps}")
        print(f"title_tokens_raw={raw_title} title_tokens_norm={norm_title}")
        print(f"angle_other_raw={raw_angle_other} angle_other_norm={norm_angle_other}")

        warnings: List[str] = []
        raw_missing_total = raw_bracket_gap + raw_ellipsis + raw_gap_tag + raw_x_gap + raw_xs_gap
        raw_big_total = raw_bracket_big + raw_ellipsis + raw_dots + raw_big_gap_tag + raw_x_big + raw_xs_big
        raw_cdli_total = (
            raw_cdli_sz
            + raw_cdli_s_comma
            + raw_cdli_t_comma
            + raw_cdli_vowel
            + raw_cdli_xx
            + raw_cdli_h
        )
        norm_cdli_total = (
            norm_cdli_sz
            + norm_cdli_s_comma
            + norm_cdli_t_comma
            + norm_cdli_vowel
            + norm_cdli_xx
            + norm_cdli_h
        )
        if not args.use_aligned and label in {"B", "C"}:
            if (raw_missing_total + raw_big_total) > 0 and (norm_gap + norm_big_gap) == 0:
                warnings.append("<gap>/<big_gap> が正規化後に検出できません")
            if raw_big_total > 0 and norm_big_gap == 0:
                warnings.append("<big_gap> が正規化後に検出できません")
        if raw_cdli_total > 0 and norm_cdli_total > 0:
            warnings.append("CDLI/ORACC の ASCII 表記が正規化後に残っています")
        if norm_bare_gap > 0 or norm_bare_big_gap > 0:
            warnings.append("gap/big_gap が山括弧なしで残っています")
        if raw_line_num > 0 and norm_line_num > 0:
            warnings.append("行番号（1'/1''）が正規化後に残っています")
        if raw_dot > 0 and norm_dot == 0:
            warnings.append("ドット区切りトークンが消失しています")
        if raw_caps > 0 and norm_caps == 0:
            warnings.append("ALL CAPS が消失しています")
        if raw_title > 0 and norm_title == 0:
            warnings.append("先頭大文字（固有名詞）が消失しています")
        if norm_angle_other > 0:
            warnings.append("想定外の山括弧トークンが残っています")

        for msg in warnings:
            print(f"WARN: {msg}")

        if args.max_examples > 0:
            diff_rows = [i for i, (raw, norm) in enumerate(zip(raw_texts, norm_texts)) if raw != norm]
            gap_rows = indices_where(
                norm_texts, lambda x: ("<gap>" in x) or ("<big_gap>" in x)
            )
            dot_rows = indices_where(norm_texts, lambda x: _DOT_TOKEN_RE.search(x) is not None)
            caps_rows = indices_where(norm_texts, has_all_caps)
            title_rows = indices_where(norm_texts, has_title_case)
            det_rows = [
                i
                for i, (raw, norm) in enumerate(zip(raw_texts, norm_texts))
                if _PAREN_DET_RE.search(raw) or _BRACE_DET_RE.search(norm)
            ]
            xspace_rows = indices_where(raw_texts, lambda x: _X_TOKEN_RUN_RE.search(x) is not None)
            cdli_rows = indices_where(
                raw_texts,
                lambda x: _CDLI_SZ_RE.search(x)
                or _CDLI_S_COMMA_RE.search(x)
                or _CDLI_T_COMMA_RE.search(x)
                or _CDLI_VOWEL_DIGIT_RE.search(x)
                or _CDLI_SUBSCRIPT_X_RE.search(x)
                or _CDLI_H_RE.search(x),
            )
            bare_gap_rows = indices_where(
                norm_texts,
                lambda x: _BARE_GAP_RE.search(x) or _BARE_BIG_GAP_RE.search(x),
            )

            print_examples(
                "diff",
                diff_rows,
                raw_texts,
                norm_texts,
                ids,
                args.max_examples,
                rng,
                args.example_limit,
            )
            print_examples(
                "cdli_raw",
                cdli_rows,
                raw_texts,
                norm_texts,
                ids,
                args.max_examples,
                rng,
                args.example_limit,
            )
            print_examples(
                "gap",
                gap_rows,
                raw_texts,
                norm_texts,
                ids,
                args.max_examples,
                rng,
                args.example_limit,
            )
            print_examples(
                "bare_gap",
                bare_gap_rows,
                raw_texts,
                norm_texts,
                ids,
                args.max_examples,
                rng,
                args.example_limit,
            )
            print_examples(
                "xspace",
                xspace_rows,
                raw_texts,
                norm_texts,
                ids,
                args.max_examples,
                rng,
                args.example_limit,
            )
            print_examples(
                "dot",
                dot_rows,
                raw_texts,
                norm_texts,
                ids,
                args.max_examples,
                rng,
                args.example_limit,
            )
            print_examples(
                "caps",
                caps_rows,
                raw_texts,
                norm_texts,
                ids,
                args.max_examples,
                rng,
                args.example_limit,
            )
            print_examples(
                "title",
                title_rows,
                raw_texts,
                norm_texts,
                ids,
                args.max_examples,
                rng,
                args.example_limit,
            )
            print_examples(
                "det",
                det_rows,
                raw_texts,
                norm_texts,
                ids,
                args.max_examples,
                rng,
                args.example_limit,
            )

    def build_texts(frame: pd.DataFrame) -> Tuple[List[str], List[str] | None]:
        texts = frame[src_col].fillna("").astype(str).tolist()
        ids = None
        if id_col in frame.columns:
            ids = frame[id_col].fillna("").astype(str).tolist()
        return texts, ids

    if args.use_aligned:
        variant_col = args.variant_col or "src_norm_variant"
        if variant_col in df.columns:
            if args.variants:
                allowed = {v.strip().upper() for v in str(args.variants).split(",") if v.strip()}
                df = df[df[variant_col].astype(str).str.upper().isin(allowed)]
            variants = sorted(df[variant_col].dropna().astype(str).unique().tolist())
            if not variants:
                variants = ["ALIGNED"]
        else:
            variants = [str(args.variants).strip().upper()] if args.variants else ["ALIGNED"]

        for variant in variants:
            if variant_col in df.columns:
                sub = df[df[variant_col].astype(str) == variant]
            else:
                sub = df
            raw_texts, ids = build_texts(sub)
            norm_texts = raw_texts
            inspect_variant(variant, raw_texts, norm_texts, ids)
    else:
        variant_spec = args.variants or cfg.get("variants", "A,B,C")
        variants = [v.strip().upper() for v in str(variant_spec).split(",") if v.strip()]
        if not variants:
            variants = ["A"]

        raw_texts, ids = build_texts(df)
        for variant in variants:
            norm_texts = [normalize_transliteration(text, variant) for text in raw_texts]
            inspect_variant(variant, raw_texts, norm_texts, ids)


if __name__ == "__main__":
    main()
