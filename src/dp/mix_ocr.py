"""aligned_train と OCR 追加並列を品質優先で混合する。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .utils import get_artifacts_dir, load_config


_GAP_RE = re.compile(r"<\s*(gap|big_gap)\s*>", re.IGNORECASE)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def list_ocr_parts(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("part-*.parquet"))
    if path.exists():
        return [path]
    pattern = f"{path.stem}-part-*.parquet"
    return sorted(path.parent.glob(pattern))


def compute_gap_rate(texts: Iterable[str]) -> float:
    total = 0
    hits = 0
    for text in texts:
        total += 1
        if _GAP_RE.search(text or ""):
            hits += 1
    return hits / max(total, 1)


def compute_caps_token_rate(texts: Iterable[str]) -> float:
    total = 0
    caps = 0
    for text in texts:
        for token in (text or "").split():
            total += 1
            if len(token) >= 2 and token.isupper():
                caps += 1
    return caps / max(total, 1)


def compute_avg_len(texts: Iterable[str]) -> float:
    total = 0
    count = 0
    for text in texts:
        count += 1
        total += len(text or "")
    return total / max(count, 1)


def _get_float(cfg: Dict[str, object], key: str, default: float) -> float:
    value = cfg.get(key, default)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _get_int(cfg: Dict[str, object], key: str, default: int) -> int:
    value = cfg.get(key, default)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _quality_sort_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "tranche_rank",
        "quality_score",
        "doc_match_score",
        "align_score",
        "metadata_top_score",
        "token_align_score",
    ]
    return [col for col in candidates if col in df.columns]


def _dedupe_pairs(df: pd.DataFrame) -> pd.DataFrame:
    if not {"src_sent", "tgt_sent"}.issubset(df.columns):
        return df
    return df.drop_duplicates(subset=["src_sent", "tgt_sent"]).reset_index(drop=True)


def _parse_ratio_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw in text.split(","):
        part = raw.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        try:
            out[key] = float(value.strip())
        except ValueError:
            continue
    return out


def _tranche_rank(tranche: str) -> int:
    order = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5}
    return order.get(str(tranche or "").upper(), 999)


def _quality_select(
    df: pd.DataFrame,
    target_n: int,
    max_per_oare_id: int,
    max_per_pdf: int,
) -> pd.DataFrame:
    if target_n <= 0 or df.empty:
        return df.head(0).copy()

    df = df.copy()
    if "tranche" in df.columns:
        df["tranche_rank"] = df["tranche"].astype(str).map(_tranche_rank)
    sort_cols = _quality_sort_columns(df)
    ascending = [True if col == "tranche_rank" else False for col in sort_cols]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)

    selected_rows: List[int] = []
    oare_counter: Counter[str] = Counter()
    pdf_counter: Counter[str] = Counter()

    for idx, row in df.iterrows():
        oare_id = str(row.get("oare_id", "") or "")
        pdf_name = str(row.get("pdf_name", "") or "")
        if max_per_oare_id > 0 and oare_id and oare_counter[oare_id] >= max_per_oare_id:
            continue
        if max_per_pdf > 0 and pdf_name and pdf_counter[pdf_name] >= max_per_pdf:
            continue
        selected_rows.append(idx)
        if oare_id:
            oare_counter[oare_id] += 1
        if pdf_name:
            pdf_counter[pdf_name] += 1
        if len(selected_rows) >= target_n:
            break

    if len(selected_rows) < target_n:
        remaining = [idx for idx in range(len(df)) if idx not in set(selected_rows)]
        selected_rows.extend(remaining[: max(0, target_n - len(selected_rows))])

    return df.iloc[selected_rows].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix aligned_train with OCR high tier data.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--aligned", default=None, help="Aligned train path.")
    parser.add_argument("--ocr", default=None, help="OCR parts directory or prefix.")
    parser.add_argument("--out", default=None, help="Output path.")
    parser.add_argument("--tiers", default=None, help="Comma separated tiers (default: high).")
    parser.add_argument("--tranches", default=None, help="Comma separated tranches (T1,T2,...).")
    parser.add_argument("--ratio", type=float, default=None, help="OCR ratio vs baseline.")
    parser.add_argument("--ratio-by-tranche", default=None, help="Comma separated ratio map, e.g. T1=0.1,T2=0.1.")
    parser.add_argument("--variants", default=None, help="Variants to keep (A,B,C).")
    parser.add_argument("--source-subtypes", default=None, help="Comma separated OCR source_subtype filter.")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR mixing.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = get_artifacts_dir(cfg)

    aligned_path = (
        Path(args.aligned)
        if args.aligned
        else Path(cfg.get("ocr_mix_aligned_path", artifacts_dir / "aligned" / "aligned_train.parquet"))
    )
    ocr_path = (
        Path(args.ocr)
        if args.ocr
        else Path(cfg.get("ocr_mix_ocr_glob", artifacts_dir / "ocr_pairs" / "part-0000.parquet"))
    )
    out_path = (
        Path(args.out)
        if args.out
        else Path(cfg.get("ocr_mix_out", artifacts_dir / "ocr_pairs" / "mixed_train.parquet"))
    )

    enable_ocr = bool(cfg.get("ocr_mix_enable", True))
    if args.disable_ocr:
        enable_ocr = False
    ratio = args.ratio if args.ratio is not None else _get_float(cfg, "ocr_mix_ratio", 0.2)
    if ratio <= 0:
        enable_ocr = False

    tiers = args.tiers or str(cfg.get("ocr_mix_quality_tiers", "high"))
    tier_set = {t.strip().lower() for t in tiers.split(",") if t.strip()}
    tranches = args.tranches or str(cfg.get("ocr_mix_tranches", ""))
    tranche_set = {t.strip().upper() for t in tranches.split(",") if t.strip()}
    ratio_by_tranche = _parse_ratio_map(
        args.ratio_by_tranche or str(cfg.get("ocr_mix_ratio_by_tranche", ""))
    )
    variants = args.variants or str(cfg.get("ocr_mix_variants", ""))
    variant_set = {v.strip().upper() for v in variants.split(",") if v.strip()}
    source_subtypes = args.source_subtypes or str(cfg.get("ocr_mix_source_subtypes", ""))
    source_subtype_set = {v.strip() for v in source_subtypes.split(",") if v.strip()}

    max_per_oare_id = _get_int(cfg, "ocr_mix_max_per_oare_id", 6)
    max_per_pdf = _get_int(cfg, "ocr_mix_max_per_pdf", 20)

    aligned_df = read_table(aligned_path)
    if variant_set and "src_norm_variant" in aligned_df.columns:
        aligned_df = aligned_df[aligned_df["src_norm_variant"].isin(variant_set)].reset_index(drop=True)

    aligned_df = aligned_df.copy()
    aligned_df["source"] = "aligned"

    if not enable_ocr:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".parquet":
            aligned_df.to_parquet(out_path, index=False)
        else:
            aligned_df.to_csv(out_path, index=False)
        print("OCR mixing disabled. Saved baseline only.")
        print(f"Output: {out_path}")
        return

    ocr_parts = list_ocr_parts(ocr_path)
    if not ocr_parts:
        raise FileNotFoundError(f"OCR parts not found at: {ocr_path}")

    ocr_frames: List[pd.DataFrame] = []
    tier_counts: Counter[str] = Counter()
    tranche_counts_total: Counter[str] = Counter()
    tranche_counts_selected: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    subtype_counts_total: Counter[str] = Counter()
    subtype_counts_selected: Counter[str] = Counter()
    for part_path in ocr_parts:
        df = pd.read_parquet(part_path)
        if "quality_tier" in df.columns and tier_set:
            df = df[df["quality_tier"].fillna("").astype(str).str.lower().isin(tier_set)]
        if tranche_set and "tranche" in df.columns:
            df = df[df["tranche"].fillna("").astype(str).str.upper().isin(tranche_set)]
        if variant_set and "src_norm_variant" in df.columns:
            df = df[df["src_norm_variant"].isin(variant_set)]
        if source_subtype_set and "source_subtype" in df.columns:
            df = df[df["source_subtype"].fillna("").astype(str).isin(source_subtype_set)]
        if df.empty:
            continue
        df = _dedupe_pairs(df)
        ocr_frames.append(df)
        if "quality_tier" in df.columns:
            tier_counts.update(df["quality_tier"].fillna("").astype(str).str.lower().tolist())
        if "tranche" in df.columns:
            tranche_counts_total.update(df["tranche"].fillna("").astype(str).str.upper().tolist())
        if "candidate_reason" in df.columns:
            reason_counts.update(df["candidate_reason"].fillna("").astype(str).tolist())
        if "source_subtype" in df.columns:
            subtype_counts_total.update(df["source_subtype"].fillna("").astype(str).tolist())

    if ocr_frames:
        ocr_all = pd.concat(ocr_frames, ignore_index=True)
    else:
        ocr_all = pd.DataFrame()

    target_n = max(int(round(len(aligned_df) * ratio)), 0)
    if not ocr_all.empty:
        ocr_all = ocr_all.copy()
        ocr_all["source"] = "ocr"
        if ratio_by_tranche and "tranche" in ocr_all.columns:
            tranche_frames: List[pd.DataFrame] = []
            for tranche, tranche_ratio in ratio_by_tranche.items():
                tranche_df = ocr_all[ocr_all["tranche"].fillna("").astype(str).str.upper() == tranche.upper()].reset_index(drop=True)
                if tranche_df.empty:
                    continue
                tranche_target_n = max(int(round(len(aligned_df) * tranche_ratio)), 0)
                if tranche_target_n <= 0:
                    continue
                selected = _quality_select(
                    tranche_df,
                    tranche_target_n,
                    max_per_oare_id=max_per_oare_id,
                    max_per_pdf=max_per_pdf,
                )
                tranche_frames.append(selected)
                if "tranche" in selected.columns:
                    tranche_counts_selected.update(selected["tranche"].fillna("").astype(str).str.upper().tolist())
                if "source_subtype" in selected.columns:
                    subtype_counts_selected.update(selected["source_subtype"].fillna("").astype(str).tolist())
            ocr_sample = pd.concat(tranche_frames, ignore_index=True) if tranche_frames else pd.DataFrame()
            ocr_sample = _dedupe_pairs(ocr_sample)
        else:
            ocr_sample = _quality_select(ocr_all, target_n, max_per_oare_id=max_per_oare_id, max_per_pdf=max_per_pdf)
            if "tranche" in ocr_sample.columns:
                tranche_counts_selected.update(ocr_sample["tranche"].fillna("").astype(str).str.upper().tolist())
            if "source_subtype" in ocr_sample.columns:
                subtype_counts_selected.update(ocr_sample["source_subtype"].fillna("").astype(str).tolist())
    else:
        ocr_sample = pd.DataFrame()

    all_cols = sorted(set(aligned_df.columns).union(set(ocr_sample.columns)))
    for col in all_cols:
        if col not in aligned_df.columns:
            aligned_df[col] = None
        if col not in ocr_sample.columns:
            ocr_sample[col] = None

    mixed_df = pd.concat([aligned_df[all_cols], ocr_sample[all_cols]], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        mixed_df.to_parquet(out_path, index=False)
    else:
        mixed_df.to_csv(out_path, index=False)

    stats = {
        "aligned_rows": int(len(aligned_df)),
        "ocr_rows_total": int(len(ocr_all)),
        "ocr_rows_selected": int(len(ocr_sample)),
        "mix_rows": int(len(mixed_df)),
        "ocr_ratio": ratio,
        "ratio_by_tranche": ratio_by_tranche,
        "max_per_oare_id": max_per_oare_id,
        "max_per_pdf": max_per_pdf,
        "tier_counts": dict(tier_counts),
        "tranche_counts_total": dict(tranche_counts_total),
        "tranche_counts_selected": dict(tranche_counts_selected),
        "candidate_reason_counts": dict(reason_counts),
        "source_subtype_total": dict(subtype_counts_total),
        "source_subtype_selected": dict(subtype_counts_selected),
        "tranche_filter": sorted(tranche_set),
        "source_subtype_filter": sorted(source_subtype_set),
        "aligned_gap_rate": compute_gap_rate(aligned_df.get("src_sent", [])),
        "ocr_gap_rate": compute_gap_rate(ocr_sample.get("src_sent", [])),
        "aligned_caps_rate": compute_caps_token_rate(aligned_df.get("src_sent", [])),
        "ocr_caps_rate": compute_caps_token_rate(ocr_sample.get("src_sent", [])),
        "aligned_src_avg_len": compute_avg_len(aligned_df.get("src_sent", [])),
        "ocr_src_avg_len": compute_avg_len(ocr_sample.get("src_sent", [])),
        "aligned_tgt_avg_len": compute_avg_len(aligned_df.get("tgt_sent", [])),
        "ocr_tgt_avg_len": compute_avg_len(ocr_sample.get("tgt_sent", [])),
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"Aligned rows: {len(aligned_df)}")
    print(f"OCR rows total: {len(ocr_all)}")
    print(f"OCR rows selected: {len(ocr_sample)}")
    print(f"Output: {out_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
