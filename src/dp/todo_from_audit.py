"""Utilities to turn `val_audit.csv` into an actionable TODO list.

This module is intended to be used *after* you generated `val_audit.csv`.
It does not depend on the training code, so you can drop it into the repo and run
it right away.

The TODO list is useful for:
- spotting likely alignment errors (off-by-one shifts / merge candidates)
- spotting deterministic issues that can be handled by preprocessing / postproc
  (numbers/names mismatch, gloss leak, copy-src, truncation, template collapse)

The output is a CSV sorted by descending `priority`.

The implementation is deliberately defensive: it only requires a small subset of
columns and will gracefully degrade if some columns are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValTodoConfig:
    """Configuration for TODO extraction."""

    # Hard limits to keep the todo file readable.
    max_rows: Optional[int] = 500
    max_rows_per_doc: Optional[int] = 8

    # Filter out extremely low-signal items.
    min_priority: float = 0.01

    # Whether to include some low-priority categories.
    include_near_match: bool = True
    include_style: bool = False


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return int(x)
    except Exception:
        return default


def _contains_tag(type_secondary: str, tag: str) -> bool:
    # type_secondary is sometimes something like "TAG_NAME_DROP|TAG_NUM_DROP"
    return tag in type_secondary.split("|") if type_secondary else False


def _with_doc_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add prev/next context columns when `oare_id` + `doc_pos` exist."""

    if "oare_id" not in df.columns or "doc_pos" not in df.columns:
        return df

    out = df.copy()
    out = out.sort_values(["oare_id", "doc_pos"], kind="mergesort")

    # Use the cleaned-ish columns if present.
    ref_col = "ref" if "ref" in out.columns else None
    src_col = "src_no_gloss" if "src_no_gloss" in out.columns else ("src" if "src" in out.columns else None)

    if ref_col:
        out["ref_prev"] = out.groupby("oare_id")[ref_col].shift(1)
        out["ref_next"] = out.groupby("oare_id")[ref_col].shift(-1)

    if src_col:
        out["src_prev"] = out.groupby("oare_id")[src_col].shift(1)
        out["src_next"] = out.groupby("oare_id")[src_col].shift(-1)

    return out


def _infer_alignment_target(row: pd.Series) -> Tuple[Optional[int], str]:
    """Return (suggested_doc_pos, note)."""

    doc_pos = row.get("doc_pos")
    best_offset = row.get("best_offset")
    if doc_pos is None or best_offset is None:
        return None, ""

    try:
        doc_pos_i = int(doc_pos)
        best_off_i = int(best_offset)
    except Exception:
        return None, ""

    if best_off_i == 0:
        return None, ""

    target = doc_pos_i + best_off_i
    direction = "prev" if best_off_i < 0 else "next"
    note = f"Check alignment: prediction matches {direction} ref better (offset={best_off_i}, target_doc_pos={target})."
    return target, note


def build_val_todo_df(audit_df: pd.DataFrame, cfg: ValTodoConfig = ValTodoConfig()) -> pd.DataFrame:
    """Build a TODO dataframe from an audit dataframe."""

    if audit_df is None or len(audit_df) == 0:
        return pd.DataFrame()

    df = _with_doc_context(audit_df)

    # Normalize expected string columns to avoid NaN propagation.
    for c in ["type_primary", "type_secondary", "fix_route", "t90_reason"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace("nan", "")

    todo_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        tp = _as_str(row.get("type_primary"))
        ts = _as_str(row.get("type_secondary"))
        fr = _as_str(row.get("fix_route"))
        t90 = _as_str(row.get("t90_reason"))

        sim_self = _as_float(row.get("sim_self"), default=0.0)
        len_ratio = _as_float(row.get("len_ratio"), default=1.0)
        best_delta = _as_float(row.get("best_delta"), default=0.0)
        best_sim = _as_float(row.get("best_sim"), default=0.0)

        ref_digit = _as_int(row.get("ref_digit"), default=0)
        pred_digit = _as_int(row.get("pred_digit"), default=0)
        ref_name_cnt = _as_int(row.get("ref_name_cnt"), default=0)
        pred_name_cnt = _as_int(row.get("pred_name_cnt"), default=0)

        pred_template_count = _as_int(row.get("pred_template_count"), default=0)
        pred_template_rank = _as_int(row.get("pred_template_rank"), default=0)

        todo_type = ""
        priority = 0.0
        action = ""

        # 1) Alignment candidates (data issues)
        if tp.startswith("T30_ALIGN_SHIFT"):
            todo_type = "ALIGN_SHIFT"
            priority = max(best_delta, 0.0)
            suggested_pos, note = _infer_alignment_target(row)
            action = note

        elif tp.startswith("T31_ALIGN_MERGE"):
            todo_type = "ALIGN_MERGE"
            sim_merge_next = _as_float(row.get("sim_merge_next"), default=0.0)
            sim_merge_prev = _as_float(row.get("sim_merge_prev"), default=0.0)
            priority = max(sim_merge_next - sim_self, sim_merge_prev - sim_self, 0.0)
            action = "Check segmentation: merging with neighbor ref may match better (see sim_merge_*)."

        # 2) Deterministic preprocess / postprocess candidates
        elif t90 == "T90_NUMERIC" or _contains_tag(ts, "TAG_NUM_DROP") or _contains_tag(ts, "TAG_NUM_HALLUC"):
            todo_type = "NUMERIC_PRESERVE"
            priority = float(abs(ref_digit - pred_digit)) + (0.25 if ref_digit == pred_digit else 0.0)
            action = "Numbers differ between ref/pred. Consider number placeholders or copy constraints."

        elif t90 == "T90_NAME" or _contains_tag(ts, "TAG_NAME_DROP") or _contains_tag(ts, "TAG_NAME_HALLUC"):
            todo_type = "NAME_PRESERVE"
            priority = float(abs(ref_name_cnt - pred_name_cnt)) + (0.25 if ref_name_cnt == pred_name_cnt else 0.0)
            action = "Name counts differ. Consider PN/GN placeholders, adding PN/GN to gloss hints, or copy constraints."

        elif t90 == "T90_GLOSS_LEAK" or _as_int(row.get("pred_gloss_leak"), 0) == 1:
            todo_type = "GLOSS_LEAK"
            priority = 1.0
            action = "Prediction contains <LEX>…</LEX>. Strip it in output post-process and/or adjust training prompts."

        elif t90 == "T90_COPY_SRC" or _as_int(row.get("pred_translit_leak"), 0) == 1:
            todo_type = "COPY_SRC"
            priority = max(1.0 - sim_self, 0.0)
            action = "Model copied transliteration/source-like tokens. Consider stronger constraints or cleaning." 

        elif t90 == "T90_TEMPLATE_COLLAPSE" or tp in {"T22_REPETITION"} or pred_template_count >= 5:
            todo_type = "TEMPLATE_OR_REPETITION"
            priority = float(max(pred_template_count, 1))
            action = "Many predictions share a template. Consider stronger repetition constraints / decoding tweaks." 

        elif tp == "T21_TRUNCATION":
            todo_type = "TRUNCATION"
            priority = max(1.0 - len_ratio, 0.0)
            action = "Prediction seems too short. Consider decode length limits / EOS settings."

        elif tp == "T20_MULTI_SENTENCE":
            todo_type = "MULTI_SENTENCE"
            priority = 0.5
            action = "Prediction contains multiple sentences. Consider force_single_sentence / sentence merging."

        elif tp == "T11_MODERN_NOTATION":
            todo_type = "MODERN_NOTATION"
            priority = 0.5
            action = "Prediction includes modern notation (e.g., x). Consider normalization / constraints." 

        elif tp == "T10_GAP_MARKER":
            todo_type = "GAP_MARKER"
            priority = 0.5
            action = "Gap marker mismatch. Consider normalizing xxx / […] consistently in preprocessing."

        elif tp == "T04_NEAR_MATCH" and cfg.include_near_match:
            todo_type = "NEAR_MATCH"
            priority = sim_self
            action = "Almost correct. Consider small output normalization (diacritics, punctuation, hyphens)."

        elif tp in {"T01_SPACE_OR_PUNCT", "T03_CASE_OR_MINOR_STYLE"} and cfg.include_style:
            todo_type = "STYLE_MINOR"
            priority = sim_self
            action = "Minor style difference. Consider output normalization."

        # Skip low-signal / semantic-only items by default.
        if not todo_type:
            continue

        if priority < cfg.min_priority:
            continue

        rec: Dict[str, Any] = {
            "todo_type": todo_type,
            "priority": priority,
            "fix_route": fr,
            "type_primary": tp,
            "type_secondary": ts,
            "t90_reason": t90,
            "sim_self": sim_self,
            "best_sim": best_sim,
            "best_delta": best_delta,
            "best_offset": row.get("best_offset"),
            "len_ratio": len_ratio,
            "ref_digit": ref_digit,
            "pred_digit": pred_digit,
            "ref_name_cnt": ref_name_cnt,
            "pred_name_cnt": pred_name_cnt,
            "pred_template_count": pred_template_count,
            "pred_template_rank": pred_template_rank,
            "action": action,
            "row_index": idx,
        }

        # Attach identifiers / text context when present.
        for c in [
            "oare_id",
            "doc_pos",
            "src",
            "src_no_gloss",
            "ref",
            "pred",
            "ref_prev",
            "ref_next",
            "src_prev",
            "src_next",
        ]:
            if c in df.columns:
                rec[c] = row.get(c)

        todo_rows.append(rec)

    if not todo_rows:
        return pd.DataFrame()

    out = pd.DataFrame(todo_rows)

    # Sort by priority descending, then by best_delta etc.
    sort_cols = ["priority"]
    for c in ["best_delta", "best_sim", "sim_self"]:
        if c in out.columns:
            sort_cols.append(c)

    out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort")

    # Apply per-doc cap.
    if cfg.max_rows_per_doc is not None and "oare_id" in out.columns:
        out = (
            out.groupby("oare_id", group_keys=False)
            .head(int(cfg.max_rows_per_doc))
            .reset_index(drop=True)
        )

    # Apply global cap.
    if cfg.max_rows is not None:
        out = out.head(int(cfg.max_rows)).reset_index(drop=True)

    # Add rank column.
    out.insert(0, "todo_rank", np.arange(1, len(out) + 1))

    return out
