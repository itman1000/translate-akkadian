"""Mismatch audit utilities for Deep Past NMT.

This module is intentionally dependency-light (stdlib + pandas) so it can run
in Kaggle/Colab without extra installs.

It produces a per-row audit table with:
- multiple normalization levels (n0/n1/n2)
- lightweight similarity scores (char n-gram Jaccard)
- heuristic mismatch type labels (primary + secondary)
- optional "T90" subtyping tags (template collapse / numeric/name issues / gloss leak)
- alignment-shift hints (best_offset / best_sim / best_delta)

The goal is to make "preprocess debugging" fast:
identify whether mismatches are driven by formatting (unicode/space/punct),
missing gap markers, modern notation leakage, decoding accidents, or alignment shifts.

NOTE:
  This is an *audit* / triage utility. It is not intended to be a perfect linguistic
  analysis of Akkadian transliteration or English.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd


# -------------------------
# Normalization helpers
# -------------------------

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")

_SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "´": "'",
    "ʾ": "'",
    "ʼ": "'",
    "ʻ": "'",
    "′": "'",
}

_DASHES = {"–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-"}

_GAP_PAT = re.compile(r"<\s*(big_)?gap\s*>", re.IGNORECASE)
_LEX_BLOCK = re.compile(r"\s*<LEX>.*?</LEX>\s*", re.IGNORECASE | re.DOTALL)

# used only for n2 (classification-only)
_PUNCT_STRIP = re.compile(r"[!\"#$%&'()*+,\-./:;=?@\[\\\]^_`{|}~]")


def _to_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def strip_lex_gloss(text: str) -> str:
    """Remove the appended <LEX> ... </LEX> gloss payload."""

    return re.sub(r"\s+", " ", _LEX_BLOCK.sub(" ", _to_str(text))).strip()


def _canon_gap_tokens(text: str) -> str:
    return _GAP_PAT.sub(lambda m: "<big_gap>" if m.group(1) else "<gap>", text)


def norm_unicode(text: str) -> str:
    """Minimal Unicode/whitespace normalization."""

    s = _to_str(text)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_SUBSCRIPT_MAP)
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    for k, v in _DASHES.items():
        s = s.replace(k, v)
    s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")
    s = re.sub(r"[ \t\r\n]+", " ", s).strip()
    return s


def norm_n0(text: str) -> str:
    """n0: unicode + whitespace + gap token canonicalization."""

    return _canon_gap_tokens(norm_unicode(text))


def norm_n1(text: str) -> str:
    """n1: n0 + punctuation spacing normalization (keep case)."""

    s = norm_n0(text)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _mask_gaps(text: str) -> str:
    s = _canon_gap_tokens(text)
    return s.replace("<big_gap>", "__BIGGAP__").replace("<gap>", "__GAP__")


def _unmask_gaps(text: str) -> str:
    return text.replace("__BIGGAP__", "<big_gap>").replace("__GAP__", "<gap>")


def norm_n2(text: str) -> str:
    """n2: classification-only aggressive normalization.

    - lowercasing
    - broad punctuation stripping
    - keep <gap>/<big_gap> markers (masked during stripping)

    NOTE: n2 is NOT intended for scoring, only for robust mismatch triage.
    """

    s = norm_n1(text)
    s = _mask_gaps(s)
    s = s.lower()
    s = _PUNCT_STRIP.sub(" ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = _unmask_gaps(s)
    return s


# -------------------------
# Similarity (lightweight)
# -------------------------


def _char_ngrams(text: str, n: int) -> Iterable[str]:
    s = re.sub(r"\s+", "", _to_str(text))
    if not s:
        return []
    if len(s) <= n:
        return [s]
    return (s[i : i + n] for i in range(len(s) - n + 1))


def ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    """Char n-gram Jaccard similarity in [0,1]."""

    a = _to_str(a)
    b = _to_str(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = set(_char_ngrams(a, n))
    sb = set(_char_ngrams(b, n))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# -------------------------
# Flags / heuristics
# -------------------------


_MODERN_CHARS = set("˹˺")


def has_modern_marks(text: str) -> bool:
    """Detect modern editorial marks that often should not appear in predictions."""

    s0 = norm_n0(text)
    tmp = _mask_gaps(s0)
    if "<<" in s0 or ">>" in s0:
        return True
    if any(ch in s0 for ch in _MODERN_CHARS):
        return True
    if "[" in s0 or "]" in s0:
        return True
    # any <...> other than gaps
    if "<" in tmp or ">" in tmp:
        return True
    return False


_DECIMAL_DOT = re.compile(r"(?<=\d)\.(?=\d)")
_ELLIPSIS = re.compile(r"\.{3,}")

# Detect a *sentence boundary* marker followed by whitespace and a likely sentence start.
# This avoids false positives from decimals (1.5) and ellipses (...).
_SENT_BOUNDARY = re.compile(r"([!?]|\.)(?=\s+[\"\)\'\]]*[A-Z])")


def is_multi_sentence(text: str) -> bool:
    """Return True if the text likely contains 2+ sentences.

    Notes:
      - Newlines are treated as multi-sentence.
      - Decimal dots (e.g. 0.3333, 1.5) are ignored.
      - Ellipses ("..." or "…") are collapsed.

    This is intended for *audit classification*, not strict sentence segmentation.
    """

    s = _to_str(text)
    if "\n" in s or "\r" in s:
        return True

    s0 = norm_n0(s)
    if not s0:
        return False

    # protect decimals and ellipses before boundary detection
    s0 = _DECIMAL_DOT.sub("<DEC>", s0)
    s0 = s0.replace("…", "<ELL>")
    s0 = _ELLIPSIS.sub("<ELL>", s0)

    return bool(_SENT_BOUNDARY.search(s0))


def repetition_score(text: str) -> float:
    """Detect decoding loops by crude repetition statistics (>=1.0 is suspicious)."""

    s0 = norm_n0(text).lower()
    toks = re.findall(r"[a-z0-9']+", s0)
    if len(toks) < 8:
        return 0.0
    max_run = 1
    run = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    unique_ratio = len(set(toks)) / max(1, len(toks))
    return max(max_run / 5.0, (0.6 - unique_ratio) * 2)


# --- T90 secondary signals (cheap heuristics) ---


_DIGIT_PAT = re.compile(r"\d")

# transliteration-ish all-caps tokens with dot segments like "KÙ.BABBAR" or "URU.KI".
# We keep this as a *warning* signal (not an error), and we intentionally avoid
# flagging decimal numbers like "11.5".
_PUNCT_TRIM = "\"'`“”‘’()[]{}<>,;:!?."


def _is_allcaps_dot_logogram_token(token: str) -> bool:
    t = _to_str(token).strip(_PUNCT_TRIM)
    if not t or "." not in t:
        return False
    # ignore decimal-like tokens (pure digits and dots)
    if t.replace(".", "").isdigit():
        return False
    parts = [p for p in t.split(".") if p]
    if len(parts) < 2:
        return False
    # ignore English abbreviations like "U.S" / "A.B" where all segments are 1 char
    if all(len(p) == 1 for p in parts):
        return False
    # cased letters must be uppercase (handles Š/Ḫ/Í/Ù etc)
    if not t.isupper():
        return False
    return True

# very common in <LEX> hints
_LEX_LEAK_PAT = re.compile(r"<\s*/?\s*LEX\s*>", re.IGNORECASE)

# lemma=gloss patterns (e.g. kaspu=silver)
_LEMMA_EQ_PAT = re.compile(r"\b[a-z\-]{2,}\s*=\s*[a-z\-]{2,}\b", re.IGNORECASE)

# tokens that often indicate the model copied the source/gloss payload
_GLOSS_TOKEN_HINT_PAT = re.compile(r"\b(?:\||=)\b")

# word-ish tokens for name extraction
_WORD_PAT = re.compile(r"[A-Za-zŠṢḪṬšṣḫṭʾʼ'-]+")

# uppercase letters including common Akkadian diacritics
_UPPER_PAT = re.compile(r"^[A-ZŠṢḪṬ]")
_LOWER_PAT = re.compile(r"[a-zšṣḫṭ]")

_STOP_CAP_WORDS = {
    # sentence function words
    "If",
    "The",
    "A",
    "An",
    "In",
    "On",
    "At",
    "To",
    "From",
    "And",
    "Or",
    "But",
    "For",
    "When",
    "As",
    "This",
    "That",
    "These",
    "Those",
    "He",
    "She",
    "We",
    "They",
    "I",
    "You",
    "It",
    "His",
    "Her",
    "Our",
    "Your",
    "Their",
    # frequent generic starters in this corpus
    "Seal",
    "Thus",
    "Witnessed",
}


def digit_count(text: str) -> int:
    return len(_DIGIT_PAT.findall(_to_str(text)))


def looks_like_gloss_leak(text: str) -> bool:
    s = _to_str(text)
    if not s:
        return False
    if _LEX_LEAK_PAT.search(s):
        return True
    if _LEMMA_EQ_PAT.search(s):
        return True
    # pipes are very uncommon in normal English output for this task
    if "|" in s:
        return True
    return False


def looks_like_transliteration_leak(text: str) -> bool:
    s = _to_str(text)
    if not s:
        return False
    if "{" in s or "}" in s:
        return True
    if "<gap>" in s or "<big_gap>" in s:
        return True
    # Look for ALLCAPS dot-segment tokens (e.g., KÙ.BABBAR, SÍG.ḪI.A).
    # This avoids false positives on decimals like "11.5".
    for tok in s.split():
        if _is_allcaps_dot_logogram_token(tok):
            return True
    return False



def extract_name_tokens(text: str, *, max_tokens: int = 8) -> List[str]:
    """Extract "name-like" capitalized tokens from an English sentence.

    Heuristic:
      - ignore first token (sentence-initial)
      - token starts with uppercase (incl. Š Ṣ Ḫ Ṭ)
      - and has lowercase letters OR contains a hyphen
      - exclude a small stoplist of generic capitalized starters

    Returns up to max_tokens unique tokens (stable order).
    """

    s = norm_n1(text)
    toks = _WORD_PAT.findall(s)
    if not toks:
        return []
    out: List[str] = []
    seen = set()
    for i, t in enumerate(toks):
        if i == 0:
            continue
        if t in _STOP_CAP_WORDS:
            continue
        if not _UPPER_PAT.search(t):
            continue
        if ("-" not in t) and (not _LOWER_PAT.search(t)):
            # ALLCAPS token like "CITY" (rare), treat as not a proper name here
            continue
        key = t
        if key not in seen:
            out.append(t)
            seen.add(key)
        if len(out) >= max_tokens:
            break
    return out


@dataclass(frozen=True)
class AuditConfig:
    # alignment suspicion
    sim_th: float = 0.85
    margin: float = 0.08
    ngram_n: int = 4
    max_shift: int = 3  # check +/-K within group for shift

    # "template collapse" signal (T90 subtyping)
    template_min_count: int = 5
    template_sim_max: float = 0.55

    # "almost correct" bucket (separate from T90)
    near_match_th: float = 0.90  # classify as near-match when sim_self >= this


def _infer_order_cols(df: pd.DataFrame) -> List[str]:
    """Try to find columns that preserve within-document sentence order."""

    candidates = [
        "sent_idx",
        "sentence_idx",
        "line_start",
        "line_end",
        "line_no",
        "seg_idx",
        "position",
    ]
    return [c for c in candidates if c in df.columns]


def _shift_series(out: pd.DataFrame, group_col: Optional[str], col: str, k: int) -> pd.Series:
    if group_col and group_col in out.columns:
        return out.groupby(group_col)[col].shift(k)
    return out[col].shift(k)


def build_audit_df(
    df: pd.DataFrame,
    *,
    src_col: str,
    ref_col: str,
    pred_col: str,
    group_col: Optional[str] = None,
    cfg: AuditConfig = AuditConfig(),
) -> pd.DataFrame:
    """Return an audit dataframe with mismatch types and diagnostics."""

    out = df.copy()

    # Keep original columns as-is; only add *derived* columns.
    # NOTE: historically we created `src_raw` here, but that was confusing when
    # the caller already provided `src_no_gloss`. We now avoid `src_raw`.

    # Optional convenience: if the caller passes `src` that still includes <LEX>,
    # create `src_no_gloss` only when it does not already exist.
    if "src_no_gloss" not in out.columns and src_col in out.columns:
        out["src_no_gloss"] = out[src_col].map(strip_lex_gloss)

    out["ref_raw"] = out[ref_col].map(_to_str)
    out["pred_raw"] = out[pred_col].map(_to_str)

    # normalization views
    out["ref_n0"] = out["ref_raw"].map(norm_n0)
    out["pred_n0"] = out["pred_raw"].map(norm_n0)
    out["ref_n1"] = out["ref_raw"].map(norm_n1)
    out["pred_n1"] = out["pred_raw"].map(norm_n1)
    out["ref_n2"] = out["ref_raw"].map(norm_n2)
    out["pred_n2"] = out["pred_raw"].map(norm_n2)

    # lightweight sim (use n2)
    out["sim_self"] = [
        ngram_jaccard(a, b, n=cfg.ngram_n) for a, b in zip(out["pred_n2"], out["ref_n2"])
    ]

    out["len_ref"] = out["ref_n0"].str.len()
    out["len_pred"] = out["pred_n0"].str.len()
    out["len_ratio"] = (out["len_pred"] + 1) / (out["len_ref"] + 1)

    out["ref_gap"] = out["ref_n0"].str.count("<gap>")
    out["ref_big_gap"] = out["ref_n0"].str.count("<big_gap>")
    out["pred_gap"] = out["pred_n0"].str.count("<gap>")
    out["pred_big_gap"] = out["pred_n0"].str.count("<big_gap>")

    out["pred_has_modern"] = out["pred_raw"].map(has_modern_marks)
    out["pred_multi_sentence"] = out["pred_raw"].map(is_multi_sentence)
    out["pred_repetition"] = out["pred_raw"].map(repetition_score)

    # T90-ish secondary signals
    out["ref_digit"] = out["ref_raw"].map(digit_count)
    out["pred_digit"] = out["pred_raw"].map(digit_count)
    out["pred_gloss_leak"] = out["pred_raw"].map(looks_like_gloss_leak)
    out["pred_translit_leak"] = out["pred_raw"].map(looks_like_transliteration_leak)

    # name tokens
    ref_names = out["ref_raw"].map(extract_name_tokens)
    pred_names = out["pred_raw"].map(extract_name_tokens)
    out["ref_name_cnt"] = ref_names.map(len)
    out["pred_name_cnt"] = pred_names.map(len)
    out["ref_name_sample"] = ref_names.map(lambda xs: ";".join(xs[:3]))
    out["pred_name_sample"] = pred_names.map(lambda xs: ";".join(xs[:3]))

    # template collapse signal (based on pred_n2)
    pred_counts = out["pred_n2"].value_counts(dropna=False)
    out["pred_template_count"] = out["pred_n2"].map(pred_counts).fillna(0).astype(int)
    top = pred_counts.head(50)
    rank_map = {k: i + 1 for i, k in enumerate(top.index.tolist())}
    out["pred_template_rank"] = out["pred_n2"].map(rank_map).fillna(0).astype(int)

    # within-document ordering for shift detection
    if group_col and group_col in out.columns:
        order_cols = _infer_order_cols(out)
        if order_cols:
            out = out.sort_values([group_col] + order_cols).reset_index(drop=True)
        out["doc_pos"] = out.groupby(group_col).cumcount()
    else:
        out["doc_pos"] = range(len(out))

    # shift similarities for offsets within +/-K
    max_k = int(cfg.max_shift) if int(cfg.max_shift) > 0 else 1
    # keep k=1 columns for quick inspection
    prev_ref1 = _shift_series(out, group_col, "ref_n2", 1).fillna("")
    next_ref1 = _shift_series(out, group_col, "ref_n2", -1).fillna("")
    out["sim_prev"] = [
        ngram_jaccard(a, b, n=cfg.ngram_n) for a, b in zip(out["pred_n2"], prev_ref1)
    ]
    out["sim_next"] = [
        ngram_jaccard(a, b, n=cfg.ngram_n) for a, b in zip(out["pred_n2"], next_ref1)
    ]

    # optional merge hint (2-sentence)
    merge_next = out["ref_n2"] + " " + _shift_series(out, group_col, "ref_n2", -1).fillna("")
    merge_prev = _shift_series(out, group_col, "ref_n2", 1).fillna("") + " " + out["ref_n2"]
    out["sim_merge_next"] = [
        ngram_jaccard(a, b, n=cfg.ngram_n) for a, b in zip(out["pred_n2"], merge_next)
    ]
    out["sim_merge_prev"] = [
        ngram_jaccard(a, b, n=cfg.ngram_n) for a, b in zip(out["pred_n2"], merge_prev)
    ]

    # best_offset / best_sim / best_delta from shift candidates
    # (offset=0 means self)
    sim_by_off: dict[int, List[float]] = {0: out["sim_self"].tolist()}
    if max_k >= 1:
        sim_by_off[-1] = out["sim_prev"].tolist()
        sim_by_off[1] = out["sim_next"].tolist()
    for k in range(2, max_k + 1):
        prev_ref = _shift_series(out, group_col, "ref_n2", k).fillna("")
        next_ref = _shift_series(out, group_col, "ref_n2", -k).fillna("")
        sim_by_off[-k] = [
            ngram_jaccard(a, b, n=cfg.ngram_n) for a, b in zip(out["pred_n2"], prev_ref)
        ]
        sim_by_off[k] = [
            ngram_jaccard(a, b, n=cfg.ngram_n) for a, b in zip(out["pred_n2"], next_ref)
        ]

    best_off: List[int] = []
    best_sim: List[float] = []
    n_rows = len(out)
    offs = sorted(sim_by_off.keys())
    for i in range(n_rows):
        bo = 0
        bs = sim_by_off[0][i]
        for off in offs:
            s = sim_by_off[off][i]
            if s > bs:
                bs = s
                bo = off
        best_off.append(int(bo))
        best_sim.append(float(bs))

    out["best_offset"] = best_off
    out["best_sim"] = best_sim
    out["best_delta"] = out["best_sim"] - out["sim_self"]

    def _classify_row(r: pd.Series) -> Tuple[str, str, str, str]:
        sec: List[str] = []

        # generic tags
        if bool(r.get("pred_multi_sentence")):
            sec.append("TAG_MULTI_SENT")
        if float(r.get("pred_repetition") or 0.0) >= 1.0:
            sec.append("TAG_REPEAT")
        if bool(r.get("pred_has_modern")):
            sec.append("TAG_MODERN")
        if (int(r.get("pred_gap") or 0), int(r.get("pred_big_gap") or 0)) != (
            int(r.get("ref_gap") or 0),
            int(r.get("ref_big_gap") or 0),
        ):
            sec.append("TAG_GAP_MISMATCH")
        if float(r.get("len_ratio") or 1.0) < 0.35:
            sec.append("TAG_TRUNC")
        if float(r.get("len_ratio") or 1.0) > 2.5:
            sec.append("TAG_TOO_LONG")

        # "T90"-ish tags (still useful even if primary becomes something else)
        if bool(r.get("pred_gloss_leak")):
            sec.append("TAG_GLOSS_LEAK")
        if bool(r.get("pred_translit_leak")):
            sec.append("TAG_COPY_SRC")

        ref_d = int(r.get("ref_digit") or 0)
        pred_d = int(r.get("pred_digit") or 0)
        if ref_d > 0 and pred_d == 0:
            sec.append("TAG_NUM_DROP")
        elif pred_d > 0 and ref_d == 0:
            sec.append("TAG_NUM_HALLUC")
        elif abs(pred_d - ref_d) >= 3:
            sec.append("TAG_NUM_DIFF")

        ref_nc = int(r.get("ref_name_cnt") or 0)
        pred_nc = int(r.get("pred_name_cnt") or 0)
        if ref_nc >= 1 and pred_nc == 0:
            sec.append("TAG_NAME_DROP")
        elif pred_nc >= 1 and ref_nc == 0:
            sec.append("TAG_NAME_HALLUC")
        elif ref_nc >= 2 and pred_nc <= max(0, ref_nc - 2):
            sec.append("TAG_NAME_DROP")

        # template collapse (frequent pred + low sim)
        if (
            int(r.get("pred_template_count") or 0) >= int(cfg.template_min_count)
            and float(r.get("sim_self") or 0.0) < float(cfg.template_sim_max)
        ):
            sec.append("TAG_TEMPLATE")

        # --------------- primary type ---------------
        # style-only matches
        if r.get("pred_raw", "") == r.get("ref_raw", ""):
            primary = "T00_EXACT"
        elif r.get("pred_n0", "") == r.get("ref_n0", ""):
            primary = "T01_UNICODE_OR_WS"
        elif r.get("pred_n1", "") == r.get("ref_n1", ""):
            primary = "T02_PUNCT_SPACING"
        elif r.get("pred_n2", "") == r.get("ref_n2", ""):
            primary = "T03_CASE_OR_MINOR_STYLE"
        else:
            # alignment suspicions (shift)
            bo = int(r.get("best_offset") or 0)
            bs = float(r.get("best_sim") or 0.0)
            bd = float(r.get("best_delta") or 0.0)

            if bo != 0 and (bs > cfg.sim_th) and (bd > cfg.margin):
                primary = "T30_ALIGN_SHIFT_PREV" if bo < 0 else "T30_ALIGN_SHIFT_NEXT"
                sec.append("TAG_NEEDS_ALIGN")
                sec.append(f"TAG_SHIFT_K{abs(bo)}")
            # alignment suspicions (merge)
            elif (float(r.get("sim_merge_next") or 0.0) > cfg.sim_th) and (
                (float(r.get("sim_merge_next") or 0.0) - float(r.get("sim_self") or 0.0)) > cfg.margin
            ):
                primary = "T31_ALIGN_MERGE_WITH_NEXT"
                sec.append("TAG_NEEDS_ALIGN")
            elif (float(r.get("sim_merge_prev") or 0.0) > cfg.sim_th) and (
                (float(r.get("sim_merge_prev") or 0.0) - float(r.get("sim_self") or 0.0)) > cfg.margin
            ):
                primary = "T31_ALIGN_MERGE_WITH_PREV"
                sec.append("TAG_NEEDS_ALIGN")
            # decode accidents
            elif bool(r.get("pred_multi_sentence")):
                primary = "T20_MULTI_SENTENCE"
            elif float(r.get("pred_repetition") or 0.0) >= 1.0:
                primary = "T22_REPETITION"
            elif (int(r.get("pred_gap") or 0), int(r.get("pred_big_gap") or 0)) != (
                int(r.get("ref_gap") or 0),
                int(r.get("ref_big_gap") or 0),
            ):
                primary = "T10_GAP_MARKER"
            elif bool(r.get("pred_has_modern")):
                primary = "T11_MODERN_NOTATION"
            elif float(r.get("len_ratio") or 1.0) < 0.35:
                primary = "T21_TRUNCATION"
            elif float(r.get("sim_self") or 0.0) >= float(getattr(cfg, "near_match_th", 0.90)):
                primary = "T04_NEAR_MATCH"
            else:
                primary = "T90_SEMANTIC_OR_MODEL"

        # T90 subtyping for quicker triage
        t90_reason = ""
        if primary == "T90_SEMANTIC_OR_MODEL":
            sec_set = set(sec)
            if "TAG_GLOSS_LEAK" in sec_set:
                t90_reason = "T90_GLOSS_LEAK"
            elif "TAG_COPY_SRC" in sec_set:
                t90_reason = "T90_COPY_SRC"
            elif "TAG_TEMPLATE" in sec_set:
                t90_reason = "T90_TEMPLATE_COLLAPSE"
            elif ("TAG_NUM_DROP" in sec_set) or ("TAG_NUM_HALLUC" in sec_set) or ("TAG_NUM_DIFF" in sec_set):
                t90_reason = "T90_NUMERIC"
            elif ("TAG_NAME_DROP" in sec_set) or ("TAG_NAME_HALLUC" in sec_set):
                t90_reason = "T90_NAME"
            else:
                t90_reason = "T90_OTHER"

        route = {
            "T00_EXACT": "OK",
            "T01_UNICODE_OR_WS": "OUTPUT_PP",
            "T02_PUNCT_SPACING": "OUTPUT_PP",
            "T03_CASE_OR_MINOR_STYLE": "OUTPUT_PP",
            "T04_NEAR_MATCH": "OUTPUT_PP",
            "T10_GAP_MARKER": "INPUT_PP+OUTPUT_PP",
            "T11_MODERN_NOTATION": "INPUT_PP+OUTPUT_PP",
            "T20_MULTI_SENTENCE": "DECODE+OUTPUT_PP",
            "T21_TRUNCATION": "DECODE",
            "T22_REPETITION": "DECODE",
            "T30_ALIGN_SHIFT_PREV": "ALIGN",
            "T30_ALIGN_SHIFT_NEXT": "ALIGN",
            "T31_ALIGN_MERGE_WITH_NEXT": "ALIGN",
            "T31_ALIGN_MERGE_WITH_PREV": "ALIGN",
            "T90_SEMANTIC_OR_MODEL": "MODEL",
        }.get(primary, "MODEL")

        return primary, "|".join(sorted(set(sec))), route, t90_reason

    tmp = out.apply(_classify_row, axis=1, result_type="expand")
    out["type_primary"] = tmp[0]
    out["type_secondary"] = tmp[1]
    out["fix_route"] = tmp[2]
    out["t90_reason"] = tmp[3]

    return out
