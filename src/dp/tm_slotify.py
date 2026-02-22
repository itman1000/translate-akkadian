"""TM++ slotification utilities (leak-safe by design).

This module implements a conservative "slotify -> retrieve -> restore" workflow
to use translation-memory (TM) pairs as *candidate generators*.

Key idea
--------
- Build a TF-IDF char n-gram index over **slotified source** strings.
- For each query source, retrieve top-K similar sources.
- Convert retrieved target (English) into a slotified template using the retrieved
  source slots, then **restore** the slots using the query slot map.

Leak safety
-----------
This module itself does not read any "gold" from validation/test.
Leak prevention is achieved operationally by building the TM index from
*train-only* data (e.g., per-fold train split). The CLI wrappers (tm_index /
tm_generate_candidates) also provide guardrails like excluding exact self-matches.

The implementation is dependency-light and follows the repository style:
usable via `python -m dp.tm_*` CLIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import re

import pandas as pd

from .audit_mismatch import norm_n0, strip_lex_gloss
from .utils import clean_text


# -------------------------
# Token helpers
# -------------------------

# Split a token into (prefix_punct, core, suffix_punct).
# This is intentionally conservative: we only strip *common* edge punctuation.
# We keep angle brackets so placeholders like "<PN_1>" stay in "core".
_EDGE_PUNCT_RE = re.compile(r'^([\"\'\(\[\{]*)(.*?)([\"\'\)\]\},\.;:\?!]*)$')

# Standalone numeric token patterns (avoid touching transliteration like "lu2").
# - digits
# - optional decimal part
# - optional simple fraction
_NUM_CORE_RE = re.compile(r"^\d+(?:[.,]\d+)?(?:/\d+)?$")

# Placeholder core patterns we generate/expect.
_PLACEHOLDER_CORE_RE = re.compile(r"^<(PN|NUM|UNIT)_(\d+)>$")


def _split_edge_punct(token: str) -> Tuple[str, str, str]:
    m = _EDGE_PUNCT_RE.match(token)
    if not m:
        return "", token, ""
    return m.group(1), m.group(2), m.group(3)


def _is_placeholder_core(core: str) -> bool:
    return bool(_PLACEHOLDER_CORE_RE.match(core)) or core in {"<gap>", "<big_gap>"}


# -------------------------
# Lexicon-driven PN slotting
# -------------------------

def _load_lexicon_forms(path: Path, types: Sequence[str]) -> List[str]:
    df = pd.read_csv(path, usecols=["type", "form"]).dropna(subset=["type", "form"])
    want = {t.strip().upper() for t in types if str(t).strip()}
    if not want:
        return []
    df = df[df["type"].astype(str).str.upper().isin(want)]
    forms = (
        df["form"]
        .astype(str)
        .map(lambda x: re.sub(r"\s+", " ", x).strip())
        .tolist()
    )
    forms = [f for f in forms if f]
    # de-dup
    return sorted(set(forms))


@lru_cache(maxsize=4)
def _build_pn_token_map_cached(lexicon_path: str, types_key: str) -> Tuple[Dict[Tuple[str, ...], str], int]:
    """Build a token sequence -> entity-type map for PN slotification.

    We intentionally **unify** PN/GN into a single type "PN" to match the plan:
    "人名/地名 → <PN1> <PN2> ...".
    """
    from .placeholders import build_token_map

    path = Path(lexicon_path)
    if not path.exists():
        return {}, 1

    types = [t for t in types_key.split(",") if t.strip()]
    forms = _load_lexicon_forms(path, types=types)

    # Unify all to PN
    forms_by_type = {"PN": forms}
    token_map, max_len = build_token_map(forms_by_type)
    return token_map, max_len


# -------------------------
# Slot maps
# -------------------------

@dataclass
class SlotifyConfig:
    """Controls how we slotify source strings.

    Parameters
    ----------
    lexicon_path:
      Optional OA lexicon CSV that contains at least columns: "type", "form".
      The provided helper file `OA_Lexicon_eBL_PN_GN_no_word_overlap.csv` is ideal.

    lexicon_types:
      Which types to use from the lexicon (default: PN,GN).
      NOTE: even if GN is included, we unify them into a single slot type "PN".

    placeholder_filter_mode:
      Placeholder matching filter (currently reserved; kept for API stability).
      We keep this field to allow future tightening without changing CLI signatures.

    strip_lex_from_src:
      If True, removes appended "<LEX> ... </LEX>" blocks before slotifying.
      This is important when the training pipeline uses gloss augmentation.

    slot_numbers / slot_units:
      Whether to replace standalone numeric tokens / unit tokens with slots.

    unit_tokens:
      Lowercased unit tokens to slotify.
    """

    lexicon_path: Optional[Path] = None
    lexicon_types: Tuple[str, ...] = ("PN", "GN")
    placeholder_filter_mode: str = "all"

    strip_lex_from_src: bool = False

    slot_numbers: bool = True
    slot_units: bool = True

    unit_tokens: Tuple[str, ...] = (
        # weights/volume/time tokens that frequently appear in the reference style
        "mina",
        "minas",
        "shekel",
        "shekels",
        "talent",
        "talents",
        "grain",
        "grains",
        "litre",
        "litres",
        "week",
        "weeks",
        "month",
        "months",
        "day",
        "days",
        "year",
        "years",
    )

    def to_dict(self) -> Dict[str, object]:
        return {
            "lexicon_path": str(self.lexicon_path) if self.lexicon_path else "",
            "lexicon_types": ",".join(self.lexicon_types),
            "placeholder_filter_mode": str(self.placeholder_filter_mode),
            "strip_lex_from_src": bool(self.strip_lex_from_src),
            "slot_numbers": bool(self.slot_numbers),
            "slot_units": bool(self.slot_units),
            "unit_tokens": list(self.unit_tokens),
        }

    @staticmethod
    def from_dict(payload: Dict[str, object]) -> "SlotifyConfig":
        lexicon_path = str(payload.get("lexicon_path") or "").strip()
        lexicon_types = str(payload.get("lexicon_types") or "PN,GN").strip()
        unit_tokens = payload.get("unit_tokens")
        if isinstance(unit_tokens, (list, tuple)):
            unit_tokens_t = tuple(str(x).strip().lower() for x in unit_tokens if str(x).strip())
        else:
            unit_tokens_t = SlotifyConfig().unit_tokens

        return SlotifyConfig(
            lexicon_path=Path(lexicon_path) if lexicon_path else None,
            lexicon_types=tuple(t.strip().upper() for t in lexicon_types.split(",") if t.strip()) or ("PN", "GN"),
            placeholder_filter_mode=str(payload.get("placeholder_filter_mode") or "all"),
            strip_lex_from_src=bool(payload.get("strip_lex_from_src", False)),
            slot_numbers=bool(payload.get("slot_numbers", True)),
            slot_units=bool(payload.get("slot_units", True)),
            unit_tokens=unit_tokens_t,
        )


@dataclass
class SlotMap:
    """Holds placeholder -> original text mappings."""

    # Unified PN (includes PN+GN)
    pn_map: Dict[str, str]
    # numbers (standalone numeric tokens)
    num_map: Dict[str, str]
    # units (English unit words)
    unit_map: Dict[str, str]

    def combined(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        out.update(self.pn_map)
        out.update(self.num_map)
        out.update(self.unit_map)
        return out

    def counts(self) -> Dict[str, int]:
        return {
            "pn": len(self.pn_map),
            "num": len(self.num_map),
            "unit": len(self.unit_map),
        }


# -------------------------
# Slotify / restore
# -------------------------

def _slotify_pn(text: str, cfg: SlotifyConfig) -> Tuple[str, Dict[str, str]]:
    """Slotify PN/GN entities in a *source* string using lexicon forms."""
    if not cfg.lexicon_path:
        return text, {}

    lex_path = Path(cfg.lexicon_path)
    if not lex_path.exists():
        return text, {}

    # Build token_map once.
    types_key = ",".join(cfg.lexicon_types)
    token_map, max_len = _build_pn_token_map_cached(str(lex_path), types_key)
    if not token_map:
        return text, {}

    # We implement a punctuation-tolerant greedy matcher.
    # The lexicon forms are whitespace token sequences; we match against token "core"
    # (edge punctuation stripped) to avoid missing "Name," etc.
    tokens = text.split()
    out_tokens: List[str] = []

    counter = 0
    text_to_ph: Dict[str, str] = {}

    i = 0
    while i < len(tokens):
        matched = False
        # longest-first
        for length in range(min(max_len, len(tokens) - i), 0, -1):
            seq_tokens = tokens[i : i + length]
            seq_core: List[str] = []
            for tok in seq_tokens:
                _pre, core, _suf = _split_edge_punct(tok)
                seq_core.append(core)
            seq = tuple(seq_core)
            if seq not in token_map:
                continue

            # unify to PN
            text_val = " ".join(seq_core)
            if text_val in text_to_ph:
                ph = text_to_ph[text_val]
            else:
                counter += 1
                ph = f"<PN_{counter}>"
                text_to_ph[text_val] = ph
            # Preserve punctuation: prefix from first token, suffix from last token.
            pre0, _c0, _s0 = _split_edge_punct(seq_tokens[0])
            _pL, _cL, sufL = _split_edge_punct(seq_tokens[-1])
            out_tokens.append(pre0 + ph + sufL)

            i += length
            matched = True
            break

        if not matched:
            out_tokens.append(tokens[i])
            i += 1

    pn_map = {ph: txt for txt, ph in text_to_ph.items()}
    return " ".join(out_tokens), pn_map


def _slotify_numbers_and_units(text: str, cfg: SlotifyConfig) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    unit_set = {t.strip().lower() for t in cfg.unit_tokens if str(t).strip()}

    tokens = text.split()
    out: List[str] = []

    num_counter = 0
    unit_counter = 0
    num_text_to_ph: Dict[str, str] = {}
    unit_text_to_ph: Dict[str, str] = {}

    for tok in tokens:
        pre, core, suf = _split_edge_punct(tok)

        # never touch our placeholders/gaps
        if _is_placeholder_core(core):
            out.append(tok)
            continue

        replaced = None

        if cfg.slot_numbers and _NUM_CORE_RE.match(core):
            if core in num_text_to_ph:
                replaced = num_text_to_ph[core]
            else:
                num_counter += 1
                replaced = f"<NUM_{num_counter}>"
                num_text_to_ph[core] = replaced

        if replaced is None and cfg.slot_units and core.lower() in unit_set:
            key = core  # preserve original casing in map
            if key in unit_text_to_ph:
                replaced = unit_text_to_ph[key]
            else:
                unit_counter += 1
                replaced = f"<UNIT_{unit_counter}>"
                unit_text_to_ph[key] = replaced

        if replaced is None:
            out.append(tok)
        else:
            out.append(pre + replaced + suf)

    num_map = {ph: txt for txt, ph in num_text_to_ph.items()}
    unit_map = {ph: txt for txt, ph in unit_text_to_ph.items()}
    return " ".join(out), num_map, unit_map


def slotify_src(text: str, cfg: SlotifyConfig) -> Tuple[str, SlotMap]:
    """Slotify a *source* text and return (slot_text, slot_map)."""
    s = "" if text is None else str(text)

    if cfg.strip_lex_from_src:
        # Reuse the same logic as audit_mismatch (keeps gaps).
        s = strip_lex_gloss(s)

    s = norm_n0(s)
    s = clean_text(s)

    # PN/GN (unified) first (so that digits inside <PN_1> etc are protected).
    s2, pn_map = _slotify_pn(s, cfg)
    s3, num_map, unit_map = _slotify_numbers_and_units(s2, cfg)

    return s3, SlotMap(pn_map=pn_map, num_map=num_map, unit_map=unit_map)


def slotify_tgt_using_src_map(tgt_text: str, src_slots: SlotMap, cfg: SlotifyConfig) -> str:
    """Convert a target string into a slotified template using a source SlotMap.

    This function replaces:
    - PN tokens that appeared in the *source* (and thus likely appear in target)
    - numeric tokens from the source
    - unit tokens from the source

    It is intentionally conservative; if a token is not found, we keep it.
    """
    s = "" if tgt_text is None else str(tgt_text)
    s = norm_n0(s)
    s = clean_text(s)
    if not s:
        return ""

    tokens = s.split()

    # Build reverse maps (original -> placeholder)
    pn_rev = {v: k for k, v in src_slots.pn_map.items()}
    num_rev = {v: k for k, v in src_slots.num_map.items()}
    unit_rev = {v: k for k, v in src_slots.unit_map.items()}

    # For PN, we need to support multi-token sequences (e.g., "Aḫ šalim").
    # Greedy longest-first over PN originals.
    pn_entries: List[Tuple[List[str], str]] = []
    for orig, ph in pn_rev.items():
        tks = orig.split()
        if tks:
            pn_entries.append((tks, ph))
    pn_entries.sort(key=lambda x: len(x[0]), reverse=True)

    out: List[str] = []
    i = 0
    while i < len(tokens):
        matched = False

        # Try PN multi-token match first
        for orig_tokens, ph in pn_entries:
            L = len(orig_tokens)
            if i + L > len(tokens):
                continue
            ok = True
            for j in range(L):
                _pre, core, _suf = _split_edge_punct(tokens[i + j])
                if core != orig_tokens[j]:
                    ok = False
                    break
            if not ok:
                continue

            pre0, _c0, _s0 = _split_edge_punct(tokens[i])
            _pL, _cL, sufL = _split_edge_punct(tokens[i + L - 1])
            out.append(pre0 + ph + sufL)
            i += L
            matched = True
            break

        if matched:
            continue

        # Single-token replacements: numbers / units
        tok = tokens[i]
        pre, core, suf = _split_edge_punct(tok)
        repl = None
        if cfg.slot_numbers and core in num_rev:
            repl = num_rev[core]
        elif cfg.slot_units and core in unit_rev:
            repl = unit_rev[core]

        if repl is None:
            out.append(tok)
        else:
            out.append(pre + repl + suf)

        i += 1

    return " ".join(out)


def restore_slots(text: str, primary: SlotMap, fallback: Optional[SlotMap] = None) -> str:
    """Restore placeholders in `text` using `primary` and optional `fallback`.

    We replace placeholder *cores* (<PN_1>, <NUM_1>, <UNIT_1>) even if edge punctuation exists,
    preserving the punctuation.
    """
    s = "" if text is None else str(text)
    s = clean_text(norm_n0(s))
    if not s:
        return ""

    primary_map = primary.combined()
    fallback_map = fallback.combined() if fallback else {}

    out_tokens: List[str] = []
    for tok in s.split():
        pre, core, suf = _split_edge_punct(tok)
        if _PLACEHOLDER_CORE_RE.match(core):
            if core in primary_map:
                out_tokens.append(pre + primary_map[core] + suf)
                continue
            if core in fallback_map:
                out_tokens.append(pre + fallback_map[core] + suf)
                continue
        out_tokens.append(tok)

    return " ".join(out_tokens)


def count_placeholders(slot_text: str) -> Dict[str, int]:
    """Count placeholders in a slotified text (for lightweight features)."""
    s = "" if slot_text is None else str(slot_text)
    counts = {"pn": 0, "num": 0, "unit": 0}
    for _pre, core, _suf in map(_split_edge_punct, s.split()):
        m = _PLACEHOLDER_CORE_RE.match(core)
        if not m:
            continue
        typ = m.group(1)
        if typ == "PN":
            counts["pn"] += 1
        elif typ == "NUM":
            counts["num"] += 1
        elif typ == "UNIT":
            counts["unit"] += 1
    return counts
