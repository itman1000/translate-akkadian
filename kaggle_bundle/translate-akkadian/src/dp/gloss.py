"""Lexicon/dictionary based gloss augmentation.

This module is optional and designed to be safe-by-default:

* Nothing changes unless enabled via config / CLI.
* If enabled, it appends short English gloss hints to the source text.

The intended usage is to load ``OA_Lexicon_eBL.csv`` (surface forms -> lexemes)
and ``eBL_Dictionary.csv`` (lexeme -> English definition), then add a compact
lexeme=gloss list to the source input:

    ... <LEX> amāru=to see | kaspu=silver </LEX>

This can help a seq2seq model with rare vocabulary and spelling/format variants.

New in this version:
- Filter controls to reduce noise:
  - match_types (OA_Lexicon type; default: word only)
  - stop lemmas (built-in + user-specified)
  - optional frequency filter (keep only rare-ish lemmas)
- Default max_hints reduced to 6 for stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import re

import pandas as pd


_ROMAN_RE = re.compile(r"^[IVX]+$")
_QUOTE_GLOSS_RE = re.compile(r'"([^"\n\r]{1,160})"')
_APOSTROPHE_RE = re.compile(r"[’‘ʾʼ]")


# Conservative default stop lemmas (function words) to reduce gloss noise.
# You can disable/override from CLI/config.
DEFAULT_STOP_LEMMAS: Tuple[str, ...] = (
    "ana",  # to, for
    "ina",  # in, by
    "u",  # and
    "ša",  # rel.
    "ma",
    "mā",
    "kīma",  # like, as
    "kima",
    # Very frequent negation; tends to over-trigger templatic English outputs.
    "lā",  # not, no; without
    "la",  # fallback (ASCII)
)


@dataclass(frozen=True)
class GlossAugmentConfig:
    """Configuration for gloss augmentation."""

    enabled: bool = False
    # Paths (optional; callers usually pass explicit paths)
    oa_lexicon_path: Optional[Path] = None
    ebl_dictionary_path: Optional[Path] = None

    # What to match in OA_Lexicon_eBL.csv
    match_columns: Tuple[str, ...] = ("form", "norm")
    match_types: Tuple[str, ...] = ("word",)  # default: lexical words only

    # Match controls
    max_match_len: int = 4  # limit token-span matching (speed/noise control)

    # Noise filters
    stop_lemmas: Tuple[str, ...] = DEFAULT_STOP_LEMMAS
    min_lemma_chars: int = 2  # e.g., drops 'u'
    exclude_bound_morphemes: bool = True  # drops lemmas like '-am'
    max_lemma_freq: Optional[int] = None  # if set, keep lemmas with freq <= N

    # Output formatting
    prefix: str = " <LEX> "
    suffix: str = " </LEX>"
    pair_sep: str = " | "
    kv_sep: str = "="

    # Limits
    max_hints: int = 6
    max_total_chars: int = 220
    max_gloss_chars: int = 60

    # Definition selection
    prefer_roman: str = "I"  # when sense (I/II/III...) is unknown


@dataclass(frozen=True)
class _GlossResources:
    token_map: Dict[Tuple[str, ...], Tuple[str, Optional[str]]]
    max_len: int
    defs_by_lemma: Dict[str, Dict[str, str]]
    lemma_display: Dict[str, str]


def _clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _canonicalize_lemma(lemma: str) -> str:
    """Normalize lemma for stable joins between OA lexeme and eBL dictionary.

    Notes:
    - eBL_Dictionary.csv often uses plain 'h' where OA uses 'ḫ'.
    - It also usually omits the final nominative '-m' in lemmas (amārum -> amāru).

    We keep bound morphemes (starting with '-') as-is.
    """

    out = _APOSTROPHE_RE.sub("'", str(lemma))
    out = out.replace("ḫ", "h").replace("Ḫ", "H")
    out = _clean_ws(out)

    if out.startswith("-"):
        return out
    if len(out) >= 4 and out.endswith("m"):
        out = out[:-1]
    return out


def _parse_dict_word(word: str) -> Optional[Tuple[str, str]]:
    """Split 'lemma ROMAN' into (lemma, roman)."""

    text = _clean_ws(str(word))
    if not text or " " not in text:
        return None
    lemma, roman = text.rsplit(" ", 1)
    if not _ROMAN_RE.match(roman):
        return None
    return _canonicalize_lemma(lemma), roman


def _simplify_definition(definition: str, max_chars: int) -> str:
    """Extract a compact gloss from a full dictionary definition."""

    text = _clean_ws(str(definition))
    if not text:
        return ""

    m = _QUOTE_GLOSS_RE.search(text)
    if m:
        gloss = m.group(1)
    else:
        # Remove leading numbering (e.g., "1.")
        text2 = re.sub(r"^\s*\d+\s*\.\s*", "", text)
        # Cut early at typical metadata separators.
        cut = len(text2)
        for sep in (";", "[", "("):
            pos = text2.find(sep)
            if pos != -1:
                cut = min(cut, pos)
        gloss = text2[:cut].strip()

    gloss = gloss.strip(" .;,:\t")
    if len(gloss) > max_chars:
        # Avoid multi-byte ellipsis to keep it simple.
        gloss = gloss[: max(0, max_chars - 3)].rstrip() + "..."
    return gloss


@lru_cache(maxsize=4)
def _load_dict_defs_cached(ebl_dictionary_path: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Load eBL dictionary into (defs_by_lemma, lemma_display)."""

    dict_path = Path(ebl_dictionary_path)
    if not dict_path.exists():
        raise FileNotFoundError(f"eBL dictionary not found: {dict_path}")

    dict_df = pd.read_csv(dict_path, usecols=["word", "definition"]).dropna()

    defs_by_lemma: Dict[str, Dict[str, str]] = {}
    lemma_display: Dict[str, str] = {}
    for _, row in dict_df.iterrows():
        parsed = _parse_dict_word(row["word"])
        if not parsed:
            continue
        lemma_key, roman = parsed
        definition = _clean_ws(row["definition"])
        if not definition:
            continue
        defs_by_lemma.setdefault(lemma_key, {})
        # Keep first definition per sense unless empty.
        defs_by_lemma[lemma_key].setdefault(roman, definition)
        lemma_display.setdefault(lemma_key, lemma_key)

    return defs_by_lemma, lemma_display


@lru_cache(maxsize=4)
def _load_oa_token_map_cached(
    oa_lexicon_path: str,
    match_columns: Tuple[str, ...],
    match_types: Tuple[str, ...],
) -> Tuple[Dict[Tuple[str, ...], Tuple[str, Optional[str]]], int]:
    """Load OA lexicon into (token_map, max_len)."""

    oa_path = Path(oa_lexicon_path)
    if not oa_path.exists():
        raise FileNotFoundError(f"OA lexicon not found: {oa_path}")

    usecols = {"type", "lexeme", "I_IV"}
    for col in match_columns:
        usecols.add(col)

    oa_df = pd.read_csv(oa_path, usecols=sorted(usecols)).dropna(subset=["lexeme"])
    types_set = {t.strip().lower() for t in match_types}
    oa_df = oa_df[oa_df["type"].astype(str).str.lower().isin(types_set)]

    token_map: Dict[Tuple[str, ...], Tuple[str, Optional[str]]] = {}
    max_len = 1
    for _, row in oa_df.iterrows():
        lexeme = _clean_ws(row.get("lexeme", ""))
        if not lexeme:
            continue
        roman = row.get("I_IV")
        roman_str: Optional[str] = None
        if roman is not None and str(roman).strip() != "":
            roman_str = str(roman).strip().upper()

        for col in match_columns:
            text = _clean_ws(row.get(col, ""))
            if not text:
                continue
            tokens = tuple(text.split())
            if not tokens:
                continue
            # Prefer existing mapping (first wins) to keep deterministic.
            token_map.setdefault(tokens, (lexeme, roman_str))
            max_len = max(max_len, len(tokens))

    return token_map, max_len


def build_lemma_frequency(
    texts: Iterable[str],
    *,
    oa_lexicon_path: Path,
    match_columns: Tuple[str, ...] = ("form", "norm"),
    match_types: Tuple[str, ...] = ("word",),
    max_match_len: int = 4,
) -> Dict[str, int]:
    """Compute a rough lemma frequency table from a corpus.

    We tokenize by whitespace and perform greedy longest-match against OA token_map.

    This is meant for **filtering very frequent function words** from gloss hints.
    It does not need to be perfect to be useful.
    """

    token_map, max_len = _load_oa_token_map_cached(
        str(oa_lexicon_path),
        tuple(match_columns),
        tuple(match_types),
    )

    max_L = max(1, min(int(max_len), int(max_match_len)))

    freq: Dict[str, int] = {}

    for text in texts:
        sent = _clean_ws(str(text))
        if not sent:
            continue
        tokens = sent.split()
        i = 0
        while i < len(tokens):
            matched = False
            for L in range(min(max_L, len(tokens) - i), 0, -1):
                seq = tuple(tokens[i : i + L])
                hit = token_map.get(seq)
                if not hit:
                    continue
                lexeme, _roman = hit
                lemma_key = _canonicalize_lemma(lexeme)
                freq[lemma_key] = freq.get(lemma_key, 0) + 1
                i += L
                matched = True
                break
            if not matched:
                i += 1

    return freq


def build_gloss_augmenter(
    cfg: GlossAugmentConfig,
    *,
    oa_lexicon_path: Path,
    ebl_dictionary_path: Path,
    lemma_freq: Optional[Dict[str, int]] = None,
) -> "callable[[str], str]":
    """Create a callable that augments a source sentence with gloss hints."""

    if not cfg.enabled:
        return lambda x: x

    defs_by_lemma, lemma_display = _load_dict_defs_cached(str(ebl_dictionary_path))
    token_map, oa_max_len = _load_oa_token_map_cached(
        str(oa_lexicon_path),
        tuple(cfg.match_columns),
        tuple(cfg.match_types),
    )

    resources = _GlossResources(
        token_map=token_map,
        max_len=oa_max_len,
        defs_by_lemma=defs_by_lemma,
        lemma_display=lemma_display,
    )

    stop_set = {_canonicalize_lemma(x) for x in cfg.stop_lemmas if str(x).strip()}

    def _lookup_definition(lemma_key: str, roman: Optional[str]) -> Optional[str]:
        senses = resources.defs_by_lemma.get(lemma_key)
        if not senses:
            return None
        if roman and roman in senses:
            return senses[roman]
        pref = str(cfg.prefer_roman).strip().upper()
        if pref and pref in senses:
            return senses[pref]
        # fallback: smallest roman by string (I, II, III...)
        return senses[sorted(senses.keys())[0]]

    @lru_cache(maxsize=8192)
    def _lookup_simplified_gloss(lemma_key: str, roman: str) -> Optional[str]:
        definition = _lookup_definition(lemma_key, roman or None)
        if not definition:
            return None
        gloss = _simplify_definition(definition, int(cfg.max_gloss_chars))
        if not gloss:
            return None
        return gloss

    max_L = max(1, min(int(resources.max_len), int(cfg.max_match_len)))

    def _augment_one(text: str) -> str:
        sent = _clean_ws(text)
        if not sent:
            return sent

        tokens = sent.split()
        # Collect candidates then sort by (freq asc, match_len desc, lemma_len desc, position asc)
        candidates: List[Tuple[int, int, int, int, str, str]] = []

        for i in range(len(tokens)):
            for L in range(min(max_L, len(tokens) - i), 0, -1):
                seq = tuple(tokens[i : i + L])
                hit = resources.token_map.get(seq)
                if not hit:
                    continue
                lexeme, roman = hit
                lemma_key = _canonicalize_lemma(lexeme)

                if cfg.exclude_bound_morphemes and lemma_key.startswith("-"):
                    continue
                if cfg.min_lemma_chars and len(lemma_key) < int(cfg.min_lemma_chars):
                    continue
                if lemma_key in stop_set:
                    continue

                freq = lemma_freq.get(lemma_key, 0) if lemma_freq else 0
                if cfg.max_lemma_freq is not None and lemma_freq is not None:
                    if freq > int(cfg.max_lemma_freq):
                        continue

                roman_key = (roman or "").strip().upper()
                gloss = _lookup_simplified_gloss(lemma_key, roman_key)
                if not gloss:
                    continue

                candidates.append((freq, -L, -len(lemma_key), i, lemma_key, gloss))
                break

        if not candidates:
            return sent

        candidates.sort()

        pairs: List[Tuple[str, str]] = []
        seen_lemmas: set[str] = set()

        for _freq, _negL, _neglen, _i, lemma_key, gloss in candidates:
            if lemma_key in seen_lemmas:
                continue
            display = resources.lemma_display.get(lemma_key, lemma_key)
            pairs.append((display, gloss))
            seen_lemmas.add(lemma_key)
            if len(pairs) >= max(0, int(cfg.max_hints)):
                break

        if not pairs:
            return sent

        payload_parts: List[str] = []
        total_chars = 0
        for lemma, gloss in pairs:
            part = f"{lemma}{cfg.kv_sep}{gloss}"
            projected = total_chars + (len(cfg.pair_sep) if payload_parts else 0) + len(part)
            if projected > int(cfg.max_total_chars):
                break
            payload_parts.append(part)
            total_chars = projected

        if not payload_parts:
            return sent

        payload = cfg.pair_sep.join(payload_parts)
        return f"{sent}{cfg.prefix}{payload}{cfg.suffix}"

    return _augment_one
