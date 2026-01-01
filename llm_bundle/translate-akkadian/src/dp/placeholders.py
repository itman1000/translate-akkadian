"""Placeholder utilities for PN/GN normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def load_lexicon_forms(path: Path, types: Iterable[str]) -> Dict[str, List[str]]:
    df = pd.read_csv(path, usecols=["type", "form"]).dropna()
    types_set = {t.strip().upper() for t in types}
    df = df[df["type"].str.upper().isin(types_set)]
    forms_by_type: Dict[str, List[str]] = {t: [] for t in types_set}

    for _, row in df.iterrows():
        t = str(row["type"]).upper()
        form = str(row["form"]).strip()
        if not form:
            continue
        forms_by_type.setdefault(t, []).append(form)

    for t in forms_by_type:
        forms_by_type[t] = sorted(set(forms_by_type[t]))

    return forms_by_type


def build_token_map(forms_by_type: Dict[str, List[str]]) -> Tuple[Dict[Tuple[str, ...], str], int]:
    token_map: Dict[Tuple[str, ...], str] = {}
    max_len = 1

    # Prefer PN over GN if duplicates
    ordered_types = ["PN", "GN"] + [t for t in forms_by_type.keys() if t not in {"PN", "GN"}]
    for t in ordered_types:
        for form in forms_by_type.get(t, []):
            tokens = tuple(form.split())
            if not tokens:
                continue
            if tokens not in token_map:
                token_map[tokens] = t
                max_len = max(max_len, len(tokens))

    return token_map, max_len


def tokenize(text: str) -> List[str]:
    return text.split()


def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


def _match_allowed(text: str, entity_type: str, mode: str) -> bool:
    if mode == "all":
        return True

    has_brace = "{" in text or "}" in text
    has_paren = bool(re.search(r"\([^)]+\)", text))
    has_upper = any(ch.isupper() for ch in text)
    hyphen_count = text.count("-")
    text_len = len(text)

    if mode == "strict":
        if has_brace or has_paren or has_upper:
            return True
        if hyphen_count >= 2 and text_len >= 8:
            return True
        return False

    if mode == "pattern":
        if has_brace or has_paren or has_upper:
            return True
        if hyphen_count >= 1 and text_len >= 5:
            return True
        return False

    return True


def apply_placeholders(
    text: str,
    token_map: Dict[Tuple[str, ...], str],
    max_len: int,
    filter_mode: str = "all",
) -> Tuple[str, List[Dict[str, str]]]:
    tokens = tokenize(text)
    out_tokens: List[str] = []
    mapping: List[Dict[str, str]] = []

    counters = {"PN": 0, "GN": 0}
    text_to_placeholder: Dict[str, str] = {}

    i = 0
    while i < len(tokens):
        matched = False
        for length in range(min(max_len, len(tokens) - i), 0, -1):
            seq = tuple(tokens[i : i + length])
            if seq in token_map:
                t = token_map[seq]
                text_val = detokenize(list(seq))
                if not _match_allowed(text_val, t, filter_mode):
                    continue
                if text_val in text_to_placeholder:
                    placeholder = text_to_placeholder[text_val]
                else:
                    counters.setdefault(t, 0)
                    counters[t] += 1
                    placeholder = f"<{t}_{counters[t]}>"
                    text_to_placeholder[text_val] = placeholder
                    mapping.append({"type": t, "placeholder": placeholder, "text": text_val})
                out_tokens.append(placeholder)
                i += length
                matched = True
                break
        if not matched:
            out_tokens.append(tokens[i])
            i += 1

    return detokenize(out_tokens), mapping


def replace_target_with_mapping(text: str, mapping: List[Dict[str, str]]) -> str:
    if not mapping:
        return text
    tokens = tokenize(text)
    entries = [
        {"placeholder": m["placeholder"], "tokens": m["text"].split()} for m in mapping if m.get("text")
    ]
    entries.sort(key=lambda x: len(x["tokens"]), reverse=True)

    out: List[str] = []
    i = 0
    while i < len(tokens):
        replaced = False
        for entry in entries:
            tks = entry["tokens"]
            if not tks:
                continue
            if tokens[i : i + len(tks)] == tks:
                out.append(entry["placeholder"])
                i += len(tks)
                replaced = True
                break
        if not replaced:
            out.append(tokens[i])
            i += 1

    return detokenize(out)


def restore_placeholders(text: str, mapping: List[Dict[str, str]]) -> str:
    out = text
    for m in mapping:
        placeholder = m.get("placeholder")
        original = m.get("text")
        if placeholder and original:
            out = out.replace(placeholder, original)
    return out


def dump_mapping(mapping: List[Dict[str, str]]) -> str:
    return json.dumps(mapping, ensure_ascii=True)


def load_mapping(payload: str) -> List[Dict[str, str]]:
    try:
        return json.loads(payload)
    except Exception:
        return []
