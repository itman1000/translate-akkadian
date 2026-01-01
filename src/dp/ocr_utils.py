"""OCR 追加並列抽出で使う共通ユーティリティ。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


_SOFT_HYPHEN_RE = re.compile("\u00ad")
_MULTI_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
_TOKEN_STRIP = ".,;:!?\"'()[]{}<>"

_DOT_PLACEHOLDER = "__DOT__"
_ABBR_RE = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|cf|ca|fig|figs|no|nos|vol|pp|p|ed|eds|al|mr|mrs|ms|dr|prof|st|jr|sr|rev)\.(?=\s+\w)",
    re.IGNORECASE,
)
_CAPS_ABBR_RE = re.compile(r"\b(?:[A-Z]\.){2,}(?=\s+\w)")
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")

_STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "that",
    "is",
    "was",
    "were",
    "for",
    "with",
    "as",
    "on",
    "by",
    "from",
    "at",
    "into",
    "this",
    "these",
    "those",
    "it",
    "its",
    "be",
    "are",
    "an",
    "a",
    "or",
    "which",
    "who",
    "their",
    "has",
    "have",
    "had",
    "not",
    "but",
    "also",
    "we",
    "our",
    "they",
    "them",
    "his",
    "her",
    "he",
    "she",
}

_ANCHOR_MAP = str.maketrans(
    {
        "á": "a",
        "à": "a",
        "â": "a",
        "ä": "a",
        "Á": "A",
        "À": "A",
        "Â": "A",
        "Ä": "A",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "É": "E",
        "È": "E",
        "Ê": "E",
        "Ë": "E",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "Í": "I",
        "Ì": "I",
        "Î": "I",
        "Ï": "I",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "Ú": "U",
        "Ù": "U",
        "Û": "U",
        "Ü": "U",
        "š": "s",
        "Š": "S",
        "ṣ": "s",
        "Ṣ": "S",
        "ṭ": "t",
        "Ṭ": "T",
        "ḫ": "h",
        "Ḫ": "H",
    }
)


@dataclass(frozen=True)
class EnglishThresholds:
    min_chars: int
    min_words: int
    min_alpha_ratio: float
    max_symbol_ratio: float
    max_digit_ratio: float
    min_stopword_hits: int
    max_upper_ratio: float


@dataclass(frozen=True)
class TranslitThresholds:
    min_chars: int
    min_tokens: int
    min_marker_ratio: float
    min_upper_ratio: float
    max_symbol_ratio: float
    max_stopword_hits: int


def normalize_newlines(text: str) -> str:
    if not text:
        return ""
    out = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _SOFT_HYPHEN_RE.sub("", out)
    return out


def normalize_ocr_text(text: str) -> str:
    out = normalize_newlines(text)
    out = re.sub(r"[ \t]+\n", "\n", out)
    # ハイフネーションで改行された単語を結合する
    out = re.sub(r"(?<=\w)-\n(?=\w)", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def split_blocks(text: str) -> List[str]:
    out = normalize_ocr_text(text)
    blocks = [b.strip() for b in re.split(r"\n{2,}", out) if b.strip()]
    if len(blocks) <= 1:
        blocks = [b.strip() for b in out.split("\n") if b.strip()]
    return blocks


def flatten_block(block: str) -> str:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    return " ".join(lines)


def text_stats(text: str) -> Dict[str, float | int]:
    total = len(text)
    if total <= 0:
        return {
            "len": 0,
            "tokens": 0,
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "symbol_ratio": 0.0,
            "upper_ratio": 0.0,
        }
    alpha = sum(ch.isalpha() for ch in text)
    upper = sum(ch.isupper() for ch in text)
    digit = sum(ch.isdigit() for ch in text)
    space = sum(ch.isspace() for ch in text)
    symbol = total - alpha - digit - space
    content = max(total - space, 1)
    upper_base = max(alpha, 1)
    return {
        "len": total,
        "tokens": len(text.split()),
        "alpha_ratio": alpha / content,
        "digit_ratio": digit / content,
        "symbol_ratio": symbol / content,
        "upper_ratio": upper / upper_base,
    }


def count_stopwords(text: str) -> int:
    hits = 0
    for token in _WORD_RE.findall(text.lower()):
        if token in _STOPWORDS:
            hits += 1
    return hits


def _translit_marker_token(token: str) -> bool:
    if any(ch.isdigit() for ch in token):
        return True
    if any(ch in ".-'" for ch in token):
        return True
    if token.isupper() and len(token) >= 2:
        return True
    if re.search(r"[ŠḪṢṬšḫṣṭ]", token):
        return True
    return False


def _simple_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in text.split():
        token = raw.strip(_TOKEN_STRIP)
        if token:
            tokens.append(token)
    return tokens


def english_score(stats: Dict[str, float | int], stopword_hits: int) -> float:
    alpha = float(stats.get("alpha_ratio", 0.0))
    stop = min(stopword_hits, 3) / 3
    return alpha * 0.7 + stop * 0.3


def translit_score(marker_ratio: float, upper_ratio: float) -> float:
    return min(marker_ratio, 1.0) * 0.6 + min(upper_ratio, 1.0) * 0.4


def is_english_like(text: str, thresholds: EnglishThresholds) -> Tuple[bool, Dict[str, float | int]]:
    stats = text_stats(text)
    stopword_hits = count_stopwords(text)
    score = english_score(stats, stopword_hits)
    metrics: Dict[str, float | int] = {
        "score": score,
        "stopword_hits": stopword_hits,
        **stats,
    }
    if int(stats["len"]) < thresholds.min_chars:
        return False, metrics
    if int(stats["tokens"]) < thresholds.min_words:
        return False, metrics
    if float(stats["alpha_ratio"]) < thresholds.min_alpha_ratio:
        return False, metrics
    if float(stats["symbol_ratio"]) > thresholds.max_symbol_ratio:
        return False, metrics
    if float(stats["digit_ratio"]) > thresholds.max_digit_ratio:
        return False, metrics
    if stopword_hits < thresholds.min_stopword_hits:
        return False, metrics
    if float(stats["upper_ratio"]) > thresholds.max_upper_ratio:
        return False, metrics
    return True, metrics


def is_translit_like(
    text: str, thresholds: TranslitThresholds
) -> Tuple[bool, Dict[str, float | int]]:
    stats = text_stats(text)
    stopword_hits = count_stopwords(text)
    tokens = _simple_tokens(text)
    marker_tokens = sum(1 for token in tokens if _translit_marker_token(token))
    marker_ratio = marker_tokens / max(len(tokens), 1)
    score = translit_score(marker_ratio, float(stats["upper_ratio"]))
    metrics: Dict[str, float | int] = {
        "score": score,
        "stopword_hits": stopword_hits,
        "marker_ratio": marker_ratio,
        **stats,
    }
    if int(stats["len"]) < thresholds.min_chars:
        return False, metrics
    if len(tokens) < thresholds.min_tokens:
        return False, metrics
    if stopword_hits > thresholds.max_stopword_hits:
        return False, metrics
    if float(stats["symbol_ratio"]) > thresholds.max_symbol_ratio:
        return False, metrics
    if marker_ratio < thresholds.min_marker_ratio and float(stats["upper_ratio"]) < thresholds.min_upper_ratio:
        return False, metrics
    return True, metrics


def normalize_anchor_token(token: str) -> str:
    stripped = token.strip(_TOKEN_STRIP)
    if not stripped:
        return ""
    return stripped.translate(_ANCHOR_MAP).upper()


def extract_anchor_tokens(text: str, min_len: int = 3) -> List[str]:
    anchors: List[str] = []
    for raw in text.split():
        token = normalize_anchor_token(raw)
        if not token:
            continue
        if token.isdigit():
            anchors.append(token)
            continue
        if any(ch.isdigit() for ch in token):
            anchors.append(token)
            continue
        if token.isalpha() and len(token) >= min_len:
            anchors.append(token)
    return anchors


def anchor_overlap(src_text: str, tgt_text: str) -> float:
    src_tokens = set(extract_anchor_tokens(src_text))
    tgt_tokens = set(extract_anchor_tokens(tgt_text))
    if not src_tokens or not tgt_tokens:
        return 0.0
    inter = src_tokens.intersection(tgt_tokens)
    union = src_tokens.union(tgt_tokens)
    return len(inter) / max(len(union), 1)


def _protect_sentence_dots(text: str) -> str:
    out = _DECIMAL_DOT_RE.sub(_DOT_PLACEHOLDER, text)
    out = _CAPS_ABBR_RE.sub(lambda m: m.group(0).replace(".", _DOT_PLACEHOLDER), out)
    out = _ABBR_RE.sub(lambda m: m.group(0).replace(".", _DOT_PLACEHOLDER), out)
    return out


def count_sentence_endings(text: str) -> int:
    if not text:
        return 0
    protected = _protect_sentence_dots(text)
    return len(re.findall(r'[.!?](?=(?:["\')\]]+)?\s|$)', protected))
