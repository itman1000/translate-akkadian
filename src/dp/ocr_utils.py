"""OCR 追加並列抽出で使う共通ユーティリティ。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
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

_LANG_STOPWORDS: Dict[str, set[str]] = {
    "en": {
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
        "you",
        "your",
        "if",
        "when",
        "then",
    },
    "fr": {
        "le",
        "la",
        "les",
        "de",
        "des",
        "du",
        "et",
        "en",
        "dans",
        "pour",
        "sur",
        "par",
        "avec",
        "que",
        "qui",
        "une",
        "un",
        "au",
        "aux",
        "ce",
        "ces",
        "cette",
        "son",
        "sa",
        "ses",
        "leur",
        "leurs",
        "est",
        "sont",
        "ont",
        "pas",
        "ne",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "elles",
    },
    "de": {
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "und",
        "zu",
        "in",
        "mit",
        "auf",
        "von",
        "für",
        "fur",
        "ein",
        "eine",
        "einer",
        "einem",
        "einen",
        "ist",
        "sind",
        "war",
        "waren",
        "nicht",
        "auch",
        "als",
        "dass",
        "daß",
        "wie",
        "sie",
        "er",
        "wir",
        "ihr",
        "ihre",
        "ihren",
        "seine",
        "seiner",
    },
    "tr": {
        "ve",
        "bir",
        "bu",
        "ile",
        "için",
        "icin",
        "olarak",
        "olan",
        "da",
        "de",
        "ki",
        "mi",
        "mı",
        "mu",
        "mü",
        "veya",
        "ama",
        "fakat",
        "gibi",
        "olan",
        "oldu",
        "vardır",
        "vardir",
        "yok",
        "bu",
        "şu",
        "su",
        "o",
        "birçok",
        "bircok",
        "daha",
        "sonra",
        "önce",
        "once",
        "onun",
        "onlar",
        "biz",
        "siz",
    },
}

_LANG_DISPLAY = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "tr": "Turkish",
    "other": "Other Latin-script language",
    "unknown": "Unknown",
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


@dataclass(frozen=True)
class TranslationThresholds:
    min_chars: int
    min_words: int
    min_alpha_ratio: float
    max_symbol_ratio: float
    max_digit_ratio: float
    min_best_stopword_hits: int
    max_upper_ratio: float


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


def count_stopwords_by_lang(text: str) -> Dict[str, int]:
    hits: Dict[str, int] = {lang: 0 for lang in _LANG_STOPWORDS}
    for token in _WORD_RE.findall(text.lower()):
        for lang, stopset in _LANG_STOPWORDS.items():
            if token in stopset:
                hits[lang] += 1
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


def detect_translation_language(text: str) -> Tuple[str, Dict[str, float | int | str]]:
    stats = text_stats(text)
    lang_hits = count_stopwords_by_lang(text)
    ranked = sorted(lang_hits.items(), key=lambda item: (-item[1], item[0]))
    best_lang = ranked[0][0] if ranked else "unknown"
    best_hits = ranked[0][1] if ranked else 0
    second_hits = ranked[1][1] if len(ranked) >= 2 else 0
    tokens = max(int(stats.get("tokens", 0)), 1)
    confidence = max(best_hits - second_hits, 0) / tokens

    if best_hits <= 0:
        if float(stats.get("alpha_ratio", 0.0)) >= 0.72 and float(stats.get("upper_ratio", 0.0)) <= 0.28:
            best_lang = "other"
        else:
            best_lang = "unknown"

    metrics: Dict[str, float | int | str] = {
        "lang": best_lang,
        "lang_name": _LANG_DISPLAY.get(best_lang, "Unknown"),
        "best_stopword_hits": best_hits,
        "second_stopword_hits": second_hits,
        "lang_confidence": round(confidence, 6),
        **{f"stopwords_{lang}": count for lang, count in lang_hits.items()},
        **stats,
    }
    return best_lang, metrics


def translation_like_score(
    stats: Dict[str, float | int],
    best_stopword_hits: int,
    lang_confidence: float,
) -> float:
    alpha = float(stats.get("alpha_ratio", 0.0))
    stop = min(best_stopword_hits, 3) / 3
    conf = min(max(lang_confidence, 0.0), 1.0)
    return alpha * 0.55 + stop * 0.30 + conf * 0.15


def is_translation_like(
    text: str,
    thresholds: TranslationThresholds,
) -> Tuple[bool, Dict[str, float | int | str]]:
    lang, lang_metrics = detect_translation_language(text)
    stats = text_stats(text)
    best_stopword_hits = int(lang_metrics.get("best_stopword_hits", 0))
    lang_confidence = float(lang_metrics.get("lang_confidence", 0.0))
    score = translation_like_score(stats, best_stopword_hits, lang_confidence)
    metrics: Dict[str, float | int | str] = {
        "score": score,
        **lang_metrics,
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
    if float(stats["upper_ratio"]) > thresholds.max_upper_ratio:
        return False, metrics
    if best_stopword_hits >= thresholds.min_best_stopword_hits:
        return True, metrics

    if lang == "other" and float(stats["alpha_ratio"]) >= max(0.75, thresholds.min_alpha_ratio + 0.08):
        return True, metrics
    return False, metrics


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



def normalize_lookup_text(text: str) -> str:
    """識別子照合用にテキストを強めに正規化する。"""
    if not text:
        return ""
    out = unicodedata.normalize("NFKD", str(text))
    out = "".join(ch for ch in out if not unicodedata.combining(ch))
    out = out.translate(_ANCHOR_MAP)
    out = out.upper()
    out = out.replace("&", " AND ")
    out = re.sub(r"\bCUNEIFORM\s+TABLET\b", " ", out)
    out = re.sub(r"[^A-Z0-9]+", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def lookup_tokens(text: str) -> List[str]:
    """識別子照合用トークン列を返す。"""
    normalized = normalize_lookup_text(text)
    if not normalized:
        return []
    return normalized.split()


def split_metadata_field(text: str) -> List[str]:
    """aliases / publication_catalog などを個別の候補に分解する。"""
    if not text:
        return []
    parts: List[str] = []
    for raw in re.split(r"[|;+\n]", str(text)):
        part = raw.strip()
        if not part:
            continue
        parts.append(part)
    return parts


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def token_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a_set = {tok for tok in a if tok}
    b_set = {tok for tok in b if tok}
    if not a_set or not b_set:
        return 0.0
    inter = a_set.intersection(b_set)
    union = a_set.union(b_set)
    return len(inter) / max(len(union), 1)


def parse_pipe_list(text: str | None) -> List[str]:
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split("|") if part.strip()]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def stable_block_hash(text: str) -> str:
    normalized = normalize_newlines(text)
    normalized = _MULTI_SPACE_RE.sub(" ", normalized).strip()
    normalized = unicodedata.normalize("NFKC", normalized)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()
