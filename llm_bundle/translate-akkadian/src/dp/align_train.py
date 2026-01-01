"""train.csv から文単位の並列データを作る。"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from .utils import clean_text, get_artifacts_dir, get_data_dir, load_config


_DASH_MAP = str.maketrans({"–": "-", "—": "-", "−": "-"})
_APOSTROPHE_RE = re.compile(r"[’‘ʾʼ]")
_QUOTE_RE = re.compile(r"[“”„]")
_GAP_TAG_RE = re.compile(r"<\s*gap\s*>", re.IGNORECASE)
_BIG_GAP_TAG_RE = re.compile(r"<\s*big_gap\s*>", re.IGNORECASE)
_BRACKET_MISSING_RE = re.compile(r"\[(?P<body>[xX\.\s\?…]+)\]")
_PAREN_MISSING_RE = re.compile(r"\((?P<body>[xX\.\s\?…]+)\)")
_X_TOKEN_RUN_RE = re.compile(r"(?<![A-Za-z0-9])x(?:[\s-]+x){1,}(?![A-Za-z0-9])", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?(sup|sub|br|i|b|u|em|strong|span)\b[^>]*>", re.IGNORECASE)
_HTML_SUP_RE = re.compile(r"<\s*sup\b[^>]*>(.*?)</\s*sup\s*>", re.IGNORECASE | re.DOTALL)
_HTML_SUB_RE = re.compile(r"<\s*sub\b[^>]*>(.*?)</\s*sub\s*>", re.IGNORECASE | re.DOTALL)
_BRACE_BIG_GAP_RE = re.compile(
    r"\{\s*(large[\s_-]*break|broken[\s_-]*area)\s*\}",
    re.IGNORECASE,
)
_DOTS_BIG_GAP_RE = re.compile(r"\.{3,}")
_DOT_TOKEN_RE = re.compile(r"(?<!\S)\.(?=\s|$)")
_PAREN_BIG_GAP_RE = re.compile(
    r"\b(large[\s_-]*break|broken[\s_-]*area|broken\s+lines?|broken\s+line|erased|illegible|missing|lacuna)\b",
    re.IGNORECASE,
)
_LINE_NUM_RE = re.compile(r"(?<!\w)\d+'+(?!\w)")
_ATF_EXCISE_RE = re.compile(r"<<.*?>>")
_ATF_SURROGATE_RE = re.compile(r"<\(([^)]+)\)>")
_ATF_INTRUSION_RE = re.compile(r"\(#.*?#\)")
_ATF_PROXIMITY_RE = re.compile(r"\$\d+")
_ATF_ALLOGRAPH_RE = re.compile(r"~[a-z0-9]+")
_ATF_MODIFIER_RE = re.compile(r"@\w+")
_ATF_LINGUISTIC_GLOSS_RE = re.compile(r"\{\{[^}]*\}\}")
_ATF_SHIFT_RE = re.compile(r"(?:(?<=^)|(?<=\s)|(?<=[({\[]))%[A-Za-z0-9]+(?=\s|$)")
_ATF_PUNCT_PAREN_RE = re.compile(r"[*/]\([^)]*\)")
_ATF_PUNCT_NAME_RE = re.compile(r"(?<!\S)P[1-4](?=\s|$)")
_ATF_PUNCT_TOKEN_RE = re.compile(r"(?<!\S)(?:\*|:'|:\"|:\.|::|:|/)(?=\s|$)")
_ANGLE_OPEN_RE = re.compile(r"<<+")
_ANGLE_CLOSE_RE = re.compile(r">>+")
_TITLE_LEAD_RE = re.compile(r"^(?:\{[^}]+\}|\([^)]*\))+")
_EMPTY_HYPHEN_SEG_RE = re.compile(r"-(?:\s+-)+")
_SZ_RE = re.compile(r"sz", re.IGNORECASE)
_H_RE = re.compile(r"[Hh]")
_S_COMMA_RE = re.compile(r"s,")
_S_COMMA_UPPER_RE = re.compile(r"S,")
_S_APOST_RE = re.compile(r"s'")
_S_APOST_UPPER_RE = re.compile(r"S'")
_T_COMMA_RE = re.compile(r"t,")
_T_COMMA_UPPER_RE = re.compile(r"T,")
_VOWEL_DIGIT_RE = re.compile(r"([aAeEiIuU])([23])")
_VOWEL_SUBSCRIPT_RE = re.compile(r"([aAeEiIuU])([₂₃])")
_SUBSCRIPT_X_RE = re.compile(r"Xx")
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUBSCRIPT_SET = set("₀₁₂₃₄₅₆₇₈₉ₓ")
_SUBSCRIPT_VOWEL_DIGIT_MAP = {"₂": "2", "₃": "3"}
_VOWEL_DIGIT_MAP = {
    "a2": "á",
    "a3": "à",
    "e2": "é",
    "e3": "è",
    "i2": "í",
    "i3": "ì",
    "u2": "ú",
    "u3": "ù",
    "A2": "Á",
    "A3": "À",
    "E2": "É",
    "E3": "È",
    "I2": "Í",
    "I3": "Ì",
    "U2": "Ú",
    "U3": "Ù",
}
_DOT_PLACEHOLDER = "__DOT__"
_SENT_SPLIT_MARK = "__SPLIT__"
_ABBR_RE = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|cf|ca|fig|figs|no|nos|vol|pp|p|ed|eds|al|mr|mrs|ms|dr|prof|st|jr|sr|rev)\.(?=\s+\w)",
    re.IGNORECASE,
)
_CAPS_ABBR_RE = re.compile(r"\b(?:[A-Z]\.){2,}(?=\s+\w)")
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")


@dataclass(frozen=True)
class QualityGateConfig:
    min_len_ratio: float
    max_len_ratio: float
    min_src_tokens: int
    min_tgt_chars: int
    min_alpha_ratio: float
    max_digit_ratio: float
    min_align_score: Optional[float]
    min_token_align_score: Optional[float]
    min_src_alpha_ratio: Optional[float]
    max_src_digit_ratio: Optional[float]
    max_src_symbol_ratio: Optional[float]
    max_tgt_symbol_ratio: Optional[float]
    max_tgt_sentence_endings: Optional[int]
    min_tgt_tokens: Optional[int]
    max_tgt_tokens: Optional[int]
    max_src_tokens: Optional[int]
    max_src_chars: Optional[int]
    max_tgt_chars: Optional[int]
    drop_tgt_trailing_quote: bool
    drop_tgt_odd_quote: bool


@dataclass(frozen=True)
class OareSentenceHint:
    """OARE の文単位アライン補助情報。

    `Sentences_Oare_FirstWord_LinNum.csv` に含まれる「文ごとの英訳」と
    「その文の先頭語（spelling / transcription）」を使って、train の文書を
    より安定して文単位へ切り出すために利用する。
    """

    translation: str
    first_word_spelling: Optional[str]
    first_word_transcription: Optional[str]
    first_word_number: Optional[int]
    sentence_obj_in_text: Optional[float]
    line_number: Optional[str]


def _parse_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _parse_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text == "":
        return None
    return int(float(text))


def _count_sentence_endings(text: str) -> int:
    if not text:
        return 0
    protected = _protect_sentence_dots(text)
    return len(re.findall(r'[.!?](?=(?:["\')\]]+)?\s|$)', protected))


def _ends_with_quote(text: str) -> bool:
    if not text:
        return False
    trimmed = text.rstrip()
    return trimmed.endswith('"') or trimmed.endswith("'")


def _has_odd_double_quote(text: str) -> bool:
    if not text:
        return False
    return text.count('"') % 2 == 1


def _text_stats(text: str) -> dict[str, float | int]:
    total = len(text)
    if total <= 0:
        return {
            "len": 0,
            "tokens": 0,
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "symbol_ratio": 0.0,
            "control": 0,
        }
    alpha = sum(ch.isalpha() for ch in text)
    digit = sum(ch.isdigit() for ch in text)
    space = sum(ch.isspace() for ch in text)
    symbol = total - alpha - digit - space
    content = max(total - space, 1)
    control = sum(1 for ch in text if ord(ch) < 32 and not ch.isspace())
    return {
        "len": total,
        "tokens": len(text.split()),
        "alpha_ratio": alpha / content,
        "digit_ratio": digit / content,
        "symbol_ratio": symbol / content,
        "control": control,
    }


def _bracket_missing_to_gap(match: re.Match) -> str:
    """欠損表現の括弧内を <gap>/<big_gap> に寄せる。"""
    body = match.group("body")
    if "…" in body or re.search(r"\.{3,}", body):
        return "<big_gap>"
    x_count = len(re.findall(r"[xX]", body))
    if x_count >= 4:
        return "<big_gap>"
    if x_count >= 1:
        return "<gap>"
    return "<big_gap>"


def _x_tokens_to_gap(match: re.Match) -> str:
    """空白区切りの x 連続を <gap>/<big_gap> に寄せる。"""
    seq = match.group(0)
    x_count = len(re.findall(r"[xX]", seq))
    if x_count >= 2:
        return "<big_gap>"
    return "<gap>"


def _normalize_subscript_digits(text: str) -> str:
    """下付き数字を ASCII にそろえる（母音の₂/₃は別処理）。"""
    return text.translate(_SUBSCRIPT_MAP)


def _normalize_cdli_oracc(text: str) -> str:
    """CDLI/ORACC系の表記ゆれを Unicode に寄せる。"""
    out = _S_COMMA_RE.sub("ṣ", text)
    out = _S_COMMA_UPPER_RE.sub("Ṣ", out)
    out = _S_APOST_RE.sub("ś", out)
    out = _S_APOST_UPPER_RE.sub("Ś", out)
    out = _T_COMMA_RE.sub("ṭ", out)
    out = _T_COMMA_UPPER_RE.sub("Ṭ", out)

    def _sz_to_scaron(match: re.Match) -> str:
        token = match.group(0)
        return "Š" if token.isupper() else "š"

    out = _SZ_RE.sub(_sz_to_scaron, out)

    def _h_to_unicode(match: re.Match) -> str:
        token = match.group(0)
        return "Ḫ" if token.isupper() else "ḫ"

    out = _H_RE.sub(_h_to_unicode, out)

    def _vowel_digit(match: re.Match) -> str:
        key = match.group(1) + match.group(2)
        return _VOWEL_DIGIT_MAP.get(key, match.group(0))

    out = out.translate(_SUPERSCRIPT_MAP)

    def _vowel_subscript(match: re.Match) -> str:
        key = match.group(1) + _SUBSCRIPT_VOWEL_DIGIT_MAP.get(match.group(2), "")
        return _VOWEL_DIGIT_MAP.get(key, match.group(0))

    out = _VOWEL_SUBSCRIPT_RE.sub(_vowel_subscript, out)
    out = _VOWEL_DIGIT_RE.sub(_vowel_digit, out)
    out = _SUBSCRIPT_X_RE.sub("ₓ", out)
    out = _normalize_subscript_digits(out)
    return out


_RAW_DETERMINATIVES = [
    "d",
    "mul",
    "ki",
    "lu2",
    "e2",
    "uru",
    "kur",
    "mi",
    "m",
    "gesz",
    "geš",
    "ĝeš",
    "tug2",
    "TÚG",
    "dub",
    "id2",
    "musen",
    "mušen",
    "na4",
    "kus",
    "kuš",
    "u2",
    "HI",
]


def _normalize_det_token(token: str) -> str:
    return _normalize_cdli_oracc(token).casefold()


_DETERMINATIVE_WHITELIST = {_normalize_det_token(token) for token in _RAW_DETERMINATIVES}


def _is_simple_determinative(text: str) -> bool:
    # 決定詞は短いトークンが中心なので、長すぎる語は除外する
    if len(text) > 6:
        return False
    for ch in text:
        if ch.isalnum():
            continue
        if ch in _SUBSCRIPT_SET:
            continue
        return False
    return _normalize_det_token(text) in _DETERMINATIVE_WHITELIST


def _is_sign_name(text: str) -> bool:
    """括弧内のサイン名らしさを判定する。"""
    if not text or " " in text:
        return False
    has_upper = False
    for ch in text:
        if ch.isalpha():
            if ch.islower():
                return False
            has_upper = True
            continue
        if ch.isdigit():
            continue
        if ch in {".", "×", "x", "X", "ₓ", "-"}:
            continue
        return False
    return has_upper


def _normalize_angle_brackets(text: str) -> str:
    text = _ANGLE_OPEN_RE.sub("<", text)
    text = _ANGLE_CLOSE_RE.sub(">", text)
    return text


def _normalize_html_sup_sub(text: str) -> str:
    """HTMLの上付き/下付きは決定詞検出に活用する。"""

    def _strip_tags(inner: str) -> str:
        return re.sub(r"<[^>]+>", "", inner)

    def _sup_repl(match: re.Match) -> str:
        inner = _strip_tags(match.group(1)).strip()
        if not inner:
            return ""
        if _is_simple_determinative(inner):
            return "{" + inner + "}"
        return inner

    def _sub_repl(match: re.Match) -> str:
        inner = _strip_tags(match.group(1)).strip()
        if not inner:
            return ""
        return inner

    text = _HTML_SUP_RE.sub(_sup_repl, text)
    text = _HTML_SUB_RE.sub(_sub_repl, text)
    return text


def _strip_doc_glosses(text: str) -> str:
    """{(...)} の文書グロスを除去する（入れ子対応）。"""
    chars: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        if i + 1 < n and text[i] == "{" and text[i + 1] == "(":
            depth = 1
            j = i + 2
            removed = False
            while j < n:
                ch = text[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        if j + 1 < n and text[j + 1] == "}":
                            chars.append(" ")
                            i = j + 2
                            removed = True
                        break
                j += 1
            if removed:
                continue
        chars.append(text[i])
        i += 1
    return "".join(chars)


def _is_number_token(token: str) -> bool:
    """翻訳側のスラッシュ判定向けに数値らしさを判定する。"""
    return bool(re.fullmatch(r"\d+(?:\.\d+)?[.,]?", token))


def _drop_translation_slash_tokens(text: str) -> str:
    """翻訳文中の行区切りスラッシュを除去する（分数は保持）。"""
    tokens = text.split()
    if not tokens:
        return text
    kept: List[str] = []
    for idx, token in enumerate(tokens):
        if token == "/":
            prev = tokens[idx - 1] if idx > 0 else ""
            next_ = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            if _is_number_token(prev) and _is_number_token(next_):
                kept.append(token)
            continue
        kept.append(token)
    return " ".join(kept)


def _is_title_token(token: str) -> bool:
    token = _TITLE_LEAD_RE.sub("", token)
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


def _protect_title_hyphens(text: str, placeholder: str) -> str:
    tokens = text.split()
    for idx, token in enumerate(tokens):
        if "-" not in token:
            continue
        if _is_title_token(token):
            tokens[idx] = token.replace("-", placeholder)
    return " ".join(tokens)


def normalize_transliteration(text: str, variant: str) -> str:
    if not text:
        return ""
    out = text.translate(_DASH_MAP)
    out = _APOSTROPHE_RE.sub("'", out)
    out = _QUOTE_RE.sub('"', out)
    out = _normalize_html_sup_sub(out)
    out = _normalize_cdli_oracc(out)

    variant = variant.upper()
    # ORACC インライン注記（グロス/シフト/句読点）を除去する
    out = _ATF_EXCISE_RE.sub(" ", out)
    out = _ATF_INTRUSION_RE.sub(" ", out)
    out = _ATF_SURROGATE_RE.sub(r"\1", out)
    out = _ATF_LINGUISTIC_GLOSS_RE.sub(" ", out)
    out = _strip_doc_glosses(out)
    out = _ATF_SHIFT_RE.sub(" ", out)
    out = _ATF_PUNCT_PAREN_RE.sub(" ", out)
    out = _ATF_PUNCT_NAME_RE.sub(" ", out)
    out = _ATF_PUNCT_TOKEN_RE.sub(" ", out)
    out = out.replace("|", "")

    out = _ATF_PROXIMITY_RE.sub("", out)
    out = out.replace("$", "")
    out = _ATF_ALLOGRAPH_RE.sub("", out)
    out = _ATF_MODIFIER_RE.sub("", out)
    out = out.replace("~", "")

    out = out.replace(";", " ")
    out = out.replace("=", " ")

    out = re.sub(r"[!?#*]", "", out)
    out = _LINE_NUM_RE.sub("", out)
    out = out.replace("˹", "").replace("˺", "")

    # 既存の欠損トークン表記を正規化
    out = _GAP_TAG_RE.sub("<gap>", out)
    out = _BIG_GAP_TAG_RE.sub("<big_gap>", out)
    out = _BRACE_BIG_GAP_RE.sub("<big_gap>", out)

    # 括弧内の欠損表記やコメントを整理
    out = _PAREN_MISSING_RE.sub(_bracket_missing_to_gap, out)

    def _normalize_parenthetical(match: re.Match) -> str:
        inner = (match.group(1) or "").strip()
        if not inner:
            return ""
        if _is_simple_determinative(inner):
            if variant in {"B", "C"}:
                return "{" + inner + "}"
            return "(" + inner + ")"
        if _PAREN_BIG_GAP_RE.search(inner):
            return "<big_gap>"
        if _is_sign_name(inner):
            return inner
        return " "

    out = re.sub(r"\(([^)]+)\)", _normalize_parenthetical, out)

    # 欠損/ギャップのマーカー
    out = _BRACKET_MISSING_RE.sub(_bracket_missing_to_gap, out)
    out = out.replace("…", "<big_gap>")
    out = _DOTS_BIG_GAP_RE.sub("<big_gap>", out)
    out = _X_TOKEN_RUN_RE.sub(_x_tokens_to_gap, out)

    # x/xxx の連続をギャップに置換
    def _x_to_gap(match: re.Match) -> str:
        seq = match.group(0)
        if len(seq) >= 2:
            return "<big_gap>"
        return "<gap>"

    out = re.sub(
        r"(?<![A-Za-z0-9])x{2,}(?![A-Za-z0-9])",
        _x_to_gap,
        out,
        flags=re.IGNORECASE,
    )

    # ここで生成した <gap>/<big_gap> を保護
    out = _normalize_angle_brackets(out)
    out = out.replace("<big_gap>", "__BIG_GAP__").replace("<gap>", "__GAP__")

    # HTML風タグはタグ本体のみ除去
    out = _HTML_TAG_RE.sub("", out)

    # 欠損トークン保護後に山括弧を除去し、中身のみ残す
    out = _normalize_angle_brackets(out)
    out = re.sub(r"<([^>]+)>", r"\1", out)

    # その他のケースでも角括弧を除去し、中身のみ残す
    out = re.sub(r"\[([^\]]+)\]", r"\1", out)

    # 欠損トークンを復元
    out = out.replace("__BIG_GAP__", "<big_gap>").replace("__GAP__", "<gap>")
    # excise による空のハイフン区切りを潰す
    out = _EMPTY_HYPHEN_SEG_RE.sub("-", out)

    # 決定詞の + マーカーを除去
    out = re.sub(r"\{\+\s*", "{", out)

    out = out.replace("//", " ")
    out = out.replace("/", " ")
    out = out.replace("+", "-")

    # 区切り文字を正規化
    out = out.replace(":", " ")
    out = _DOT_TOKEN_RE.sub(" ", out)

    if variant == "C":
        # より強い正規化: ハイフンを単純化し、残った括弧を削除
        out = out.replace("(", "").replace(")", "")
        placeholder = "__HYPHEN_KEEP__"
        out = _protect_title_hyphens(out, placeholder)
        out = out.replace("-", " ").replace(placeholder, "-")

    out = re.sub(r"\s+", " ", out).strip()
    return out


def normalize_translation(text: str) -> str:
    if not text:
        return ""
    out = text.translate(_DASH_MAP)
    out = _APOSTROPHE_RE.sub("'", out)
    out = _QUOTE_RE.sub('"', out)
    out = _LINE_NUM_RE.sub("", out)
    out = out.replace("˹", "").replace("˺", "")
    out = _HTML_TAG_RE.sub("", out)
    out = _normalize_angle_brackets(out)
    out = re.sub(r"<([^>]+)>", r"\1", out)
    out = re.sub(r"\[([^\]]+)\]", r"\1", out)
    out = _drop_translation_slash_tokens(out)
    out = clean_text(out)
    out = _strip_dangling_quotes(out)
    return out


def _strip_dangling_quotes(text: str) -> str:
    out = re.sub(r'("{2,})$', '"', text)
    out = re.sub(r"('{2,})$", "'", out)
    if out.endswith('"') and '"' not in out[:-1]:
        out = out[:-1].rstrip()
    if out.endswith("'") and "'" not in out[:-1]:
        out = out[:-1].rstrip()
    if out.count('"') % 2 == 1 and out.endswith('"'):
        out = out[:-1].rstrip()
    if out.count("'") % 2 == 1 and out.endswith("'"):
        out = out[:-1].rstrip()
    return out


def _protect_sentence_dots(text: str) -> str:
    out = _DECIMAL_DOT_RE.sub(_DOT_PLACEHOLDER, text)
    out = _CAPS_ABBR_RE.sub(lambda m: m.group(0).replace(".", _DOT_PLACEHOLDER), out)
    out = _ABBR_RE.sub(lambda m: m.group(0).replace(".", _DOT_PLACEHOLDER), out)
    return out


def _restore_sentence_dots(text: str) -> str:
    return text.replace(_DOT_PLACEHOLDER, ".")


def split_english(text: str, min_len: int = 20) -> List[str]:
    cleaned = normalize_translation(text)
    if not cleaned:
        return []

    protected = _protect_sentence_dots(cleaned)
    protected = re.sub(r'([.!?]["\']*)(\s+)', r"\1" + _SENT_SPLIT_MARK, protected)
    parts = [p.strip() for p in protected.split(_SENT_SPLIT_MARK)]
    candidates: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.count(";") >= 2 or len(part) >= 300:
            chunks = [p.strip() for p in part.split(";") if p.strip()]
            for i, chunk in enumerate(chunks):
                if i < len(chunks) - 1:
                    chunk = chunk + ";"
                candidates.append(chunk)
        else:
            candidates.append(part)

    candidates = [_restore_sentence_dots(seg) for seg in candidates]
    merged: List[str] = []
    for seg in candidates:
        if merged and len(seg) < min_len:
            merged[-1] = (merged[-1] + " " + seg).strip()
        else:
            merged.append(seg)

    return merged


def merge_targets_to_match(tgt_sents: List[str], src_tokens: List[str]) -> Tuple[List[str], bool]:
    if len(src_tokens) <= 0:
        return tgt_sents, False
    merged = list(tgt_sents)
    changed = False
    while len(merged) > len(src_tokens):
        merged[-2] = (merged[-2].rstrip() + " " + merged[-1].lstrip()).strip()
        merged.pop()
        changed = True
    return merged, changed


def segment_source_tokens(
    tokens: List[str],
    target_lengths: List[int],
    min_tokens: int = 3,
    min_ratio: float = 0.25,
    max_ratio: float = 4.0,
) -> List[List[str]]:
    if not target_lengths:
        return [tokens]

    n = len(target_lengths)
    total_tokens = len(tokens)
    if n == 1 or total_tokens == 0:
        return [tokens]

    total_tgt = sum(target_lengths)
    desired = [total_tokens * (t / total_tgt) for t in target_lengths]

    inf = float("inf")
    dp = [[inf] * (total_tokens + 1) for _ in range(n + 1)]
    back = [[None] * (total_tokens + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        di = desired[i - 1]
        base_min = max(1, int(di * min_ratio))
        base_max = max(base_min, int(di * max_ratio) + 1)
        for k in range(0, total_tokens + 1):
            if dp[i - 1][k] == inf:
                continue
            remaining = total_tokens - k
            remaining_segs = n - i
            if i == n:
                min_len = max(1, remaining)
                max_len = remaining
            else:
                min_len = max(base_min, 1)
                max_len = min(base_max, remaining - remaining_segs)
                min_len = min(min_len, max_len)
            if max_len < 1:
                continue

            for l in range(min_len, max_len + 1):
                if remaining - l < remaining_segs:
                    continue
                penalty = 0.0
                if l < min_tokens:
                    penalty += (min_tokens - l) * 5.0
                cost = dp[i - 1][k] + (l - di) ** 2 + penalty
                nxt = k + l
                if cost < dp[i][nxt]:
                    dp[i][nxt] = cost
                    back[i][nxt] = l

    if dp[n][total_tokens] == inf:
        # フォールバック: 比例分割
        cuts = []
        acc = 0
        for di in desired[:-1]:
            acc += di
            cuts.append(int(round(acc)))
        cuts = [min(max(1, c), total_tokens - (n - idx - 1)) for idx, c in enumerate(cuts)]
        boundaries = sorted(set(cuts))
    else:
        boundaries = []
        pos = total_tokens
        for i in range(n, 0, -1):
            l = back[i][pos]
            if l is None:
                boundaries = []
                break
            pos -= l
            boundaries.append(pos)
        boundaries = sorted(boundaries)[1:]

    segments = []
    start = 0
    for b in boundaries:
        segments.append(tokens[start:b])
        start = b
    segments.append(tokens[start:])
    return segments




def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        # pandas / numpy の NaN 判定
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def _optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    try:
        return int(float(value))
    except Exception:
        return None


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def load_oare_sentence_hints(path: Path) -> dict[str, List[OareSentenceHint]]:
    """Sentences_Oare_FirstWord_LinNum.csv を読み、text_uuid -> 文リストへまとめる。"""
    if not path.exists():
        raise FileNotFoundError(f"OARE sentence file not found: {path}")

    df = pd.read_csv(path)

    required = [
        "text_uuid",
        "sentence_obj_in_text",
        "translation",
        "first_word_transcription",
        "first_word_spelling",
        "first_word_number",
        "line_number",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OARE sentence file missing columns: {missing}")

    df = df[required].copy()
    df["translation"] = df["translation"].fillna("").astype(str)
    df = df[df["translation"].str.strip() != ""].copy()

    # ソート（安定ソート）
    df["sentence_obj_in_text_num"] = pd.to_numeric(df["sentence_obj_in_text"], errors="coerce")
    df["first_word_number_num"] = pd.to_numeric(df["first_word_number"], errors="coerce")
    df = df.sort_values(
        ["text_uuid", "sentence_obj_in_text_num", "first_word_number_num", "line_number"],
        kind="mergesort",
    )

    out: dict[str, List[OareSentenceHint]] = {}
    for text_uuid, g in df.groupby("text_uuid", sort=False):
        hints: List[OareSentenceHint] = []
        for _, r in g.iterrows():
            hints.append(
                OareSentenceHint(
                    translation=str(r.get("translation", "")),
                    first_word_spelling=_optional_str(r.get("first_word_spelling")),
                    first_word_transcription=_optional_str(r.get("first_word_transcription")),
                    first_word_number=_optional_int(r.get("first_word_number")),
                    sentence_obj_in_text=_optional_float(r.get("sentence_obj_in_text")),
                    line_number=_optional_str(r.get("line_number")),
                )
            )
        if hints:
            out[str(text_uuid)] = hints
    return out


def _find_subsequence_positions(tokens: List[str], pattern: List[str], start: int) -> List[int]:
    if not pattern:
        return []
    if start < 0:
        start = 0
    if start >= len(tokens):
        return []
    out: List[int] = []
    limit = len(tokens) - len(pattern)
    for i in range(start, limit + 1):
        if tokens[i : i + len(pattern)] == pattern:
            out.append(i)
    return out


def _hint_first_word_tokens(hint: OareSentenceHint, variant: str) -> List[str]:
    raw = hint.first_word_spelling or hint.first_word_transcription
    if not raw:
        return []
    norm = normalize_transliteration(raw, variant)
    return norm.split()


def _build_oare_sentence_anchors(
    src_tokens: List[str],
    hints: List[OareSentenceHint],
    variant: str,
) -> Tuple[List[Tuple[int, int]], dict[str, int]]:
    """先頭語の出現位置から「文開始 anchor」を作る。

    返り値は (sentence_index, token_index) のリスト。
    sentence_index は tgt_sents / hints のインデックス基準。
    """
    if not hints:
        return [(0, 0), (0, len(src_tokens))]

    total = len(src_tokens)
    max_num = 0
    for h in hints:
        if h.first_word_number is not None:
            max_num = max(max_num, h.first_word_number)

    anchors: List[Tuple[int, int]] = [(0, 0)]
    chosen_positions: List[int] = []
    # 先頭文は 0 から開始する（先頭語が見つからないケースがあるため）
    last_search = 0
    for i in range(1, len(hints)):
        pat = _hint_first_word_tokens(hints[i], variant)
        if not pat:
            continue

        candidates = _find_subsequence_positions(src_tokens, pat, last_search)
        if not candidates and len(pat) > 1:
            # フォールバック: 先頭トークンのみ
            candidates = _find_subsequence_positions(src_tokens, [pat[0]], last_search)

        if not candidates:
            continue

        expected: Optional[float] = None
        if max_num > 0 and hints[i].first_word_number is not None:
            expected = (hints[i].first_word_number / max_num) * total

        if expected is None:
            chosen = candidates[0]
        else:
            chosen = min(candidates, key=lambda x: (abs(x - expected), x))

        chosen_positions.append(chosen)

        # anchor は単調増加のみ採用
        if chosen <= anchors[-1][1]:
            continue
        anchors.append((i, chosen))
        last_search = chosen + max(len(pat), 1)

    anchors.append((len(hints), total))

    duplicate_anchor_positions = len(chosen_positions) - len(set(chosen_positions))
    meta = {
        "duplicate_anchor_positions": duplicate_anchor_positions,
        "anchor_candidates": len(chosen_positions),
    }
    return anchors, meta


def segment_source_tokens_with_oare_hints(
    src_tokens: List[str],
    tgt_sents: List[str],
    hints: List[OareSentenceHint],
    variant: str,
) -> Tuple[List[List[str]], dict[str, object]]:
    """OARE の先頭語ヒントを使って src を文単位へ切る。

    - anchor が十分取れれば、その anchor を固定して区間ごとに DP 分割する
    - anchor が取れなければ従来通り DP 分割にフォールバックする

    NOTE: ここで返す `meta` は **品質フラグではなくメタ情報**。
    `--drop-flagged` の挙動に影響しないよう、flags 列には入れない。
    """
    meta: dict[str, object] = {"anchors_used": 0, "status": "init"}

    if not src_tokens:
        meta["status"] = "empty_src_tokens"
        return [[] for _ in tgt_sents], meta

    if len(tgt_sents) != len(hints) or not tgt_sents:
        # lengths が合わない場合は安全にフォールバック
        lengths = [len(s) for s in tgt_sents] if tgt_sents else [0]
        meta["status"] = "len_mismatch"
        return segment_source_tokens(src_tokens, lengths), meta

    anchors, anchor_meta = _build_oare_sentence_anchors(src_tokens, hints, variant)
    used = max(len(anchors) - 2, 0)
    meta["anchors_used"] = used
    meta["anchors_found"] = used
    meta.update(anchor_meta)

    if used <= 0:
        lengths = [len(s) for s in tgt_sents]
        meta["status"] = "no_anchor"
        return segment_source_tokens(src_tokens, lengths), meta

    segments: List[List[str]] = []
    for (s0, t0), (s1, t1) in zip(anchors[:-1], anchors[1:]):
        chunk_tgts = tgt_sents[s0:s1]
        chunk_tokens = src_tokens[t0:t1]
        if not chunk_tgts:
            continue
        if len(chunk_tgts) == 1:
            segments.append(chunk_tokens)
            continue
        lengths = [len(s) for s in chunk_tgts]
        chunk_segs = segment_source_tokens(chunk_tokens, lengths)
        # 念のため長さが合わない場合はフォールバック
        if len(chunk_segs) != len(chunk_tgts):
            chunk_segs = segment_source_tokens(chunk_tokens, lengths)
        segments.extend(chunk_segs)

    if len(segments) != len(tgt_sents):
        lengths = [len(s) for s in tgt_sents]
        meta["status"] = "fallback_dp"
        return segment_source_tokens(src_tokens, lengths), meta

    meta["status"] = "anchored"
    return segments, meta


def _len_ratio_outlier_score(min_len_ratio: float, max_len_ratio: float, gate_cfg: QualityGateConfig) -> float:
    if min_len_ratio <= 0:
        return 1e9
    score = 0.0
    if min_len_ratio < gate_cfg.min_len_ratio:
        score = max(score, gate_cfg.min_len_ratio / max(min_len_ratio, 1e-9))
    if max_len_ratio > gate_cfg.max_len_ratio:
        score = max(score, max_len_ratio / max(gate_cfg.max_len_ratio, 1e-9))
    return score


def compute_flags(
    src_stats: dict[str, float | int],
    tgt_stats: dict[str, float | int],
    len_ratio: float,
    align_score: float,
    token_align_score: float,
    tgt_sentence_ends: int,
    gate_cfg: QualityGateConfig,
) -> List[str]:
    flags = []
    if src_stats["len"] <= 0:
        flags.append("empty_src")
    if tgt_stats["len"] <= 0:
        flags.append("empty_tgt")
    if len_ratio < gate_cfg.min_len_ratio or len_ratio > gate_cfg.max_len_ratio:
        flags.append("len_ratio")

    if gate_cfg.min_align_score is not None and align_score < gate_cfg.min_align_score:
        flags.append("align_score")
    if gate_cfg.min_token_align_score is not None and token_align_score < gate_cfg.min_token_align_score:
        flags.append("token_align")

    if int(src_stats["tokens"]) < gate_cfg.min_src_tokens:
        flags.append("short_src")
    if gate_cfg.max_src_tokens is not None and int(src_stats["tokens"]) > gate_cfg.max_src_tokens:
        flags.append("long_src_tokens")
    if gate_cfg.max_src_chars is not None and int(src_stats["len"]) > gate_cfg.max_src_chars:
        flags.append("long_src_chars")

    if int(tgt_stats["len"]) < gate_cfg.min_tgt_chars:
        flags.append("short_tgt")
    if gate_cfg.min_tgt_tokens is not None and int(tgt_stats["tokens"]) < gate_cfg.min_tgt_tokens:
        flags.append("short_tgt_tokens")
    if gate_cfg.max_tgt_tokens is not None and int(tgt_stats["tokens"]) > gate_cfg.max_tgt_tokens:
        flags.append("long_tgt_tokens")
    if gate_cfg.max_tgt_chars is not None and int(tgt_stats["len"]) > gate_cfg.max_tgt_chars:
        flags.append("long_tgt_chars")

    if gate_cfg.min_src_alpha_ratio is not None:
        if float(src_stats["alpha_ratio"]) < gate_cfg.min_src_alpha_ratio:
            flags.append("low_alpha_src")
    if gate_cfg.max_src_digit_ratio is not None:
        if float(src_stats["digit_ratio"]) > gate_cfg.max_src_digit_ratio:
            flags.append("digit_heavy_src")
    if gate_cfg.max_src_symbol_ratio is not None:
        if float(src_stats["symbol_ratio"]) > gate_cfg.max_src_symbol_ratio:
            flags.append("symbol_heavy_src")

    if gate_cfg.min_alpha_ratio > 0.0:
        if float(tgt_stats["alpha_ratio"]) < gate_cfg.min_alpha_ratio:
            flags.append("low_alpha_tgt")
    if gate_cfg.max_digit_ratio > 0.0:
        if float(tgt_stats["digit_ratio"]) > gate_cfg.max_digit_ratio:
            flags.append("digit_heavy_tgt")
    if gate_cfg.max_tgt_symbol_ratio is not None:
        if float(tgt_stats["symbol_ratio"]) > gate_cfg.max_tgt_symbol_ratio:
            flags.append("symbol_heavy_tgt")

    if gate_cfg.max_tgt_sentence_endings is not None:
        if tgt_sentence_ends > gate_cfg.max_tgt_sentence_endings:
            flags.append("multi_sentence_tgt")

    if int(src_stats["control"]) > 0:
        flags.append("control_src")
    if int(tgt_stats["control"]) > 0:
        flags.append("control_tgt")

    return flags


def _append_flag(flags: str, flag: str) -> str:
    if not flags:
        return flag
    parts = [p for p in flags.split(";") if p]
    if flag in parts:
        return ";".join(parts)
    parts.append(flag)
    return ";".join(parts)


def apply_duplicate_flags(
    df: pd.DataFrame,
    max_tgt_per_src: Optional[int],
    max_src_per_tgt: Optional[int],
) -> pd.DataFrame:
    if df.empty:
        return df

    if max_tgt_per_src is not None:
        src_counts = df.groupby("src_sent")["tgt_sent"].nunique()
        bad_src = src_counts[src_counts > max_tgt_per_src].index
        if len(bad_src) > 0:
            mask = df["src_sent"].isin(bad_src)
            df.loc[mask, "flags"] = (
                df.loc[mask, "flags"].fillna("").map(lambda v: _append_flag(v, "src_multi_tgt"))
            )

    if max_src_per_tgt is not None:
        tgt_counts = df.groupby("tgt_sent")["src_sent"].nunique()
        bad_tgt = tgt_counts[tgt_counts > max_src_per_tgt].index
        if len(bad_tgt) > 0:
            mask = df["tgt_sent"].isin(bad_tgt)
            df.loc[mask, "flags"] = (
                df.loc[mask, "flags"].fillna("").map(lambda v: _append_flag(v, "tgt_multi_src"))
            )

    return df


def summarize_series(series: pd.Series, label: str) -> None:
    if series.empty:
        print(f"[{label}] no data")
        return
    q = series.quantile([0.5, 0.9, 0.95]).to_dict()
    print(
        f"[{label}] n={len(series)} min={series.min():.4f} max={series.max():.4f} "
        f"p50={q[0.5]:.4f} p90={q[0.9]:.4f} p95={q[0.95]:.4f}"
    )


def count_flags(flags_series: pd.Series) -> Counter:
    counts: Counter = Counter()
    for value in flags_series.fillna(""):
        for flag in str(value).split(";"):
            flag = flag.strip()
            if flag:
                counts[flag] += 1
    return counts


def log_alignment_stats(
    df: pd.DataFrame,
    gate_cfg: QualityGateConfig,
    label: str,
    max_tgt_per_src: Optional[int],
    max_src_per_tgt: Optional[int],
) -> None:
    print(f"=== アライン品質ログ ({label}) ===")
    if "align_score" in df.columns:
        summarize_series(df["align_score"], "align_score")
        if gate_cfg.min_align_score is not None:
            low_ratio = (df["align_score"] < gate_cfg.min_align_score).mean()
            print(f"align_score_below_min={low_ratio:.1%}")
    if "len_ratio" in df.columns:
        summarize_series(df["len_ratio"], "len_ratio")
        out_ratio = (
            (df["len_ratio"] < gate_cfg.min_len_ratio)
            | (df["len_ratio"] > gate_cfg.max_len_ratio)
        ).mean()
        print(f"len_ratio_outside={out_ratio:.1%}")

    if "flags" in df.columns:
        flags = df["flags"].fillna("").astype(str)
        counts = count_flags(flags)
        if counts:
            print("flags_top10:")
            for name, count in counts.most_common(10):
                ratio = count / len(df) if len(df) else 0.0
                print(f"- {name}: {count} ({ratio:.1%})")
        multi_src_rate = flags.str.contains("src_multi_tgt").mean()
        multi_tgt_rate = flags.str.contains("tgt_multi_src").mean()
        print(
            f"src_multi_tgt_rate={multi_src_rate:.1%} "
            f"tgt_multi_src_rate={multi_tgt_rate:.1%} "
            f"max_tgt_per_src={max_tgt_per_src} max_src_per_tgt={max_src_per_tgt}"
        )

    if "tgt_sent" in df.columns:
        trimmed = df["tgt_sent"].fillna("").astype(str).str.rstrip()
        trailing_quote_rate = (trimmed.str.endswith('"') | trimmed.str.endswith("'")).mean()
        print(f"trailing_quote_rate={trailing_quote_rate:.1%}")
        odd_double_quote_rate = trimmed.map(lambda s: s.count('"') % 2 == 1).mean()
        print(f"odd_double_quote_rate={odd_double_quote_rate:.1%}")

    if "oare_id" in df.columns:
        counts = df.groupby("oare_id").size()
        if len(counts) > 0:
            q = counts.quantile([0.5, 0.9, 0.99]).to_dict()
            print(
                f"sentences_per_oare_id p50={q[0.5]:.0f} p90={q[0.9]:.0f} "
                f"p99={q[0.99]:.0f} max={counts.max():.0f}"
            )


def _align_document_candidate(
    oare_id: str,
    src_text: str,
    tgt_text: str,
    variant: str,
    gate_cfg: QualityGateConfig,
    oare_hints: Optional[List[OareSentenceHint]] = None,
    debug: bool = False,
) -> Union[List[dict], Tuple[List[dict], dict]]:
    src_norm = normalize_transliteration(src_text, variant)
    src_tokens = src_norm.split()

    hint_mode = False
    hint_flags: List[str] = []
    oare_meta: dict[str, object] = {}
    filtered_hints: Optional[List[OareSentenceHint]] = None
    expected_sentences_raw = len(oare_hints) if oare_hints else 0

    if oare_hints:
        tgt_sents_from_hint: List[str] = []
        filtered: List[OareSentenceHint] = []
        for hint in oare_hints:
            t = clean_text(normalize_translation(str(hint.translation)))
            t = _strip_dangling_quotes(t)
            t = clean_text(t)
            if t:
                tgt_sents_from_hint.append(t)
                filtered.append(hint)
        if tgt_sents_from_hint:
            hint_mode = True
            hint_flags.append("oare_sentence_translation")
            filtered_hints = filtered
            tgt_sents = tgt_sents_from_hint
        else:
            hint_flags.append("oare_sentence_empty")
            tgt_sents = []
    else:
        tgt_sents = []

    if not hint_mode:
        tgt_norm = normalize_translation(tgt_text)
        tgt_sents = split_english(tgt_norm)
        if not tgt_sents:
            tgt_sents = [tgt_norm] if tgt_norm else []

    tgt_sents, merged_flag = merge_targets_to_match(tgt_sents, src_tokens)

    target_lengths = [len(s) for s in tgt_sents] if tgt_sents else [len(tgt_text or "")]

    if hint_mode and (not merged_flag) and filtered_hints is not None and len(filtered_hints) == len(tgt_sents):
        segments, oare_meta = segment_source_tokens_with_oare_hints(
            src_tokens,
            tgt_sents,
            filtered_hints,
            variant,
        )
    else:
        if hint_mode:
            if merged_flag:
                hint_flags.append("oare_hint_dropped_after_merge")
            elif filtered_hints is None or len(filtered_hints) != len(tgt_sents):
                hint_flags.append("oare_hint_len_mismatch")
        segments = segment_source_tokens(src_tokens, target_lengths)

    align_method = "oare" if hint_mode else "dp"
    oare_anchors_used = int(oare_meta.get("anchors_used") or 0) if hint_mode else 0
    meta_parts: List[str] = []
    if hint_mode:
        meta_parts.extend(hint_flags)
        status = str(oare_meta.get("status") or "")
        if status:
            meta_parts.append(f"oare_{status}")
        meta_parts.append(f"oare_anchors_{oare_anchors_used}")
    align_meta = ";".join([p for p in meta_parts if p])

    tgt_clean: List[str] = []
    for t in tgt_sents:
        t_clean = clean_text(t)
        t_clean = _strip_dangling_quotes(t_clean)
        t_clean = clean_text(t_clean)
        tgt_clean.append(t_clean)

    seg_lengths = [len(seg) for seg in segments]
    empty_src_segments = sum(1 for l in seg_lengths if l <= 0)
    empty_tgt_segments = sum(1 for t in tgt_clean if not t)
    min_seg_tokens = min(seg_lengths) if seg_lengths else 0
    max_seg_tokens = max(seg_lengths) if seg_lengths else 0

    rows = []
    len_ratios: List[float] = []
    for idx, tgt_sent in enumerate(tgt_sents):
        src_seg = " ".join(segments[idx]) if idx < len(segments) else ""
        src_seg = clean_text(src_seg)
        tgt_sent = tgt_clean[idx] if idx < len(tgt_clean) else ""
        src_stats = _text_stats(src_seg)
        tgt_stats = _text_stats(tgt_sent)
        tgt_sentence_ends = _count_sentence_endings(tgt_sent)
        tgt_trailing_quote = _ends_with_quote(tgt_sent)
        tgt_odd_double_quote = _has_odd_double_quote(tgt_sent)

        src_len = int(src_stats["len"])
        tgt_len = int(tgt_stats["len"])
        len_ratio = src_len / max(tgt_len, 1)
        len_ratios.append(float(len_ratio))
        align_score = min(src_len, tgt_len) / max(src_len, tgt_len, 1)
        if int(src_stats["tokens"]) > 0 and int(tgt_stats["tokens"]) > 0:
            token_align_score = min(int(src_stats["tokens"]), int(tgt_stats["tokens"])) / max(
                int(src_stats["tokens"]), int(tgt_stats["tokens"])
            )
        else:
            token_align_score = 0.0

        flags = compute_flags(
            src_stats,
            tgt_stats,
            len_ratio,
            align_score,
            token_align_score,
            tgt_sentence_ends,
            gate_cfg,
        )
        if merged_flag:
            flags.append("merged_tgt")
        if gate_cfg.drop_tgt_trailing_quote and tgt_trailing_quote:
            flags.append("trailing_quote")
        if gate_cfg.drop_tgt_odd_quote and tgt_odd_double_quote:
            flags.append("odd_quote")

        rows.append(
            {
                "oare_id": oare_id,
                "src_sent": src_seg,
                "tgt_sent": tgt_sent,
                "src_norm_variant": variant,
                "align_method": align_method,
                "align_meta": align_meta,
                "oare_anchors_used": oare_anchors_used,
                "align_score": round(align_score, 6),
                "len_ratio": round(len_ratio, 6),
                "token_align_score": round(token_align_score, 6),
                "src_tokens": int(src_stats["tokens"]),
                "tgt_tokens": int(tgt_stats["tokens"]),
                "src_alpha_ratio": round(float(src_stats["alpha_ratio"]), 6),
                "src_digit_ratio": round(float(src_stats["digit_ratio"]), 6),
                "src_symbol_ratio": round(float(src_stats["symbol_ratio"]), 6),
                "tgt_alpha_ratio": round(float(tgt_stats["alpha_ratio"]), 6),
                "tgt_digit_ratio": round(float(tgt_stats["digit_ratio"]), 6),
                "tgt_symbol_ratio": round(float(tgt_stats["symbol_ratio"]), 6),
                "tgt_sentence_ends": int(tgt_sentence_ends),
                "flags": ";".join(sorted(set(flags))),
            }
        )
    expected_sentences = len(filtered_hints) if filtered_hints is not None else 0
    actual_sentences = len(tgt_sents)
    min_len_ratio = min(len_ratios) if len_ratios else 0.0
    max_len_ratio = max(len_ratios) if len_ratios else 0.0
    zero_len_ratio_count = sum(1 for lr in len_ratios if lr == 0.0)

    if debug:
        debug_info = {
            "oare_id": oare_id,
            "variant": variant,
            "hint_mode": hint_mode,
            "expected_sentences": expected_sentences,
            "expected_sentences_raw": expected_sentences_raw,
            "actual_sentences": actual_sentences,
            "anchors_found": int(oare_meta.get("anchors_found") or 0) if hint_mode else 0,
            "duplicate_anchor_positions": int(oare_meta.get("duplicate_anchor_positions") or 0)
            if hint_mode
            else 0,
            "empty_src_segments": empty_src_segments,
            "empty_tgt_segments": empty_tgt_segments,
            "min_seg_tokens": min_seg_tokens,
            "max_seg_tokens": max_seg_tokens,
            "min_len_ratio": round(min_len_ratio, 6),
            "max_len_ratio": round(max_len_ratio, 6),
            "zero_len_ratio_count": zero_len_ratio_count,
            "align_method": align_method,
            "align_status": str(oare_meta.get("status") or ""),
            "align_meta": align_meta,
        }
        return rows, debug_info

    return rows





@dataclass(frozen=True)
class _AlignCandidateSummary:
    n_total: int
    n_good: int
    good_ratio: float
    mean_align_good: float
    mean_token_align_good: float
    min_len_ratio: float
    max_len_ratio: float
    len_ratio_outlier: float
    score_key: Tuple[float, float, float, float, float]


def _is_empty_flags(value: object) -> bool:
    if value is None:
        return True
    text = str(value)
    return text.strip() == ""


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _summarize_candidate(rows: List[dict], gate_cfg: QualityGateConfig) -> _AlignCandidateSummary:
    n_total = len(rows)
    flags = [row.get("flags") for row in rows]
    good_mask = [_is_empty_flags(f) for f in flags]
    n_good = int(sum(1 for ok in good_mask if ok))
    good_ratio = (n_good / n_total) if n_total else 0.0

    align_good: List[float] = []
    token_good: List[float] = []
    len_ratios: List[float] = []
    for row, ok in zip(rows, good_mask):
        try:
            len_ratios.append(float(row.get("len_ratio") or 0.0))
        except Exception:
            len_ratios.append(0.0)
        if not ok:
            continue
        try:
            align_good.append(float(row.get("align_score") or 0.0))
        except Exception:
            align_good.append(0.0)
        try:
            token_good.append(float(row.get("token_align_score") or 0.0))
        except Exception:
            token_good.append(0.0)

    min_lr = min(len_ratios) if len_ratios else 0.0
    max_lr = max(len_ratios) if len_ratios else 0.0
    outlier = float(_len_ratio_outlier_score(min_lr, max_lr, gate_cfg))

    mean_align_good = _mean(align_good)
    mean_token_good = _mean(token_good)

    # まずは「drop-flagged 後に残る行数」を最大化し、同点なら品質（align_score 等）で決める。
    score_key = (float(n_good), float(good_ratio), mean_align_good, mean_token_good, -outlier)
    return _AlignCandidateSummary(
        n_total=n_total,
        n_good=n_good,
        good_ratio=float(good_ratio),
        mean_align_good=mean_align_good,
        mean_token_align_good=mean_token_good,
        min_len_ratio=float(min_lr),
        max_len_ratio=float(max_lr),
        len_ratio_outlier=float(outlier),
        score_key=score_key,
    )


def _extract_oare_fallback_reason(meta: str) -> Optional[str]:
    if not meta:
        return None
    for part in str(meta).split(";"):
        part = part.strip()
        if part.startswith("oare_fallback_"):
            return part[len("oare_fallback_") :]
    return None


def align_document(
    oare_id: str,
    src_text: str,
    tgt_text: str,
    variant: str,
    gate_cfg: QualityGateConfig,
    oare_hints: Optional[List[OareSentenceHint]] = None,
    debug: bool = False,
    oare_doc_select: bool = True,
    oare_min_anchors: int = 1,
    oare_tie_break: str = "oare",
) -> Union[List[dict], Tuple[List[dict], dict]]:
    """文書1件を文ペアへ変換する（OARE hints があれば doc-level に自動選択）。

    (A) no_anchor / 非anchored な場合は **OARE を無効化**して baseline DP にフォールバック
    (B) anchored でも、品質ゲートを通過する行数（drop-flagged後）が baseline より悪ければフォールバック

    `oare_doc_select=False` にすると従来通りの動作（OARE hints があれば常に使用）に戻る。
    """

    # OARE が無い、または doc-level 選択を使わないなら従来通り
    if not oare_hints or not oare_doc_select:
        return _align_document_candidate(
            oare_id,
            src_text,
            tgt_text,
            variant,
            gate_cfg,
            oare_hints=oare_hints,
            debug=debug,
        )

    # まず OARE を試す（meta を見るため debug=True で実行）
    rows_oare, dbg_oare = _align_document_candidate(
        oare_id,
        src_text,
        tgt_text,
        variant,
        gate_cfg,
        oare_hints=oare_hints,
        debug=True,
    )

    # OARE 側に sentence translations が無い場合は baseline と同等（そのまま返す）
    if not bool(dbg_oare.get("hint_mode")):
        if debug:
            dbg_oare = dict(dbg_oare)
            dbg_oare["oare_available"] = False
            dbg_oare["selected_method"] = "dp"
            dbg_oare["selected_reason"] = "oare_no_sentence_translations"
            return rows_oare, dbg_oare
        return rows_oare

    # baseline（train.csv の translation を split）も作り、doc-level に比較
    rows_dp, dbg_dp = _align_document_candidate(
        oare_id,
        src_text,
        tgt_text,
        variant,
        gate_cfg,
        oare_hints=None,
        debug=True,
    )

    anchors_found = int(dbg_oare.get("anchors_found") or 0)
    oare_status = str(dbg_oare.get("align_status") or "")
    eligible = (anchors_found >= int(oare_min_anchors)) and (oare_status == "anchored")

    # score 用サマリ（比較とデバッグ）
    sum_oare = _summarize_candidate(rows_oare, gate_cfg)
    sum_dp = _summarize_candidate(rows_dp, gate_cfg)

    selected_method = "oare"
    selected_reason = "oare_selected"

    if not eligible:
        selected_method = "dp"
        if anchors_found < int(oare_min_anchors):
            selected_reason = "no_anchor"
        else:
            selected_reason = f"status_{oare_status or 'unknown'}"
    else:
        if sum_oare.score_key > sum_dp.score_key:
            selected_method = "oare"
            selected_reason = "oare_better"
        elif sum_oare.score_key < sum_dp.score_key:
            selected_method = "dp"
            selected_reason = "dp_better"
        else:
            prefer = str(oare_tie_break or "oare").lower()
            if prefer not in {"oare", "dp"}:
                prefer = "oare"
            selected_method = prefer
            selected_reason = f"tie_{prefer}"

    selected_rows = rows_oare if selected_method == "oare" else rows_dp

    # 出力 rows に「選択結果」を印として残す（集計しやすいように align_meta に入れる）
    if selected_method == "oare":
        for row in selected_rows:
            row["align_meta"] = _append_flag(str(row.get("align_meta") or ""), "oare_selected")
    else:
        for row in selected_rows:
            meta = str(row.get("align_meta") or "")
            meta = _append_flag(meta, "oare_available")
            meta = _append_flag(meta, f"oare_fallback_{selected_reason}")
            row["align_meta"] = meta

    if debug:
        # base は「実際に採用した方」の debug_info を採用しつつ、比較情報も付与
        out_dbg = dict(dbg_oare if selected_method == "oare" else dbg_dp)
        out_dbg["oare_available"] = True
        out_dbg["selected_method"] = selected_method
        out_dbg["selected_reason"] = selected_reason

        out_dbg["oare_candidate_status"] = oare_status
        out_dbg["oare_candidate_anchors_found"] = anchors_found

        out_dbg["oare_n_total"] = sum_oare.n_total
        out_dbg["oare_n_good"] = sum_oare.n_good
        out_dbg["oare_good_ratio"] = round(sum_oare.good_ratio, 6)
        out_dbg["oare_mean_align_good"] = round(sum_oare.mean_align_good, 6)
        out_dbg["oare_mean_token_align_good"] = round(sum_oare.mean_token_align_good, 6)
        out_dbg["oare_min_len_ratio"] = round(sum_oare.min_len_ratio, 6)
        out_dbg["oare_max_len_ratio"] = round(sum_oare.max_len_ratio, 6)

        out_dbg["dp_n_total"] = sum_dp.n_total
        out_dbg["dp_n_good"] = sum_dp.n_good
        out_dbg["dp_good_ratio"] = round(sum_dp.good_ratio, 6)
        out_dbg["dp_mean_align_good"] = round(sum_dp.mean_align_good, 6)
        out_dbg["dp_mean_token_align_good"] = round(sum_dp.mean_token_align_good, 6)
        out_dbg["dp_min_len_ratio"] = round(sum_dp.min_len_ratio, 6)
        out_dbg["dp_max_len_ratio"] = round(sum_dp.max_len_ratio, 6)

        return selected_rows, out_dbg

    return selected_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Align train.csv to sentence pairs.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    parser.add_argument("--out", default=None, help="Output file path.")
    parser.add_argument("--format", default=None, choices=["parquet", "csv"], help="Output format.")
    parser.add_argument("--variant", default=None, help="Normalization variants to export (e.g., A,B,C).")
    parser.add_argument("--drop-flagged", action="store_true", help="Drop rows with any flags.")
    parser.add_argument("--sample", type=int, default=None, help="Use first N rows for quick runs.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-oare-sentences",
        action="store_true",
        help="Use Sentences_Oare_FirstWord_LinNum.csv (sentence-level hints) when available.",
    )
    group.add_argument(
        "--no-oare-sentences",
        action="store_true",
        help="Disable OARE sentence hints even if the file exists.",
    )
    parser.add_argument(
        "--oare-sentences-path",
        default=None,
        help="Path to Sentences_Oare_FirstWord_LinNum.csv (default: data/Sentences_Oare_FirstWord_LinNum.csv).",
    )
    parser.add_argument(
        "--oare-debug",
        action="store_true",
        help="Log per-document OARE hint diagnostics to a CSV for debugging.",
    )
    parser.add_argument(
        "--oare-debug-top-n",
        type=int,
        default=None,
        help="Number of outlier documents to show in stdout (default: 3).",
    )
    parser.add_argument(
        "--oare-debug-out",
        default=None,
        help="Output path for OARE debug CSV (default: <artifacts>/aligned/oare_debug.csv).",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, args.data_dir)
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found at: {train_path}")

    train_df = pd.read_csv(train_path)
    if args.sample:
        train_df = train_df.head(args.sample)


    # OARE sentence-level hints (optional)
    oare_hint_map: dict[str, List[OareSentenceHint]] = {}
    oare_path_cfg = cfg.get("oare_sentences_path")
    if args.oare_sentences_path:
        oare_path = Path(args.oare_sentences_path)
    elif oare_path_cfg:
        oare_path = Path(str(oare_path_cfg))
        if not oare_path.is_absolute():
            oare_path = data_dir / str(oare_path_cfg)
    else:
        oare_path = data_dir / "Sentences_Oare_FirstWord_LinNum.csv"

    cfg_use_oare = cfg.get("use_oare_sentences", None)
    if args.no_oare_sentences:
        use_oare = False
    elif args.use_oare_sentences:
        use_oare = True
    elif cfg_use_oare is None:
        # config が無い場合は「ファイルがあれば自動で使う」
        use_oare = oare_path.exists()
    else:
        use_oare = bool(cfg_use_oare)

    if use_oare:
        if oare_path.exists():
            oare_hint_map = load_oare_sentence_hints(oare_path)
            print(f"Loaded OARE sentence hints: {len(oare_hint_map)} texts from {oare_path}")
        else:
            print(f"[WARN] use_oare_sentences enabled but file not found: {oare_path}")

    oare_debug = bool(args.oare_debug or cfg.get("oare_debug", False))
    oare_debug_top_n = (
        args.oare_debug_top_n
        if args.oare_debug_top_n is not None
        else _parse_optional_int(cfg.get("oare_debug_top_n")) or 3
    )
    oare_debug_out = args.oare_debug_out or cfg.get("oare_debug_path")

    variant_spec = args.variant or cfg.get("variants") or "A,B,C"
    variants = [v.strip().upper() for v in str(variant_spec).split(",") if v.strip()]
    if not variants:
        variants = ["A"]

    gate_cfg = QualityGateConfig(
        min_len_ratio=float(cfg.get("min_len_ratio", 0.25)),
        max_len_ratio=float(cfg.get("max_len_ratio", 4.0)),
        min_src_tokens=int(cfg.get("min_src_tokens", 3)),
        min_tgt_chars=int(cfg.get("min_tgt_chars", 20)),
        min_alpha_ratio=float(cfg.get("min_alpha_ratio", 0.3)),
        max_digit_ratio=float(cfg.get("max_digit_ratio", 0.5)),
        min_align_score=_parse_optional_float(cfg.get("min_align_score")),
        min_token_align_score=_parse_optional_float(cfg.get("min_token_align_score")),
        min_src_alpha_ratio=_parse_optional_float(cfg.get("min_src_alpha_ratio")),
        max_src_digit_ratio=_parse_optional_float(cfg.get("max_src_digit_ratio")),
        max_src_symbol_ratio=_parse_optional_float(cfg.get("max_src_symbol_ratio")),
        max_tgt_symbol_ratio=_parse_optional_float(cfg.get("max_tgt_symbol_ratio")),
        max_tgt_sentence_endings=_parse_optional_int(cfg.get("max_tgt_sentence_endings")),
        min_tgt_tokens=_parse_optional_int(cfg.get("min_tgt_tokens")),
        max_tgt_tokens=_parse_optional_int(cfg.get("max_tgt_tokens")),
        max_src_tokens=_parse_optional_int(cfg.get("max_src_tokens")),
        max_src_chars=_parse_optional_int(cfg.get("max_src_chars")),
        max_tgt_chars=_parse_optional_int(cfg.get("max_tgt_chars")),
        drop_tgt_trailing_quote=bool(cfg.get("drop_tgt_trailing_quote", False)),
        drop_tgt_odd_quote=bool(cfg.get("drop_tgt_odd_quote", False)),
    )
    max_tgt_per_src = _parse_optional_int(cfg.get("max_tgt_per_src"))
    max_src_per_tgt = _parse_optional_int(cfg.get("max_src_per_tgt"))


    # OARE sentence hints: doc-level selection (A/B safety)
    # - enabled by default when use_oare_sentences is on
    # - set oare_doc_select=false to revert to the legacy behavior
    oare_doc_select = bool(cfg.get("oare_doc_select", True))
    oare_min_anchors = int(cfg.get("oare_min_anchors", 1))
    oare_tie_break = str(cfg.get("oare_tie_break", "oare") or "oare")

    all_rows = []
    oare_debug_rows: List[dict] = []
    used_oare_docs = 0
    oare_selected_by_variant: Counter = Counter()
    oare_fallback_reasons: Counter = Counter()
    for _, row in train_df.iterrows():
        oare_id = str(row.get("oare_id", ""))
        src = str(row.get("transliteration", ""))
        tgt = str(row.get("translation", ""))
        oare_hints = oare_hint_map.get(oare_id) if use_oare else None
        if use_oare and oare_hints:
            used_oare_docs += 1
        for variant in variants:
            if oare_debug:
                rows, debug_info = align_document(
                    oare_id,
                    src,
                    tgt,
                    variant,
                    gate_cfg,
                    oare_hints=oare_hints,
                    debug=True,
                    oare_doc_select=oare_doc_select,
                    oare_min_anchors=oare_min_anchors,
                    oare_tie_break=oare_tie_break,
                )
                all_rows.extend(rows)
                oare_debug_rows.append(debug_info)

                if use_oare and oare_hints and oare_doc_select:
                    sel = str(debug_info.get("selected_method") or "")
                    if sel == "oare":
                        oare_selected_by_variant[variant] += 1
                    elif sel == "dp":
                        reason = str(debug_info.get("selected_reason") or "")
                        if reason:
                            oare_fallback_reasons[reason] += 1
            else:
                rows = align_document(
                    oare_id,
                    src,
                    tgt,
                    variant,
                    gate_cfg,
                    oare_hints=oare_hints,
                    oare_doc_select=oare_doc_select,
                    oare_min_anchors=oare_min_anchors,
                    oare_tie_break=oare_tie_break,
                )
                all_rows.extend(rows)

                if use_oare and oare_hints and oare_doc_select and rows:
                    if str(rows[0].get("align_method") or "") == "oare":
                        oare_selected_by_variant[variant] += 1
                    else:
                        reason = _extract_oare_fallback_reason(str(rows[0].get("align_meta") or ""))
                        if reason:
                            oare_fallback_reasons[reason] += 1

    if use_oare:
        print(f"OARE sentence hints available for {used_oare_docs}/{len(train_df)} documents (any variant).")
        if oare_doc_select and used_oare_docs > 0:
            total_doc_variants = used_oare_docs * len(variants)
            selected_total = sum(oare_selected_by_variant.values())
            print(
                "OARE doc-select enabled: "
                f"selected {selected_total}/{total_doc_variants} doc-variants "
                f"(min_anchors={oare_min_anchors}, tie_break={oare_tie_break})."
            )
            if oare_selected_by_variant:
                parts = ", ".join([f"{v}={int(oare_selected_by_variant.get(v, 0))}" for v in variants])
                print(f"OARE selected by variant: {parts}")
            if oare_fallback_reasons:
                print("OARE fallback reasons (top):")
                for name, count in oare_fallback_reasons.most_common(8):
                    print(f"- {name}: {count}")
        elif not oare_doc_select:
            print("OARE doc-select disabled (legacy behavior).")
    aligned_df = pd.DataFrame(all_rows)
    aligned_df = apply_duplicate_flags(aligned_df, max_tgt_per_src, max_src_per_tgt)
    log_alignment_stats(aligned_df, gate_cfg, "before_drop", max_tgt_per_src, max_src_per_tgt)
    if args.drop_flagged:
        aligned_df = aligned_df[aligned_df["flags"] == ""].reset_index(drop=True)
        log_alignment_stats(aligned_df, gate_cfg, "after_drop", max_tgt_per_src, max_src_per_tgt)
    else:
        log_alignment_stats(aligned_df, gate_cfg, "output", max_tgt_per_src, max_src_per_tgt)

    if oare_debug and oare_debug_rows:
        debug_dir = get_artifacts_dir(cfg) / "aligned"
        debug_dir.mkdir(parents=True, exist_ok=True)
        if oare_debug_out:
            debug_path = Path(str(oare_debug_out))
        else:
            debug_path = debug_dir / "oare_debug.csv"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_df = pd.DataFrame(oare_debug_rows)
        debug_df.to_csv(debug_path, index=False)
        print(f"[oare_debug] saved: {debug_path} rows={len(debug_df)}")

        outliers = []
        for row in oare_debug_rows:
            if not row.get("hint_mode"):
                continue
            min_lr = float(row.get("min_len_ratio", 0.0) or 0.0)
            max_lr = float(row.get("max_len_ratio", 0.0) or 0.0)
            score = _len_ratio_outlier_score(min_lr, max_lr, gate_cfg)
            if score > 0:
                row_copy = dict(row)
                row_copy["_outlier_score"] = score
                outliers.append(row_copy)

        if outliers and (oare_debug_top_n or 0) > 0:
            outliers.sort(key=lambda r: r["_outlier_score"], reverse=True)
            top_n = min(len(outliers), max(1, int(oare_debug_top_n)))
            print("[oare_debug] len_ratio outliers (top):")
            for row in outliers[:top_n]:
                print(
                    "- "
                    f"oare_id={row.get('oare_id')} "
                    f"variant={row.get('variant')} "
                    f"min_lr={row.get('min_len_ratio')} max_lr={row.get('max_len_ratio')} "
                    f"empty_src={row.get('empty_src_segments')} empty_tgt={row.get('empty_tgt_segments')} "
                    f"anchors_found={row.get('anchors_found')} dup_anchors={row.get('duplicate_anchor_positions')}"
                )

    out_format = args.format or str(cfg.get("aligned_format", "parquet")).lower()
    if out_format not in {"parquet", "csv"}:
        out_format = "parquet"

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = get_artifacts_dir(cfg) / "aligned"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"aligned_train.{out_format}"
        out_path = out_dir / out_name

    if out_format == "parquet":
        try:
            aligned_df.to_parquet(out_path, index=False)
        except Exception as exc:
            raise RuntimeError("Failed to write parquet. Install pyarrow.") from exc
    else:
        aligned_df.to_csv(out_path, index=False)

    print(f"Saved aligned data: {out_path}")


if __name__ == "__main__":
    main()
