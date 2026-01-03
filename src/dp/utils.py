"""最小パイプライン向けの小さなユーティリティ群。"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple


def find_repo_root() -> Path:
    """現在の作業ディレクトリからリポジトリのルートを探す。"""
    cwd = Path.cwd().resolve()
    for path in [cwd] + list(cwd.parents):
        if (path / "data").exists() or (path / "docs").exists():
            return path
    return cwd


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    suffix = cfg_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(cfg_path.read_text())
            return data or {}
        except ImportError:
            return parse_simple_yaml(cfg_path.read_text())
    if suffix == ".json":
        return json.loads(cfg_path.read_text())
    if suffix == ".toml":
        try:
            import tomllib  # Python 3.11+のみ

            return tomllib.loads(cfg_path.read_text())
        except Exception as exc:
            raise RuntimeError("Failed to parse TOML config") from exc
    raise ValueError(f"Unsupported config format: {suffix}")


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """外部依存なしでフラットな YAML マッピングを解析する。"""
    cfg: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        cfg[key] = coerce_value(value)
    return cfg


def coerce_value(value: str) -> Any:
    if value == "" or value.lower() in {"null", "none", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def get_data_dir(cfg: Dict[str, Any], data_dir: str | None) -> Path:
    if data_dir:
        return Path(data_dir)
    if "data_dir" in cfg:
        return Path(cfg["data_dir"])
    return find_repo_root() / "data"


def get_artifacts_dir(cfg: Dict[str, Any], artifacts_dir: str | None = None) -> Path:
    if artifacts_dir:
        return Path(artifacts_dir)
    if "artifacts_dir" in cfg:
        return Path(cfg["artifacts_dir"])
    return find_repo_root() / "artifacts"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_id(prefix: str | None = None) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        return f"{stamp}_{prefix}"
    return stamp


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# 出力正規化用の簡易ルール
_DASH_MAP = str.maketrans({"–": "-", "—": "-", "−": "-"})
_APOSTROPHE_RE = re.compile(r"[’‘ʾʼ]")
_QUOTE_RE = re.compile(r"[“”„]")
_FRACTION_SLASH = "\u2044"

_VULGAR_FRACTIONS: dict[str, tuple[int, int]] = {
    "\u00bc": (1, 4),
    "\u00bd": (1, 2),
    "\u00be": (3, 4),
    "\u2150": (1, 7),
    "\u2151": (1, 9),
    "\u2152": (1, 10),
    "\u2153": (1, 3),
    "\u2154": (2, 3),
    "\u2155": (1, 5),
    "\u2156": (2, 5),
    "\u2157": (3, 5),
    "\u2158": (4, 5),
    "\u2159": (1, 6),
    "\u215a": (5, 6),
    "\u215b": (1, 8),
    "\u215c": (3, 8),
    "\u215d": (5, 8),
    "\u215e": (7, 8),
}
_VULGAR_FRAC_CHARS = "".join(_VULGAR_FRACTIONS.keys())
_VULGAR_MIXED_RE = re.compile(rf"(\d+)\s*([{re.escape(_VULGAR_FRAC_CHARS)}])")
_VULGAR_STANDALONE_RE = re.compile(rf"(?<!\d)([{re.escape(_VULGAR_FRAC_CHARS)}])")

_MIXED_FRAC_RE = re.compile(r"\b(\d+)\s+(\d+)\s*/\s*(\d+)\b")
_SIMPLE_FRAC_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")

_UNIT_TOKENS = {
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
}
_UNIT_JOIN_RE = re.compile(
    rf"(\d+(?:\.\d+)?)(?P<unit>{'|'.join(sorted(_UNIT_TOKENS, key=len, reverse=True))})\b",
    re.IGNORECASE,
)


def normalize_translation_output(
    text: str,
    *,
    normalize_fractions: bool = True,
    normalize_units: bool = True,
) -> str:
    """翻訳出力の表記ゆれを軽く正規化する。"""
    if not text:
        return ""
    out = str(text)
    if normalize_fractions:
        out = _normalize_vulgar_fractions(out)
    out = unicodedata.normalize("NFKC", out)
    out = out.translate(_DASH_MAP)
    out = _APOSTROPHE_RE.sub("'", out)
    out = _QUOTE_RE.sub('"', out)
    if normalize_fractions:
        if _FRACTION_SLASH in out:
            out = out.replace(_FRACTION_SLASH, "/")
        out = _normalize_ascii_fractions(out)
    if normalize_units:
        out = _normalize_units(out)
    out = clean_text(out)
    return out


def _fraction_to_decimal(whole: int, num: int, den: int) -> str | None:
    if den <= 0:
        return None
    total = whole * den + num
    if den == 2:
        integer = total // 2
        if total % 2 == 0:
            return str(integer)
        return f"{integer}.5"
    if den == 3:
        integer = total // 3
        rem = total % 3
        if rem == 0:
            return str(integer)
        frac = "3333" if rem == 1 else "6666"
        return f"{integer}.{frac}"
    if den == 6:
        integer = total // 6
        rem = total % 6
        if rem == 0:
            return str(integer)
        if rem == 5:
            return f"{integer}.83333"
    return None


def _format_fraction(whole: int, num: int, den: int) -> str:
    dec = _fraction_to_decimal(whole, num, den)
    if dec is not None:
        return dec
    if whole > 0:
        return f"{whole} {num} / {den}"
    return f"{num} / {den}"


def _normalize_vulgar_fractions(text: str) -> str:
    def _mixed_repl(match: re.Match[str]) -> str:
        whole = int(match.group(1))
        frac = match.group(2)
        num, den = _VULGAR_FRACTIONS[frac]
        return _format_fraction(whole, num, den)

    def _standalone_repl(match: re.Match[str]) -> str:
        frac = match.group(1)
        num, den = _VULGAR_FRACTIONS[frac]
        return _format_fraction(0, num, den)

    out = _VULGAR_MIXED_RE.sub(_mixed_repl, text)
    out = _VULGAR_STANDALONE_RE.sub(_standalone_repl, out)
    return out


def _normalize_ascii_fractions(text: str) -> str:
    def _mixed_repl(match: re.Match[str]) -> str:
        whole = int(match.group(1))
        num = int(match.group(2))
        den = int(match.group(3))
        return _format_fraction(whole, num, den)

    def _simple_repl(match: re.Match[str]) -> str:
        num = int(match.group(1))
        den = int(match.group(2))
        return _format_fraction(0, num, den)

    out = _MIXED_FRAC_RE.sub(_mixed_repl, text)
    out = _SIMPLE_FRAC_RE.sub(_simple_repl, out)
    return out


def _normalize_units(text: str) -> str:
    def _unit_repl(match: re.Match[str]) -> str:
        num = match.group(1)
        unit = match.group("unit")
        return f"{num} {unit}"

    return _UNIT_JOIN_RE.sub(_unit_repl, text)


# Common English abbreviations that may contain '.' but are not sentence boundaries.
# This is intentionally small/conservative.
_SENTENCE_ABBREVIATIONS: set[str] = {
    "e.g",
    "i.e",
    "etc",
    "cf",
    "vs",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "no",
    "fig",
    "eq",
    "vol",
    "pp",
    "p",
}


def _iter_sentence_end_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return spans (start_idx, end_idx, punct_char) that likely end sentences.

    Heuristics:
    - Treat '.', '!', '?' as candidates.
    - Ignore decimal points like '3.5'.
    - Collapse runs like '...' into a single span.
    - Ignore common abbreviations like 'e.g.' / 'Dr.'

    The returned indices are inclusive (end_idx points to the last punctuation char).
    """

    s = clean_text(text)
    if not s:
        return []

    spans: List[Tuple[int, int, str]] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch not in ".!?":
            i += 1
            continue

        # Decimal point: digit '.' digit
        if ch == "." and 0 < i < n - 1 and s[i - 1].isdigit() and s[i + 1].isdigit():
            i += 1
            continue

        # Collapse repeated punctuation: '...' / '!!' / '??'
        j = i
        while j + 1 < n and s[j + 1] == ch:
            j += 1

        # Abbreviation guard for '.' (check token immediately before punctuation)
        if ch == ".":
            # token = last whitespace-separated chunk before i
            prev = s[:i].rstrip()
            token = prev.split()[-1] if prev else ""
            token = token.rstrip("\"'”’)]}>,")
            if token.lower() in _SENTENCE_ABBREVIATIONS:
                i = j + 1
                continue

        spans.append((i, j, ch))
        i = j + 1

    return spans



def count_sentence_endings(text: str) -> int:
    """Count likely sentence-ending punctuation spans in text.

    This is used for logging / guardrails.
    - Ignores decimal points like '3.5'
    - Ignores a small set of common abbreviations (e.g., 'e.g.')
    """

    return len(_iter_sentence_end_spans(text))


def enforce_single_sentence(text: str, *, mode: str = "merge") -> str:
    """Force a prediction into a single sentence.

    This is a *postprocess* step for Kaggle submission safety.

    Parameters
    ----------
    mode:
      - 'merge': replace internal sentence boundaries with ';' and keep only the last final punctuation.
      - 'truncate': keep only the first sentence.

    Notes
    -----
    - We try to avoid truncating at decimal points (e.g., '3.5').
    - We intentionally do NOT attempt full sentence segmentation.
    """

    s = clean_text(text)
    if not s:
        return ""

    spans = _iter_sentence_end_spans(s)
    if len(spans) <= 1:
        return s

    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"merge", "truncate"}:
        mode_norm = "merge"

    if mode_norm == "truncate":
        start, end, _ch = spans[0]
        out = s[: end + 1].strip()
        return out

    # merge
    chars = list(s)
    for start, end, _ch in spans[:-1]:
        # Replace the whole punctuation span with a single ';'
        chars[start : end + 1] = [";"]
    out = "".join(chars)
    # normalize spaces around ';'
    out = re.sub(r"\s*;\s*", "; ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def first_sentence(text: str) -> str:
    # Backward-compatible helper used by the dummy baseline.
    return enforce_single_sentence(text, mode="truncate")
