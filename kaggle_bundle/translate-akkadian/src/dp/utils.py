"""最小パイプライン向けの小さなユーティリティ群。"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import re
import sys
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
