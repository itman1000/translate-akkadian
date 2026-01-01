"""最小パイプライン向けの小さなユーティリティ群。"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict


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


def first_sentence(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    for match in re.finditer(r"[.!?]", cleaned):
        end = match.end()
        return cleaned[:end].strip()
    return cleaned
