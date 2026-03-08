"""推論側で辞書グロス（<LEX> ... </LEX>）を付与するためのヘルパー。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .gloss import DEFAULT_STOP_LEMMAS, GlossAugmentConfig, build_gloss_augmenter, build_lemma_frequency
from .utils import find_repo_root


def add_gloss_args(parser: argparse.ArgumentParser) -> None:
    """推論系スクリプト向けの gloss オプションを追加する。"""

    # 任意: 辞書グロスのヒントをソース末尾に付与
    parser.add_argument(
        "--use-gloss",
        action="store_true",
        help="Append compact English gloss hints using OA_Lexicon + eBL_Dictionary.",
    )
    parser.add_argument(
        "--no-gloss",
        action="store_true",
        help="Disable gloss augmentation even if enabled in config/checkpoint meta.",
    )
    parser.add_argument(
        "--oa-lexicon",
        default=None,
        help="Path to OA_Lexicon_eBL.csv (default: <data-dir>/OA_Lexicon_eBL.csv).",
    )
    parser.add_argument(
        "--ebl-dictionary",
        default=None,
        help="Path to eBL_Dictionary.csv (default: <data-dir>/eBL_Dictionary.csv).",
    )
    parser.add_argument(
        "--gloss-max-hints",
        type=int,
        default=None,
        help="Max number of gloss hints appended per sentence.",
    )
    parser.add_argument(
        "--gloss-max-total-chars",
        type=int,
        default=None,
        help="Max total chars for appended gloss payload.",
    )
    parser.add_argument(
        "--gloss-match-types",
        default=None,
        help=(
            "Comma/space-separated OA_Lexicon types to match (e.g., 'word,PN'). "
            "Default: word"
        ),
    )
    parser.add_argument(
        "--gloss-max-match-len",
        type=int,
        default=None,
        help="Max token span to match for gloss lookup (default: 4).",
    )
    parser.add_argument(
        "--gloss-stop-lemmas",
        default=None,
        help="Comma/space-separated lemma stopwords to exclude from gloss hints.",
    )
    parser.add_argument(
        "--gloss-stop-lemmas-file",
        default=None,
        help="Path to a newline-separated lemma stopword file (UTF-8).",
    )
    parser.add_argument(
        "--gloss-no-default-stop-lemmas",
        action="store_true",
        help="Disable built-in stop lemmas (function words) for gloss.",
    )
    parser.add_argument(
        "--gloss-max-lemma-freq",
        type=int,
        default=None,
        help=(
            "If set, only keep gloss hints for lemmas whose corpus frequency <= N "
            "(uses <ckpt>/gloss_lemma_freq.json when available)."
        ),
    )
    parser.add_argument(
        "--gloss-min-lemma-chars",
        type=int,
        default=None,
        help="Minimum lemma length (chars) to allow in gloss hints (default: 2).",
    )


def maybe_apply_gloss(
    src_texts: List[str],
    *,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    data_dir: Path,
    ckpt_dir: Path,
) -> List[str]:
    """必要なら src_texts に gloss を付与して返す（安全に失敗する）。"""

    if not src_texts:
        return src_texts

    if bool(getattr(args, "no_gloss", False)):
        return src_texts

    meta = _load_train_meta(ckpt_dir / "train_meta.json")
    meta_gloss = meta.get("gloss") if isinstance(meta, dict) else None

    # 明示指定 → config → ckpt の順に有効化を判定
    want_gloss = False
    if bool(getattr(args, "use_gloss", False)):
        want_gloss = True
    elif _as_bool(cfg.get("use_gloss", False)) or _as_bool(cfg.get("gloss_enabled", False)) or _as_bool(
        cfg.get("infer_use_gloss", False)
    ):
        want_gloss = True
    elif _as_bool(meta_gloss.get("enabled", False) if isinstance(meta_gloss, dict) else False):
        want_gloss = True
    elif (ckpt_dir / "gloss_lemma_freq.json").exists():
        # train_meta.json が無くても artifacts があれば gloss 学習の可能性が高い
        want_gloss = True

    if not want_gloss:
        return src_texts

    # config や CLI で明示されていない場合は train_meta の値をデフォルトとして使う
    meta_defaults = meta_gloss if isinstance(meta_gloss, dict) else {}
    auto_enabled = (
        not bool(getattr(args, "use_gloss", False))
        and not _as_bool(cfg.get("use_gloss", False))
        and _as_bool(meta_defaults.get("enabled", False))
    )
    if auto_enabled:
        print("[gloss] checkpoint の train_meta.json により gloss を自動有効化しました（無効化: --no-gloss）")

    oa_lexicon_path = _resolve_optional_path(
        data_dir,
        _first_not_none(
            getattr(args, "oa_lexicon", None),
            cfg.get("oa_lexicon_path", cfg.get("lexicon_path")),
            meta_defaults.get("oa_lexicon"),
        ),
        "OA_Lexicon_eBL.csv",
    )
    ebl_dictionary_path = _resolve_optional_path(
        data_dir,
        _first_not_none(
            getattr(args, "ebl_dictionary", None),
            cfg.get("ebl_dictionary_path", cfg.get("ebl_dict_path")),
            meta_defaults.get("ebl_dictionary"),
        ),
        "eBL_Dictionary.csv",
    )

    # OA_Lexicon の type フィルタ
    match_types_value = _first_not_none(
        getattr(args, "gloss_match_types", None),
        cfg.get("gloss_match_types"),
        meta_defaults.get("match_types"),
    )
    match_types = _parse_list(match_types_value) or ["word"]

    # stop lemmas: 明示指定がない限り、train_meta の stop_lemmas を優先して一致させる
    stop_explicit = bool(getattr(args, "gloss_stop_lemmas", None)) or bool(
        getattr(args, "gloss_stop_lemmas_file", None)
    ) or bool(getattr(args, "gloss_no_default_stop_lemmas", False))
    stop_explicit = stop_explicit or bool(cfg.get("gloss_stop_lemmas")) or bool(cfg.get("gloss_stop_lemmas_file")) or _as_bool(
        cfg.get("gloss_no_default_stop_lemmas", False)
    )

    stop_lemmas: List[str] = []
    if not stop_explicit and meta_defaults.get("stop_lemmas"):
        stop_lemmas.extend([str(x) for x in meta_defaults.get("stop_lemmas") or []])
    else:
        use_default_stop = not (
            bool(getattr(args, "gloss_no_default_stop_lemmas", False))
            or _as_bool(cfg.get("gloss_no_default_stop_lemmas", False))
        )
        if use_default_stop:
            stop_lemmas.extend(list(DEFAULT_STOP_LEMMAS))
        stop_lemmas.extend(_parse_list(cfg.get("gloss_stop_lemmas")))
        stop_lemmas.extend(_parse_list(getattr(args, "gloss_stop_lemmas", None)))

        stop_file_value = _first_not_none(getattr(args, "gloss_stop_lemmas_file", None), cfg.get("gloss_stop_lemmas_file"))
        if stop_file_value:
            stop_path = Path(str(stop_file_value))
            if not stop_path.is_absolute():
                cand = data_dir / stop_path
                if cand.exists():
                    stop_path = cand
            if stop_path.exists():
                for line in stop_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    stop_lemmas.append(line)
            else:
                print(f"[WARN] gloss_stop_lemmas_file not found: {stop_path}")

    stop_lemmas = _dedupe_keep_order([s for s in stop_lemmas if str(s).strip()])

    # 主要パラメータ
    gloss_max_hints = _parse_int_with_fallback(
        getattr(args, "gloss_max_hints", None),
        cfg.get("gloss_max_hints"),
        meta_defaults.get("max_hints"),
        6,
    )
    gloss_max_total_chars = _parse_int_with_fallback(
        getattr(args, "gloss_max_total_chars", None),
        cfg.get("gloss_max_total_chars"),
        meta_defaults.get("max_total_chars"),
        220,
    )
    gloss_max_match_len = _parse_int_with_fallback(
        getattr(args, "gloss_max_match_len", None),
        cfg.get("gloss_max_match_len"),
        meta_defaults.get("max_match_len"),
        4,
    )
    gloss_min_lemma_chars = _parse_int_with_fallback(
        getattr(args, "gloss_min_lemma_chars", None),
        cfg.get("gloss_min_lemma_chars"),
        meta_defaults.get("min_lemma_chars"),
        2,
    )

    gloss_max_lemma_freq = _parse_optional_int_with_fallback(
        getattr(args, "gloss_max_lemma_freq", None),
        cfg.get("gloss_max_lemma_freq"),
        meta_defaults.get("max_lemma_freq"),
    )

    # bound morphemes の除外（config があれば優先、無ければ meta の値に合わせる）
    exclude_bound_morphemes = bool(meta_defaults.get("exclude_bound_morphemes", True))
    if "gloss_include_bound_morphemes" in cfg:
        exclude_bound_morphemes = not _as_bool(cfg.get("gloss_include_bound_morphemes", False))

    gloss_cfg = GlossAugmentConfig(
        enabled=True,
        match_types=tuple(match_types),
        max_match_len=int(gloss_max_match_len),
        stop_lemmas=tuple(stop_lemmas),
        min_lemma_chars=int(gloss_min_lemma_chars),
        exclude_bound_morphemes=bool(exclude_bound_morphemes),
        max_lemma_freq=int(gloss_max_lemma_freq) if gloss_max_lemma_freq is not None else None,
        max_hints=int(gloss_max_hints),
        max_total_chars=int(gloss_max_total_chars),
    )

    lemma_freq: Optional[Dict[str, int]] = None
    lemma_freq_src = "none"
    freq_path = ckpt_dir / "gloss_lemma_freq.json"
    if freq_path.exists():
        loaded = _load_json_dict(freq_path)
        if loaded is not None:
            lemma_freq = loaded
            lemma_freq_src = "ckpt"
    elif gloss_cfg.max_lemma_freq is not None:
        # ckpt に頻度表が無い場合は、入力から簡易計算（完全一致にはならないが、無いよりはマシ）
        try:
            lemma_freq = build_lemma_frequency(
                src_texts,
                oa_lexicon_path=oa_lexicon_path,
                match_columns=gloss_cfg.match_columns,
                match_types=gloss_cfg.match_types,
                max_match_len=gloss_cfg.max_match_len,
            )
            lemma_freq_src = "input"
        except Exception as exc:
            print(f"[WARN] failed to compute gloss lemma frequency: {exc}")

    try:
        gloss_augment = build_gloss_augmenter(
            gloss_cfg,
            oa_lexicon_path=oa_lexicon_path,
            ebl_dictionary_path=ebl_dictionary_path,
            lemma_freq=lemma_freq,
        )
    except Exception as exc:
        print(f"[WARN] gloss enabled but failed to initialize resources: {exc}")
        return src_texts

    print(
        "[gloss] enabled:"
        f" oa={oa_lexicon_path.name} dict={ebl_dictionary_path.name}"
        f" types={list(gloss_cfg.match_types)} max_hints={gloss_cfg.max_hints}"
        f" max_total_chars={gloss_cfg.max_total_chars} max_match_len={gloss_cfg.max_match_len}"
        f" max_lemma_freq={gloss_cfg.max_lemma_freq} lemma_freq={lemma_freq_src}"
    )

    def _guarded(text: str) -> str:
        # すでに <LEX> が付いている入力（val_predictions など）を二重付与しない
        if "<LEX>" in text and "</LEX>" in text:
            return text
        return gloss_augment(text)

    return [_guarded(str(x)) for x in src_texts]


def _first_not_none(*values: object) -> Optional[object]:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def _parse_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        raw = [str(x) for x in v]
    else:
        s = str(v).strip()
        if not s:
            return []
        raw = re.split(r"[,\s]+", s)
    out: List[str] = []
    for x in raw:
        x2 = str(x).strip()
        if x2:
            out.append(x2)
    return out


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _resolve_optional_path(data_dir: Path, value: object, default_name: str) -> Path:
    repo_data_dir = find_repo_root() / "data"

    def _first_existing(candidates: List[Path]) -> Optional[Path]:
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    # 値未指定時は data_dir 優先、無ければ repo/data を使う。
    if value is None or str(value).strip() == "":
        candidates = [data_dir / default_name, repo_data_dir / default_name]
        found = _first_existing(candidates)
        if found is not None:
            return found
        return candidates[0]

    p = Path(str(value))
    if p.exists():
        return p

    # 相対パス指定なら data_dir / repo/data への相対解決を試す。
    if not p.is_absolute():
        rel_candidates = [data_dir / p, repo_data_dir / p]
        found = _first_existing(rel_candidates)
        if found is not None:
            return found

    # Colab 由来などの無効な絶対パスを想定して、同名ファイルへフォールバックする。
    if p.name:
        name_candidates = [data_dir / p.name, repo_data_dir / p.name]
        found = _first_existing(name_candidates)
        if found is not None:
            print(f"[gloss] path fallback: {p} -> {found}")
            return found

    return p


def _load_train_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] failed to read train_meta.json: {exc}")
        return {}


def _load_json_dict(path: Path) -> Optional[Dict[str, int]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] failed to read {path.name}: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    out: Dict[str, int] = {}
    for k, v in data.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def _parse_int_with_fallback(*values: object) -> int:
    # values のうち最初に int 変換できたものを返す（全てダメなら末尾の default を期待）
    default = int(values[-1]) if values else 0
    for v in values[:-1]:
        if v is None or str(v).strip() == "":
            continue
        try:
            return int(float(v))
        except Exception:
            continue
    return default


def _parse_optional_int_with_fallback(*values: object) -> Optional[int]:
    for v in values:
        if v is None or str(v).strip() == "":
            continue
        try:
            return int(float(v))
        except Exception:
            continue
    return None
