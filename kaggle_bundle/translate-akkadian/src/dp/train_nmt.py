"""NMT 用の Seq2Seq 学習スクリプト。"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .align_train import normalize_transliteration
from .gloss import (
    DEFAULT_STOP_LEMMAS,
    GlossAugmentConfig,
    build_gloss_augmenter,
    build_lemma_frequency,
)
from .utils import (
    clean_text,
    count_sentence_endings,
    enforce_single_sentence,
    get_artifacts_dir,
    get_data_dir,
    load_config,
    now_id,
    set_seed,
)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def log_text_summary(df: pd.DataFrame, col: str, label: str) -> None:
    """テキスト列の簡易サマリを出力する。"""
    if col not in df.columns:
        print(f"[WARN] {label}: column not found: {col}")
        return
    if len(df) == 0:
        print(f"[WARN] {label}: 0 rows")
        return
    texts = df[col].fillna("").astype(str)
    empty_ratio = (texts.str.strip() == "").mean()
    lens = texts.map(len)
    token_counts = texts.str.split().map(len)
    q_len = lens.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    q_tok = token_counts.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    unique_ratio = texts.nunique() / max(len(texts), 1)
    print(
        f"[{label}] rows={len(texts)} empty={empty_ratio:.1%} unique={unique_ratio:.2f} "
        f"len_med={q_len[0.5]:.0f} p90={q_len[0.9]:.0f} p95={q_len[0.95]:.0f} p99={q_len[0.99]:.0f} "
        f"tok_med={q_tok[0.5]:.0f} p90={q_tok[0.9]:.0f}"
    )


_GAP_TAG_RE = re.compile(r"<\s*gap\s*>", re.IGNORECASE)
_BIG_GAP_TAG_RE = re.compile(r"<\s*big_gap\s*>", re.IGNORECASE)
_DET_RE = re.compile(r"\{[^}]+\}|\([^)]*\)")


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def summarize_lengths(
    values: List[int],
    label: str,
    max_len: Optional[int],
    hit_values: Optional[List[int]] = None,
) -> None:
    if not values:
        print(f"[{label}] no data")
        return
    series = pd.Series(values)
    q = series.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    base = (
        f"[{label}] n={len(series)} max={series.max():.0f} "
        f"p50={q[0.5]:.0f} p90={q[0.9]:.0f} p95={q[0.95]:.0f} p99={q[0.99]:.0f}"
    )
    if max_len is None:
        print(base)
    else:
        trunc_rate = (series > max_len).mean()
        hit_rate = None
        if hit_values is not None:
            hit_series = pd.Series(hit_values)
            hit_rate = (hit_series == max_len).mean()
        if hit_rate is None:
            hit_rate = (series == max_len).mean()
        print(f"{base} hit_max={hit_rate:.1%} trunc={trunc_rate:.1%}")


def collect_token_lengths(
    texts: List[str],
    tokenizer: Any,
    batch_size: int,
) -> List[int]:
    lengths: List[int] = []
    for batch in batch_iter(texts, batch_size):
        enc = tokenizer(batch, add_special_tokens=True, padding=False, truncation=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


def log_token_length_stats(
    texts: List[str],
    tokenizer: Any,
    max_len: Optional[int],
    label: str,
    batch_size: int,
) -> None:
    raw_lengths = collect_token_lengths(texts, tokenizer, batch_size)
    if max_len is None:
        summarize_lengths(raw_lengths, label, max_len)
        return
    truncated_lengths: List[int] = []
    for batch in batch_iter(texts, batch_size):
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=max_len,
        )
        truncated_lengths.extend(len(ids) for ids in enc["input_ids"])
    summarize_lengths(raw_lengths, label, max_len, hit_values=truncated_lengths)


def decode_predictions(pred_output: Any, tokenizer: Any) -> Optional[List[str]]:
    preds = pred_output.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    try:
        import numpy as np  # type: ignore

        def _sanitize_token_ids(token_ids: "np.ndarray") -> "np.ndarray":
            """デコード前に tokenizer が扱えない token id を安全に置換する。

            ByT5Tokenizer は id を ``chr(id - offset)`` の形で復元するため、
            `-100` のような負の値が混ざると `chr()` が例外を投げる。
            Seq2SeqTrainer / gather 処理の都合で、生成結果が `-100` で
            pad されるケースがあるので、ここで pad_token_id に寄せる。
            """

            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                # ほぼ起きないが、念のため
                pad_id = 0

            # まず負の値（典型: -100）を除去
            if (token_ids < 0).any():
                token_ids = np.where(token_ids < 0, pad_id, token_ids)

            # 念のため、vocab サイズ外も置換（通常は発生しない）
            vocab_size = getattr(tokenizer, "vocab_size", None)
            if vocab_size is not None:
                unk_id = tokenizer.unk_token_id
                if unk_id is None:
                    unk_id = pad_id
                if (token_ids >= vocab_size).any():
                    token_ids = np.where(token_ids >= vocab_size, unk_id, token_ids)

            return token_ids

        preds_arr = np.asarray(preds)
        if preds_arr.ndim == 3:
            preds_arr = preds_arr.argmax(axis=-1)
        if np.issubdtype(preds_arr.dtype, np.floating):
            preds_arr = np.rint(preds_arr).astype("int64")
        else:
            preds_arr = preds_arr.astype("int64", copy=False)

        preds_arr = _sanitize_token_ids(preds_arr)
        decoded = tokenizer.batch_decode(preds_arr, skip_special_tokens=True)
    except Exception as exc:
        print(f"[WARN] decode_predictions failed: {exc}")
        return None
    return [clean_text(text) for text in decoded]


def compute_bleu_chrf(preds: List[str], refs: List[str]) -> Optional[Tuple[float, float, float]]:
    try:
        import sacrebleu  # type: ignore
    except ImportError:
        return None

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    gm = math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))
    return float(bleu), float(chrf), float(gm)


def log_val_samples(
    src_texts: List[str],
    ref_texts: List[str],
    pred_texts: List[str],
    sample_size: int,
) -> None:
    size = min(sample_size, len(pred_texts))
    if size <= 0:
        return
    print("=== val サンプル ===")
    for idx in range(size):
        print(f"[{idx}]")
        print(f"src: {src_texts[idx]}")
        print(f"ref: {ref_texts[idx]}")
        print(f"pred: {pred_texts[idx]}")


def has_gap(text: str) -> bool:
    return bool(_GAP_TAG_RE.search(text) or _BIG_GAP_TAG_RE.search(text))


def has_determinative(text: str) -> bool:
    return bool(_DET_RE.search(text))


def count_all_caps_tokens(texts: List[str]) -> Tuple[int, int]:
    total = 0
    caps = 0
    for text in texts:
        for token in text.split():
            total += 1
            if len(token) >= 2 and token.isupper():
                caps += 1
    return caps, total


def log_invariant_rates(
    src_texts: List[str],
    ref_texts: List[str],
    pred_texts: List[str],
) -> None:
    src_gap = [has_gap(text) for text in src_texts]
    ref_gap = [has_gap(text) for text in ref_texts]
    pred_gap = [has_gap(text) for text in pred_texts]
    gap_mismatch = [(r != p) for r, p in zip(ref_gap, pred_gap)]
    pred_det = [has_determinative(text) for text in pred_texts]
    pred_punct = [count_sentence_endings(text) for text in pred_texts]

    print("=== 不変条件チェック ===")
    print(
        f"src_has_gap_rate={sum(src_gap) / len(src_gap):.1%} "
        f"ref_has_gap_rate={sum(ref_gap) / len(ref_gap):.1%} "
        f"pred_has_gap_rate={sum(pred_gap) / len(pred_gap):.1%} "
        f"gap_mismatch_rate={sum(gap_mismatch) / len(gap_mismatch):.1%}"
    )
    print(f"pred_determinative_rate={sum(pred_det) / len(pred_det):.1%}")
    caps_count, token_total = count_all_caps_tokens(pred_texts)
    if token_total > 0:
        print(f"pred_allcaps_token_rate={caps_count / token_total:.1%}")
    else:
        print("pred_allcaps_token_rate=0.0%")
    punct_series = pd.Series(pred_punct)
    p_punct = punct_series.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    multi_sentence_rate = (punct_series >= 2).mean()
    print(
        f"pred_punct_p50={p_punct[0.5]:.0f} p90={p_punct[0.9]:.0f} "
        f"p95={p_punct[0.95]:.0f} p99={p_punct[0.99]:.0f} "
        f"pred_multi_sentence_rate={multi_sentence_rate:.1%}"
    )


def log_collapse_stats(pred_texts: List[str], ref_texts: List[str]) -> None:
    pred_series = pd.Series(pred_texts)
    ref_series = pd.Series(ref_texts)
    distinct_rate = pred_series.nunique() / max(len(pred_series), 1)
    avg_pred_len = pred_series.str.len().mean()
    avg_ref_len = ref_series.str.len().mean()
    len_ratio = avg_pred_len / avg_ref_len if avg_ref_len else 0.0
    print("=== 予測の崩壊チェック ===")
    print(
        f"distinct_pred_rate={distinct_rate:.1%} "
        f"avg_pred_len={avg_pred_len:.1f} avg_ref_len={avg_ref_len:.1f} "
        f"len_ratio_pred_ref={len_ratio:.2f}"
    )
    top_templates = Counter(pred_texts).most_common(10)
    if top_templates:
        print("top10_pred_templates:")
        for text, count in top_templates:
            ratio = count / len(pred_texts)
            print(f"- {count} ({ratio:.1%}): {text}")


def prepare_df(
    df: pd.DataFrame,
    src_col: str,
    tgt_col: str,
    variant: Optional[str],
    drop_flagged: bool,
    norm_variant: Optional[str],
    max_src_chars: Optional[int],
    max_tgt_chars: Optional[int],
) -> pd.DataFrame:
    work = df.copy()
    if variant and "src_norm_variant" in work.columns:
        work = work[work["src_norm_variant"].astype(str) == variant]
    if drop_flagged and "flags" in work.columns:
        work = work[work["flags"].fillna("").astype(str).str.strip() == ""]

    if src_col not in work.columns or tgt_col not in work.columns:
        raise ValueError(f"data must include {src_col} and {tgt_col}")

    if norm_variant:
        work[src_col] = work[src_col].fillna("").astype(str).map(
            lambda x: normalize_transliteration(x, norm_variant)
        )
    else:
        work[src_col] = work[src_col].fillna("").astype(str).map(clean_text)

    work[tgt_col] = work[tgt_col].fillna("").astype(str).map(clean_text)

    work = work[(work[src_col] != "") & (work[tgt_col] != "")].reset_index(drop=True)

    if max_src_chars is not None:
        work = work[work[src_col].str.len() <= max_src_chars]
    if max_tgt_chars is not None:
        work = work[work[tgt_col].str.len() <= max_tgt_chars]

    return work


def update_model_config(model_cfg: Any, cfg: Dict[str, Any]) -> None:
    # モデルが持つ設定だけを上書きする
    candidates: Dict[str, Optional[float]] = {
        "dropout_rate": parse_optional_float(cfg.get("dropout_rate")),
        "attention_dropout_rate": parse_optional_float(cfg.get("attention_dropout_rate")),
        "activation_dropout": parse_optional_float(cfg.get("activation_dropout")),
        "classifier_dropout": parse_optional_float(cfg.get("classifier_dropout")),
        "encoder_layerdrop": parse_optional_float(cfg.get("encoder_layerdrop")),
        "decoder_layerdrop": parse_optional_float(cfg.get("decoder_layerdrop")),
        "layerdrop": parse_optional_float(cfg.get("layerdrop")),
    }
    for key, value in candidates.items():
        if value is None:
            continue
        if hasattr(model_cfg, key):
            setattr(model_cfg, key, value)


def estimate_total_steps(
    train_len: int, batch_size: int, grad_accum: int, epochs: float
) -> int:
    denom = max(1, batch_size * max(1, grad_accum))
    steps_per_epoch = max(1, math.ceil(train_len / denom))
    return max(1, int(steps_per_epoch * max(1.0, epochs)))


def filter_training_args(
    kwargs: Dict[str, Any],
    supported: set[str],
    has_val: bool,
    total_steps: int,
) -> Dict[str, Any]:
    filtered = {k: v for k, v in kwargs.items() if k in supported}

    if "evaluation_strategy" not in supported:
        if "evaluate_during_training" in supported:
            filtered["evaluate_during_training"] = bool(has_val)
        if "do_eval" in supported and has_val:
            filtered["do_eval"] = True

    if "warmup_ratio" not in supported and "warmup_steps" in supported:
        warmup_ratio = kwargs.get("warmup_ratio", 0.0) or 0.0
        filtered["warmup_steps"] = int(float(warmup_ratio) * total_steps)

    return filtered


def split_train_val(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
    *,
    split_unit: str = "row",
    doc_id_col: str = "oare_id",
    source_col: str = "source",
    exclude_sources: Optional[Iterable[str]] = None,
    holdout_all_sources: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """df を train/val に分割する。

    - split_unit="row": 行単位で分割（従来動作）。
    - split_unit="doc": 文書ID（doc_id_col）単位で分割し、選ばれた文書の全行が val に入る。

    exclude_sources（例: ["ocr"]）は検証候補から除外し、train に残す。
    split_unit="doc" かつ holdout_all_sources=True の場合は、除外ソースであっても
    val 文書の行を train から外す（文書リーク防止）。
    """

    if val_ratio <= 0:
        return df.reset_index(drop=True), None

    split_unit_norm = str(split_unit or "row").strip().lower()

    exclude_set = {
        str(s).strip().lower()
        for s in (exclude_sources or [])
        if s is not None and str(s).strip()
    }

    if source_col in df.columns and exclude_set:
        source_series = df[source_col].astype(str).str.strip().str.lower()
        is_excluded = source_series.isin(exclude_set)
    else:
        is_excluded = pd.Series(False, index=df.index)

    candidate_df = df.loc[~is_excluded]
    if len(candidate_df) == 0:
        # 検証対象がないため学習のみ
        return df.reset_index(drop=True), None

    if split_unit_norm in {"doc", "document"}:
        if doc_id_col not in df.columns:
            print(f"[val_split][warn] doc_id_col='{doc_id_col}' not found; fallback to row split.")
        else:
            docs = (
                candidate_df[doc_id_col]
                .dropna()
                .astype(str)
                .map(str.strip)
                .loc[lambda s: s != ""]
                .unique()
                .tolist()
            )
            docs = sorted(docs)
            if docs:
                n_val_docs = max(1, int(len(docs) * float(val_ratio)))
                rng = random.Random(int(seed))
                rng.shuffle(docs)
                val_docs = set(docs[:n_val_docs])

                val_mask = candidate_df[doc_id_col].astype(str).isin(val_docs)
                val_df = candidate_df.loc[val_mask].copy()

                if holdout_all_sources:
                    train_mask = ~df[doc_id_col].astype(str).isin(val_docs)
                    train_df = df.loc[train_mask].copy()
                else:
                    train_df = df.drop(val_df.index).copy()

                train_df = train_df.reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)
                return train_df, val_df

            print(
                f"[val_split][warn] no valid doc ids found in col '{doc_id_col}'; fallback to row split."
            )

    # デフォルト: 行分割
    val_df = candidate_df.sample(frac=val_ratio, random_state=seed)
    train_df = df.drop(val_df.index).copy()
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Seq2Seq NMT model.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--train", required=True, help="Training CSV/Parquet path.")
    parser.add_argument("--val", default=None, help="Optional validation CSV/Parquet path.")
    parser.add_argument("--out", default=None, help="Output directory.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Data directory (for OA_Lexicon_eBL.csv / eBL_Dictionary.csv, etc.). "
            "If omitted, uses config.data_dir or <repo>/data."
        ),
    )
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--tgt-col", default=None, help="Target column name.")
    parser.add_argument("--variant", default=None, help="Normalization variant filter (A/B/C).")
    parser.add_argument("--drop-flagged", action="store_true", help="Drop flagged rows.")
    parser.add_argument("--norm-variant", default=None, help="Normalize source with A/B/C.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Use first N rows.")
    parser.add_argument("--max-val-rows", type=int, default=None, help="Use first N rows.")
    parser.add_argument("--model-name-or-path", default=None, help="Override model name/path.")
    parser.add_argument(
        "--post-eval-mode",
        default=None,
        choices=["none", "quick", "full"],
        help=(
            "Post-training evaluation mode. "
            "none: skip post eval; quick: metrics only; full: metrics + samples/diagnostics."
        ),
    )

    # 任意: 辞書グロスのヒントをソース末尾に付与
    parser.add_argument(
        "--use-gloss",
        action="store_true",
        help="Append compact English gloss hints using OA_Lexicon + eBL_Dictionary.",
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
            "(computed from train src before gloss)."
        ),
    )
    parser.add_argument(
        "--gloss-min-lemma-chars",
        type=int,
        default=None,
        help="Minimum lemma length (chars) to allow in gloss hints (default: 2).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, args.data_dir)

    # 事後評価の制御（後方互換のためデフォルトは full）
    post_eval_mode = (
        args.post_eval_mode
        or str(cfg.get("post_eval_mode", "full")).strip().lower()
    )
    if post_eval_mode not in {"none", "quick", "full"}:
        print(f"[WARN] unknown post_eval_mode={post_eval_mode!r}; fallback to 'full'")
        post_eval_mode = "full"
    post_eval_max_rows = cfg.get("post_eval_max_rows")
    if post_eval_max_rows is not None:
        post_eval_max_rows = parse_int(post_eval_max_rows, 0) or None

    src_col = args.src_col or cfg.get("src_col", "src_sent")
    tgt_col = args.tgt_col or cfg.get("tgt_col", "tgt_sent")
    variant = args.variant or cfg.get("variant")
    drop_flagged = bool(args.drop_flagged or cfg.get("drop_flagged", False))
    norm_variant = args.norm_variant or cfg.get("norm_variant")
    val_ratio = args.val_ratio if args.val_ratio is not None else parse_float(cfg.get("val_ratio"), 0.0)
    seed = args.seed if args.seed is not None else parse_int(cfg.get("seed"), 42)

    # 検証分割の設定
    val_split_unit = str(cfg.get("val_split_unit", "row")).strip().lower()
    val_doc_id_col = str(cfg.get("val_doc_id_col", "oare_id")).strip()
    val_source_col = str(cfg.get("val_source_col", "source")).strip()

    _ves = cfg.get("val_exclude_sources", ["ocr"])
    if _ves is None:
        val_exclude_sources = []
    elif isinstance(_ves, str):
        val_exclude_sources = [s.strip() for s in _ves.replace(';', ',').split(',') if s.strip()]
    else:
        try:
            val_exclude_sources = [str(s).strip() for s in _ves if str(s).strip()]
        except TypeError:
            val_exclude_sources = [str(_ves).strip()] if str(_ves).strip() else []

    val_holdout_all_sources = bool(cfg.get("val_holdout_all_sources", True))

    max_src_chars = cfg.get("max_src_chars")
    max_tgt_chars = cfg.get("max_tgt_chars")
    if max_src_chars is not None:
        max_src_chars = parse_int(max_src_chars, 0) or None
    if max_tgt_chars is not None:
        max_tgt_chars = parse_int(max_tgt_chars, 0) or None

    if variant:
        variant = str(variant).upper()
    if norm_variant:
        norm_variant = str(norm_variant).upper()

    set_seed(seed)

    train_df = prepare_df(
        read_table(Path(args.train)),
        src_col=src_col,
        tgt_col=tgt_col,
        variant=variant,
        drop_flagged=drop_flagged,
        norm_variant=norm_variant,
        max_src_chars=max_src_chars,
        max_tgt_chars=max_tgt_chars,
    )

    if args.max_train_rows:
        train_df = train_df.head(args.max_train_rows).reset_index(drop=True)

    if args.val:
        val_df = prepare_df(
            read_table(Path(args.val)),
            src_col=src_col,
            tgt_col=tgt_col,
            variant=variant,
            drop_flagged=drop_flagged,
            norm_variant=norm_variant,
            max_src_chars=max_src_chars,
            max_tgt_chars=max_tgt_chars,
        )
    else:
        train_df, val_df = split_train_val(
            train_df,
            val_ratio=val_ratio,
            seed=seed,
            split_unit=val_split_unit,
            doc_id_col=val_doc_id_col,
            source_col=val_source_col,
            exclude_sources=val_exclude_sources,
            holdout_all_sources=val_holdout_all_sources,
        )

        if val_df is not None:
            # 再現性のための簡易ログ
            msg = [
                f"[val_split] unit={val_split_unit} ratio={val_ratio} seed={seed}",
                f"doc_id_col={val_doc_id_col!r}",
            ]
            if val_source_col in train_df.columns:
                msg.append(f"source_col={val_source_col!r} exclude_sources={val_exclude_sources}")
            msg.append(f"train_rows={len(train_df)} val_rows={len(val_df)}")
            if val_doc_id_col in val_df.columns:
                msg.append(f"val_docs={val_df[val_doc_id_col].nunique()}")
            print(' '.join(msg))

    if val_df is not None and args.max_val_rows:
        val_df = val_df.head(args.max_val_rows).reset_index(drop=True)

    # 再現性のために検証文書IDを保存（doc 分割時のみ意味がある）
    val_doc_ids: Optional[List[str]] = None
    if val_df is not None and val_doc_id_col in val_df.columns:
        try:
            val_doc_ids = sorted(val_df[val_doc_id_col].dropna().astype(str).unique().tolist())
        except Exception:
            val_doc_ids = None

    # 任意: ソースにグロスヒントを付与
    gloss_enabled = bool(
        args.use_gloss
        or bool(cfg.get("use_gloss", False))
        or bool(cfg.get("gloss_enabled", False))
    )

    gloss_cfg: Optional[GlossAugmentConfig] = None
    gloss_lemma_freq: Optional[Dict[str, int]] = None
    gloss_meta: Optional[Dict[str, Any]] = None

    if gloss_enabled:
        # リソースパスを解決
        def _resolve_optional(path_value: Any, default_name: str) -> Path:
            if path_value is None or str(path_value).strip() == "":
                return data_dir / default_name
            p = Path(str(path_value))
            if not p.is_absolute():
                cand = data_dir / p
                if cand.exists():
                    return cand
            return p

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

        oa_lexicon_path = _resolve_optional(
            args.oa_lexicon if args.oa_lexicon is not None else cfg.get("oa_lexicon_path", cfg.get("lexicon_path")),
            "OA_Lexicon_eBL.csv",
        )
        ebl_dictionary_path = _resolve_optional(
            args.ebl_dictionary if args.ebl_dictionary is not None else cfg.get("ebl_dictionary_path", cfg.get("ebl_dict_path")),
            "eBL_Dictionary.csv",
        )

        # マッチタイプ（デフォルトは "word" のみ。必要なら PN/GN などを追加）
        match_types_value = (
            args.gloss_match_types
            if args.gloss_match_types is not None
            else cfg.get("gloss_match_types")
        )
        match_types = _parse_list(match_types_value) or ["word"]

        # 除外するレマ
        use_default_stop = not (
            args.gloss_no_default_stop_lemmas
            or _as_bool(cfg.get("gloss_no_default_stop_lemmas", False))
        )
        stop_lemmas: List[str] = []
        if use_default_stop:
            stop_lemmas.extend(list(DEFAULT_STOP_LEMMAS))
        stop_lemmas.extend(_parse_list(cfg.get("gloss_stop_lemmas")))
        stop_lemmas.extend(_parse_list(args.gloss_stop_lemmas))

        stop_file_value = (
            args.gloss_stop_lemmas_file
            if args.gloss_stop_lemmas_file is not None
            else cfg.get("gloss_stop_lemmas_file")
        )
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

        stop_lemmas = _dedupe_keep_order([s for s in stop_lemmas if s.strip()])

        # 主要パラメータ
        gloss_max_hints = parse_int(
            args.gloss_max_hints if args.gloss_max_hints is not None else cfg.get("gloss_max_hints", 6),
            6,
        )
        gloss_max_total_chars = parse_int(
            args.gloss_max_total_chars if args.gloss_max_total_chars is not None else cfg.get("gloss_max_total_chars", 220),
            220,
        )
        gloss_max_match_len = parse_int(
            args.gloss_max_match_len if args.gloss_max_match_len is not None else cfg.get("gloss_max_match_len", 4),
            4,
        )
        gloss_max_lemma_freq = (
            args.gloss_max_lemma_freq
            if args.gloss_max_lemma_freq is not None
            else parse_optional_int(cfg.get("gloss_max_lemma_freq"))
        )
        gloss_min_lemma_chars = parse_int(
            args.gloss_min_lemma_chars if args.gloss_min_lemma_chars is not None else cfg.get("gloss_min_lemma_chars", 2),
            2,
        )
        gloss_exclude_bound = not _as_bool(cfg.get("gloss_include_bound_morphemes", False))

        gloss_cfg = GlossAugmentConfig(
            enabled=True,
            match_types=tuple(match_types),
            max_match_len=gloss_max_match_len,
            stop_lemmas=tuple(stop_lemmas),
            min_lemma_chars=gloss_min_lemma_chars,
            exclude_bound_morphemes=gloss_exclude_bound,
            max_lemma_freq=gloss_max_lemma_freq,
            max_hints=gloss_max_hints,
            max_total_chars=gloss_max_total_chars,
        )

        # 任意: gloss 付与前の train src からレマ頻度を算出し、フィルタに使う
        if gloss_cfg.max_lemma_freq is not None:
            gloss_lemma_freq = build_lemma_frequency(
                train_df[src_col].fillna("").astype(str).tolist(),
                oa_lexicon_path=oa_lexicon_path,
                match_columns=gloss_cfg.match_columns,
                match_types=gloss_cfg.match_types,
                max_match_len=gloss_cfg.max_match_len,
            )
            print(
                f"[gloss] lemma_freq computed: size={len(gloss_lemma_freq)} max_lemma_freq={gloss_cfg.max_lemma_freq}"
            )

        gloss_augment = build_gloss_augmenter(
            gloss_cfg,
            oa_lexicon_path=oa_lexicon_path,
            ebl_dictionary_path=ebl_dictionary_path,
            lemma_freq=gloss_lemma_freq,
        )
        train_df[src_col] = train_df[src_col].fillna("").astype(str).map(gloss_augment)
        if val_df is not None:
            val_df[src_col] = val_df[src_col].fillna("").astype(str).map(gloss_augment)

        gloss_meta = {
            "enabled": True,
            "oa_lexicon": str(oa_lexicon_path),
            "ebl_dictionary": str(ebl_dictionary_path),
            "match_types": list(gloss_cfg.match_types),
            "max_match_len": gloss_cfg.max_match_len,
            "stop_lemmas": list(gloss_cfg.stop_lemmas),
            "min_lemma_chars": gloss_cfg.min_lemma_chars,
            "exclude_bound_morphemes": gloss_cfg.exclude_bound_morphemes,
            "max_lemma_freq": gloss_cfg.max_lemma_freq,
            "lemma_freq_size": 0 if gloss_lemma_freq is None else len(gloss_lemma_freq),
            "max_hints": gloss_cfg.max_hints,
            "max_total_chars": gloss_cfg.max_total_chars,
        }

        print(
            "[gloss] enabled: "
            f"oa_lexicon={oa_lexicon_path} ebl_dictionary={ebl_dictionary_path} "
            f"types={gloss_cfg.match_types} max_match_len={gloss_cfg.max_match_len} "
            f"stop_lemmas={len(gloss_cfg.stop_lemmas)} max_lemma_freq={gloss_cfg.max_lemma_freq} "
            f"max_hints={gloss_cfg.max_hints} max_total_chars={gloss_cfg.max_total_chars}"
        )
    print("=== データ概要 ===")
    print(f"train_rows={len(train_df)} val_rows={0 if val_df is None else len(val_df)}")
    log_text_summary(train_df, src_col, "train_src")
    log_text_summary(train_df, tgt_col, "train_tgt")
    if val_df is not None and len(val_df) > 0:
        log_text_summary(val_df, src_col, "val_src")
        log_text_summary(val_df, tgt_col, "val_tgt")

    try:
        import torch  # type: ignore
        from datasets import Dataset  # type: ignore
        from transformers import (  # type: ignore
            AutoConfig,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            set_seed as hf_set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "transformers/torch/datasets が見つかりません。requirements.txt をインストールしてください。"
        ) from exc

    hf_set_seed(seed)

    model_name_or_path = args.model_name_or_path or cfg.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("model_name_or_path is required in config or --model-name-or-path")

    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    print(f"device: {device_name}")

    model_config = AutoConfig.from_pretrained(model_name_or_path)
    update_model_config(model_config, cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=model_config)

    # 生成 / パディング / 停止条件まわりの安全化
    #
    # - 一部の環境では model.config.pad_token_id が None のままになり、
    #   Trainer の pad/gather の過程で -100 が混ざってデコードが落ちることがある。
    # - eos_token_id / decoder_start_token_id が None の場合、生成が EOS で
    #   止まらず max_length まで伸びやすい。
    #
    # ByT5 は通常 pad/eos を持つが、念のため tokenizer を正としてそろえる。
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None:
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = pad_id
    if eos_id is not None:
        if getattr(model.config, "eos_token_id", None) is None:
            model.config.eos_token_id = eos_id
    # T5/ByT5 は decoder_start_token_id == pad_token_id が一般的
    if pad_id is not None and getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = pad_id

    # transformers>=4.26 以降は generation_config を参照するケースもある
    if getattr(model, "generation_config", None) is not None:
        if pad_id is not None and getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = pad_id
        if eos_id is not None and getattr(model.generation_config, "eos_token_id", None) is None:
            model.generation_config.eos_token_id = eos_id
        if pad_id is not None and getattr(model.generation_config, "decoder_start_token_id", None) is None:
            model.generation_config.decoder_start_token_id = pad_id

    max_source_length = parse_int(cfg.get("max_source_length"), 256)
    max_target_length = parse_int(cfg.get("max_target_length"), 256)
    print(
        "=== 学習設定 ===\n"
        f"model={model_name_or_path} src_len={max_source_length} tgt_len={max_target_length} "
        f"val_ratio={val_ratio} seed={seed}"
    )
    token_stats_batch = parse_int(cfg.get("token_stats_batch_size"), 256)
    print("=== トークン長統計 ===")
    train_src_texts = train_df[src_col].fillna("").astype(str).tolist()
    train_tgt_texts = train_df[tgt_col].fillna("").astype(str).tolist()
    log_token_length_stats(
        train_src_texts,
        tokenizer,
        max_source_length,
        "train_src_tokens",
        token_stats_batch,
    )
    log_token_length_stats(
        train_tgt_texts,
        tokenizer,
        max_target_length,
        "train_tgt_tokens",
        token_stats_batch,
    )
    if val_df is not None and len(val_df) > 0:
        val_src_texts = val_df[src_col].fillna("").astype(str).tolist()
        val_tgt_texts = val_df[tgt_col].fillna("").astype(str).tolist()
        log_token_length_stats(
            val_src_texts,
            tokenizer,
            max_source_length,
            "val_src_tokens",
            token_stats_batch,
        )
        log_token_length_stats(
            val_tgt_texts,
            tokenizer,
            max_target_length,
            "val_tgt_tokens",
            token_stats_batch,
        )

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        inputs = tokenizer(
            batch[src_col],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[tgt_col],
            max_length=max_target_length,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = Dataset.from_pandas(train_df[[src_col, tgt_col]], preserve_index=False)
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=[src_col, tgt_col])

    val_ds = None
    if val_df is not None and len(val_df) > 0:
        val_ds = Dataset.from_pandas(val_df[[src_col, tgt_col]], preserve_index=False)
        val_ds = val_ds.map(preprocess, batched=True, remove_columns=[src_col, tgt_col])

    per_device_train_batch_size = parse_int(cfg.get("per_device_train_batch_size"), 8)
    per_device_eval_batch_size = parse_int(cfg.get("per_device_eval_batch_size"), 8)
    gradient_accumulation_steps = parse_int(cfg.get("gradient_accumulation_steps"), 1)
    learning_rate = parse_float(cfg.get("learning_rate"), 5e-4)
    num_train_epochs = parse_float(cfg.get("num_train_epochs"), 3.0)
    warmup_ratio = parse_float(cfg.get("warmup_ratio"), 0.0)
    weight_decay = parse_float(cfg.get("weight_decay"), 0.0)
    max_grad_norm = parse_float(cfg.get("max_grad_norm"), 1.0)
    logging_steps = parse_int(cfg.get("logging_steps"), 100)
    eval_steps = parse_int(cfg.get("eval_steps"), 0)
    save_steps = parse_int(cfg.get("save_steps"), 0)
    save_total_limit = parse_int(cfg.get("save_total_limit"), 2)
    lr_scheduler_type = str(cfg.get("lr_scheduler_type", "linear"))
    step_decay_steps = parse_int(cfg.get("step_decay_steps"), 0)
    step_decay_gamma = parse_float(cfg.get("step_decay_gamma"), 0.0)
    predict_with_generate = bool(cfg.get("predict_with_generate", True))
    generation_max_length = parse_int(cfg.get("generation_max_length"), max_target_length)
    generation_max_new_tokens = parse_optional_int(cfg.get("generation_max_new_tokens"))
    if generation_max_new_tokens is None:
        generation_max_new_tokens = parse_optional_int(cfg.get("max_new_tokens"))
    use_max_new_tokens = generation_max_new_tokens is not None and generation_max_new_tokens > 0
    # デフォルトは最も良かった "beam2_cfg" デコードに合わせて調整済み
    num_beams = parse_int(cfg.get("num_beams"), 2)
    length_penalty = parse_float(cfg.get("length_penalty"), 0.8)
    no_repeat_ngram_size = parse_int(cfg.get("no_repeat_ngram_size"), 20)
    repetition_penalty = parse_float(cfg.get("repetition_penalty"), 1.15)
    early_stopping = bool(cfg.get("early_stopping", False))
    gen_limit = generation_max_new_tokens if use_max_new_tokens else generation_max_length
    fp16 = bool(cfg.get("fp16", False)) and torch.cuda.is_available()
    bf16 = bool(cfg.get("bf16", False)) and torch.cuda.is_available()
    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))

    # 後処理: 提出の安全策として単文化する
    force_single_sentence = bool(cfg.get("force_single_sentence", True))
    single_sentence_mode = str(cfg.get("single_sentence_mode", "merge")).strip().lower()
    if single_sentence_mode not in {"merge", "truncate"}:
        single_sentence_mode = "merge"

    print(
        "=== ハイパーパラメータ ===\n"
        f"lr={learning_rate} batch={per_device_train_batch_size} grad_accum={gradient_accumulation_steps} "
        f"weight_decay={weight_decay} warmup_ratio={warmup_ratio} max_grad_norm={max_grad_norm} "
        f"scheduler={lr_scheduler_type} step_decay_steps={step_decay_steps} step_decay_gamma={step_decay_gamma} "
        f"fp16={fp16} bf16={bf16} grad_ckpt={gradient_checkpointing}"
    )
    print(
        "=== デコード設定 ===\n"
        f"gen_limit={gen_limit} use_max_new_tokens={use_max_new_tokens} "
        f"beams={num_beams} length_penalty={length_penalty} early_stopping={early_stopping} "
        f"no_repeat_ngram_size={no_repeat_ngram_size} repetition_penalty={repetition_penalty}"
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    eval_strategy = "no"
    save_strategy = "no"
    if val_ds is not None:
        eval_strategy = "steps"
        save_strategy = "steps"

    if eval_steps <= 0:
        eval_steps = max(50, len(train_ds) // max(1, per_device_train_batch_size))
    if save_steps <= 0:
        save_steps = eval_steps

    out_dir = Path(args.out) if args.out else get_artifacts_dir(cfg) / "nmt" / now_id("byt5")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 再現性のために検証文書IDを保存
    if val_doc_ids is not None:
        try:
            val_ids_path = out_dir / "val_doc_ids.json"
            val_ids_path.write_text(
                json.dumps(val_doc_ids, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[val_split] saved val_doc_ids: {val_ids_path}")
        except Exception as exc:
            print(f"[WARN] failed to save val_doc_ids: {exc}")

    # 頻度ベースのフィルタを使った場合、推論で再利用できるよう計算結果を保存
    if gloss_enabled and gloss_lemma_freq is not None:
        try:
            freq_path = out_dir / "gloss_lemma_freq.json"
            freq_path.write_text(
                json.dumps(gloss_lemma_freq, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            print(f"[gloss] saved lemma_freq: {freq_path}")
        except Exception as exc:
            print(f"[WARN] failed to save gloss lemma_freq: {exc}")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    compute_metrics = None
    try:
        import numpy as np  # type: ignore
        import sacrebleu  # type: ignore

        def _decode_safe(token_ids: np.ndarray, label: str) -> Optional[List[str]]:
            try:
                return tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            except Exception as exc:
                print(f"[WARN] metric decode failed ({label}): {exc}")
                return None

        def _sanitize_token_ids(token_ids: np.ndarray) -> np.ndarray:
            """デコード前の安全化（-100 や範囲外 id を pad/unk に寄せる）。"""

            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = 0
            if (token_ids < 0).any():
                token_ids = np.where(token_ids < 0, pad_id, token_ids)
            vocab_size = getattr(tokenizer, "vocab_size", None)
            if vocab_size is not None:
                unk_id = tokenizer.unk_token_id
                if unk_id is None:
                    unk_id = pad_id
                if (token_ids >= vocab_size).any():
                    token_ids = np.where(token_ids >= vocab_size, unk_id, token_ids)
            return token_ids

        def _compute_metrics(pred) -> Dict[str, float]:
            preds = pred.predictions
            if isinstance(preds, tuple):
                preds = preds[0]
            preds_arr = np.asarray(preds)
            if preds_arr.ndim == 3:
                preds_arr = preds_arr.argmax(axis=-1)
            if np.issubdtype(preds_arr.dtype, np.floating):
                preds_arr = np.rint(preds_arr).astype("int64")
            else:
                preds_arr = preds_arr.astype("int64", copy=False)
            preds_arr = _sanitize_token_ids(preds_arr)
            decoded_preds = _decode_safe(preds_arr, "preds")
            if decoded_preds is None:
                return {}

            labels = pred.label_ids
            labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
            labels = _sanitize_token_ids(labels)
            decoded_labels = _decode_safe(labels, "labels")
            if decoded_labels is None:
                return {}

            decoded_preds = [clean_text(text) for text in decoded_preds]
            decoded_labels = [clean_text(text) for text in decoded_labels]
            if force_single_sentence:
                decoded_preds = [
                    enforce_single_sentence(text, mode=single_sentence_mode) for text in decoded_preds
                ]
            bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
            chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2).score
            gm = math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))
            return {"bleu": float(bleu), "chrf": float(chrf), "gm": float(gm)}

        compute_metrics = _compute_metrics
    except Exception:
        compute_metrics = None

    training_args_kwargs = {
        "output_dir": str(out_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "logging_steps": logging_steps,
        "evaluation_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "predict_with_generate": predict_with_generate,
        "generation_max_length": generation_max_length,
        "generation_num_beams": num_beams,
        "fp16": fp16,
        "bf16": bf16,
        "lr_scheduler_type": lr_scheduler_type,
        "report_to": [],
    }

    total_steps = estimate_total_steps(
        len(train_ds),
        per_device_train_batch_size,
        gradient_accumulation_steps,
        num_train_epochs,
    )
    import inspect

    supported_keys = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters.keys())
    supported_keys.discard("self")
    training_args_kwargs = filter_training_args(
        training_args_kwargs,
        supported_keys,
        has_val=val_ds is not None,
        total_steps=total_steps,
    )

    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    if step_decay_steps > 0 and step_decay_gamma > 0:
        class StepDecayTrainer(Seq2SeqTrainer):
            def __init__(self, step_size: int, gamma: float, **kwargs: Any) -> None:
                self.step_decay_step_size = step_size
                self.step_decay_gamma = gamma
                super().__init__(**kwargs)

            def create_scheduler(self, num_training_steps: int, optimizer=None):
                if optimizer is None:
                    optimizer = self.optimizer
                from torch.optim.lr_scheduler import StepLR  # type: ignore

                self.lr_scheduler = StepLR(
                    optimizer, step_size=self.step_decay_step_size, gamma=self.step_decay_gamma
                )
                return self.lr_scheduler

        trainer = StepDecayTrainer(
            step_size=step_decay_steps,
            gamma=step_decay_gamma,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = train_result.metrics
    if val_ds is not None and val_df is not None and len(val_df) > 0 and post_eval_mode != "none":
        # -------------
        # 注意:
        # 以前は (1) trainer.evaluate() と (2) trainer.predict() の両方で
        # val 全体に対して generate を回しており、2回分の計算コストがかかっていた。
        # ここでは predict を1回だけ実行し、その metrics/loss を採用する。
        # -------------

        def _fmt(value: Any) -> str:
            if value is None:
                return "na"
            try:
                return f"{float(value):.4f}"
            except (TypeError, ValueError):
                return str(value)

        # 任意: 反復を速くするためサブセットで近似評価
        val_df_eval = val_df
        val_ds_eval = val_ds
        if post_eval_max_rows is not None and 0 < post_eval_max_rows < len(val_df):
            val_df_eval = val_df.sample(n=post_eval_max_rows, random_state=seed).reset_index(drop=True)
            val_ds_eval = Dataset.from_pandas(val_df_eval[[src_col, tgt_col]], preserve_index=False)
            val_ds_eval = val_ds_eval.map(preprocess, batched=True, remove_columns=[src_col, tgt_col])
            print(
                f"[post_eval] use subset: {len(val_df_eval)}/{len(val_df)} rows "
                f"(post_eval_max_rows={post_eval_max_rows})"
            )

        val_gen_kwargs = {
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
        }
        if use_max_new_tokens:
            val_gen_kwargs["max_new_tokens"] = gen_limit
        else:
            val_gen_kwargs["max_length"] = generation_max_length

        original_predict_with_generate = trainer.args.predict_with_generate
        trainer.args.predict_with_generate = True

        # 1回だけ predict し、loss/metrics をここから拾う
        val_pred_output = trainer.predict(val_ds_eval, metric_key_prefix="eval", **val_gen_kwargs)

        # --- 生成停止の診断（1行） ---
        try:
            import numpy as np  # type: ignore

            pred_ids = val_pred_output.predictions
            if isinstance(pred_ids, tuple):
                pred_ids = pred_ids[0]
            pred_arr = np.asarray(pred_ids)
            if pred_arr.ndim == 3:
                pred_arr = pred_arr.argmax(axis=-1)
            if np.issubdtype(pred_arr.dtype, np.floating):
                pred_arr = np.rint(pred_arr).astype("int64")
            else:
                pred_arr = pred_arr.astype("int64", copy=False)

            pad_id = tokenizer.pad_token_id
            eos_id = tokenizer.eos_token_id
            if pad_id is None:
                pad_id = getattr(model.config, "pad_token_id", None)
            if eos_id is None:
                eos_id = getattr(model.config, "eos_token_id", None)

            if pad_id is None:
                print("[gen_stop] pad_id=None (skip)")
            else:
                # -100 などの負値が混ざると長さが壊れるので pad 扱いに寄せる
                if (pred_arr < 0).any():
                    pred_arr = np.where(pred_arr < 0, pad_id, pred_arr)

                returned_max_len = int(pred_arr.shape[1])
                non_pad_len = (pred_arr != pad_id).sum(axis=1)

                if eos_id is None:
                    eos_rate = float("nan")
                    eos_present = None
                else:
                    eos_present = (pred_arr == eos_id).any(axis=1)
                    eos_rate = float(eos_present.mean())

                # config 由来の想定上限（目安）
                cfg_limit_total_len = (int(gen_limit) + 1) if use_max_new_tokens else int(generation_max_length)

                if eos_present is None:
                    hit_limit_rate = float((non_pad_len >= returned_max_len).mean())
                else:
                    hit_limit_rate = float(((~eos_present) & (non_pad_len >= returned_max_len)).mean())

                print(
                    f"[gen_stop] pad_id={pad_id} eos_id={eos_id} "
                    f"eos_rate={eos_rate:.1%} hit_limit_rate={hit_limit_rate:.1%} "
                    f"returned_max_len={returned_max_len} cfg_limit_total_len={cfg_limit_total_len}"
                )
        except Exception as e:
            print(f"[WARN] gen_stop diagnostics failed: {e}")
        # --- /生成停止の診断 ---
        if getattr(val_pred_output, "metrics", None):
            metrics.update(val_pred_output.metrics)

        eval_bleu = metrics.get("eval_bleu")
        eval_chrf = metrics.get("eval_chrf")
        eval_gm = metrics.get("eval_gm")
        eval_loss = metrics.get("eval_loss")

        if eval_bleu is not None or eval_chrf is not None or eval_gm is not None:
            print(
                "[eval_metrics] "
                f"bleu={_fmt(eval_bleu)} chrf={_fmt(eval_chrf)} gm={_fmt(eval_gm)}"
            )
        if eval_loss is not None:
            try:
                print(f"Val loss: {float(eval_loss):.4f}")
            except (TypeError, ValueError):
                print(f"Val loss: {eval_loss}")

        # quick モードはメトリクス出力だけで終了
        if post_eval_mode == "quick":
            trainer.args.predict_with_generate = original_predict_with_generate
        else:
            # full モード: サンプル表示/不変条件/崩壊検知/デコード比較まで実施
            val_src_texts = val_df_eval[src_col].fillna("").astype(str).map(clean_text).tolist()
            val_ref_texts = val_df_eval[tgt_col].fillna("").astype(str).map(clean_text).tolist()

            # デコード（注意: Seq2SeqTrainer は gather/pad の都合で -100 を混ぜることがある）
            val_preds = decode_predictions(val_pred_output, tokenizer)
            if val_preds is None:
                print("[WARN] val_pred decode skipped; downstream logs are skipped")
            else:
                if force_single_sentence:
                    val_preds = [
                        enforce_single_sentence(text, mode=single_sentence_mode)
                        for text in val_preds
                    ]
                # compute_metrics が無効だった場合のフォールバック（環境差対策）
                if eval_bleu is None or eval_chrf is None or eval_gm is None:
                    fallback = compute_bleu_chrf(val_preds, val_ref_texts)
                    if fallback is not None:
                        bleu, chrf, gm = fallback
                        print(f"[val_metrics] bleu={bleu:.4f} chrf={chrf:.4f} gm={gm:.4f}")
                        metrics["val_bleu"] = bleu
                        metrics["val_chrf"] = chrf
                        metrics["val_gm"] = gm

                val_sample_size = parse_int(cfg.get("val_sample_size"), 20)
                log_val_samples(val_src_texts, val_ref_texts, val_preds, val_sample_size)
                log_invariant_rates(val_src_texts, val_ref_texts, val_preds)
                log_collapse_stats(val_preds, val_ref_texts)

            # ---- 生成長 / EOS 診断（軽量: 既に生成済みのIDを使用） ----
            try:
                import numpy as np  # type: ignore

                pred_ids = val_pred_output.predictions
                if isinstance(pred_ids, tuple):
                    pred_ids = pred_ids[0]
                pred_arr = np.asarray(pred_ids)
                if pred_arr.ndim == 3:
                    pred_arr = pred_arr.argmax(axis=-1)
                if np.issubdtype(pred_arr.dtype, np.floating):
                    pred_arr = np.rint(pred_arr).astype("int64")
                else:
                    pred_arr = pred_arr.astype("int64", copy=False)

                pad_id = tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = 0
                neg_rate = float((pred_arr < 0).mean()) if pred_arr.size else 0.0
                if neg_rate > 0:
                    # これがあると「長さが固定に見える」統計が出やすい
                    print(f"[gen_len][note] negative token ids detected (rate={neg_rate:.2%}); treat as padding")
                    pred_arr = np.where(pred_arr < 0, pad_id, pred_arr)

                lens = (pred_arr != pad_id).sum(axis=1)
                series = pd.Series(lens.tolist())
                q = series.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
                # 目安: max_new_tokens を使うなら gen_limit に張り付く率
                hit_max_rate = float((series >= gen_limit).mean()) if len(series) else 0.0

                eos_id = tokenizer.eos_token_id
                eos_missing = 0
                if eos_id is not None:
                    for seq in pred_arr.tolist():
                        if eos_id not in seq:
                            eos_missing += 1
                    eos_missing_rate = eos_missing / max(1, len(pred_arr))
                    print(
                        "[gen_len] "
                        f"limit={gen_limit} hit_limit_rate={hit_max_rate:.1%} eos_missing={eos_missing_rate:.1%} "
                        f"max={series.max():.0f} p50={q[0.5]:.0f} p90={q[0.9]:.0f} p95={q[0.95]:.0f} p99={q[0.99]:.0f}"
                    )
                else:
                    print(
                        "[gen_len] "
                        f"limit={gen_limit} hit_limit_rate={hit_max_rate:.1%} "
                        f"max={series.max():.0f} p50={q[0.5]:.0f} p90={q[0.9]:.0f} p95={q[0.95]:.0f} p99={q[0.99]:.0f}"
                    )
            except Exception as exc:
                print(f"[WARN] gen_len diagnostics skipped: {exc}")

            decode_sample_size = parse_int(cfg.get("decode_sample_size"), parse_int(cfg.get("val_sample_size"), 20))
            sample_size = min(decode_sample_size, len(val_df_eval))
            if sample_size > 0:
                sample_df = val_df_eval.head(sample_size).reset_index(drop=True)
                sample_refs = sample_df[tgt_col].fillna("").astype(str).map(clean_text).tolist()
                sample_ds = Dataset.from_pandas(sample_df[[src_col, tgt_col]], preserve_index=False)
                sample_ds = sample_ds.map(preprocess, batched=True, remove_columns=[src_col, tgt_col])

                def run_decode_case(name: str, gen_kwargs: Dict[str, Any]) -> None:
                    pred_out = trainer.predict(sample_ds, metric_key_prefix=f"decode_{name}", **gen_kwargs)
                    preds = decode_predictions(pred_out, tokenizer)
                    if preds is None:
                        print(f"[decode:{name}] skipped (decode failed)")
                        return

                    if force_single_sentence:
                        preds = [
                            enforce_single_sentence(text, mode=single_sentence_mode)
                            for text in preds
                        ]

                    sample_metrics = compute_bleu_chrf(preds, sample_refs)

                    try:
                        import numpy as np  # type: ignore

                        pred_ids = pred_out.predictions
                        if isinstance(pred_ids, tuple):
                            pred_ids = pred_ids[0]
                        pred_arr = np.asarray(pred_ids)
                        if pred_arr.ndim == 3:
                            pred_arr = pred_arr.argmax(axis=-1)
                        if np.issubdtype(pred_arr.dtype, np.floating):
                            pred_arr = np.rint(pred_arr).astype("int64")
                        else:
                            pred_arr = pred_arr.astype("int64", copy=False)
                        pad_id = tokenizer.pad_token_id
                        if pad_id is None:
                            pad_id = 0
                        # gather/pad の -100 を pad として扱う（これをしないと「長さ固定」に見える）
                        if (pred_arr < 0).any():
                            pred_arr = np.where(pred_arr < 0, pad_id, pred_arr)
                        lens = (pred_arr != pad_id).sum(axis=1).tolist()
                    except Exception:
                        # フォールバック（型が想定外でも落とさない）
                        pred_ids = pred_out.predictions
                        if isinstance(pred_ids, tuple):
                            pred_ids = pred_ids[0]
                        try:
                            lens = [len(x) for x in pred_ids]
                        except Exception:
                            lens = [0 for _ in range(len(sample_refs))]

                    series = pd.Series(lens)
                    q = series.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
                    if sample_metrics is None:
                        metric_str = "bleu=na chrf=na gm=na"
                    else:
                        bleu, chrf, gm = sample_metrics
                        metric_str = f"bleu={bleu:.4f} chrf={chrf:.4f} gm={gm:.4f}"
                    print(
                        f"[decode:{name}] {metric_str} "
                        f"pred_len_tok_p50={q[0.5]:.0f} p90={q[0.9]:.0f} "
                        f"p95={q[0.95]:.0f} p99={q[0.99]:.0f} max={series.max():.0f}"
                    )

                def build_gen_kwargs(
                    beams: int,
                    length_penalty_value: float,
                    no_repeat_value: int,
                    rep_penalty: float,
                ) -> Dict[str, Any]:
                    gen_kwargs = {
                        "num_beams": beams,
                        "length_penalty": length_penalty_value,
                        "early_stopping": early_stopping,
                        "no_repeat_ngram_size": no_repeat_value,
                        "repetition_penalty": rep_penalty,
                    }
                    if use_max_new_tokens:
                        gen_kwargs["max_new_tokens"] = gen_limit
                    else:
                        gen_kwargs["max_length"] = generation_max_length
                    return gen_kwargs

                print("=== デコード比較 ===")
                run_decode_case(
                    "greedy",
                    build_gen_kwargs(1, 1.0, 0, 1.0),
                )
                run_decode_case(
                    "beam2_free",
                    build_gen_kwargs(2, 1.0, 0, 1.0),
                )
                run_decode_case(
                    "beam2_cfg",
                    build_gen_kwargs(
                        num_beams,
                        length_penalty,
                        no_repeat_ngram_size,
                        repetition_penalty,
                    ),
                )
                run_decode_case(
                    "beam4_free",
                    build_gen_kwargs(4, 1.0, 0, 1.0),
                )

            trainer.args.predict_with_generate = original_predict_with_generate

    meta = {
        "train_rows": len(train_df),
        "val_rows": 0 if val_df is None else len(val_df),
        "val_split_unit": val_split_unit,
        "val_doc_id_col": val_doc_id_col,
        "val_source_col": val_source_col,
        "val_exclude_sources": list(val_exclude_sources),
        "val_holdout_all_sources": bool(val_holdout_all_sources),
        "train_doc_count": int(train_df[val_doc_id_col].nunique()) if val_doc_id_col in train_df.columns else None,
        "val_doc_count": int(val_df[val_doc_id_col].nunique()) if (val_df is not None and val_doc_id_col in val_df.columns) else None,
        "val_doc_ids": val_doc_ids,
        "src_col": src_col,
        "tgt_col": tgt_col,
        "variant": variant,
        "drop_flagged": drop_flagged,
        "norm_variant": norm_variant,
    }

    if gloss_meta is not None:
        meta["gloss"] = gloss_meta
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"Saved model: {out_dir}")
    print(f"Train rows: {len(train_df)} / Val rows: {0 if val_df is None else len(val_df)}")


if __name__ == "__main__":
    main()
