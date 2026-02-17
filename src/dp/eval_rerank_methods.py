"""Compare candidate-selection methods (MBR / noisy-channel) on a validation set.

This script is intended for the "Step 0+" stage:
  - You already generated an n-best candidate pool (multiple candidates per id)
  - You want to select *one* prediction per id without looking at the gold

Implemented methods:
  1) MBR(chrF++):
     Pick the candidate that maximizes the expected chrF++ similarity to other
     candidates (a "consensus" decode). This does NOT require a reverse model.

  2) Noisy-channel (reverse score):
     Pick the candidate that maximizes log p(x | y) using a reverse model
     (English -> Akkadian). Optionally, you can combine it with a forward
     score column from the candidate CSV.

The script can also compute BLEU/chrF++/gm against the gold (for analysis).

Candidate CSV format (flexible):
  Required:
    - id column (default: 'id')
    - text column (default auto-detect; prefers 'translation')
  Optional:
    - 'tag' (otherwise inferred from filename)
    - a forward score column (e.g., 'gen_score', 'seq_score', 'score')

Example:
  python -m dp.eval_rerank_methods \
    --config configs/nmt_byt5_small.yaml \
    --val artifacts/oracle/val_for_oracle.csv \
    --id-col id --src-col src_sent --gold-col tgt_sent \
    --cands artifacts/oracle/cands_beam64.csv artifacts/oracle/cands_sample_t0.9_p0.95.csv \
    --reverse-ckpt artifacts/nmt/byt5_reverse \
    --out artifacts/rerank_methods
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .train_real import compute_metrics
from .utils import (
    clean_text,
    enforce_single_sentence,
    load_config,
    normalize_translation_output,
)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _infer_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_candidates(
    paths: Sequence[Path],
    *,
    id_col: str,
    text_col: Optional[str],
    tag_col: str = "tag",
    tag_from_filename: bool = True,
) -> pd.DataFrame:
    """Load and concatenate candidate CSVs with light column normalization."""

    frames: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"candidate file not found: {p}")
        df = pd.read_csv(p)
        # Normalize id col
        if id_col not in df.columns:
            alt = _infer_first_existing(df, ["id", "ID", "Id", "sample_id", "row_id"])
            if alt is None:
                raise ValueError(f"{p} must contain an id column (expected '{id_col}')")
            df = df.rename(columns={alt: id_col})

        # Normalize text col
        tcol = text_col
        if tcol is None:
            tcol = _infer_first_existing(
                df,
                [
                    "translation",
                    "pred",
                    "prediction",
                    "text",
                    "output",
                    "hyp",
                ],
            )
        if tcol is None or tcol not in df.columns:
            raise ValueError(f"{p} must contain a text column (e.g., 'translation')")
        if tcol != "translation":
            df = df.rename(columns={tcol: "translation"})

        # Tag
        if tag_col in df.columns:
            if tag_col != "tag":
                df = df.rename(columns={tag_col: "tag"})
        else:
            if tag_from_filename:
                df["tag"] = p.stem
            else:
                df["tag"] = "cands"

        df[id_col] = df[id_col].astype(str)
        df["translation"] = df["translation"].fillna("").astype(str)

        # Keep useful columns only (ignore the rest)
        keep = [id_col, "translation", "tag"]
        for col in [
            "rank",
            "run",
            "gen_score",
            "seq_score",
            "sequence_score",
            "score",
        ]:
            if col in df.columns:
                keep.append(col)
        df = df[keep]
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def build_postprocess(cfg: Dict[str, Any]):
    force_one = bool(cfg.get("force_single_sentence", True))
    mode = str(cfg.get("single_sentence_mode", "merge")).strip().lower()
    normalize_output = bool(cfg.get("normalize_output", False))
    normalize_fractions = bool(cfg.get("normalize_output_fractions", True))
    normalize_units = bool(cfg.get("normalize_output_units", True))

    def _pp(text: str) -> str:
        if normalize_output:
            text2 = normalize_translation_output(
                text,
                normalize_fractions=normalize_fractions,
                normalize_units=normalize_units,
            )
        else:
            text2 = clean_text(text)
        if force_one:
            text2 = enforce_single_sentence(text2, mode=mode)
        return text2

    return _pp


_LEX_RE = re.compile(r"<LEX>.*?</LEX>", flags=re.DOTALL | re.IGNORECASE)


def strip_lex(text: str) -> str:
    return _LEX_RE.sub(" ", text)


@dataclass
class MBRConfig:
    ref_size: int = 32
    max_exact: int = 64
    seed: int = 42


def mbr_select_chrFpp(
    texts: List[str],
    *,
    metric: Any,
    cfg: MBRConfig,
    weights: Optional[List[float]] = None,
    seed_extra: int = 0,
) -> Tuple[int, float]:
    """Return (best_index, best_utility) by MBR using chrF++.

    We approximate the expectation by sampling a reference subset when the
    unique-candidate count is large.
    """

    if not texts:
        return -1, float("-inf")
    if len(texts) == 1:
        return 0, 0.0

    # Deduplicate with counts (keeps an approximate posterior mass signal).
    counter = Counter(texts)
    uniq = list(counter.keys())
    w = [float(counter[t]) for t in uniq]
    if weights is not None and len(weights) == len(texts):
        # Optional: incorporate external weights by summing over duplicates.
        acc: Dict[str, float] = {}
        for t, ww in zip(texts, weights):
            acc[t] = acc.get(t, 0.0) + float(ww)
        uniq = list(acc.keys())
        w = [float(acc[t]) for t in uniq]

    k = len(uniq)
    if k == 1:
        return texts.index(uniq[0]), 0.0

    # Seed per group (deterministic across runs)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for MBR sampling") from exc

    base_seed = int(cfg.seed) + int(seed_extra)
    rng = np.random.default_rng(base_seed)

    # Prepare reference indices
    if k <= cfg.max_exact:
        ref_idx = list(range(k))
    else:
        ref_n = int(min(cfg.ref_size, k))
        prob = np.asarray(w, dtype="float64")
        prob = prob / max(1e-12, prob.sum())
        # without replacement
        ref_idx = rng.choice(k, size=ref_n, replace=False, p=prob).tolist()

    ref_w = [w[i] for i in ref_idx]
    ref_w_sum = sum(ref_w) if ref_w else 1.0

    best_u = None
    best_i = 0
    # Utility of a candidate: weighted average chrF++ vs reference subset
    for i in range(k):
        hyp = uniq[i]
        total = 0.0
        for j, ww in zip(ref_idx, ref_w):
            ref = uniq[j]
            # sacrebleu CHRF sentence_score returns .score
            s = float(metric.sentence_score(hyp, [ref]).score)
            total += ww * s
        u = total / ref_w_sum
        if best_u is None or u > best_u:
            best_u = u
            best_i = i
        elif best_u is not None and abs(u - best_u) < 1e-12:
            # tie-break: prefer higher weight (more frequent), then shorter string
            if w[i] > w[best_i] or (w[i] == w[best_i] and len(hyp) < len(uniq[best_i])):
                best_u = u
                best_i = i

    chosen_text = uniq[best_i]
    # Return index in the ORIGINAL list for downstream bookkeeping
    orig_idx = texts.index(chosen_text)
    return orig_idx, float(best_u if best_u is not None else 0.0)


def _stable_hash_int(text: str) -> int:
    """Deterministic (process-stable) hash -> int."""

    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def score_reverse_logp(
    *,
    model_dir: Path,
    inputs: List[str],
    targets: List[str],
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    bf16: bool,
    device: str,
) -> List[float]:
    """Compute per-example reverse scores ~ log p(target | input).

    Returns a list of scores (higher is better). We use negative average NLL.
    """

    try:
        import torch  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError("torch/transformers are required for noisy-channel scoring") from exc

    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same length")

    torch_device = torch.device(device)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    dtype = torch.bfloat16 if bf16 and torch_device.type == "cuda" else None
    # transformers の版によっては `torch_dtype` より `dtype` が推奨されるため、
    # まず `dtype` を試し、ダメなら `torch_dtype` にフォールバックする。
    if dtype is not None:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), dtype=dtype)
        except TypeError:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), torch_dtype=dtype)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    model.to(torch_device)
    model.eval()

    scores: List[float] = []

    # 例ごとの loss を計算するため（reduction="none"）
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    def batches(n: int) -> Iterable[Tuple[int, int]]:
        for i in range(0, n, batch_size):
            yield i, min(n, i + batch_size)

    from contextlib import nullcontext

    use_amp = bf16 and torch_device.type == "cuda"

    def _autocast_ctx():
        if not use_amp:
            return nullcontext()
        # 新 API を優先（FutureWarning 回避）。無ければ旧 API にフォールバック。
        try:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        except AttributeError:
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)

    with torch.inference_mode():
        for i, j in batches(len(inputs)):
            batch_in = inputs[i:j]
            batch_tgt = targets[i:j]
            enc = tok(
                batch_in,
                max_length=max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            dec = tok(
                text_target=batch_tgt,
                max_length=max_target_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(torch_device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(torch_device)
            labels = dec["input_ids"].to(torch_device)
            labels = labels.clone()
            labels[labels == tok.pad_token_id] = -100

            with _autocast_ctx():
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels, return_dict=True)

            logits = out.logits  # (bsz, tgt_len, vocab)
            vocab = logits.size(-1)
            loss_tok = ce(logits.view(-1, vocab), labels.view(-1)).view(labels.size())
            valid = (labels != -100)
            denom = valid.sum(dim=1).clamp(min=1)
            nll = (loss_tok * valid).sum(dim=1) / denom
            batch_scores = (-nll).detach().float().cpu().tolist()
            scores.extend([float(x) for x in batch_scores])

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MBR(chrF++) and noisy-channel reranking.")
    parser.add_argument("--config", default=None, help="Optional config file (for postprocess settings).")
    parser.add_argument("--val", required=True, help="Validation CSV/Parquet path.")
    parser.add_argument("--cands", nargs="+", required=True, help="Candidate CSV path(s).")
    parser.add_argument("--out", required=True, help="Output directory.")

    parser.add_argument("--id-col", default="id", help="ID column name in val/cands.")
    parser.add_argument("--src-col", default=None, help="Source column name in val (for noisy-channel).")
    parser.add_argument("--gold-col", default=None, help="Gold/target column name in val.")
    parser.add_argument("--cand-text-col", default=None, help="Candidate text column name (auto if omitted).")
    parser.add_argument("--cand-tag-col", default="tag", help="Candidate tag column name.")

    # MBR
    parser.add_argument("--run-mbr", action="store_true", help="Run MBR(chrF++) selection.")
    parser.add_argument("--mbr-ref-size", type=int, default=32, help="MBR reference subset size.")
    parser.add_argument("--mbr-max-exact", type=int, default=64, help="Compute full MBR when uniq<=N.")
    parser.add_argument("--mbr-seed", type=int, default=42, help="Random seed for MBR subset.")

    # Noisy-channel
    parser.add_argument("--run-noisy", action="store_true", help="Run noisy-channel (reverse model) selection.")
    parser.add_argument("--reverse-ckpt", default=None, help="Reverse model dir (English->Akkadian).")
    parser.add_argument("--noisy-batch-size", type=int, default=8, help="Batch size for reverse scoring.")
    parser.add_argument("--noisy-max-source-length", type=int, default=None, help="Max length for reverse input.")
    parser.add_argument("--noisy-max-target-length", type=int, default=None, help="Max length for reverse target.")
    parser.add_argument("--noisy-bf16", action="store_true", help="Use bf16 for reverse scoring.")
    parser.add_argument("--device", default=None, help="Device for reverse scoring (e.g., cuda, cpu).")
    parser.add_argument(
        "--strip-lex-from-src",
        action="store_true",
        help="Strip <LEX>...</LEX> blocks from source before reverse scoring.",
    )
    parser.add_argument(
        "--max-cands-per-id",
        type=int,
        default=None,
        help="Optional pruning: score at most N candidates per id (keeps earliest rows).",
    )
    parser.add_argument(
        "--prune-strategy",
        default="head",
        choices=["head", "top_fwd"],
        help=(
            "How to prune candidates when --max-cands-per-id is set. "
            "'head': keep earliest rows per id (fast, depends on CSV ordering). "
            "'top_fwd': keep top-N by forward score per id (requires fwd score column)."
        ),
    )
    parser.add_argument(
        "--lambda-fwd",
        type=float,
        default=0.0,
        help="Optional: add lambda * forward_score_column when selecting.",
    )
    parser.add_argument(
        "--lambda-fwd-grid",
        default=None,
        help=(
            "Optional: sweep multiple lambda values without re-scoring reverse model. "
            "Comma-separated list, e.g. '0,0.5,1,2'. If set, --lambda-fwd is ignored."
        ),
    )
    parser.add_argument(
        "--fwd-score-col",
        default=None,
        help="Forward score column in candidate CSV (auto-detect if omitted).",
    )

    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    pp = build_postprocess(cfg)

    val_path = Path(args.val)
    val_df = read_table(val_path)
    id_col = args.id_col
    if id_col not in val_df.columns:
        raise ValueError(f"val must contain id column: {id_col}")
    val_df[id_col] = val_df[id_col].astype(str)

    # Resolve columns
    src_col = args.src_col or cfg.get("src_col") or cfg.get("infer_src_col")
    gold_col = (
        args.gold_col
        or cfg.get("tgt_col")
        or cfg.get("target_col")
        or cfg.get("label_col")
        or ("translation" if "translation" in val_df.columns else None)
    )
    if gold_col is None or gold_col not in val_df.columns:
        raise ValueError("val must contain gold column (e.g., --gold-col translation)")

    # Load candidates
    cand_paths = [Path(p) for p in args.cands]
    cands = load_candidates(
        cand_paths,
        id_col=id_col,
        text_col=args.cand_text_col,
        tag_col=args.cand_tag_col,
        tag_from_filename=True,
    )

    # Postprocess candidates (like submit)
    cands["translation"] = cands["translation"].astype(str).map(pp)

    # Basic candidate stats
    cand_counts = cands.groupby(id_col).size()
    print("=== candidate pool ===")
    print(
        f"ids={cand_counts.shape[0]} rows={len(cands)} avg_cands_per_id={cand_counts.mean():.1f} "
        f"p90={cand_counts.quantile(0.9):.0f} min={cand_counts.min()} max={cand_counts.max()}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_out: Dict[str, Any] = {
        "val": str(val_path),
        "n_val": int(len(val_df)),
        "n_cand_rows": int(len(cands)),
        "avg_cands_per_id": float(cand_counts.mean()),
    }

    gold_list = val_df[gold_col].fillna("").astype(str).tolist()
    ids_order = val_df[id_col].astype(str).tolist()

    # ---------- MBR ----------
    if args.run_mbr:
        try:
            from sacrebleu.metrics import CHRF
        except ImportError as exc:
            raise RuntimeError("sacrebleu is required for MBR") from exc

        chrf = CHRF(word_order=2)
        mbr_cfg = MBRConfig(ref_size=args.mbr_ref_size, max_exact=args.mbr_max_exact, seed=args.mbr_seed)

        chosen: Dict[str, Tuple[str, float, str]] = {}
        # (translation, utility, tag)
        for sid, group in cands.groupby(id_col, sort=False):
            texts = group["translation"].astype(str).tolist()
            tags = group["tag"].astype(str).tolist()
            # Stable seed per id
            seed_extra = _stable_hash_int(str(sid)) % 1000003
            idx, util = mbr_select_chrFpp(texts, metric=chrf, cfg=mbr_cfg, seed_extra=seed_extra)
            if idx < 0:
                continue
            chosen[sid] = (texts[idx], util, tags[idx])

        preds = [chosen.get(s, ("", 0.0, ""))[0] for s in ids_order]
        m = compute_metrics(preds, gold_list)
        metrics_out["mbr"] = {"bleu": m["bleu"], "chrf": m["chrf"], "gm": m["score"]}
        print(f"[mbr_chrF++] bleu={m['bleu']:.4f} chrf={m['chrf']:.4f} gm={m['score']:.4f}")

        mbr_df = pd.DataFrame(
            {
                id_col: ids_order,
                "translation": preds,
                "mbr_utility": [chosen.get(s, ("", float('nan'), ""))[1] for s in ids_order],
                "tag": [chosen.get(s, ("", 0.0, ""))[2] for s in ids_order],
            }
        )
        mbr_df.to_csv(out_dir / "pred_mbr.csv", index=False)
        metrics_out["mbr_out"] = "pred_mbr.csv"

    # ---------- Noisy-channel (reverse) ----------
    if args.run_noisy:
        if not args.reverse_ckpt:
            print("[noisy] skipped: --reverse-ckpt is not provided")
        else:
            if src_col is None or src_col not in val_df.columns:
                raise ValueError("--src-col is required for noisy-channel scoring")

            # Parse lambda sweep (cheap once reverse scores are computed)
            if args.lambda_fwd_grid:
                lambdas: List[float] = []
                for part in str(args.lambda_fwd_grid).split(","):
                    part = part.strip()
                    if not part:
                        continue
                    lambdas.append(float(part))
                if not lambdas:
                    raise ValueError("--lambda-fwd-grid was provided but could not be parsed")
            else:
                lambdas = [float(args.lambda_fwd)]
            need_fwd = any(abs(x) > 1e-12 for x in lambdas) or (str(args.prune_strategy) == "top_fwd")

            # Prepare candidates to score
            fwd_col = args.fwd_score_col
            if need_fwd:
                if fwd_col is None:
                    fwd_col = _infer_first_existing(
                        cands,
                        [
                            "gen_score",
                            "seq_score",
                            "sequence_score",
                            # last resort (note: in TF-IDF baseline it's similarity)
                            "score",
                        ],
                    )
                if fwd_col is None or fwd_col not in cands.columns:
                    raise ValueError(
                        "Forward score is required (lambda!=0 or prune_strategy=top_fwd) but no score column was found. "
                        "Generate candidates with --save-forward-score (adds 'seq_score') or pass --fwd-score-col."
                    )
                keep_cols = [id_col, "translation", "tag", fwd_col]
                score_df = cands[keep_cols].copy()
                score_df["fwd_score"] = pd.to_numeric(score_df[fwd_col], errors="coerce")
                n_nan = int(score_df["fwd_score"].isna().sum())
                if n_nan > 0:
                    frac = 100.0 * n_nan / max(1, len(score_df))
                    print(
                        f"[WARN] forward スコア列 '{fwd_col}' に NaN が {n_nan}/{len(score_df)} 件 ({frac:.2f}%) あります。"
                        "NaN は -inf として扱います。"
                        "全ての候補CSVにスコア列が入るよう、`dp.infer_nmt_nbest --save-forward-score` で再生成してください。"
                    )
                # 重要: NaN を 0.0 にすると、負の log-prob スコアより過大評価されてしまう。
                score_df["fwd_score"] = score_df["fwd_score"].fillna(float("-inf"))
            else:
                score_df = cands[[id_col, "translation", "tag"]].copy()
                score_df["fwd_score"] = 0.0

            # Optional pruning (reduce reverse-scoring cost)
            if args.max_cands_per_id is not None and args.max_cands_per_id > 0:
                n = int(args.max_cands_per_id)
                if str(args.prune_strategy) == "top_fwd":
                    # keep best forward-scored candidates
                    score_df = (
                        score_df.sort_values([id_col, "fwd_score"], ascending=[True, False])
                        .groupby(id_col, sort=False)
                        .head(n)
                        .reset_index(drop=True)
                    )
                else:
                    # keep earliest rows (depends on candidate CSV ordering)
                    score_df = score_df.groupby(id_col, sort=False).head(n).reset_index(drop=True)

            # Attach source text per id
            src_map = {
                str(k): str(v)
                for k, v in zip(val_df[id_col].astype(str).tolist(), val_df[src_col].fillna("").astype(str).tolist())
            }
            score_df["src"] = score_df[id_col].astype(str).map(lambda x: src_map.get(x, ""))
            if args.strip_lex_from_src:
                score_df["src"] = score_df["src"].astype(str).map(strip_lex)

            # (forward score already attached as score_df['fwd_score'])

            # Reverse scoring
            if args.device:
                device = str(args.device)
            else:
                try:
                    import torch  # type: ignore

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            max_src = args.noisy_max_source_length or int(cfg.get("max_source_length", 384))
            max_tgt = args.noisy_max_target_length or int(cfg.get("max_target_length", cfg.get("max_source_length", 384)))
            bf16 = bool(args.noisy_bf16 or cfg.get("bf16", False))

            rev_scores = score_reverse_logp(
                model_dir=Path(args.reverse_ckpt),
                inputs=score_df["translation"].astype(str).tolist(),
                targets=score_df["src"].astype(str).tolist(),
                batch_size=int(args.noisy_batch_size),
                max_source_length=int(max_src),
                max_target_length=int(max_tgt),
                bf16=bf16,
                device=str(device),
            )
            score_df["rev_score"] = rev_scores

            # Evaluate one or many lambdas (no extra reverse scoring)
            grid_metrics: Dict[str, Any] = {}
            best = None
            best_lambda = None
            best_preds: List[str] = []
            best_rows: Optional[pd.DataFrame] = None

            for lam in lambdas:
                score_df["total_score"] = score_df["rev_score"] + float(lam) * score_df["fwd_score"]
                chosen_rows = (
                    score_df.sort_values([id_col, "total_score"], ascending=[True, False])
                    .groupby(id_col)
                    .head(1)
                )
                chosen_map = {
                    str(r[id_col]): (
                        str(r["translation"]),
                        float(r["total_score"]),
                        float(r["rev_score"]),
                        float(r.get("fwd_score", 0.0)),
                        str(r["tag"]),
                    )
                    for r in chosen_rows.to_dict(orient="records")
                }
                preds = [chosen_map.get(s, ("", float("nan"), float("nan"), float("nan"), ""))[0] for s in ids_order]
                m = compute_metrics(preds, gold_list)
                key = f"lambda={lam:g}"
                grid_metrics[key] = {"bleu": m["bleu"], "chrf": m["chrf"], "gm": m["score"]}
                print(f"[noisy_channel lam={lam:g}] bleu={m['bleu']:.4f} chrf={m['chrf']:.4f} gm={m['score']:.4f}")

                if best is None or float(m["score"]) > float(best):
                    best = float(m["score"])
                    best_lambda = float(lam)
                    best_preds = preds
                    best_rows = chosen_rows.copy()

            metrics_out["noisy"] = {
                "lambda_grid": lambdas,
                "results": grid_metrics,
                "best_lambda": best_lambda,
                "best_gm": best,
                "fwd_score_col": fwd_col or "",
                "max_cands_per_id": int(args.max_cands_per_id) if args.max_cands_per_id else None,
                "prune_strategy": str(args.prune_strategy),
            }

            # Save best predictions
            if best_rows is None:
                best_rows = score_df.groupby(id_col, sort=False).head(0)
            best_map = {
                str(r[id_col]): (str(r["translation"]), float(r["total_score"]), float(r["rev_score"]), float(r.get("fwd_score", 0.0)), str(r["tag"]))
                for r in best_rows.to_dict(orient="records")
            }
            out_df = pd.DataFrame(
                {
                    id_col: ids_order,
                    "translation": best_preds,
                    "total_score": [best_map.get(s, ("", float('nan'), float('nan'), float('nan'), ""))[1] for s in ids_order],
                    "rev_score": [best_map.get(s, ("", float('nan'), float('nan'), float('nan'), ""))[2] for s in ids_order],
                    "fwd_score": [best_map.get(s, ("", float('nan'), float('nan'), float('nan'), ""))[3] for s in ids_order],
                    "tag": [best_map.get(s, ("", float('nan'), float('nan'), float('nan'), ""))[4] for s in ids_order],
                }
            )
            out_df.to_csv(out_dir / "pred_noisy_channel.csv", index=False)
            metrics_out["noisy_out"] = "pred_noisy_channel.csv"

            # Save scored candidates for debugging (can be large)
            score_df.to_csv(out_dir / "noisy_scored_candidates.csv", index=False)
            metrics_out["noisy_scored_candidates"] = "noisy_scored_candidates.csv"

    # ---------- Save summary ----------
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, sort_keys=True))
    print(f"Saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
