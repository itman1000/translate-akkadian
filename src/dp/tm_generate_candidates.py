"""Generate candidate translations via TM++ (slotified retrieval).

This is the "improvement A" candidate generator described in plan_A-E.md:
TM++ (スロット化検索) as an additional candidate source.

Pipeline
--------
1) Slotify query src (PN/NUM/UNIT).
2) Retrieve top-K similar slotified src from the TM index (TF-IDF char n-gram).
3) Slotify retrieved target using retrieved src slot map (template).
4) Restore slots using query slot map (fallback to retrieved map).
5) Postprocess (normalize/enforce single sentence) and write a long CSV.

Optional: forward-scoring (adds 'seq_score')
------------------------------------------
If you pass --fwd-ckpt, this script will compute a forward sequence score
(avg log-prob per token) for each TM++ candidate using teacher forcing
(negative average NLL, higher is better). This makes TM++ candidates usable in:

- dp.eval_rerank_methods --prune-strategy top_fwd
- dp.eval_rerank_methods --lambda-fwd != 0 (noisy-channel with forward term)

The output CSV is compatible with:
- dp.oracle_eval (needs id + translation)
- dp.eval_rerank_methods (id + translation + optional tag + optional seq_score + extra features)
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("joblib is required (it should come with scikit-learn)") from exc

from .tm_slotify import SlotifyConfig, restore_slots, slotify_src, slotify_tgt_using_src_map
from .utils import (
    clean_text,
    enforce_single_sentence,
    ensure_dir,
    find_repo_root,
    load_config,
    normalize_translation_output,
)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _parse_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate TM++ candidates (slotified retrieval).")
    p.add_argument("--index", required=True, help="TM++ index joblib path.")
    p.add_argument("--queries", required=True, help="Queries file (csv/parquet).")
    p.add_argument("--id-col", required=True, help="Query id column.")
    p.add_argument("--src-col", required=True, help="Query source column.")
    p.add_argument("--out", required=True, help="Output candidate CSV path.")
    p.add_argument("--topk", type=int, default=50, help="Top-K retrieved pairs per query (default: 50).")
    p.add_argument("--min-sim", type=float, default=0.0, help="Drop retrieved items with sim < min_sim.")
    p.add_argument("--tag", default="tmpp", help="Candidate tag (default: tmpp).")
    p.add_argument("--exclude-exact-src", action="store_true", help="Exclude exact slotified self-matches.")
    p.add_argument("--dedupe", action="store_true", help="Deduplicate translations per id (recommended).")
    p.add_argument("--debug-cols", action="store_true", help="Include debug columns (tm_src/tm_tgt/template).")

    # Optional overrides / compatibility knobs
    p.add_argument("--config", default=None, help="Optional NMT config to reuse postprocess settings.")
    p.add_argument("--lexicon", default=None, help="Override lexicon path for PN slotification.")
    p.add_argument("--strip-lex-from-src", action="store_true", help="Strip <LEX>...</LEX> from query src (retrieval only).")

    # Optional: forward score for noisy-channel/top_fwd pruning
    p.add_argument(
        "--fwd-ckpt",
        default=None,
        help="Optional forward model ckpt dir to score TM++ candidates (adds 'seq_score').",
    )
    p.add_argument(
        "--fwd-batch-size",
        type=int,
        default=None,
        help="Batch size for forward scoring (default: from config or 8).",
    )
    p.add_argument(
        "--fwd-max-source-length",
        type=int,
        default=None,
        help="Max source length for forward scoring (default: from config or 384).",
    )
    p.add_argument(
        "--fwd-max-target-length",
        type=int,
        default=None,
        help="Max target length for forward scoring (default: from config or generation_max_length).",
    )
    p.add_argument(
        "--fwd-bf16",
        action="store_true",
        help="Use bf16 autocast for forward scoring on CUDA (default: follow config bf16).",
    )
    p.add_argument(
        "--fwd-device",
        default=None,
        help="Device for forward scoring (e.g., cuda, cpu). If omitted, auto-detect.",
    )
    return p


def _make_postprocess_fn(cfg: Dict[str, Any]):
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


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= len(scores):
        return np.argsort(-scores)
    part = np.argpartition(scores, -k)[-k:]
    part = part[np.argsort(-scores[part])]
    return part


def _score_forward_logp(
    *,
    model_dir: Path,
    inputs: List[str],
    targets: List[str],
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    bf16: bool,
    device: Optional[str],
) -> List[float]:
    """Compute per-example forward scores ~ log p(target | input).

    Returns a list of scores (higher is better). We use negative average NLL.
    """

    try:
        import torch  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch/transformers are required for forward scoring (--fwd-ckpt).") from exc

    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same length")

    if device:
        torch_device = torch.device(str(device))
    else:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    dtype = torch.bfloat16 if bf16 and torch_device.type == "cuda" else None

    # Load model with dtype when supported (transformers version-dependent).
    if dtype is not None:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), dtype=dtype)
        except TypeError:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), torch_dtype=dtype)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))

    # Align special token ids (safety; consistent with dp.infer_nmt_nbest)
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id
    if pad_id is not None:
        model.config.pad_token_id = pad_id
        if getattr(model.config, "decoder_start_token_id", None) is None:
            model.config.decoder_start_token_id = pad_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = pad_id
            if getattr(model.generation_config, "decoder_start_token_id", None) is None:
                model.generation_config.decoder_start_token_id = pad_id
    if eos_id is not None:
        model.config.eos_token_id = eos_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.eos_token_id = eos_id

    model.to(torch_device)
    model.eval()

    scores: List[float] = []

    # Per-example loss (reduction="none")
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    def batches(n: int) -> Iterable[Tuple[int, int]]:
        for i in range(0, n, batch_size):
            yield i, min(n, i + batch_size)

    use_amp = bf16 and torch_device.type == "cuda"

    def _autocast_ctx():
        if not use_amp:
            return nullcontext()
        # Prefer the new API (FutureWarning avoidance). Fallback to old API if needed.
        try:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        except AttributeError:  # pragma: no cover
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)

    with torch.inference_mode():
        for bi, (i, j) in enumerate(batches(len(inputs))):
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
            if tok.pad_token_id is not None:
                labels[labels == tok.pad_token_id] = -100

            with _autocast_ctx():
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels, return_dict=True)

            logits = out.logits  # (bsz, tgt_len, vocab)
            vocab = logits.size(-1)
            loss_tok = ce(logits.view(-1, vocab), labels.view(-1)).view(labels.size())
            valid = labels != -100
            denom = valid.sum(dim=1).clamp(min=1)
            nll = (loss_tok * valid).sum(dim=1) / denom
            batch_scores = (-nll).detach().float().cpu().tolist()
            scores.extend([float(x) for x in batch_scores])

            if (bi + 1) % 100 == 0:
                print(f"[tm_generate_candidates] forward scoring... {j}/{len(inputs)}")

    return scores


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    repo_root = find_repo_root()

    # Load config once (postprocess + forward-scoring defaults)
    cfg: Dict[str, Any] = load_config(args.config) if args.config else {}
    postprocess = _make_postprocess_fn(cfg)

    # --- load TM++ index ---
    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = repo_root / index_path
    payload: Dict[str, Any] = joblib.load(index_path)

    vectorizer = payload["vectorizer"]
    X = payload["X"]
    meta: pd.DataFrame = payload["meta"]
    slot_cfg = SlotifyConfig.from_dict(payload.get("slotify", {}))

    # Overrides (useful when moving the index between environments)
    if args.lexicon:
        slot_cfg.lexicon_path = Path(args.lexicon)
    if args.strip_lex_from_src:
        slot_cfg.strip_lex_from_src = True

    # --- load queries ---
    q_path = Path(args.queries)
    if not q_path.is_absolute():
        q_path = repo_root / q_path
    q_df = _read_table(q_path)

    missing = [c for c in (args.id_col, args.src_col) if c not in q_df.columns]
    if missing:
        raise ValueError(f"Missing query columns: {missing} (columns={list(q_df.columns)[:50]}...)")

    ids = q_df[args.id_col].astype(str).tolist()
    srcs = q_df[args.src_col].fillna("").astype(str).tolist()

    topk = int(args.topk)
    min_sim = float(args.min_sim)

    rows: List[Dict[str, Any]] = []

    for qi, (qid, src) in enumerate(zip(ids, srcs)):
        slot_q, q_map = slotify_src(src, slot_cfg)
        q_counts = q_map.counts()

        q_vec = vectorizer.transform([slot_q])
        sims = (X @ q_vec.T).toarray().ravel()
        idxs = _topk_indices(sims, topk)

        seen: set[str] = set()
        rank = 0

        for idx in idxs:
            sim = float(sims[int(idx)])
            if sim < min_sim:
                break

            # Optional self-match exclusion (guardrail if the index accidentally includes the query set).
            if args.exclude_exact_src:
                try:
                    if str(meta["src_slot"].iat[int(idx)]) == slot_q:
                        continue
                except Exception:
                    pass

            tm_src = str(meta["src"].iat[int(idx)])
            tm_tgt = str(meta["tgt"].iat[int(idx)])

            # Build TM slot map from retrieved source (on-demand).
            _slot_tm_src, tm_map = slotify_src(tm_src, slot_cfg)
            tm_counts = tm_map.counts()

            # Convert TM target to a slotified template, then restore using query map.
            tm_template = slotify_tgt_using_src_map(tm_tgt, tm_map, slot_cfg)
            restored = restore_slots(tm_template, primary=q_map, fallback=tm_map)
            pred = postprocess(restored)

            if not pred:
                continue

            if args.dedupe:
                if pred in seen:
                    continue
                seen.add(pred)

            rank += 1
            row: Dict[str, Any] = {
                "id": qid,
                "translation": pred,
                "tag": str(args.tag),
                "run": 1,
                "rank": rank,
                "tm_sim": sim,
                "slot_match_num": min(q_counts.get("num", 0), tm_counts.get("num", 0)),
                "slot_match_pn": min(q_counts.get("pn", 0), tm_counts.get("pn", 0)),
                "slot_match_unit": min(q_counts.get("unit", 0), tm_counts.get("unit", 0)),
            }

            if args.debug_cols:
                row.update(
                    {
                        "tm_row": int(idx),
                        "tm_src": tm_src,
                        "tm_tgt": tm_tgt,
                        "tm_template": tm_template,
                        "query_slot": slot_q,
                    }
                )

            rows.append(row)

            if rank >= topk:
                break

        if (qi + 1) % 200 == 0:
            print(f"[tm_generate_candidates] processed {qi+1}/{len(ids)}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    ensure_dir(out_path.parent)

    out_df = pd.DataFrame(rows)
    if len(out_df) == 0:
        raise RuntimeError("No candidates were generated (check --topk/--min-sim and index/query paths).")

    # --- forward scoring (optional) ---
    if args.fwd_ckpt:
        fwd_dir = Path(args.fwd_ckpt)
        if not fwd_dir.is_absolute():
            fwd_dir = repo_root / fwd_dir
        if not fwd_dir.exists():
            raise FileNotFoundError(f"Forward ckpt dir not found: {fwd_dir}")

        # id -> raw src (NOT slotified; keep consistent with dp.infer_nmt_nbest inputs)
        id_to_src: Dict[str, str] = {}
        for qid, src in zip(ids, srcs):
            if qid not in id_to_src:
                id_to_src[qid] = src

        fwd_inputs = [id_to_src.get(str(x), "") for x in out_df["id"].astype(str).tolist()]
        fwd_targets = out_df["translation"].astype(str).tolist()

        bs = int(args.fwd_batch_size) if args.fwd_batch_size is not None else _parse_int(
            cfg.get("infer_batch_size", cfg.get("per_device_eval_batch_size", 8)),
            8,
        )
        max_src = int(args.fwd_max_source_length) if args.fwd_max_source_length is not None else _parse_int(
            cfg.get("max_source_length", 384),
            384,
        )
        max_tgt_default = cfg.get("max_target_length", cfg.get("generation_max_length", cfg.get("max_source_length", 384)))
        max_tgt = int(args.fwd_max_target_length) if args.fwd_max_target_length is not None else _parse_int(
            max_tgt_default,
            _parse_int(cfg.get("max_source_length", 384), 384),
        )
        bf16 = bool(args.fwd_bf16 or cfg.get("bf16", False))

        print(
            f"[tm_generate_candidates] scoring forward seq_score with ckpt={fwd_dir} "
            f"rows={len(out_df)} batch={bs} device={args.fwd_device or 'auto'} bf16={bf16} "
            f"max_src={max_src} max_tgt={max_tgt}"
        )
        seq_scores = _score_forward_logp(
            model_dir=fwd_dir,
            inputs=fwd_inputs,
            targets=fwd_targets,
            batch_size=bs,
            max_source_length=max_src,
            max_target_length=max_tgt,
            bf16=bf16,
            device=args.fwd_device,
        )
        if len(seq_scores) != len(out_df):
            raise RuntimeError("Internal error: seq_score length mismatch")
        out_df["seq_score"] = seq_scores

    # Keep a stable column order (useful when concatenating candidate CSVs).
    base_cols: List[str] = [
        "id",
        "translation",
        "tag",
        "run",
        "rank",
    ]
    if "seq_score" in out_df.columns:
        base_cols.append("seq_score")
    base_cols += [
        "tm_sim",
        "slot_match_num",
        "slot_match_pn",
        "slot_match_unit",
    ]
    extra_cols = [c for c in out_df.columns if c not in base_cols]
    out_df = out_df[base_cols + extra_cols]

    out_df.to_csv(out_path, index=False)
    print(f"[tm_generate_candidates] saved: {out_path} rows={len(out_df)} unique_ids={out_df['id'].nunique()}")


if __name__ == "__main__":  # pragma: no cover
    main()
