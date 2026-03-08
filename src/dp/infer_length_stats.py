#!/usr/bin/env python3
"""
Infer + diagnostic stats for Deep Past (Akkadian -> English) Kaggle code competition.

Purpose:
- Measure source token lengths and truncation rate (max_source_length).
- Measure generated token lengths and whether generation hits max length / misses EOS.
- Optionally apply lexicon-based gloss augmentation (same mechanism as dp.train_nmt).

This is meant to be run inside a Kaggle notebook (P100/T4), with PYTHONPATH pointing to repo/src.

Example (Kaggle cell):
  %%bash
  set -eux
  source /kaggle/working/kaggle_env.sh
  cd "$REPO_DIR"
  export PYTHONPATH="$REPO_DIR/src"
  python /kaggle/working/infer_length_stats.py \
    --config "$CFG" \
    --ckpt "$FWD_CKPT" \
    --data-dir "$COMP_DATA_DIR" \
    --use-gloss \
    --run-generate \
    --out-csv /kaggle/working/infer_stats_test.csv

Notes:
- For the hidden test in code competitions, you cannot "see" the real input text,
  but you *can* log aggregated stats like truncation rate safely.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _parse_int(v: Any, default: int) -> int:
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _parse_float(v: Any, default: float) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _quantiles(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan"), "mean": float("nan")}
    s = pd.Series(xs, dtype="float64")
    return {
        "mean": float(s.mean()),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


_LEX_RE = re.compile(r"<LEX>\s*(.*?)\s*</LEX>")


def _count_gloss_hints(text: str) -> int:
    """
    Count number of "lemma=gloss" pairs inside <LEX> ... </LEX>.

    This is heuristic (works with default dp.gloss formatting: "a=b | c=d").
    """
    m = _LEX_RE.search(text)
    if not m:
        return 0
    inner = m.group(1).strip()
    if not inner:
        return 0
    # split by pipe first
    parts = [p.strip() for p in inner.split("|") if p.strip()]
    return len(parts)


def _gloss_chars(text: str) -> int:
    m = _LEX_RE.search(text)
    if not m:
        return 0
    return len(m.group(0))


def _resolve_cfg_path(repo_dir: Path, cfg_env: str) -> Path:
    p = Path(cfg_env)
    if not p.is_absolute():
        p = repo_dir / p
    return p


def _resolve_data_dir(cfg: Dict[str, Any], arg_data_dir: Optional[str]) -> Path:
    # Minimal fallback: dp.utils.get_data_dir does more, but we keep this standalone.
    if arg_data_dir:
        return Path(arg_data_dir)
    v = cfg.get("data_dir") or cfg.get("data-dir")
    if v:
        return Path(str(v))
    raise ValueError("data-dir is required (either --data-dir or config.data_dir)")


def _resolve_optional_path(data_dir: Path, v: Any, default_name: str) -> Path:
    """
    Mirrors dp.train_nmt behavior:
    - If v is empty, use data_dir/default_name.
    - If v is relative and exists under data_dir, prefer that.
    """
    if v is None or str(v).strip() == "":
        return data_dir / default_name
    p = Path(str(v))
    if not p.is_absolute():
        cand = data_dir / p
        if cand.exists():
            return cand
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config YAML used in training/inference.")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint dir (transformers format).")
    ap.add_argument("--data-dir", default=None, help="Competition data dir (contains test.csv, OA_Lexicon_eBL.csv, eBL_Dictionary.csv).")
    ap.add_argument("--input-csv", default=None, help="Override input CSV/Parquet path (default: data-dir/test.csv).")
    ap.add_argument("--src-col", default=None, help="Source column (default from config or 'transliteration').")
    ap.add_argument("--id-col", default=None, help="ID column (default from config or 'id').")
    ap.add_argument("--ref-col", default=None, help="Optional reference column to compute metrics (e.g., 'translation' or 'ref').")
    ap.add_argument("--norm-variant", default=None, help="A/B/C transliteration normalization (override config).")
    ap.add_argument("--batch-size", type=int, default=None, help="Eval batch size (override config).")
    ap.add_argument("--max-source-length", type=int, default=None, help="Max source length (override config).")
    ap.add_argument("--run-generate", action="store_true", help="Actually run model.generate to collect output-length stats.")
    ap.add_argument("--max-rows", type=int, default=None, help="Only process first N rows (debug).")
    ap.add_argument("--out-csv", default=None, help="Write per-row stats to this CSV path.")
    ap.add_argument("--pred-out", default=None, help="If set and --run-generate, also save predictions (id,translation) CSV here.")

    # Gloss controls
    ap.add_argument("--use-gloss", action="store_true", help="Apply lexicon-based gloss augmentation to src before tokenization.")
    ap.add_argument("--oa-lexicon", default=None, help="Path to OA_Lexicon_eBL.csv (default: data-dir/OA_Lexicon_eBL.csv).")
    ap.add_argument("--ebl-dictionary", default=None, help="Path to eBL_Dictionary.csv (default: data-dir/eBL_Dictionary.csv).")
    ap.add_argument("--gloss-max-hints", type=int, default=None, help="Override gloss max_hints (default from config or dp.gloss default).")
    ap.add_argument("--gloss-max-total-chars", type=int, default=None, help="Override gloss max_total_chars.")
    ap.add_argument("--gloss-max-match-len", type=int, default=None, help="Override gloss max_match_len.")

    args = ap.parse_args()

    # Import dp lazily (requires PYTHONPATH)
    from dp.utils import load_config, clean_text, enforce_single_sentence  # type: ignore
    from dp.align_train import normalize_transliteration  # type: ignore

    repo_dir = Path(os.environ.get("REPO_DIR", ".")).resolve()
    cfg_path = _resolve_cfg_path(repo_dir, args.config)
    cfg = load_config(str(cfg_path))

    data_dir = _resolve_data_dir(cfg, args.data_dir)
    input_path = Path(args.input_csv) if args.input_csv else (data_dir / "test.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Resolve columns
    src_col = args.src_col or cfg.get("infer_src_col") or cfg.get("src_col") or "transliteration"
    id_col = args.id_col or cfg.get("infer_id_col") or "id"
    ref_col = args.ref_col

    # Load input
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)

    if src_col not in df.columns:
        raise ValueError(f"Missing src_col='{src_col}' in {input_path}. columns={list(df.columns)}")

    ids = df[id_col].astype(str).tolist() if id_col in df.columns else [str(i) for i in range(len(df))]
    raw_srcs = df[src_col].fillna("").astype(str).tolist()

    norm_variant = args.norm_variant or cfg.get("norm_variant") or cfg.get("infer_norm_variant")
    norm_variant = str(norm_variant).strip().upper() if norm_variant else ""
    if norm_variant:
        srcs_base = [normalize_transliteration(s, norm_variant) for s in raw_srcs]
    else:
        srcs_base = [clean_text(s) for s in raw_srcs]

    # Gloss augmentation (optional)
    srcs_aug = srcs_base
    gloss_enabled = bool(args.use_gloss)
    gloss_meta: Dict[str, Any] = {"enabled": False}
    if gloss_enabled:
        from dp.gloss import GlossAugmentConfig, build_gloss_augmenter  # type: ignore

        oa_path = _resolve_optional_path(data_dir, args.oa_lexicon or cfg.get("oa_lexicon_path"), "OA_Lexicon_eBL.csv")
        ebl_path = _resolve_optional_path(data_dir, args.ebl_dictionary or cfg.get("ebl_dictionary_path"), "eBL_Dictionary.csv")

        if not oa_path.exists() or not ebl_path.exists():
            print(f"[WARN] gloss requested but lexicon files not found. oa={oa_path} ebl={ebl_path}. Disabling gloss.")
            gloss_enabled = False
        else:
            # Build config from defaults + optional overrides
            # Note: for deeper alignment with training, add more cfg keys here as needed.
            gcfg = GlossAugmentConfig(
                enabled=True,
                max_hints=_parse_int(args.gloss_max_hints, _parse_int(cfg.get("gloss_max_hints"), 6)),
                max_total_chars=_parse_int(args.gloss_max_total_chars, _parse_int(cfg.get("gloss_max_total_chars"), 220)),
                max_match_len=_parse_int(args.gloss_max_match_len, _parse_int(cfg.get("gloss_max_match_len"), 4)),
            )
            gloss_fn = build_gloss_augmenter(gcfg, oa_lexicon_path=oa_path, ebl_dictionary_path=ebl_path, lemma_freq=None)
            srcs_aug = [gloss_fn(s) for s in srcs_base]
            gloss_meta = {
                "enabled": True,
                "oa_lexicon": str(oa_path),
                "ebl_dictionary": str(ebl_path),
                "max_hints": gcfg.max_hints,
                "max_total_chars": gcfg.max_total_chars,
                "max_match_len": gcfg.max_match_len,
            }

    # Basic text-level stats
    gloss_hints = [_count_gloss_hints(s) for s in srcs_aug]
    gloss_used = [h > 0 for h in gloss_hints]
    print(f"[rows] n={len(ids)} src_col={src_col} id_col={id_col} norm_variant={norm_variant or '(none)'}")
    if gloss_meta.get("enabled"):
        used_ratio = sum(gloss_used) / max(1, len(gloss_used))
        print(
            f"[gloss] enabled: used_rows={sum(gloss_used)}/{len(gloss_used)} ({used_ratio:.1%}) "
            f"avg_hints={sum(gloss_hints)/max(1,len(gloss_hints)):.2f} "
            f"oa={gloss_meta.get('oa_lexicon')} ebl={gloss_meta.get('ebl_dictionary')}"
        )
    else:
        print("[gloss] disabled")

    # Load model/tokenizer
    import torch  # type: ignore
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.ckpt)))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(Path(args.ckpt))).to(device)
    model.eval()

    batch_size = args.batch_size or _parse_int(cfg.get("per_device_eval_batch_size"), 8)
    max_source_length = args.max_source_length or _parse_int(cfg.get("max_source_length"), 384)

    # Generation config
    generation_max_length = _parse_int(cfg.get("generation_max_length"), 320)
    generation_max_new_tokens = _parse_int(cfg.get("generation_max_new_tokens"), 0)
    num_beams = _parse_int(cfg.get("num_beams"), 4)
    length_penalty = _parse_float(cfg.get("length_penalty"), 1.0)
    early_stopping = bool(cfg.get("early_stopping", False))
    no_repeat_ngram_size = _parse_int(cfg.get("no_repeat_ngram_size"), 0)
    repetition_penalty = _parse_float(cfg.get("repetition_penalty"), 1.0)

    force_single_sentence = bool(cfg.get("force_single_sentence", True))
    single_sentence_mode = str(cfg.get("single_sentence_mode", "merge")).strip().lower()
    if single_sentence_mode not in {"merge", "truncate"}:
        single_sentence_mode = "merge"

    print(f"[tokenizer] max_source_length={max_source_length} batch_size={batch_size}")
    if args.run_generate:
        print(
            "[generate] "
            f"beams={num_beams} len_pen={length_penalty} early_stop={early_stopping} "
            f"no_repeat_ngram={no_repeat_ngram_size} rep_pen={repetition_penalty} "
            f"max_new_tokens={generation_max_new_tokens if generation_max_new_tokens>0 else '(disabled)'} "
            f"max_length={generation_max_length if generation_max_new_tokens<=0 else '(n/a)'}"
        )

    # Per-row stats
    rows: List[Dict[str, Any]] = []
    preds: List[str] = []

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    def _nonpad_len(seq: List[int]) -> int:
        if pad_id is None:
            return len(seq)
        n = 0
        for t in seq:
            if t == pad_id:
                break
            n += 1
        return n

    t0 = time.time()
    n_trunc = 0
    src_len_raw_all: List[int] = []
    src_len_used_all: List[int] = []
    out_len_all: List[int] = []
    out_no_eos = 0
    out_hit_limit = 0

    # Iterate in batches
    for b0 in range(0, len(srcs_aug), batch_size):
        b_ids = ids[b0 : b0 + batch_size]
        b_src = srcs_aug[b0 : b0 + batch_size]
        b_src_base = srcs_base[b0 : b0 + batch_size]
        b_gloss_h = gloss_hints[b0 : b0 + batch_size]

        # Tokenize without truncation to measure true lengths
        enc_raw = tokenizer(
            b_src,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=True,
        )
        raw_lens = [len(x) for x in enc_raw["input_ids"]]
        src_len_raw_all.extend(raw_lens)

        # Tokenize with truncation for actual inference
        enc = tokenizer(
            b_src,
            truncation=True,
            padding=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        # Used length per row = attention_mask sum (nonpad tokens)
        used_lens = enc["attention_mask"].sum(dim=1).tolist()
        used_lens = [int(x) for x in used_lens]
        src_len_used_all.extend(used_lens)

        trunc_flags = [rl > max_source_length for rl in raw_lens]
        n_trunc += sum(1 for f in trunc_flags if f)

        # Prepare row-level records early (so we can attach output later)
        for i in range(len(b_ids)):
            rows.append(
                {
                    "id": b_ids[i],
                    "gloss_hints": int(b_gloss_h[i]),
                    "gloss_chars": int(_gloss_chars(b_src[i])),
                    "src_chars_base": int(len(b_src_base[i])),
                    "src_chars_aug": int(len(b_src[i])),
                    "src_ws_tokens_aug": int(len(b_src[i].split())),
                    "src_tok_len_raw": int(raw_lens[i]),
                    "src_tok_len_used": int(used_lens[i]),
                    "src_truncated": bool(trunc_flags[i]),
                }
            )

        if not args.run_generate:
            continue

        # Move to device and generate
        enc = {k: v.to(device) for k, v in enc.items()}
        gen_kwargs: Dict[str, Any] = dict(
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )
        if generation_max_new_tokens and generation_max_new_tokens > 0:
            gen_kwargs["max_new_tokens"] = generation_max_new_tokens
            limit_type = "max_new_tokens"
            limit_val = int(generation_max_new_tokens)
        else:
            gen_kwargs["max_length"] = generation_max_length
            limit_type = "max_length"
            limit_val = int(generation_max_length)

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        out_list = out.detach().cpu().tolist()
        # out is padded-to-batch max len, so compute per-row nonpad length.
        out_lens = [_nonpad_len(seq) for seq in out_list]
        out_len_all.extend(out_lens)

        # Detect EOS presence (before padding)
        for i, seq in enumerate(out_list):
            seq2 = seq[: out_lens[i]]
            has_eos = (eos_id is not None) and (eos_id in seq2)
            if not has_eos:
                out_no_eos += 1
            # "hit limit" heuristic: no eos AND length is near the configured limit.
            if (not has_eos) and (out_lens[i] >= limit_val):
                out_hit_limit += 1

        # Decode
        dec = tokenizer.batch_decode(out, skip_special_tokens=True)
        dec = [clean_text(x) for x in dec]
        if force_single_sentence:
            dec = [enforce_single_sentence(x, mode=single_sentence_mode) for x in dec]
        preds.extend(dec)

        # Attach output stats to rows (same order)
        base_idx = len(rows) - len(b_ids)
        for i in range(len(b_ids)):
            rows[base_idx + i]["out_tok_len"] = int(out_lens[i])
            rows[base_idx + i]["out_chars"] = int(len(dec[i]))
            rows[base_idx + i]["out_ws_tokens"] = int(len(dec[i].split()))
            rows[base_idx + i]["out_has_eos"] = bool((eos_id is not None) and (eos_id in out_list[i][: out_lens[i]]))
            rows[base_idx + i]["out_text"] = dec[i]  # might be long; keep if you save CSV

    dt = time.time() - t0

    # Print summaries
    src_raw_q = _quantiles(src_len_raw_all)
    src_used_q = _quantiles(src_len_used_all)
    print(
        "[src_tok_len] raw: "
        f"mean={src_raw_q['mean']:.1f} p50={src_raw_q['p50']:.0f} p90={src_raw_q['p90']:.0f} "
        f"p95={src_raw_q['p95']:.0f} p99={src_raw_q['p99']:.0f} max={src_raw_q['max']:.0f}"
    )
    print(
        "[src_tok_len] used(trunc+pad removed): "
        f"mean={src_used_q['mean']:.1f} p50={src_used_q['p50']:.0f} p90={src_used_q['p90']:.0f} "
        f"p95={src_used_q['p95']:.0f} p99={src_used_q['p99']:.0f} max={src_used_q['max']:.0f}"
    )
    print(f"[src_trunc] truncated_rows={n_trunc}/{len(ids)} ({n_trunc/max(1,len(ids)):.1%}) max_source_length={max_source_length}")

    if args.run_generate:
        out_q = _quantiles(out_len_all)
        print(
            "[out_tok_len] "
            f"mean={out_q['mean']:.1f} p50={out_q['p50']:.0f} p90={out_q['p90']:.0f} "
            f"p95={out_q['p95']:.0f} p99={out_q['p99']:.0f} max={out_q['max']:.0f}"
        )
        print(f"[out_eos] no_eos_rows={out_no_eos}/{len(ids)} ({out_no_eos/max(1,len(ids)):.1%})")
        print(f"[out_limit_hit] est_hit_rows={out_hit_limit}/{len(ids)} ({out_hit_limit/max(1,len(ids)):.1%})")
        print(f"[time] total_sec={dt:.1f} sec_per_1k_rows={dt/max(1,len(ids))/1000.0:.3f}")

    # Optional: compute metrics if ref column is provided
    if ref_col and ref_col in df.columns and args.run_generate:
        try:
            import sacrebleu  # type: ignore
            refs = df[ref_col].fillna("").astype(str).map(clean_text).tolist()
            bleu = sacrebleu.corpus_bleu(preds, [refs]).score
            chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
            gm = math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))
            print(f"[metrics] ref_col={ref_col} bleu={bleu:.4f} chrf={chrf:.4f} gm={gm:.4f}")
        except Exception as exc:
            print(f"[WARN] metrics failed: {exc}")

    # Save outputs
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[saved] stats_csv={out_path}")

    if args.pred_out and args.run_generate:
        pred_path = Path(args.pred_out)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": ids, "translation": preds}).to_csv(pred_path, index=False)
        print(f"[saved] pred_csv={pred_path}")


if __name__ == "__main__":
    main()
