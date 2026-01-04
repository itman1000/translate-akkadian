"""NMT 用の推論スクリプト。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .align_train import normalize_transliteration
from .utils import clean_text, get_data_dir, load_config
from .nmt_ensemble import EnsembleGenConfig, ensemble_generate


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


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def log_texts_summary(texts: List[str], label: str) -> None:
    """推論テキストの簡易サマリを出力する。"""
    if not texts:
        print(f"[WARN] {label}: 0 texts")
        return
    series = pd.Series(texts).fillna("")
    empty_ratio = (series.str.strip() == "").mean()
    lens = series.str.len()
    tokens = series.str.split().map(len)
    q_len = lens.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    q_tok = tokens.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()

    def _control(text: str) -> bool:
        return any(ord(ch) < 32 for ch in text)

    def _unique_ratio(text: str) -> float:
        parts = text.split()
        if not parts:
            return 0.0
        return len(set(parts)) / len(parts)

    control_ratio = series.map(_control).mean()
    unique_ratios = series.map(_unique_ratio)
    low_unique_ratio = (unique_ratios < 0.5).mean()
    mean_unique_ratio = unique_ratios.mean()

    print(
        f"[{label}] count={len(series)} empty={empty_ratio:.1%} control={control_ratio:.1%} "
        f"len_med={q_len[0.5]:.0f} p90={q_len[0.9]:.0f} p95={q_len[0.95]:.0f} "
        f"tok_med={q_tok[0.5]:.0f} p90={q_tok[0.9]:.0f} "
        f"uniq_mean={mean_unique_ratio:.2f} uniq_low={low_unique_ratio:.1%}"
    )


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NMT inference.")
    parser.add_argument("--config", required=True, help="Config file path.")
    # Single checkpoint OR logits-averaging ensemble.
    # - single: --ckpt artifacts/nmt/byt5_small
    # - ensemble: --ckpts artifacts/nmt/m1,artifacts/nmt/m2,...
    parser.add_argument("--ckpt", default=None, help="Model directory path (single model).")
    parser.add_argument(
        "--ckpts",
        default=None,
        help="Comma-separated model directory paths for logits-averaging ensemble (overrides --ckpt).",
    )
    parser.add_argument("--test", default=None, help="Test CSV/Parquet path.")
    parser.add_argument("--out", required=True, help="Output predictions CSV path.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--id-col", default=None, help="ID column name.")
    parser.add_argument(
        "--ref-col",
        default=None,
        help="Optional reference column (e.g., 'translation') to compute BLEU/CHRF/GM for sanity-check.",
    )
    parser.add_argument("--norm-variant", default=None, help="Normalize source with A/B/C.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--max-source-length", type=int, default=None, help="Max source length.")
    parser.add_argument("--max-target-length", type=int, default=None, help="Max target length.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens.")
    parser.add_argument("--length-penalty", type=float, default=None, help="Length penalty.")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None, help="No-repeat ngram size.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty.")
    parser.add_argument("--num-beams", type=int, default=None, help="Beam size.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use first N rows.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, args.data_dir)

    test_path = Path(args.test) if args.test else data_dir / "test.csv"
    df = read_table(test_path)

    src_col = args.src_col or cfg.get("infer_src_col", cfg.get("src_col", "transliteration"))
    id_col = args.id_col or cfg.get("infer_id_col", "id")
    norm_variant = args.norm_variant or cfg.get("norm_variant")
    if norm_variant:
        norm_variant = str(norm_variant).upper()

    if src_col not in df.columns:
        raise ValueError(f"test data must include {src_col}")

    if args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)

    ids = df[id_col].astype(str).tolist() if id_col in df.columns else [str(i) for i in range(len(df))]
    src_texts = df[src_col].fillna("").astype(str).tolist()
    if norm_variant:
        src_texts = [normalize_transliteration(text, norm_variant) for text in src_texts]
    else:
        src_texts = [clean_text(text) for text in src_texts]

    try:
        import torch  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers/torch が見つかりません。requirements.txt をインストールしてください。"
        ) from exc

    # --- load model(s) ---
    ckpt_dirs: list[Path] = []
    if args.ckpts:
        ckpt_dirs = [Path(p.strip()) for p in str(args.ckpts).split(",") if p.strip()]
    else:
        cfg_ckpts = cfg.get("ensemble_ckpts") or cfg.get("ensemble_ckpt_dirs")
        if isinstance(cfg_ckpts, str) and cfg_ckpts.strip():
            ckpt_dirs = [Path(p.strip()) for p in cfg_ckpts.split(",") if p.strip()]
        elif isinstance(cfg_ckpts, (list, tuple)):
            ckpt_dirs = [Path(str(p)) for p in cfg_ckpts if str(p).strip()]

    if not ckpt_dirs:
        if args.ckpt is None:
            raise ValueError("--ckpt または --ckpts (もしくは config の ensemble_ckpts) を指定してください")
        ckpt_dirs = [Path(args.ckpt)]

    missing = [p for p in ckpt_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Model dir not found: {missing[0]}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dirs[0]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: list[Any] = []
    for p in ckpt_dirs:
        m = AutoModelForSeq2SeqLM.from_pretrained(str(p))
        # Align special token ids across checkpoints (important for generation).
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
        if pad_id is not None:
            m.config.pad_token_id = pad_id
            if getattr(m.config, "decoder_start_token_id", None) is None:
                m.config.decoder_start_token_id = pad_id
            if getattr(m, "generation_config", None) is not None:
                m.generation_config.pad_token_id = pad_id
                if getattr(m.generation_config, "decoder_start_token_id", None) is None:
                    m.generation_config.decoder_start_token_id = pad_id
        if eos_id is not None:
            m.config.eos_token_id = eos_id
            if getattr(m, "generation_config", None) is not None:
                m.generation_config.eos_token_id = eos_id
        m.to(device)
        m.eval()
        models.append(m)

    use_ensemble = len(models) > 1
    if use_ensemble:
        print(f"[ensemble] logits-avg: n_models={len(models)}")
    else:
        model = models[0]

    batch_size = args.batch_size or parse_int(cfg.get("infer_batch_size"), 8)
    max_source_length = args.max_source_length or parse_int(cfg.get("max_source_length"), 256)
    max_target_length = args.max_target_length or parse_int(cfg.get("generation_max_length"), 256)
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        raw = cfg.get("generation_max_new_tokens", cfg.get("max_new_tokens"))
        max_new_tokens = parse_optional_int(raw)
    use_max_new_tokens = max_new_tokens is not None and max_new_tokens > 0
    gen_limit = max_new_tokens if use_max_new_tokens else max_target_length
    num_beams = args.num_beams or parse_int(cfg.get("num_beams"), 4)
    length_penalty = args.length_penalty
    if length_penalty is None:
        raw = cfg.get("length_penalty")
        length_penalty = float(raw) if raw is not None else 1.0
    early_stopping = bool(args.early_stopping or cfg.get("early_stopping", False))
    no_repeat_ngram_size = args.no_repeat_ngram_size
    if no_repeat_ngram_size is None:
        raw = cfg.get("no_repeat_ngram_size")
        no_repeat_ngram_size = int(raw) if raw is not None else 0
    repetition_penalty = args.repetition_penalty
    if repetition_penalty is None:
        raw = cfg.get("repetition_penalty")
        repetition_penalty = float(raw) if raw is not None else 1.0

    print("=== 推論設定 ===")
    print(
        f"rows={len(src_texts)} batch={batch_size} src_len={max_source_length} tgt_len={max_target_length} "
        f"gen_limit={gen_limit} use_max_new_tokens={use_max_new_tokens} "
        f"beams={num_beams} length_penalty={length_penalty} early_stopping={early_stopping} "
        f"no_repeat_ngram_size={no_repeat_ngram_size} repetition_penalty={repetition_penalty}"
    )
    log_texts_summary(src_texts, "src")

    preds: List[str] = []
    gen_token_lens: List[int] = []
    eos_early = 0
    eos_missing = 0
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    base_model = models[0] if use_ensemble else model
    decoder_start_id = getattr(base_model.config, "decoder_start_token_id", None) or pad_id
    early_cutoff = min(32, max(8, int(gen_limit * 0.25)))
    with torch.no_grad():
        for batch in batch_iter(src_texts, batch_size):
            inputs = tokenizer(
                batch,
                max_length=max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gen_kwargs = dict(
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
            )
            if use_max_new_tokens:
                gen_kwargs["max_new_tokens"] = gen_limit
            else:
                gen_kwargs["max_length"] = max_target_length
            if use_ensemble:
                # Logits-averaging ensemble decode.
                gen_cfg = EnsembleGenConfig(
                    max_new_tokens=gen_limit,
                    num_beams=gen_kwargs.get("num_beams", 1),
                    length_penalty=gen_kwargs.get("length_penalty", 1.0),
                    no_repeat_ngram_size=gen_kwargs.get("no_repeat_ngram_size", 0),
                    repetition_penalty=gen_kwargs.get("repetition_penalty", 1.0),
                    pad_token_id=pad_id if pad_id is not None else 0,
                    eos_token_id=eos_id if eos_id is not None else 1,
                    decoder_start_token_id=decoder_start_id if decoder_start_id is not None else 0,
                )
                outputs = ensemble_generate(
                    models=models,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    gen_cfg=gen_cfg,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            if pad_id is not None:
                lens = (outputs != pad_id).sum(dim=1).tolist()
            else:
                lens = [len(seq) for seq in outputs.tolist()]
            gen_token_lens.extend(int(x) for x in lens)
            if eos_id is not None:
                for seq in outputs.tolist():
                    try:
                        pos = seq.index(eos_id)
                    except ValueError:
                        pos = None
                    if pos is None:
                        eos_missing += 1
                    elif pos + 1 <= early_cutoff:
                        eos_early += 1
            # まれに pad/gather の都合で -100 が混ざる環境があるため安全化
            # (ByT5Tokenizer は chr(id-offset) のため負値で落ちる)
            if (outputs < 0).any():
                safe_outputs = outputs.detach().cpu().numpy()
                safe_outputs = safe_outputs.copy()
                safe_outputs[safe_outputs < 0] = (pad_id if pad_id is not None else 0)
                decoded = tokenizer.batch_decode(safe_outputs, skip_special_tokens=True)
            else:
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds.extend([clean_text(text) for text in decoded])

    if gen_token_lens:
        series = pd.Series(gen_token_lens)
        q = series.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
        hit_ratio = (series >= gen_limit).mean()
        if eos_id is not None:
            eos_early_rate = eos_early / len(series)
            eos_missing_rate = eos_missing / len(series)
            print(
                "[gen_len] "
                f"limit={gen_limit} hit_max_new_tokens_rate={hit_ratio:.1%} "
                f"eos_early<= {early_cutoff}={eos_early_rate:.1%} "
                f"eos_missing={eos_missing_rate:.1%} max={series.max():.0f} "
                f"p50={q[0.5]:.0f} p90={q[0.9]:.0f} p95={q[0.95]:.0f} p99={q[0.99]:.0f}"
            )
        else:
            print(
                "[gen_len] "
                f"limit={gen_limit} hit_max_new_tokens_rate={hit_ratio:.1%} max={series.max():.0f} "
                f"p50={q[0.5]:.0f} p90={q[0.9]:.0f} p95={q[0.95]:.0f} p99={q[0.99]:.0f}"
            )
    log_texts_summary(preds, "pred")

    # Optional: compute BLEU/CHRF/GM if a reference column is present.
    ref_col = (
        args.ref_col
        or cfg.get("ref_col")
        or cfg.get("target_col")
        or cfg.get("tgt_col")
        or cfg.get("label_col")
    )
    if ref_col is None and "translation" in df.columns:
        ref_col = "translation"
    if ref_col is not None and ref_col in df.columns:
        try:
            from sacrebleu.metrics import BLEU, CHRF

            refs = [clean_text(str(x)) for x in df[ref_col].fillna("").tolist()]
            hyps = preds
            bleu = BLEU(tokenize="none").corpus_score(hyps, [refs]).score
            chrf = CHRF(word_order=2).corpus_score(hyps, [refs]).score
            gm = math.sqrt(bleu * chrf)
            tag = f"ensemble={len(models)}" if use_ensemble else "single"
            print(f"[eval_metrics][{tag}] bleu={bleu:.4f} chrf={chrf:.4f} gm={gm:.4f}")
        except Exception as e:
            print(f"[eval_metrics] skipped (failed to compute metrics): {e}")

    out_df = pd.DataFrame({"id": ids, "translation": preds})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
