"""NMT の n-best / sampling 推論（Oracle Upper Bound 用）。

Step 0（Oracle Upper Bound）では、1入力あたり 200〜500 の候補を作り、
参照（gold）に最も近い候補を選んだときにどこまで gm が伸び得るかを測ります。

このスクリプトは、ByT5 などの Seq2Seq モデルから複数候補を生成し、
1行=1候補の long 形式 CSV を出力します。

代表的な実行例
--------------

Beam n-best（決定的）:

  python -m dp.infer_nmt_nbest \
    --config configs/train_real.yaml \
    --ckpt artifacts/nmt/byt5_large \
    --test artifacts/ablation/variant=C/fold=0/val.parquet \
    --src-col src_sent --id-col id \
    --out artifacts/oracle/cands_beam64.csv \
    --k 64 --num-beams 64 --tag beam64

Sampling（確率的・複数 run）:

  python -m dp.infer_nmt_nbest \
    --config configs/train_real.yaml \
    --ckpt artifacts/nmt/byt5_large \
    --test artifacts/ablation/variant=C/fold=0/val.parquet \
    --src-col src_sent --id-col id \
    --out artifacts/oracle/cands_sample_t0.9_p0.95.csv \
    --do-sample --temperature 0.9 --top-p 0.95 \
    --k 32 --runs 4 --seed 42 --tag sample_t0.9_p0.95

出力 CSV（long 形式）
--------------------

- id: 入力行ID
- run: 生成 run（sampling の複数試行用。beam は常に 1）
- rank: run 内順位（1..k）
- tag: 任意のラベル（後で候補プールを結合するため）
- model: ckpt 識別子（複数 ckpt を同時に回す場合に区別）
- translation: 生成文

任意で forward の系列スコア（候補生成時の log-prob）を保存できます。
`dp.eval_rerank_methods` の noisy-channel rerank で
`--lambda-fwd` を使って合成スコア `rev + lambda*fwd` を試すときに便利です。

- seq_score: 生成文の 1 token あたり平均 log-prob（高いほど良い）
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd

from .align_train import normalize_transliteration
from .gloss_infer import add_gloss_args, maybe_apply_gloss
from .utils import clean_text, get_data_dir, load_config, set_seed


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


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _model_tag(path: Path) -> str:
    # artifacts/nmt/byt5_large -> byt5_large
    name = path.name
    if not name:
        return str(path).replace("/", "_")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate n-best candidates for Oracle Upper Bound.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--ckpt", default=None, help="Model directory path (single model).")
    parser.add_argument(
        "--ckpts",
        default=None,
        help="Comma-separated model directory paths. Each ckpt is decoded independently and appended.",
    )
    parser.add_argument("--test", default=None, help="CSV/Parquet path (val/test).")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    parser.add_argument("--out", required=True, help="Output candidates CSV (long format).")
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--id-col", default=None, help="ID column name.")
    parser.add_argument("--norm-variant", default=None, help="Normalize source with A/B/C.")

    parser.add_argument("--k", type=int, default=32, help="Candidates per run (num_return_sequences).")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs (useful for sampling).")
    parser.add_argument("--tag", default="nbest", help="Tag label stored in output for later merging.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (sampling only).")

    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--max-source-length", type=int, default=None, help="Max source length.")
    parser.add_argument("--max-target-length", type=int, default=None, help="Max target length.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens.")

    parser.add_argument("--num-beams", type=int, default=None, help="Beam size (beam n-best).")
    parser.add_argument("--length-penalty", type=float, default=None, help="Length penalty.")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None, help="No-repeat ngram size.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty.")

    # sampling
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top_p.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")

    parser.add_argument(
        "--save-forward-score",
        action="store_true",
        help="Save forward sequence score (avg log-prob per token) as 'seq_score'.",
    )

    parser.add_argument("--max-rows", type=int, default=None, help="Use first N rows.")
    add_gloss_args(parser)
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
        raise ValueError(f"input data must include {src_col}")

    if args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)

    ids = df[id_col].astype(str).tolist() if id_col in df.columns else [str(i) for i in range(len(df))]
    base_src_texts = df[src_col].fillna("").astype(str).tolist()
    if norm_variant:
        base_src_texts = [normalize_transliteration(text, norm_variant) for text in base_src_texts]
    else:
        base_src_texts = [clean_text(text) for text in base_src_texts]

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

    try:
        import torch  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError("transformers/torch が見つかりません。requirements.txt をインストールしてください。") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size or parse_int(cfg.get("infer_batch_size"), 4)
    max_source_length = args.max_source_length or parse_int(cfg.get("max_source_length"), 256)
    max_target_length = args.max_target_length or parse_int(cfg.get("generation_max_length"), 256)
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        raw = cfg.get("generation_max_new_tokens", cfg.get("max_new_tokens"))
        max_new_tokens = parse_optional_int(raw)
    use_max_new_tokens = max_new_tokens is not None and max_new_tokens > 0
    gen_limit = int(max_new_tokens) if use_max_new_tokens else int(max_target_length)

    k = int(args.k)
    runs = max(1, int(args.runs))
    do_sample = bool(args.do_sample)

    # beams
    num_beams = args.num_beams
    if num_beams is None:
        num_beams = parse_int(cfg.get("num_beams"), 4)
    num_beams = int(num_beams)
    if not do_sample and k > num_beams:
        print(f"[WARN] k({k}) > num_beams({num_beams}). auto-set num_beams={k}.")
        num_beams = k

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

    # sampling
    temperature = args.temperature
    if temperature is None:
        temperature = parse_float(cfg.get("temperature"), 1.0)
    top_p = args.top_p
    if top_p is None:
        top_p = parse_float(cfg.get("top_p"), 1.0)
    top_k = args.top_k
    if top_k is None:
        top_k = parse_int(cfg.get("top_k"), 0)

    # Reproducibility for sampling
    if args.seed is not None:
        set_seed(int(args.seed))
        try:
            torch.manual_seed(int(args.seed))
        except Exception:
            pass

    print("=== infer_nmt_nbest settings ===")
    print(
        f"rows={len(base_src_texts)} batch={batch_size} k={k} runs={runs} do_sample={do_sample} "
        f"src_len={max_source_length} gen_limit={gen_limit} use_max_new_tokens={use_max_new_tokens} "
        f"beams={num_beams} len_penalty={length_penalty} early_stopping={early_stopping} "
        f"no_repeat_ngram={no_repeat_ngram_size} rep_penalty={repetition_penalty} "
        f"temp={temperature} top_p={top_p} top_k={top_k} save_fwd_score={bool(args.save_forward_score)}"
    )

    rows_out: list[dict[str, Any]] = []

    for ckpt_dir in ckpt_dirs:
        model_name = _model_tag(ckpt_dir)
        tag = str(args.tag)
        full_tag = f"{tag}|{model_name}" if len(ckpt_dirs) > 1 else tag
        print(f"[load] model={ckpt_dir} tag={full_tag}")

        # 任意: 辞書グロスをソース末尾に付与（学習で使っているなら推論でも一致させる）
        src_texts = maybe_apply_gloss(
            base_src_texts,
            args=args,
            cfg=cfg,
            data_dir=data_dir,
            ckpt_dir=ckpt_dir,
        )

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_dir))

        # Align special token ids (safety)
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
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

        model.to(device)
        model.eval()

        with torch.no_grad():
            for run_idx in range(runs):
                if do_sample and args.seed is not None:
                    # run ごとに seed をずらす（同じ run が重複しにくいように）
                    seed = int(args.seed) + int(run_idx)
                    set_seed(seed)
                    try:
                        torch.manual_seed(seed)
                    except Exception:
                        pass

                # batch decode
                offset = 0
                for batch_ids, batch_src in zip(batch_iter(ids, batch_size), batch_iter(src_texts, batch_size)):
                    inputs = tokenizer(
                        batch_src,
                        max_length=max_source_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    gen_kwargs: dict[str, Any] = dict(
                        num_return_sequences=k,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        repetition_penalty=repetition_penalty,
                    )
                    if do_sample:
                        gen_kwargs.update(
                            dict(
                                do_sample=True,
                                temperature=float(temperature),
                                top_p=float(top_p),
                            )
                        )
                        if int(top_k) > 0:
                            gen_kwargs["top_k"] = int(top_k)
                        # sampling は beam を使わない（beam-sampling をやりたい場合は num_beams を明示）
                        gen_kwargs["num_beams"] = 1
                    else:
                        gen_kwargs.update(
                            dict(
                                num_beams=num_beams,
                                length_penalty=float(length_penalty),
                                early_stopping=early_stopping,
                            )
                        )

                    if use_max_new_tokens:
                        gen_kwargs["max_new_tokens"] = int(gen_limit)
                    else:
                        gen_kwargs["max_length"] = int(max_target_length)

                    if args.save_forward_score:
                        gen_out = model.generate(
                            **inputs,
                            **gen_kwargs,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                        outputs = gen_out.sequences
                    else:
                        outputs = model.generate(**inputs, **gen_kwargs)

                    # outputs: (batch*k, seq_len)
                    if not hasattr(outputs, "shape"):
                        raise RuntimeError("Unexpected generate() output type")

                    # Forward score (avg log-prob per token) if requested.
                    seq_scores: Optional[List[float]] = None
                    if args.save_forward_score:
                        try:
                            # まずは HF のヘルパーで系列スコア（平均 log-prob）を計算する。
                            # beam search では基本的にこれが動く想定（sampling でも version により動くことがある）。
                            beam_idx = getattr(gen_out, "beam_indices", None)
                            trans = model.compute_transition_scores(
                                sequences=gen_out.sequences,
                                scores=gen_out.scores,
                                beam_indices=beam_idx,
                                normalize_logits=True,
                            )

                            # Exclude decoder start token; count non-pad tokens.
                            pad = int(pad_id if pad_id is not None else 0)
                            seq = gen_out.sequences
                            if seq.size(1) <= 1:
                                lengths = torch.ones(seq.size(0), device=seq.device, dtype=torch.long)
                            else:
                                # NOTE: sampling + top-p/top-k の場合、
                                #   - 終了後のトークンのスコアが -inf になりやすい
                                #   - ( -inf * 0 ) が NaN になって合計が壊れる
                                # ため、まず「有効トークン位置の mask」を作り、mask で 0 埋めしてから合計する。
                                toks_full = seq[:, 1:]
                                mask = toks_full != pad
                                if eos_id is not None:
                                    eos = int(eos_id)
                                    eos_pos = toks_full == eos
                                    csum = eos_pos.cumsum(dim=1)
                                    before = csum == 0
                                    first_eos = eos_pos & (csum == 1)
                                    mask = mask & (before | first_eos)
                                lengths = mask.sum(dim=1).clamp(min=1)
                            # trans has shape (bsz*k, seq_len-1)
                            if trans.size(1) > 0:
                                # mask した位置は 0 にしてから合計（-inf*0 -> NaN を防ぐ）
                                if seq.size(1) <= 1:
                                    summed = torch.zeros(seq.size(0), device=seq.device, dtype=torch.float)
                                else:
                                    # 上で作った mask を使う（shape: (bsz*k, seq_len-1)）
                                    trans_masked = trans.masked_fill(~mask, 0.0)
                                    summed = trans_masked.sum(dim=1)
                            else:
                                summed = torch.zeros(seq.size(0), device=seq.device, dtype=torch.float)
                            avg = (summed / lengths).detach().float().cpu().tolist()
                            seq_scores = [float(x) for x in avg]
                        except Exception as exc:
                            # フォールバック: sampling/greedy（num_beams==1）の場合は、
                            # beam の追跡なしで gen_out.scores から token の log-prob を直接計算する。
                            try:
                                if int(gen_kwargs.get("num_beams", 1)) != 1:
                                    raise RuntimeError("fallback requires num_beams==1")
                                import torch.nn.functional as F  # type: ignore

                                pad = int(pad_id if pad_id is not None else 0)
                                seq = gen_out.sequences
                                scores = getattr(gen_out, "scores", None)
                                if scores is None or len(scores) == 0:
                                    raise RuntimeError("gen_out.scores is empty")

                                T = len(scores)
                                # 整列: scores[t] は sequences[:, t+1] の token を予測している想定
                                toks = seq[:, 1 : 1 + T]
                                if toks.size(1) < T:
                                    pad_extra = torch.full(
                                        (toks.size(0), T - toks.size(1)),
                                        pad,
                                        device=toks.device,
                                        dtype=toks.dtype,
                                    )
                                    toks = torch.cat([toks, pad_extra], dim=1)
                                elif toks.size(1) > T:
                                    toks = toks[:, :T]

                                step_logps = []
                                for t, step_scores in enumerate(scores):
                                    logp = F.log_softmax(step_scores, dim=-1)
                                    tok = toks[:, t].clamp(min=0)
                                    step_logps.append(logp.gather(1, tok.unsqueeze(1)).squeeze(1))
                                logps = torch.stack(step_logps, dim=1)

                                mask = toks != pad
                                if eos_id is not None:
                                    eos = int(eos_id)
                                    eos_pos = toks == eos
                                    csum = eos_pos.cumsum(dim=1)
                                    before = csum == 0
                                    first_eos = eos_pos & (csum == 1)
                                    mask = mask & (before | first_eos)
                                lengths = mask.sum(dim=1).clamp(min=1)
                                # mask した位置は 0 にしてから合計（-inf*0 -> NaN を防ぐ）
                                logps_masked = logps.masked_fill(~mask, 0.0)
                                summed = logps_masked.sum(dim=1)
                                avg = (summed / lengths).detach().float().cpu().tolist()
                                seq_scores = [float(x) for x in avg]
                            except Exception as exc2:
                                # seq_score なしでも候補自体は出力する（ただし警告は出す）。
                                print(f"[WARN] failed to compute forward seq_score: {type(exc).__name__}: {exc}")
                                print(f"[WARN] fallback seq_score also failed: {type(exc2).__name__}: {exc2}")
                                seq_scores = None

                    # safe decode: -100 が混ざる環境対策
                    if (outputs < 0).any():
                        safe = outputs.detach().cpu().numpy().copy()
                        safe[safe < 0] = int(pad_id if pad_id is not None else 0)
                        decoded = tokenizer.batch_decode(safe, skip_special_tokens=True)
                    else:
                        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    decoded = [clean_text(x) for x in decoded]

                    # group into (batch, k)
                    expected = len(batch_ids) * k
                    if len(decoded) != expected:
                        # HF はまれに early stop で数がズレるケースがある（ほぼ無い）
                        print(
                            f"[WARN] decoded size mismatch: got={len(decoded)} expected={expected}. "
                            "Falling back to min length."
                        )
                    take = min(len(decoded), expected)
                    if seq_scores is not None:
                        take = min(take, len(seq_scores))

                    for i, row_id in enumerate(batch_ids):
                        base = i * k
                        for r in range(k):
                            j = base + r
                            if j >= take:
                                break
                            rows_out.append(
                                {
                                    "id": row_id,
                                    "run": int(run_idx) + 1,
                                    "rank": int(r) + 1,
                                    "tag": full_tag,
                                    "model": model_name,
                                    "translation": decoded[j],
                                    **({"seq_score": float(seq_scores[j])} if seq_scores is not None else {}),
                                }
                            )

                    offset += len(batch_ids)

        # free VRAM
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_path, index=False)

    # quick sanity stats
    total = len(rows_out)
    uniq_ids = len(set(r["id"] for r in rows_out))
    per_id = total / max(1, uniq_ids)
    print(f"Saved candidates: {out_path} rows={total} uniq_ids={uniq_ids} avg_cands_per_id={per_id:.1f}")


if __name__ == "__main__":
    main()
