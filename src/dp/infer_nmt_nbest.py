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
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd

from .align_train import normalize_transliteration
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
        raise ValueError(f"input data must include {src_col}")

    if args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)

    ids = df[id_col].astype(str).tolist() if id_col in df.columns else [str(i) for i in range(len(df))]
    src_texts = df[src_col].fillna("").astype(str).tolist()
    if norm_variant:
        src_texts = [normalize_transliteration(text, norm_variant) for text in src_texts]
    else:
        src_texts = [clean_text(text) for text in src_texts]

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
        f"rows={len(src_texts)} batch={batch_size} k={k} runs={runs} do_sample={do_sample} "
        f"src_len={max_source_length} gen_limit={gen_limit} use_max_new_tokens={use_max_new_tokens} "
        f"beams={num_beams} len_penalty={length_penalty} early_stopping={early_stopping} "
        f"no_repeat_ngram={no_repeat_ngram_size} rep_penalty={repetition_penalty} "
        f"temp={temperature} top_p={top_p} top_k={top_k}"
    )

    rows_out: list[dict[str, Any]] = []

    for ckpt_dir in ckpt_dirs:
        model_name = _model_tag(ckpt_dir)
        tag = str(args.tag)
        full_tag = f"{tag}|{model_name}" if len(ckpt_dirs) > 1 else tag
        print(f"[load] model={ckpt_dir} tag={full_tag}")

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

                    outputs = model.generate(**inputs, **gen_kwargs)
                    # outputs: (batch*k, seq_len)
                    if not hasattr(outputs, "shape"):
                        raise RuntimeError("Unexpected generate() output type")

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
