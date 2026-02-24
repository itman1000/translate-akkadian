"""Prototype Editing (編集モデル) で候補を生成する。

改善策B（Prototype Editing）の Step 3。

手順:
 1) train 側の (src,tgt) から作った index（TF-IDF char n-gram）で topK のプロトタイプを取得
 2) editor 入力を <SRC>/<TM_SRC>/<TM_TGT> 形式で作る
 3) editor モデルで beam n-best もしくは sampling を回し、候補 CSV（long 形式）を出力

出力は `dp.eval_rerank_methods` と互換の long CSV:
  - id, run, rank, tag, model, translation
  - (optional) seq_score

リーク注意:
  - index は必ず train のみで作成（fold ごと）
  - val/test を index に混ぜない
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .align_train import normalize_transliteration
from .prepare_editor_data import TmIndex, build_editor_input, load_index
from .utils import clean_text, enforce_single_sentence, load_config, set_seed


def score_forward_logp(
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
    """Compute per-example forward scores ~ log p(target | input).

    Returns a list of scores (higher is better). We use negative average NLL.

    NOTE:
      - This is intentionally aligned with noisy-channel's reverse scorer
        (`dp.eval_rerank_methods.score_reverse_logp`).
      - Use this to re-score editor candidates with the *same* forward ckpt
        used for NMT candidates, so `seq_score` is comparable across tags.
    """

    try:
        import torch  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError("torch/transformers are required for forward rescoring") from exc

    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same length")

    torch_device = torch.device(device)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    dtype = torch.bfloat16 if bf16 and torch_device.type == "cuda" else None
    # transformers の版によっては `torch_dtype` より `dtype` が推奨されるため、
    # まず `dtype` を試し、ダメなら `torch_dtype` にフォールバック。
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


def _norm_src(text: str, norm_variant: Optional[str]) -> str:
    s = str(text) if text is not None else ""
    if norm_variant:
        return normalize_transliteration(s, norm_variant)
    return clean_text(s)


def _topk_retrieve(index: TmIndex, queries: List[str], topk: int) -> Tuple[List[List[int]], List[List[float]]]:
    """Return indices and cosine sims for each query."""
    X = index.X
    vec = index.vectorizer

    # TF-IDF vectors are L2 normalized by default, so dot == cosine
    Q = vec.transform(queries)
    sims_mat = Q @ X.T  # shape: (nq, n)
    # For each row, take topk
    # Note: sims_mat is sparse; convert row-wise.
    out_idx: List[List[int]] = []
    out_sim: List[List[float]] = []
    topk = max(1, int(topk))
    for i in range(sims_mat.shape[0]):
        row = sims_mat.getrow(i)
        if row.nnz == 0:
            out_idx.append([])
            out_sim.append([])
            continue
        # Get topk among nonzeros first
        data = row.data
        cols = row.indices
        if len(data) <= topk:
            order = data.argsort()[::-1]
            idxs = cols[order].tolist()
            sims = data[order].tolist()
        else:
            import numpy as np  # type: ignore

            part = np.argpartition(data, -topk)[-topk:]
            order = part[data[part].argsort()[::-1]]
            idxs = cols[order].tolist()
            sims = data[order].tolist()
        out_idx.append([int(x) for x in idxs])
        out_sim.append([float(x) for x in sims])
    return out_idx, out_sim


def batch_iter(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _model_tag(path: Path) -> str:
    name = path.name
    if not name:
        return str(path).replace("/", "_")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate editor (Prototype Editing) candidates in long CSV format.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--ckpt", required=True, help="Editor model checkpoint dir.")
    parser.add_argument(
        "--fwd-ckpt",
        default=None,
        help=(
            "(Recommended for strict noisy-channel) Forward model checkpoint dir used to teacher-force re-score "
            "editor outputs and write comparable 'seq_score'. If provided, the output will contain 'seq_score' "
            "computed by this forward model."
        ),
    )
    parser.add_argument("--index", required=True, help="Prototype index joblib (from prepare_editor_data or TM++ index).")
    parser.add_argument("--test", required=True, help="CSV/Parquet path (val/test).")
    parser.add_argument("--out", required=True, help="Output candidates CSV.")
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--id-col", default=None, help="ID column name.")
    parser.add_argument("--norm-variant", default=None, help="Normalize source with A/B/C.")

    parser.add_argument("--topk", type=int, default=10, help="Number of prototypes to retrieve per query.")
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.0,
        help="Skip prototypes with cosine similarity below this value.",
    )
    parser.add_argument("--tag", default="editor", help="Tag label stored in output.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for generation.")
    parser.add_argument("--k", type=int, default=4, help="Candidates per prototype per run.")
    parser.add_argument("--runs", type=int, default=1, help="Independent runs (sampling only).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    # generation
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)

    parser.add_argument("--max-source-length", type=int, default=None)
    parser.add_argument("--max-target-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)

    parser.add_argument("--save-forward-score", action="store_true", help="Save seq_score (avg log-prob per token).")
    parser.add_argument("--fwd-batch-size", type=int, default=None, help="Batch size for forward re-scoring.")
    parser.add_argument("--fwd-max-source-length", type=int, default=None, help="Max source length for forward re-scoring.")
    parser.add_argument("--fwd-max-target-length", type=int, default=None, help="Max target length for forward re-scoring.")
    parser.add_argument("--fwd-bf16", action="store_true", help="Use bf16 autocast for forward re-scoring (CUDA only).")
    parser.add_argument("--fwd-device", default=None, help="Device for forward re-scoring (e.g., cuda, cpu).")
    parser.add_argument("--force-single-sentence", action="store_true", help="Apply enforce_single_sentence postprocess.")
    parser.add_argument("--single-sentence-mode", default="merge", help="merge or truncate")
    parser.add_argument("--save-prototype-cols", action="store_true", help="Include tm_src/tm_tgt/tm_sim in output rows.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Keep consistent with NMT inference: args overrides config.
    norm_variant = args.norm_variant or cfg.get("norm_variant")
    if norm_variant:
        norm_variant = str(norm_variant).upper()

    index = load_index(Path(args.index))
    if index.norm_variant is None:
        index.norm_variant = norm_variant

    df = read_table(Path(args.test))
    src_col = args.src_col or cfg.get("infer_src_col", cfg.get("src_col", "transliteration"))
    id_col = args.id_col or cfg.get("infer_id_col", "id")
    if src_col not in df.columns:
        raise ValueError(f"input data must include {src_col}")

    ids = df[id_col].astype(str).tolist() if id_col in df.columns else [str(i) for i in range(len(df))]
    src_texts_raw = df[src_col].fillna("").astype(str).tolist()
    src_texts = [_norm_src(s, index.norm_variant) for s in src_texts_raw]

    # For forward re-scoring we need per-id source (normalized the same way as NMT inference).
    id_to_src_for_score = {str(i): str(s) for i, s in zip(ids, src_texts)}

    # Retrieve prototypes
    topk = max(1, int(args.topk))
    proto_idx, proto_sim = _topk_retrieve(index, src_texts, topk=topk)

    # Build editor inputs (flatten)
    flat_inputs: List[str] = []
    flat_meta: List[Dict[str, Any]] = []
    min_sim = float(args.min_sim)
    for row_i, row_id in enumerate(ids):
        gold_src = clean_text(src_texts_raw[row_i])
        if not gold_src:
            continue
        idxs = proto_idx[row_i]
        sims = proto_sim[row_i]
        for rank0, (nb_idx, sim) in enumerate(zip(idxs, sims)):
            if float(sim) < min_sim:
                continue
            tm_src = clean_text(str(index.meta.loc[int(nb_idx), "src"]))
            tm_tgt = clean_text(str(index.meta.loc[int(nb_idx), "tgt"]))
            if not tm_src or not tm_tgt:
                continue
            editor_in = build_editor_input(gold_src, tm_src, tm_tgt)
            flat_inputs.append(editor_in)
            flat_meta.append(
                {
                    "id": row_id,
                    "proto_rank": int(rank0) + 1,
                    "tm_sim": float(sim),
                    "tm_src": tm_src,
                    "tm_tgt": tm_tgt,
                }
            )

    if not flat_inputs:
        raise RuntimeError("No editor inputs were created. Check index/test data.")

    try:
        import torch  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError("transformers/torch が見つかりません。requirements.txt をインストールしてください。") from exc

    ckpt_dir = Path(args.ckpt)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Editor ckpt dir not found: {ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_dir)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

    # Ensure special tokens exist (safe if already)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<SRC>", "<TM_SRC>", "<TM_TGT>"]})
    if getattr(model, "resize_token_embeddings", None) is not None:
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = pad_id
    if eos_id is not None and getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = eos_id
    if pad_id is not None and getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = pad_id

    # generation params
    batch_size = int(args.batch_size) if args.batch_size else parse_int(cfg.get("infer_batch_size"), 4)
    max_source_length = int(args.max_source_length) if args.max_source_length else parse_int(cfg.get("max_source_length"), 384)
    max_target_length = int(args.max_target_length) if args.max_target_length else parse_int(cfg.get("generation_max_length"), 256)
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        raw = cfg.get("generation_max_new_tokens", cfg.get("max_new_tokens"))
        if raw is not None:
            try:
                max_new_tokens = int(raw)
            except Exception:
                max_new_tokens = None

    use_max_new_tokens = max_new_tokens is not None and int(max_new_tokens) > 0
    gen_limit = int(max_new_tokens) if use_max_new_tokens else int(max_target_length)

    k = max(1, int(args.k))
    runs = max(1, int(args.runs))
    do_sample = bool(args.do_sample)

    num_beams = args.num_beams
    if num_beams is None:
        num_beams = parse_int(cfg.get("num_beams"), 4)
    num_beams = int(num_beams)
    if not do_sample and k > num_beams:
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

    temperature = args.temperature
    if temperature is None:
        raw = cfg.get("temperature")
        temperature = float(raw) if raw is not None else 1.0
    top_p = args.top_p
    if top_p is None:
        raw = cfg.get("top_p")
        top_p = float(raw) if raw is not None else 1.0
    top_k = args.top_k
    if top_k is None:
        raw = cfg.get("top_k")
        try:
            top_k = int(raw) if raw is not None else 0
        except Exception:
            top_k = 0

    seed = args.seed if args.seed is not None else cfg.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except Exception:
            seed = None

    model_name = _model_tag(ckpt_dir)
    full_tag = str(args.tag)

    rows_out: List[Dict[str, Any]] = []

    # Flattened inputs are independent; we generate for each and then attach meta
    # For sampling with multiple runs, we simply re-run generate() with different seeds.
    for run_idx in range(runs):
        if do_sample and seed is not None:
            set_seed(int(seed) + int(run_idx))
        elif seed is not None and run_idx == 0:
            set_seed(int(seed))

        offset = 0
        for batch_inputs in batch_iter(flat_inputs, batch_size=batch_size):
            batch_meta = flat_meta[offset : offset + len(batch_inputs)]
            offset += len(batch_inputs)

            enc = tokenizer(
                batch_inputs,
                max_length=max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            gen_kwargs: Dict[str, Any] = dict(
                num_return_sequences=k,
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                repetition_penalty=float(repetition_penalty),
            )
            if do_sample:
                gen_kwargs.update(
                    dict(
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        num_beams=1,
                    )
                )
                if int(top_k) > 0:
                    gen_kwargs["top_k"] = int(top_k)
            else:
                gen_kwargs.update(
                    dict(
                        num_beams=int(num_beams),
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
                    **enc,
                    **gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                outputs = gen_out.sequences
            else:
                outputs = model.generate(**enc, **gen_kwargs)
                gen_out = None

            seq_scores: Optional[List[float]] = None
            if args.save_forward_score and gen_out is not None:
                try:
                    beam_idx = getattr(gen_out, "beam_indices", None)
                    trans = model.compute_transition_scores(
                        sequences=gen_out.sequences,
                        scores=gen_out.scores,
                        beam_indices=beam_idx,
                        normalize_logits=True,
                    )
                    pad = int(pad_id if pad_id is not None else 0)
                    seq = gen_out.sequences
                    if seq.size(1) <= 1:
                        lengths = torch.ones(seq.size(0), device=seq.device, dtype=torch.long)
                        mask = None
                    else:
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

                    if trans.size(1) > 0:
                        if mask is None:
                            summed = trans.sum(dim=1)
                        else:
                            summed = trans.masked_fill(~mask, 0.0).sum(dim=1)
                    else:
                        summed = torch.zeros(seq.size(0), device=seq.device, dtype=torch.float)
                    avg = (summed / lengths).detach().float().cpu().tolist()
                    seq_scores = [float(x) for x in avg]
                except Exception as exc:
                    # Best-effort: don't fail inference
                    print(f"[WARN] failed to compute seq_score: {type(exc).__name__}: {exc}")
                    seq_scores = None

            # Decode
            if hasattr(outputs, "detach") and (outputs < 0).any():
                safe = outputs.detach().cpu().numpy().copy()
                safe[safe < 0] = int(pad_id if pad_id is not None else 0)
                decoded = tokenizer.batch_decode(safe, skip_special_tokens=True)
            else:
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded = [clean_text(x) for x in decoded]

            if args.force_single_sentence or bool(cfg.get("force_single_sentence", False)):
                mode = str(args.single_sentence_mode or cfg.get("single_sentence_mode", "merge"))
                decoded = [enforce_single_sentence(x, mode=mode) for x in decoded]

            expected = len(batch_inputs) * k
            if len(decoded) != expected:
                take = min(len(decoded), expected)
            else:
                take = expected
            if seq_scores is not None:
                take = min(take, len(seq_scores))

            # Attach
            for bi in range(len(batch_inputs)):
                base = bi * k
                meta = batch_meta[bi]
                for r in range(k):
                    j = base + r
                    if j >= take:
                        break
                    run_no = (int(meta["proto_rank"]) - 1) * runs + (run_idx + 1)
                    row: Dict[str, Any] = {
                        "id": meta["id"],
                        "run": int(run_no),
                        "rank": int(r) + 1,
                        "tag": full_tag,
                        "model": model_name,
                        "translation": decoded[j],
                    }
                    if seq_scores is not None:
                        row["seq_score"] = float(seq_scores[j])
                    if args.save_prototype_cols:
                        row.update(
                            {
                                "tm_rank": int(meta["proto_rank"]),
                                "tm_sim": float(meta["tm_sim"]),
                                "tm_src": meta["tm_src"],
                                "tm_tgt": meta["tm_tgt"],
                            }
                        )
                    rows_out.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows_out)

    # Optional: teacher-forcing re-score with a separate forward ckpt
    if args.fwd_ckpt:
        fwd_dir = Path(str(args.fwd_ckpt))
        if not fwd_dir.exists():
            raise FileNotFoundError(f"Forward ckpt dir not found: {fwd_dir}")

        # Preserve editor's own seq_score (if computed) for debugging.
        if "seq_score" in out_df.columns:
            out_df = out_df.rename(columns={"seq_score": "editor_seq_score"})

        # Free editor model memory before loading the forward model (helps avoid OOM on GPU).
        try:
            del model
        except Exception:
            pass
        try:
            if "torch" in globals():
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore

        if args.fwd_device:
            fwd_device = str(args.fwd_device)
        else:
            if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                fwd_device = "cuda"
            else:
                fwd_device = "cpu"

        fwd_batch = int(args.fwd_batch_size) if args.fwd_batch_size else int(batch_size)
        fwd_max_src = int(args.fwd_max_source_length) if args.fwd_max_source_length else int(parse_int(cfg.get("max_source_length"), 384))
        # For teacher-forcing, we want to avoid truncating target candidates.
        # Use config's max_target_length if present; otherwise fall back to generation_max_length.
        if args.fwd_max_target_length:
            fwd_max_tgt = int(args.fwd_max_target_length)
        else:
            raw_tgt = cfg.get("max_target_length", cfg.get("generation_max_length", cfg.get("max_source_length", 384)))
            fwd_max_tgt = int(parse_int(raw_tgt, 384))

        fwd_bf16 = bool(args.fwd_bf16 or cfg.get("bf16", False))

        srcs = [id_to_src_for_score.get(str(i), "") for i in out_df["id"].astype(str).tolist()]
        tgts = out_df["translation"].fillna("").astype(str).tolist()
        print(
            f"[fwd_rescore] scoring {len(out_df)} candidates with forward ckpt='{fwd_dir}' "
            f"device={fwd_device} batch={fwd_batch} max_src={fwd_max_src} max_tgt={fwd_max_tgt} bf16={fwd_bf16}"
        )
        out_df["seq_score"] = score_forward_logp(
            model_dir=fwd_dir,
            inputs=srcs,
            targets=tgts,
            batch_size=int(fwd_batch),
            max_source_length=int(fwd_max_src),
            max_target_length=int(fwd_max_tgt),
            bf16=bool(fwd_bf16),
            device=str(fwd_device),
        )

    out_df.to_csv(out_path, index=False)

    total = len(rows_out)
    uniq_ids = len(set(r["id"] for r in rows_out))
    per_id = total / max(1, uniq_ids)
    print(f"Saved editor candidates: {out_path} rows={total} uniq_ids={uniq_ids} avg_cands_per_id={per_id:.1f}")


if __name__ == "__main__":
    main()
