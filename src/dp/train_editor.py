"""Prototype Editing (編集モデル) の学習。

改善策B（Prototype Editing）の Step 2。

`dp.prepare_editor_data` が出力する (editor_input, tgt) 形式のデータから
Seq2Seq editor を学習する。

入力:  <SRC> ... <TM_SRC> ... <TM_TGT> ...
出力:  gold English (tgt)

基本的には `dp.train_nmt` の簡易版で、gloss augmentation 等は行わない。
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .utils import clean_text, get_artifacts_dir, load_config, now_id, set_seed


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


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if val_ratio <= 0 or len(df) < 2:
        return df.reset_index(drop=True), None
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(math.ceil(len(df) * val_ratio)))
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df


def estimate_total_steps(train_len: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    denom = max(1, batch_size * max(1, grad_accum))
    steps_per_epoch = max(1, int(math.ceil(train_len / denom)))
    return max(1, int(steps_per_epoch * max(1.0, float(epochs))))


def filter_training_args(
    kwargs: Dict[str, Any],
    supported: set[str],
    *,
    has_val: bool,
    total_steps: int,
) -> Dict[str, Any]:
    kwargs = dict(kwargs)

    # transformers のバージョン差分: evaluation_strategy -> eval_strategy
    if "evaluation_strategy" not in supported and "eval_strategy" in supported:
        if "evaluation_strategy" in kwargs and "eval_strategy" not in kwargs:
            kwargs["eval_strategy"] = kwargs.get("evaluation_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in supported}

    if "evaluation_strategy" not in supported and "eval_strategy" not in supported:
        if "evaluate_during_training" in supported:
            filtered["evaluate_during_training"] = bool(has_val)
        if "do_eval" in supported and has_val:
            filtered["do_eval"] = True

    if "warmup_ratio" not in supported and "warmup_steps" in supported:
        warmup_ratio = kwargs.get("warmup_ratio", 0.0) or 0.0
        filtered["warmup_steps"] = int(float(warmup_ratio) * max(1, int(total_steps)))

    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Prototype Editing (editor) seq2seq model.")
    parser.add_argument("--config", required=True, help="Config file path (reuse NMT config is OK).")
    parser.add_argument("--train", required=True, help="Editor training data (csv/parquet).")
    parser.add_argument("--src-col", default="editor_input", help="Input column name.")
    parser.add_argument("--tgt-col", default="tgt", help="Target column name.")
    parser.add_argument("--model-name-or-path", default=None, help="Base model (e.g., google/byt5-small).")
    parser.add_argument("--out-dir", default=None, help="Output directory for the trained editor model.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")

    # Lengths / batches
    parser.add_argument("--max-source-length", type=int, default=None, help="Max input length.")
    parser.add_argument("--max-target-length", type=int, default=None, help="Max target length.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)

    # Optim
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)

    # Precision / memory
    parser.add_argument("--bf16", action="store_true", help="Use bf16 if available.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 if available.")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    # Logging
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    train_path = Path(args.train)
    df = read_table(train_path)

    src_col = str(args.src_col)
    tgt_col = str(args.tgt_col)
    if src_col not in df.columns or tgt_col not in df.columns:
        raise ValueError(f"train data must include columns: {src_col}, {tgt_col} (columns={list(df.columns)})")

    # Clean
    df = df.copy()
    df[src_col] = df[src_col].fillna("").astype(str).map(clean_text)
    df[tgt_col] = df[tgt_col].fillna("").astype(str).map(clean_text)
    df = df[(df[src_col].str.strip() != "") & (df[tgt_col].str.strip() != "")].reset_index(drop=True)

    val_ratio = float(args.val_ratio)
    train_df, val_df = split_train_val(df[[src_col, tgt_col]], val_ratio=val_ratio, seed=seed)

    try:
        import numpy as np  # type: ignore
        import sacrebleu  # type: ignore
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
        raise ImportError("transformers/torch/datasets/sacrebleu が必要です。requirements.txt を確認してください。") from exc

    hf_set_seed(seed)

    model_name_or_path = args.model_name_or_path or cfg.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("--model-name-or-path (or config:model_name_or_path) is required")

    # Output dir
    exp = str(cfg.get("experiment_name", "editor"))
    out_dir = Path(args.out_dir) if args.out_dir else get_artifacts_dir(cfg) / "editor" / now_id(exp)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Add editor special tokens (safe even if tokenizer is byte-level)
    special = ["<SRC>", "<TM_SRC>", "<TM_TGT>"]
    add_res = tokenizer.add_special_tokens({"additional_special_tokens": special})

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=model_config)
    if add_res and int(add_res) > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Safety: pad/eos ids
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = pad_id
    if eos_id is not None and getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = eos_id
    if pad_id is not None and getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = pad_id
    if getattr(model, "generation_config", None) is not None:
        if pad_id is not None and getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = pad_id
        if eos_id is not None and getattr(model.generation_config, "eos_token_id", None) is None:
            model.generation_config.eos_token_id = eos_id
        if pad_id is not None and getattr(model.generation_config, "decoder_start_token_id", None) is None:
            model.generation_config.decoder_start_token_id = pad_id

    max_source_length = int(args.max_source_length) if args.max_source_length else parse_int(cfg.get("max_source_length"), 384)
    max_target_length = int(args.max_target_length) if args.max_target_length else parse_int(cfg.get("max_target_length"), 384)

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        inputs = tokenizer(batch[src_col], max_length=max_source_length, truncation=True)
        labels = tokenizer(text_target=batch[tgt_col], max_length=max_target_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(preprocess, batched=True, remove_columns=[src_col, tgt_col])
    eval_ds = None
    if val_df is not None and len(val_df) > 0:
        eval_ds = Dataset.from_pandas(val_df, preserve_index=False).map(preprocess, batched=True, remove_columns=[src_col, tgt_col])

    per_device_train_batch_size = int(args.per_device_train_batch_size) if args.per_device_train_batch_size else parse_int(cfg.get("per_device_train_batch_size"), 4)
    per_device_eval_batch_size = int(args.per_device_eval_batch_size) if args.per_device_eval_batch_size else parse_int(cfg.get("per_device_eval_batch_size"), 4)
    grad_accum = int(args.gradient_accumulation_steps) if args.gradient_accumulation_steps else parse_int(cfg.get("gradient_accumulation_steps"), 4)

    lr = float(args.learning_rate) if args.learning_rate else parse_float(cfg.get("learning_rate"), 5e-4)
    epochs = float(args.num_train_epochs) if args.num_train_epochs else parse_float(cfg.get("num_train_epochs"), 3.0)
    warmup_ratio = float(args.warmup_ratio) if args.warmup_ratio is not None else parse_float(cfg.get("warmup_ratio"), 0.0)
    weight_decay = float(args.weight_decay) if args.weight_decay is not None else parse_float(cfg.get("weight_decay"), 0.0)
    max_grad_norm = float(args.max_grad_norm) if args.max_grad_norm is not None else parse_float(cfg.get("max_grad_norm"), 1.0)

    # Generation config for eval
    gen_max = parse_int(cfg.get("generation_max_length"), max_target_length)

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = eval_pred.label_ids
        pad = int(pad_id if pad_id is not None else 0)

        preds = np.where(preds < 0, pad, preds)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels < 0, pad, labels)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [clean_text(x) for x in decoded_preds]
        decoded_labels = [clean_text(x) for x in decoded_labels]

        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2).score
        gm = math.sqrt(max(0.0, bleu) * max(0.0, chrf))
        return {"bleu": float(bleu), "chrf": float(chrf), "gm": float(gm)}

    use_bf16 = bool(args.bf16 or cfg.get("bf16", False))
    use_fp16 = bool(args.fp16 or cfg.get("fp16", False))
    grad_ckpt = bool(args.gradient_checkpointing or cfg.get("gradient_checkpointing", False))
    if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    total_steps = estimate_total_steps(
        train_len=len(train_df),
        batch_size=per_device_train_batch_size,
        grad_accum=grad_accum,
        epochs=epochs,
    )
    training_args_kwargs: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "eval_steps": int(args.eval_steps),
        "evaluation_strategy": "steps" if eval_ds is not None else "no",
        "predict_with_generate": True,
        "generation_max_length": int(gen_max),
        "save_total_limit": 2,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "report_to": [],
    }
    supported_keys = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters.keys())
    training_args_kwargs = filter_training_args(
        training_args_kwargs,
        supported_keys,
        has_val=eval_ds is not None,
        total_steps=total_steps,
    )
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    print(
        "=== train_editor ===\n"
        f"model={model_name_or_path}\n"
        f"train_rows={len(train_df)} val_rows={0 if val_df is None else len(val_df)}\n"
        f"max_source_length={max_source_length} max_target_length={max_target_length}\n"
        f"batch={per_device_train_batch_size} accum={grad_accum} lr={lr} epochs={epochs} bf16={use_bf16} fp16={use_fp16}\n"
        f"out_dir={out_dir}"
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save a small training manifest
    manifest = {
        "train_file": str(train_path),
        "src_col": src_col,
        "tgt_col": tgt_col,
        "model_name_or_path": str(model_name_or_path),
        "seed": seed,
        "val_ratio": val_ratio,
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
        "special_tokens": ["<SRC>", "<TM_SRC>", "<TM_TGT>"],
    }
    (out_dir / "editor_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved editor model: {out_dir}")


if __name__ == "__main__":
    main()
