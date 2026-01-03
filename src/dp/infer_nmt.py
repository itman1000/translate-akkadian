"""NMT 用の推論スクリプト。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .align_train import normalize_transliteration
from .gloss import (
    DEFAULT_STOP_LEMMAS,
    GlossAugmentConfig,
    build_gloss_augmenter,
    build_lemma_frequency,
)
from .utils import clean_text, enforce_single_sentence, get_data_dir, load_config, normalize_translation_output


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
    parser.add_argument("--ckpt", required=True, help="Model directory path.")
    parser.add_argument("--test", default=None, help="Test CSV/Parquet path.")
    parser.add_argument("--out", required=True, help="Output predictions CSV path.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--id-col", default=None, help="ID column name.")
    parser.add_argument("--norm-variant", default=None, help="Normalize source with A/B/C.")

    # Submission safety: force single-sentence output.
    parser.add_argument(
        "--no-force-single-sentence",
        action="store_true",
        help="Disable single-sentence postprocess for predictions.",
    )
    parser.add_argument(
        "--single-sentence-mode",
        default=None,
        choices=["merge", "truncate"],
        help="How to force one-sentence output: merge (default) or truncate.",
    )
    parser.add_argument(
        "--normalize-output",
        action="store_true",
        help="Enable output normalization (fractions/spacing).",
    )
    parser.add_argument(
        "--no-normalize-output",
        action="store_true",
        help="Disable output normalization even if config enables it.",
    )

    # Decode preset (helpful when config is minimal).
    parser.add_argument(
        "--decode-preset",
        default=None,
        choices=["cfg", "greedy", "beam2_free", "beam4_free"],
        help=(
            "Decoding preset. 'cfg' uses config/CLI values (recommended; matches train_nmt beam2_cfg). "
            "Other presets override beams/penalties for quick comparison."
        ),
    )

    # Optional: dictionary gloss hints appended to source text.
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
            "If set, only keep gloss hints for lemmas whose corpus frequency <= N. "
            "In inference, loads <ckpt>/gloss_lemma_freq.json if present; otherwise computes from input."
        ),
    )
    parser.add_argument(
        "--gloss-lemma-freq",
        default=None,
        help="Optional path to gloss_lemma_freq.json (overrides <ckpt>/gloss_lemma_freq.json).",
    )
    parser.add_argument(
        "--gloss-min-lemma-chars",
        type=int,
        default=None,
        help="Minimum lemma length (chars) to allow in gloss hints (default: 2).",
    )

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

    # Optional: append gloss hints to the source text.
    gloss_enabled = bool(
        args.use_gloss
        or bool(cfg.get("use_gloss", False))
        or bool(cfg.get("gloss_enabled", False))
    )

    if gloss_enabled:
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

        match_types_value = (
            args.gloss_match_types
            if args.gloss_match_types is not None
            else cfg.get("gloss_match_types")
        )
        match_types = _parse_list(match_types_value) or ["word"]

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

        lemma_freq: Optional[Dict[str, int]] = None
        if gloss_cfg.max_lemma_freq is not None:
            # Prefer a precomputed table (train_nmt saves it).
            freq_path_value = args.gloss_lemma_freq
            if not freq_path_value:
                # auto: <ckpt>/gloss_lemma_freq.json
                ckpt_dir = Path(args.ckpt)
                cand = ckpt_dir / "gloss_lemma_freq.json"
                if cand.exists():
                    freq_path_value = str(cand)

            if freq_path_value:
                freq_path = Path(str(freq_path_value))
                if not freq_path.is_absolute():
                    cand = data_dir / freq_path
                    if cand.exists():
                        freq_path = cand
                try:
                    raw = json.loads(freq_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        lemma_freq = {str(k): int(v) for k, v in raw.items()}
                        print(f"[gloss] loaded lemma_freq: {freq_path} (size={len(lemma_freq)})")
                except Exception as exc:
                    print(f"[WARN] failed to load gloss lemma_freq: {freq_path} ({exc})")
                    lemma_freq = None

            if lemma_freq is None:
                # Fallback: compute from the inference inputs themselves.
                lemma_freq = build_lemma_frequency(
                    src_texts,
                    oa_lexicon_path=oa_lexicon_path,
                    match_columns=gloss_cfg.match_columns,
                    match_types=gloss_cfg.match_types,
                    max_match_len=gloss_cfg.max_match_len,
                )
                print(
                    f"[gloss] lemma_freq computed from input: size={len(lemma_freq)} max_lemma_freq={gloss_cfg.max_lemma_freq}"
                )

        gloss_augment = build_gloss_augmenter(
            gloss_cfg,
            oa_lexicon_path=oa_lexicon_path,
            ebl_dictionary_path=ebl_dictionary_path,
            lemma_freq=lemma_freq,
        )
        src_texts = [gloss_augment(text) for text in src_texts]
        print(
            "[gloss] enabled: "
            f"oa_lexicon={oa_lexicon_path} ebl_dictionary={ebl_dictionary_path} "
            f"types={gloss_cfg.match_types} max_match_len={gloss_cfg.max_match_len} "
            f"stop_lemmas={len(gloss_cfg.stop_lemmas)} max_lemma_freq={gloss_cfg.max_lemma_freq} "
            f"max_hints={gloss_cfg.max_hints} max_total_chars={gloss_cfg.max_total_chars}"
        )
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers/torch が見つかりません。requirements.txt をインストールしてください。"
        ) from exc

    model_dir = Path(args.ckpt)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def _cfg_first(*keys: str, default: Any = None) -> Any:
        for k in keys:
            if k in cfg and cfg.get(k) is not None:
                return cfg.get(k)
        return default

    batch_size = args.batch_size or parse_int(_cfg_first("infer_batch_size", default=8), 8)
    max_source_length = args.max_source_length or parse_int(_cfg_first("max_source_length", default=256), 256)
    max_target_length = args.max_target_length or parse_int(_cfg_first("generation_max_length", "max_target_length", default=256), 256)
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        raw = cfg.get("generation_max_new_tokens", cfg.get("max_new_tokens"))
        max_new_tokens = parse_optional_int(raw)
    use_max_new_tokens = max_new_tokens is not None and max_new_tokens > 0
    gen_limit = max_new_tokens if use_max_new_tokens else max_target_length
    # Defaults are tuned to the best-performing 'beam2_cfg' decoding from train_nmt.
    num_beams = args.num_beams or parse_int(_cfg_first("num_beams", "generation_num_beams", default=2), 2)
    length_penalty = args.length_penalty
    if length_penalty is None:
        raw = _cfg_first("length_penalty")
        length_penalty = float(raw) if raw is not None else 0.8
    early_stopping = bool(args.early_stopping or cfg.get("early_stopping", False))
    no_repeat_ngram_size = args.no_repeat_ngram_size
    if no_repeat_ngram_size is None:
        raw = _cfg_first("no_repeat_ngram_size")
        no_repeat_ngram_size = int(raw) if raw is not None else 20
    repetition_penalty = args.repetition_penalty
    if repetition_penalty is None:
        raw = _cfg_first("repetition_penalty")
        repetition_penalty = float(raw) if raw is not None else 1.15

    # Optional decode preset override.
    decode_preset = str(args.decode_preset or _cfg_first("decode_preset", default="cfg")).strip().lower()
    if decode_preset == "greedy":
        num_beams = 1
        length_penalty = 1.0
        no_repeat_ngram_size = 0
        repetition_penalty = 1.0
    elif decode_preset == "beam2_free":
        num_beams = 2
        length_penalty = 1.0
        no_repeat_ngram_size = 0
        repetition_penalty = 1.0
    elif decode_preset == "beam4_free":
        num_beams = 4
        length_penalty = 1.0
        no_repeat_ngram_size = 0
        repetition_penalty = 1.0

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
            preds.extend(decoded)

    # Text postprocess (normalize + optional single-sentence enforcement)
    normalize_output = bool(cfg.get("normalize_output", False))
    if args.normalize_output:
        normalize_output = True
    if args.no_normalize_output:
        normalize_output = False
    normalize_fractions = bool(cfg.get("normalize_output_fractions", True))
    normalize_units = bool(cfg.get("normalize_output_units", True))
    if normalize_output:
        preds_pp = [
            normalize_translation_output(
                text,
                normalize_fractions=normalize_fractions,
                normalize_units=normalize_units,
            )
            for text in preds
        ]
        changed = sum(int(a != b) for a, b in zip(preds_pp, preds))
        changed_rate = changed / max(1, len(preds))
        preds = preds_pp
        print(
            "[postprocess] "
            f"normalize_output=True changed={changed_rate:.1%} "
            f"fractions={normalize_fractions} units={normalize_units}"
        )
    else:
        preds = [clean_text(text) for text in preds]
    force_one = bool(_cfg_first("force_single_sentence", default=True)) and not bool(args.no_force_single_sentence)
    one_sent_mode = str(args.single_sentence_mode or _cfg_first("single_sentence_mode", default="merge")).strip().lower()
    if force_one:
        preds_pp = [enforce_single_sentence(text, mode=one_sent_mode) for text in preds]
        changed = sum(int(a != b) for a, b in zip(preds_pp, preds))
        changed_rate = changed / max(1, len(preds))
        preds = preds_pp
        print(f"[postprocess] force_single_sentence=True mode={one_sent_mode} changed={changed_rate:.1%}")

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

    out_df = pd.DataFrame({"id": ids, "translation": preds})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
