"""Oracle Upper Bound 評価。

複数候補（n-best / 複数モデル / 複数デコード設定など）から、
参照（gold）に最も近い候補を **oracle** 的に選んだ場合に
どこまでスコア（BLEU / chrF++ / gm）が伸び得るかを測定します。

注意
----
Kaggle の評価は corpus-level（コーパス全体で十分統計量を集計）なので、
理論的な厳密解は「コーパス BLEU/chrF を直接最大化する組合せ」です。
しかし実務上は、候補を *sentence-level gm* で選び、最後に corpus-level gm を
計算する近似で十分に診断できます（上限の感触を掴む用途）。

入力フォーマット
----------------
候補: CSV（long 形式、1行=1候補）
- id: 入力行ID
- translation: 候補訳
- (任意) tag: 候補の出所ラベル（beam64 / sample_t0.9 ... など）
- (任意) rank/run: n-best / sampling 由来のメタ情報

gold: CSV / Parquet
- id
- translation（または --gold-text-col で指定）
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .utils import (
    clean_text,
    enforce_single_sentence,
    load_config,
    normalize_translation_output,
)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _postprocess_text(
    text: str,
    *,
    normalize_output: bool,
    normalize_fractions: bool,
    normalize_units: bool,
    force_single_sentence: bool,
    single_sentence_mode: str,
) -> str:
    if normalize_output:
        out = normalize_translation_output(
            text,
            normalize_fractions=normalize_fractions,
            normalize_units=normalize_units,
        )
    else:
        out = clean_text(text)
    if force_single_sentence:
        out = enforce_single_sentence(out, mode=single_sentence_mode)
    return out


def compute_corpus_metrics(
    hyps: List[str],
    refs: List[str],
    *,
    bleu_tokenize: str,
) -> Dict[str, float]:
    try:
        import sacrebleu  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sacrebleu is required. Install with: pip install sacrebleu") from exc

    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=bleu_tokenize)
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    gm = math.sqrt(float(bleu.score) * float(chrf.score))
    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "gm": float(gm),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle upper bound evaluation (BLEU/chrF++/gm).")
    parser.add_argument(
        "--cands",
        nargs="+",
        required=True,
        help="Candidate CSV path(s). You can pass multiple files.",
    )
    parser.add_argument("--gold", required=True, help="Gold CSV/Parquet path.")
    parser.add_argument("--config", default=None, help="Optional config for postprocess defaults.")
    parser.add_argument("--out", default=None, help="Output directory (metrics + best csv).")

    parser.add_argument("--id-col", default="id", help="ID column name.")
    parser.add_argument("--cand-text-col", default="translation", help="Candidate text column name.")
    parser.add_argument("--gold-text-col", default="translation", help="Gold text column name.")

    parser.add_argument(
        "--bleu-tokenize",
        default=None,
        help="BLEU tokenizer for sacrebleu (default: config or '13a').",
    )

    # postprocess（submit と同じ寄せを入れたい場合）
    parser.add_argument("--normalize-output", action="store_true", help="Enable output normalization.")
    parser.add_argument("--no-normalize-output", action="store_true", help="Disable output normalization.")
    parser.add_argument("--force-single-sentence", action="store_true", help="Force single-sentence.")
    parser.add_argument(
        "--no-force-single-sentence",
        action="store_true",
        help="Do not force single-sentence.",
    )
    parser.add_argument(
        "--single-sentence-mode",
        default=None,
        help="merge|truncate (default: config or 'merge').",
    )

    parser.add_argument(
        "--save-scored-cands",
        action="store_true",
        help="Save per-candidate sentence BLEU/chrF/gm (can be large).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

    # Defaults aligned with submit.py
    normalize_output = bool(cfg.get("normalize_output", False))
    if args.normalize_output:
        normalize_output = True
    if args.no_normalize_output:
        normalize_output = False

    normalize_fractions = bool(cfg.get("normalize_output_fractions", True))
    normalize_units = bool(cfg.get("normalize_output_units", True))

    force_single_sentence = bool(cfg.get("force_single_sentence", True))
    if args.force_single_sentence:
        force_single_sentence = True
    if args.no_force_single_sentence:
        force_single_sentence = False

    single_sentence_mode = str(args.single_sentence_mode or cfg.get("single_sentence_mode", "merge"))

    bleu_tokenize = args.bleu_tokenize
    if bleu_tokenize is None:
        bleu_tokenize = str(cfg.get("bleu_tokenize", "13a"))
    bleu_tokenize = str(bleu_tokenize)

    gold_path = Path(args.gold)
    gold_df = read_table(gold_path)
    if args.id_col not in gold_df.columns:
        raise ValueError(f"gold must include id col: {args.id_col}")
    if args.gold_text_col not in gold_df.columns:
        raise ValueError(f"gold must include text col: {args.gold_text_col}")

    gold_df = gold_df[[args.id_col, args.gold_text_col]].copy()
    gold_df[args.id_col] = gold_df[args.id_col].astype(str)
    gold_df[args.gold_text_col] = gold_df[args.gold_text_col].fillna("").astype(str)
    gold_df["gold"] = gold_df[args.gold_text_col].map(
        lambda x: _postprocess_text(
            x,
            normalize_output=normalize_output,
            normalize_fractions=normalize_fractions,
            normalize_units=normalize_units,
            force_single_sentence=force_single_sentence,
            single_sentence_mode=single_sentence_mode,
        )
    )

    gold_map: Dict[str, str] = dict(zip(gold_df[args.id_col].tolist(), gold_df["gold"].tolist()))
    gold_ids: List[str] = gold_df[args.id_col].tolist()

    # Load candidates
    cand_frames: list[pd.DataFrame] = []
    for p in args.cands:
        path = Path(p)
        df = pd.read_csv(path)
        if args.id_col not in df.columns:
            raise ValueError(f"cands({path}) must include id col: {args.id_col}")
        if args.cand_text_col not in df.columns:
            raise ValueError(f"cands({path}) must include text col: {args.cand_text_col}")
        df = df.copy()
        df[args.id_col] = df[args.id_col].astype(str)
        df[args.cand_text_col] = df[args.cand_text_col].fillna("").astype(str)
        if "tag" not in df.columns:
            # fileごとに出所を付けておくと後で分析しやすい
            df["tag"] = path.stem
        df["cand"] = df[args.cand_text_col].map(
            lambda x: _postprocess_text(
                x,
                normalize_output=normalize_output,
                normalize_fractions=normalize_fractions,
                normalize_units=normalize_units,
                force_single_sentence=force_single_sentence,
                single_sentence_mode=single_sentence_mode,
            )
        )
        cand_frames.append(df)

    cand_df = pd.concat(cand_frames, axis=0, ignore_index=True)
    cand_df = cand_df[cand_df[args.id_col].isin(gold_map.keys())].reset_index(drop=True)

    # remove exact-duplicate candidates per id to reduce work
    cand_df = cand_df.drop_duplicates(subset=[args.id_col, "cand"]).reset_index(drop=True)

    # sentence-level scoring
    try:
        from sacrebleu.metrics import BLEU, CHRF  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sacrebleu is required. Install with: pip install sacrebleu") from exc

    bleu_metric = BLEU(tokenize=bleu_tokenize)
    chrf_metric = CHRF(word_order=2)

    bleu_scores: list[float] = []
    chrf_scores: list[float] = []
    gm_scores: list[float] = []
    refs: list[str] = []

    ids_list = cand_df[args.id_col].tolist()
    cands_list = cand_df["cand"].tolist()
    for row_id, hyp in zip(ids_list, cands_list):
        ref = gold_map.get(row_id, "")
        refs.append(ref)
        try:
            b = float(bleu_metric.sentence_score(hyp, [ref]).score)
            c = float(chrf_metric.sentence_score(hyp, [ref]).score)
        except Exception:
            b = 0.0
            c = 0.0
        g = math.sqrt(b * c) if b > 0 and c > 0 else 0.0
        bleu_scores.append(b)
        chrf_scores.append(c)
        gm_scores.append(g)

    cand_df["sent_bleu"] = bleu_scores
    cand_df["sent_chrf"] = chrf_scores
    cand_df["sent_gm"] = gm_scores

    # oracle pick: max sentence-level gm per id
    best_idx = cand_df.groupby(args.id_col)["sent_gm"].idxmax()
    best_df = cand_df.loc[best_idx].copy().reset_index(drop=True)

    # align to gold order for corpus metrics
    best_map: Dict[str, str] = dict(zip(best_df[args.id_col].tolist(), best_df["cand"].tolist()))
    hyps_oracle = [best_map.get(i, "") for i in gold_ids]
    refs_oracle = [gold_map.get(i, "") for i in gold_ids]
    oracle_metrics = compute_corpus_metrics(hyps_oracle, refs_oracle, bleu_tokenize=bleu_tokenize)

    # Optional: per-tag top1 baseline (rank==1 があればそれを使う)
    tag_metrics: Dict[str, Dict[str, float]] = {}
    if "tag" in cand_df.columns:
        for tag, sub in cand_df.groupby("tag"):
            pick = sub
            if "rank" in sub.columns:
                # rank==1 があるならそれ
                pick = sub[sub["rank"].astype(str) == "1"]
            if len(pick) == 0:
                # fallback: 先頭候補
                pick = sub.sort_values(args.id_col).groupby(args.id_col).head(1)
            pick_map = dict(zip(pick[args.id_col].tolist(), pick["cand"].tolist()))
            hyps = [pick_map.get(i, "") for i in gold_ids]
            tag_metrics[str(tag)] = compute_corpus_metrics(hyps, refs_oracle, bleu_tokenize=bleu_tokenize)

    # Stats
    n_gold = len(gold_df)
    n_cand_rows = len(cand_df)
    per_id = cand_df.groupby(args.id_col).size()
    stats = {
        "gold_rows": int(n_gold),
        "cand_rows": int(n_cand_rows),
        "uniq_ids_in_cands": int(per_id.shape[0]),
        "avg_cands_per_id": float(per_id.mean() if len(per_id) else 0.0),
        "p50_cands_per_id": float(per_id.quantile(0.50) if len(per_id) else 0.0),
        "p90_cands_per_id": float(per_id.quantile(0.90) if len(per_id) else 0.0),
        "min_cands_per_id": int(per_id.min() if len(per_id) else 0),
        "max_cands_per_id": int(per_id.max() if len(per_id) else 0),
    }

    payload: Dict[str, Any] = {
        "oracle": oracle_metrics,
        "per_tag_top1": tag_metrics,
        "stats": stats,
        "bleu_tokenize": bleu_tokenize,
        "postprocess": {
            "normalize_output": normalize_output,
            "normalize_output_fractions": normalize_fractions,
            "normalize_output_units": normalize_units,
            "force_single_sentence": force_single_sentence,
            "single_sentence_mode": single_sentence_mode,
        },
    }

    out_dir = Path(args.out) if args.out else Path("artifacts") / "oracle_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    # Save best predictions
    best_out = best_df[[args.id_col, "cand", "sent_gm", "sent_bleu", "sent_chrf", "tag"]].copy()
    best_out = best_out.rename(columns={"cand": "translation"})
    best_out.to_csv(out_dir / "oracle_best.csv", index=False)

    if args.save_scored_cands:
        cand_df.to_csv(out_dir / "cands_scored.csv", index=False)

    print("=== Oracle Upper Bound ===")
    print(f"[oracle] bleu={oracle_metrics['bleu']:.4f} chrf={oracle_metrics['chrf']:.4f} gm={oracle_metrics['gm']:.4f}")
    print(f"[stats] gold={stats['gold_rows']} cand_rows={stats['cand_rows']} avg_cands_per_id={stats['avg_cands_per_id']:.1f} p90={stats['p90_cands_per_id']:.0f}")
    if tag_metrics:
        print("--- per-tag top1 (rough baseline) ---")
        for tag, m in sorted(tag_metrics.items(), key=lambda x: x[1]["gm"], reverse=True):
            print(f"[{tag}] gm={m['gm']:.4f} bleu={m['bleu']:.4f} chrf={m['chrf']:.4f}")
    print(f"Saved: {out_dir / 'metrics.json'}")
    print(f"Saved: {out_dir / 'oracle_best.csv'}")


if __name__ == "__main__":
    main()
