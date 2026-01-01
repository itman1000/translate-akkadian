"""SacreBLEU を使ったローカル評価。"""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions with BLEU and chrF++.")
    parser.add_argument("--pred", required=True, help="Predictions CSV path.")
    parser.add_argument("--gold", required=True, help="Gold CSV path.")
    parser.add_argument("--id-col", default="id", help="ID column name.")
    parser.add_argument("--text-col", default="translation", help="Text column name.")
    args = parser.parse_args()

    try:
        import sacrebleu  # type: ignore
    except ImportError:
        print("ERROR: sacrebleu is required for eval. Install with: pip install sacrebleu")
        sys.exit(1)

    gold = pd.read_csv(args.gold)
    pred = pd.read_csv(args.pred)

    if args.id_col in gold.columns and args.id_col in pred.columns:
        gold = gold.sort_values(args.id_col)
        pred = pred.sort_values(args.id_col)

    refs = gold[args.text_col].astype(str).tolist()
    hyps = pred[args.text_col].astype(str).tolist()

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    score = math.sqrt(bleu.score * chrf.score)

    print(f"BLEU: {bleu.score:.4f}")
    print(f"chrF++: {chrf.score:.4f}")
    print(f"Geometric Mean: {score:.4f}")


if __name__ == "__main__":
    main()
