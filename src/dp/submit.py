"""予測から submission.csv を作成する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import (
    clean_text,
    enforce_single_sentence,
    get_data_dir,
    load_config,
    normalize_translation_output,
)


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) == 1 and df.columns[0] not in {"id", "translation"}:
        df = pd.read_csv(path, header=None, names=["translation"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build submission.csv.")
    parser.add_argument("--pred", required=True, help="Predictions CSV path.")
    parser.add_argument("--out", required=True, help="Output submission CSV path.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    pred_df = load_predictions(pred_path)

    if "translation" not in pred_df.columns:
        raise ValueError("Predictions must include a 'translation' column")

    if "id" not in pred_df.columns:
        data_dir = get_data_dir(cfg, args.data_dir)
        test_path = data_dir / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"test.csv not found at: {test_path}")
        test_df = pd.read_csv(test_path)
        if len(test_df) != len(pred_df):
            raise ValueError("Predictions row count does not match test.csv")
        pred_df.insert(0, "id", test_df["id"].values)

    # Submission safety: force single-sentence output (decimals like '3.5' are preserved).
    force_one = bool(cfg.get("force_single_sentence", True))
    mode = str(cfg.get("single_sentence_mode", "merge")).strip().lower()
    normalize_output = bool(cfg.get("normalize_output", False))
    normalize_fractions = bool(cfg.get("normalize_output_fractions", True))
    normalize_units = bool(cfg.get("normalize_output_units", True))

    def _postprocess(text: str) -> str:
        if normalize_output:
            text = normalize_translation_output(
                text,
                normalize_fractions=normalize_fractions,
                normalize_units=normalize_units,
            )
        else:
            text = clean_text(text)
        if force_one:
            text = enforce_single_sentence(text, mode=mode)
        return text

    pred_df["translation"] = pred_df["translation"].astype(str).map(_postprocess)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df[["id", "translation"]].to_csv(out_path, index=False)

    print(f"Saved submission: {out_path}")


if __name__ == "__main__":
    main()
