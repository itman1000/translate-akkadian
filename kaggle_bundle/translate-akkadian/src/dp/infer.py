"""最小の推論エントリポイント（ダミーベースライン）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .utils import clean_text, first_sentence, get_artifacts_dir, get_data_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy baseline inference.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    parser.add_argument("--ckpt", required=True, help="Path to model.json.")
    parser.add_argument("--out", default=None, help="Output predictions CSV.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, args.data_dir)
    test_path = data_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found at: {test_path}")

    model_path = Path(args.ckpt)
    if not model_path.exists():
        raise FileNotFoundError(f"model.json not found at: {model_path}")

    model = json.loads(model_path.read_text())
    strategy = cfg.get("dummy_strategy", model.get("dummy_strategy", "constant"))
    fallback = cfg.get("dummy_translation", model.get("dummy_translation", "Translation unavailable."))
    fallback = clean_text(str(fallback)) or "Translation unavailable."

    test_df = pd.read_csv(test_path)
    if "id" not in test_df.columns:
        raise ValueError("test.csv must contain 'id' column")

    if strategy == "copy_src":
        if "transliteration" not in test_df.columns:
            raise ValueError("test.csv must contain 'transliteration' for copy_src")
        translations = (
            test_df["transliteration"].astype(str).map(first_sentence).map(clean_text)
        )
        translations = translations.replace("", fallback)
    else:
        translations = pd.Series([fallback] * len(test_df))

    pred_df = pd.DataFrame({"id": test_df["id"], "translation": translations})

    out_path = Path(args.out) if args.out else get_artifacts_dir(cfg) / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)

    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
