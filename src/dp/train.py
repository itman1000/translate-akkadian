"""最小の学習エントリポイント（ダミーベースライン）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .utils import (
    clean_text,
    first_sentence,
    get_artifacts_dir,
    get_data_dir,
    load_config,
    now_id,
    ensure_dir,
    set_seed,
)


def build_dummy_translation(df: pd.DataFrame) -> str:
    if "translation" not in df.columns or df.empty:
        return "Translation unavailable."
    sample = str(df["translation"].iloc[0])
    sentence = first_sentence(sample)
    if sentence:
        return sentence
    return "Translation unavailable."


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy baseline training.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    parser.add_argument("--out-dir", default=None, help="Output directory for artifacts.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    data_dir = get_data_dir(cfg, args.data_dir)
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found at: {train_path}")

    train_df = pd.read_csv(train_path)

    dummy_translation = cfg.get("dummy_translation")
    if not dummy_translation:
        dummy_translation = build_dummy_translation(train_df)
    dummy_translation = clean_text(str(dummy_translation))
    if not dummy_translation:
        dummy_translation = "Translation unavailable."

    exp_name = str(cfg.get("experiment_name", "baseline"))
    out_dir = Path(args.out_dir) if args.out_dir else get_artifacts_dir(cfg) / "exp" / now_id(exp_name)
    ensure_dir(out_dir)

    model = {
        "dummy_strategy": cfg.get("dummy_strategy", "constant"),
        "dummy_translation": dummy_translation,
        "seed": seed,
        "train_rows": int(len(train_df)),
    }

    model_path = out_dir / "model.json"
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True))

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))

    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
