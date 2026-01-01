"""Generate pseudo-parallel data using a retrieval baseline."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .align_train import normalize_transliteration
from .utils import clean_text, get_artifacts_dir, get_data_dir, load_config


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_src(text: str, variant: str | None) -> str:
    if variant:
        return normalize_transliteration(text, variant.upper())
    return clean_text(text)


def parse_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo-parallel data.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--ckpt", required=True, help="Path to model.pkl.")
    parser.add_argument("--input", default=None, help="Input CSV/Parquet path.")
    parser.add_argument("--out", default=None, help="Output file path.")
    parser.add_argument("--data-dir", default=None, help="Data directory path.")
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--id-col", default=None, help="ID column name.")
    parser.add_argument("--translation-col", default=None, help="Existing translation column name.")
    parser.add_argument("--require-untranslated", action="store_true", help="Keep rows without translation.")
    parser.add_argument("--norm-variant", default=None, help="Normalization variant A/B/C.")
    parser.add_argument("--min-confidence", type=float, default=None, help="Minimum confidence to keep.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use first N rows for quick runs.")
    parser.add_argument("--format", default=None, choices=["parquet", "csv"], help="Output format.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg, args.data_dir)
    artifacts_dir = get_artifacts_dir(cfg)

    src_col = args.src_col or cfg.get("pseudo_src_col", "transliteration")
    id_col = args.id_col or cfg.get("pseudo_id_col", "oare_id")
    translation_col = args.translation_col or cfg.get("pseudo_translation_col", "AICC_translation")
    norm_variant = args.norm_variant or cfg.get("pseudo_norm_variant", "C")
    require_untranslated = bool(args.require_untranslated or cfg.get("pseudo_require_untranslated", False))
    min_confidence = (
        args.min_confidence
        if args.min_confidence is not None
        else parse_float(cfg.get("pseudo_min_confidence"), 0.0)
    )
    fmt = args.format or cfg.get("pseudo_format", "parquet")

    input_path = Path(args.input) if args.input else Path(cfg.get("pseudo_input", ""))
    if not input_path or str(input_path) == "":
        input_path = data_dir / "published_texts.csv"

    model_path = Path(args.ckpt)
    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found: {model_path}")

    df = read_table(input_path)
    if args.max_rows:
        df = df.head(args.max_rows)

    if src_col not in df.columns:
        raise ValueError(f"input must include {src_col}")

    if require_untranslated:
        if translation_col not in df.columns:
            raise ValueError(f"input must include {translation_col} for untranslated filter")
        mask = df[translation_col].fillna("").astype(str).str.strip() == ""
        df = df[mask].reset_index(drop=True)

    ids = None
    if id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]
        id_col = "row_id"

    src_texts = df[src_col].fillna("").astype(str).tolist()
    src_norm = [normalize_src(text, norm_variant) for text in src_texts]

    with model_path.open("rb") as f:
        model = pickle.load(f)

    vectorizer = model["vectorizer"]
    x_train = model["x_train"]
    train_tgt = model["train_tgt"]

    from sklearn.neighbors import NearestNeighbors  # type: ignore

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(x_train)

    x_input = vectorizer.transform(src_norm)
    distances, indices = nn.kneighbors(x_input, n_neighbors=1)
    preds = [train_tgt[i[0]] for i in indices]
    confidences = [max(0.0, 1.0 - float(d[0])) for d in distances]

    out_df = pd.DataFrame(
        {
            id_col: ids,
            "src_sent": src_norm,
            "tgt_sent": preds,
            "confidence": confidences,
        }
    )
    if norm_variant:
        out_df["src_norm_variant"] = str(norm_variant).upper()
    out_df["source"] = input_path.stem

    if min_confidence > 0:
        out_df = out_df[out_df["confidence"] >= min_confidence].reset_index(drop=True)

    if args.out:
        out_path = Path(args.out)
    else:
        variant_tag = str(norm_variant).upper() if norm_variant else "raw"
        out_dir = artifacts_dir / "pseudo" / f"variant={variant_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pseudo_train.{fmt}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        out_df.to_parquet(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)

    print(f"Saved pseudo data: {out_path}")
    print(f"Rows: {len(out_df)} (min_confidence={min_confidence})")


if __name__ == "__main__":
    main()
