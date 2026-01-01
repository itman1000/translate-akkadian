"""Train a lightweight retrieval baseline using TF-IDF char n-grams."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from .utils import clean_text, get_artifacts_dir, load_config, now_id


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_metrics(hypotheses: list[str], references: list[str]) -> Dict[str, float]:
    try:
        import sacrebleu  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sacrebleu is required for evaluation. Install with: pip install sacrebleu") from exc

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    score = math.sqrt(bleu.score * chrf.score)
    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "score": float(score),
    }


def parse_placeholder_map(value: Any) -> Dict[str, str]:
    if value is None:
        return {}
    data: Any = None
    if isinstance(value, list):
        data = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
    else:
        return {}

    mapping: Dict[str, str] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            placeholder = item.get("placeholder")
            text = item.get("text")
            if placeholder and text:
                mapping[str(placeholder)] = str(text)
    return mapping


def restore_placeholders(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    restored = text
    for placeholder in sorted(mapping.keys(), key=len, reverse=True):
        restored = restored.replace(placeholder, mapping[placeholder])
    return restored


def to_clean_list(series: Iterable[Any]) -> list[str]:
    return [clean_text(str(value)) for value in series]


def compute_length_stats(texts: Iterable[str]) -> Dict[str, float]:
    lengths = [len(text) for text in texts]
    if not lengths:
        return {"tgt_len_mean": 0.0, "tgt_len_std": 1.0}
    mean = sum(lengths) / len(lengths)
    var = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    std = math.sqrt(var) if var > 0 else 1.0
    return {"tgt_len_mean": float(mean), "tgt_len_std": float(std)}


def parse_df_value(value: Any, default: float | int) -> float | int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return default
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF retrieval baseline.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--train", required=True, help="Training data path.")
    parser.add_argument("--val", default=None, help="Validation data path.")
    parser.add_argument("--out", default=None, help="Output directory.")
    parser.add_argument("--variant", default=None, help="Variant label (A/B/C).")
    parser.add_argument("--fold", default=None, help="Fold id.")
    parser.add_argument("--sample", type=int, default=None, help="Use first N rows for quick runs.")
    parser.add_argument("--src-col", default=None, help="Source column name.")
    parser.add_argument("--tgt-col", default=None, help="Target column name.")
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore placeholders using mapping when evaluating on validation.",
    )
    parser.add_argument(
        "--map-col",
        default=None,
        help="Column containing placeholder mapping (used with --restore).",
    )
    parser.add_argument(
        "--restore-target-col",
        default=None,
        help="Target column for restored evaluation (used with --restore).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    src_col = args.src_col or cfg.get("src_col", "src_sent")
    tgt_col = args.tgt_col or cfg.get("tgt_col", "tgt_sent")
    restore = bool(args.restore or cfg.get("restore_placeholders", False))
    map_col = args.map_col or cfg.get("map_col", "placeholder_map")
    restore_target_col = args.restore_target_col or cfg.get("restore_target_col", "tgt_sent")

    train_df = read_table(Path(args.train))
    if args.sample:
        train_df = train_df.head(args.sample)

    if src_col not in train_df.columns or tgt_col not in train_df.columns:
        raise ValueError(f"train data must include {src_col} and {tgt_col}")

    train_src = to_clean_list(train_df[src_col])
    train_tgt = to_clean_list(train_df[tgt_col])
    stats = compute_length_stats(train_tgt)

    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore

    ngram_min = int(cfg.get("ngram_min", 3))
    ngram_max = int(cfg.get("ngram_max", 5))
    min_df = parse_df_value(cfg.get("min_df", 2), 2)
    max_df = parse_df_value(cfg.get("max_df", 0.9), 0.9)
    max_features = cfg.get("max_features")
    if max_features is not None:
        max_features = int(max_features)
    analyzer = str(cfg.get("analyzer", "char")).lower()
    token_pattern = cfg.get("token_pattern")
    if token_pattern is not None:
        token_pattern = str(token_pattern)

    vectorizer_kwargs = {
        "analyzer": analyzer,
        "ngram_range": (ngram_min, ngram_max),
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
        "lowercase": False,
    }
    if token_pattern:
        vectorizer_kwargs["token_pattern"] = token_pattern

    vectorizer = TfidfVectorizer(**vectorizer_kwargs)

    x_train = vectorizer.fit_transform(train_src)

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(x_train)

    exp_name = str(cfg.get("experiment_name", "tfidf"))
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = get_artifacts_dir(cfg) / "exp" / now_id(exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = {
        "vectorizer": vectorizer,
        "x_train": x_train,
        "train_tgt": train_tgt,
        "stats": stats,
        "config": {
            "analyzer": analyzer,
            "ngram_min": ngram_min,
            "ngram_max": ngram_max,
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_features,
        },
    }

    model_path = out_dir / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    metrics = {}
    if args.val:
        val_df = read_table(Path(args.val))
        if args.sample:
            val_df = val_df.head(args.sample)

        if src_col not in val_df.columns or tgt_col not in val_df.columns:
            raise ValueError(f"val data must include {src_col} and {tgt_col}")

        val_src = to_clean_list(val_df[src_col])
        val_tgt = to_clean_list(val_df[tgt_col])

        x_val = vectorizer.transform(val_src)
        distances, indices = nn.kneighbors(x_val, n_neighbors=1)
        preds = [train_tgt[i[0]] for i in indices]

        if restore:
            if map_col not in val_df.columns:
                raise ValueError(f"val data must include {map_col} for placeholder restore")
            if restore_target_col not in val_df.columns:
                raise ValueError(f"val data must include {restore_target_col} for restored eval")
            maps = [parse_placeholder_map(value) for value in val_df[map_col]]
            restored_preds = [
                clean_text(restore_placeholders(pred, mapping))
                for pred, mapping in zip(preds, maps)
            ]
            restored_gold = to_clean_list(val_df[restore_target_col])
            metrics = compute_metrics(restored_preds, restored_gold)
            placeholder_metrics = compute_metrics(preds, val_tgt)
            for key, value in placeholder_metrics.items():
                metrics[f"{key}_placeholder"] = value
            val_pred_df = pd.DataFrame(
                {
                    "pred": restored_preds,
                    "gold": restored_gold,
                    "pred_placeholder": preds,
                    "gold_placeholder": val_tgt,
                }
            )
        else:
            metrics = compute_metrics(preds, val_tgt)
            val_pred_df = pd.DataFrame({"pred": preds, "gold": val_tgt})
        metrics["variant"] = args.variant or ""
        metrics["fold"] = args.fold or ""

        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

        val_pred_path = out_dir / "val_predictions.csv"
        val_pred_df.to_csv(val_pred_path, index=False)

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))

    print(f"Saved model: {model_path}")
    if metrics:
        print(f"Saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
