"""Hugging Face のモデルをローカルに保存する。"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a HF model locally.")
    parser.add_argument("--model", required=True, help="Model name or path.")
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
        "pytorch_model.bin",
    ]

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub が見つかりません。pip install -r requirements.txt を実行してください。"
        ) from exc

    snapshot_download(
        repo_id=args.model,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    print(f"Downloaded model files to: {out_dir}")


if __name__ == "__main__":
    main()
