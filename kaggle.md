# Kaggle 実行セル（コンペデータのみで学習 → gm計測 →（任意）submission.csv作成）

この `kaggle.md` は、Colab の `実行セル2.md` で行っている「コンペデータのみでの学習→val gm（BLEU/chrF++の幾何平均）算出」を **Kaggle Notebook** 上で再現するためのセル集です。  
（元セルでは `dp.align_train` → `dp.train_nmt` の流れになっています。）

---

## 0) Kaggle Notebook 側の事前準備

- Accelerator を **GPU**（P100/T4 など）に設定
- Notebook の「Add data」から、次を追加
  1. コンペ公式データセット：`deep-past-initiative-machine-translation`（通常は自動で見えます）
  2. **あなたのリポジトリ zip**（`translate-akkadian.zip` または `translate_akkadian.zip`）を含む Dataset
  3. （重要）**ByT5 の事前学習モデル**を含む Dataset（オフライン前提）
     - 例：`/kaggle/input/byt5-base-model/byt5-base/` のように `config.json` や `pytorch_model.bin` 等がある形

> Kaggle がオフラインの場合、`google/byt5-base` を Hugging Face からダウンロードできないため、(3) を用意してください。

---

## 1) リポジトリ展開

```bash
%%bash
set -eux

SRC="/kaggle/input/datasets/vashi1000/kaggle-bundle/translate-akkadian"
DST="/kaggle/working/repo/translate-akkadian"

mkdir -p /kaggle/working/repo
rm -rf "$DST"
cp -a "$SRC" "$DST"

ls -la "$DST" | head
```

---

## 2) 依存関係の確認（必要ならインストール）

`実行セル2.md` では `pip install -r requirements.txt` を実行しています。Kaggle には多くの依存が既に入っていますが、念のため repo 側 requirements を満たします。

```bash
%%bash
set -eux
cd /kaggle/working/repo/translate-akkadian

pip -q install -r requirements.txt

python - <<'PY'
import inspect
import transformers
from transformers import Seq2SeqTrainer

print('transformers:', transformers.__version__)
major = int(transformers.__version__.split('.')[0])
if major >= 5:
    # 実行セル2.md では v5 が不安定として 4.56.2 に下げています
    raise SystemExit(
        '[ERROR] transformers v5.x が入っています。\n'
        'この repo は transformers v4 系想定です。\n'
        'Kaggle がオフラインなら、transformers==4.x の wheel を Dataset に入れて pip install してください。'
    )

print('Seq2SeqTrainer has tokenizer kw:', 'tokenizer' in inspect.signature(Seq2SeqTrainer.__init__).parameters)
print('Seq2SeqTrainer has processing_class kw:', 'processing_class' in inspect.signature(Seq2SeqTrainer.__init__).parameters)
PY
```

---

## 3) 変数設定（コンペデータのみ） + env ファイル作成

Colab 版では `REPO_DIR`, `COMP_DATA_DIR`, `MODEL_NAME_OR_PATH`, `NMT_CONFIG`, `TRAIN_PATH` などを環境変数へ入れ、`colab_env.sh` を書き出しています。Kaggle では同様に `kaggle_env.sh` を作ります。

```python
import os
from pathlib import Path

REPO_DIR = "/kaggle/working/repo/translate-akkadian"
COMP_DATA_DIR = "/kaggle/input/deep-past-initiative-machine-translation"

# ===== ByT5 の選択 =====
BYT5_VARIANT = "base"  # "large" も可（ただし Kaggle GPU だと OOM しやすい）

# 事前学習モデル（オフライン想定: /kaggle/input 内のローカルパスを優先）
MODEL_HF = f"google/byt5-{BYT5_VARIANT}"
CAND_MODEL_DIRS = [
    f"/kaggle/input/byt5-{BYT5_VARIANT}-model/byt5-{BYT5_VARIANT}",
    f"/kaggle/input/byt5-{BYT5_VARIANT}-model",
    f"/kaggle/input/byt5-{BYT5_VARIANT}",
]
MODEL_DIR = next((p for p in CAND_MODEL_DIRS if Path(p).exists()), "")
MODEL_NAME_OR_PATH = MODEL_DIR if MODEL_DIR else MODEL_HF

# NMT 設定（上書き元）
NMT_CONFIG = "configs/nmt_byt5_small.yaml"

# 学習データ（dp.align_train の出力を使う）
TRAIN_PATH = "artifacts/aligned/aligned_train.parquet"

# コンペデータのみ（追加データは使わない）
RUN_EVACUN = "0"
RUN_OCR = "0"

# env
os.environ.update({
    "REPO_DIR": REPO_DIR,
    "COMP_DATA_DIR": COMP_DATA_DIR,
    "BYT5_VARIANT": BYT5_VARIANT,
    "MODEL_DIR": MODEL_DIR,
    "MODEL_NAME_OR_PATH": MODEL_NAME_OR_PATH,
    "NMT_CONFIG": NMT_CONFIG,
    "TRAIN_PATH": TRAIN_PATH,
    "RUN_EVACUN": RUN_EVACUN,
    "RUN_OCR": RUN_OCR,
    "PYTHONPATH": f"{REPO_DIR}/src",
})

# bash セル用の env ファイル
env_lines = [
    f'export REPO_DIR="{REPO_DIR}"',
    f'export COMP_DATA_DIR="{COMP_DATA_DIR}"',
    f'export BYT5_VARIANT="{BYT5_VARIANT}"',
    f'export MODEL_DIR="{MODEL_DIR}"',
    f'export MODEL_NAME_OR_PATH="{MODEL_NAME_OR_PATH}"',
    f'export NMT_CONFIG="{NMT_CONFIG}"',
    f'export TRAIN_PATH="{TRAIN_PATH}"',
    f'export RUN_EVACUN="{RUN_EVACUN}"',
    f'export RUN_OCR="{RUN_OCR}"',
    f'export PYTHONPATH="{REPO_DIR}/src"',
]
Path("/kaggle/working/kaggle_env.sh").write_text("\n".join(env_lines) + "\n")

print("MODEL_NAME_OR_PATH:", MODEL_NAME_OR_PATH)
print("wrote: /kaggle/working/kaggle_env.sh")
```

> `MODEL_NAME_OR_PATH` が `google/byt5-base` のままだと、Kaggle がオフラインの場合にロードで失敗します。  
> その場合は `byt5-base` の重みを Dataset として追加し、上の `CAND_MODEL_DIRS` に合う場所へ置いてください。

---

## 4) Kaggle 用 override config を生成（fp16 前提）

Colab 側セルは `cfg.update({...})` で学習設定を上書きし、`configs/nmt_byt5_small.colab.override.json` を書き出しています。Kaggle では P100/T4 を想定して **fp16** に寄せます（`bf16` は無効）。

```python
import json
import os
import sys
from pathlib import Path

# dp.utils を使って読み込む
sys.path.append(f'{os.environ["REPO_DIR"]}/src')
from dp.utils import load_config

repo_dir = Path(os.environ["REPO_DIR"])
base_path = repo_dir / os.environ.get("NMT_CONFIG", "configs/nmt_byt5_small.yaml")
cfg = load_config(str(base_path))

# Kaggle 向け安全設定（OOM したら batch をさらに下げて grad_acc を増やす）
cfg.update({
    "num_train_epochs": 5,
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    "max_grad_norm": 1.0,
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.1,
    "warmup_ratio": 0.1,

    "max_source_length": 384,
    "max_target_length": 384,

    "generation_max_length": 320,
    "generation_max_new_tokens": 320,

    # P100/T4 で OOM を避けるため、デフォルト(4)より小さめ推奨
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,

    # eval を学習途中で回さない（最後の post-eval に寄せる）
    "eval_steps": 999999,

    # Kaggle(P100/T4) は bf16 が基本使えないので fp16
    "bf16": False,
    "fp16": False,

    # 辞書グロス（必要なら OFF にして比較可能）
    "use_gloss": True,
    "gloss_max_hints": 6,
    "gloss_max_total_chars": 220,
    "gloss_max_match_len": 4,
    "gloss_max_lemma_freq": 1000000,

    # decode
    "num_beams": 4,
    "length_penalty": 0.8,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 20,

    # 出力の安全策
    "force_single_sentence": True,
    "single_sentence_mode": "merge",

    "data_dir": os.environ["COMP_DATA_DIR"],
})

out_rel = "configs/nmt_byt5_small.kaggle.override.json"
out_path = repo_dir / out_rel
out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print("wrote:", out_path)

# 以後はこの設定を使う
os.environ["NMT_CONFIG"] = out_rel

# bash 用 env を更新
env_path = Path("/kaggle/working/kaggle_env.sh")
env_lines = env_path.read_text().splitlines()
env_lines = [ln for ln in env_lines if not ln.startswith('export NMT_CONFIG=')]
env_lines.append(f'export NMT_CONFIG="{out_rel}"')
env_path.write_text("\n".join(env_lines) + "\n")
print("updated:", env_path)
```

---

## 5) コンペ train.csv を文単位に整形（aligned_train 作成）

Colab セルと同じく `dp.align_train` を実行します。

```bash
%%bash
set -eux
source /kaggle/working/kaggle_env.sh
cd "$REPO_DIR"

python -m dp.align_train \
  --config configs/align.yaml \
  --data-dir "${COMP_DATA_DIR}" \
  --variant C \
  --drop-flagged \
  --no-oare-sentences

ls -la artifacts/aligned | head
```

---

## 6) 学習（コンペデータのみ） + val gm 算出

`dp.train_nmt` を「aligned_train.parquet（＝コンペ由来）だけ」で学習します。  
`--post-eval-mode full --save-val-preds --save-val-audit` により、学習後に gm を出力し、`val_predictions.csv` を保存します。

```bash
%%bash
set -eux
source /kaggle/working/kaggle_env.sh
cd "$REPO_DIR"

OUT_DIR="/kaggle/working/models/byt5_comp_only_v1"
MODEL_NAME_OR_PATH="/kaggle/input/datasets/vashi1000/byt5-base-model/byt5-base"
MODEL_ARG="--model-name-or-path ${MODEL_NAME_OR_PATH}"

python -m dp.train_nmt \
  --config "${NMT_CONFIG}" \
  --data-dir "${COMP_DATA_DIR}" \
  --train "${TRAIN_PATH}" \
  --variant C \
  --drop-flagged \
  --out "${OUT_DIR}" \
  ${MODEL_ARG} \
  --post-eval-mode none

echo "[INFO] model saved: ${OUT_DIR}"
ls -la "${OUT_DIR}" | head
```

---

## 7) （確認用）保存された val_predictions.csv から gm を再計算

```python
import math
from pathlib import Path

import pandas as pd
import sacrebleu

out_dir = Path("/kaggle/working/models/byt5_comp_only_v1")
val_pred_path = out_dir / "val_predictions.csv"

if not val_pred_path.exists():
    raise FileNotFoundError(f"val_predictions.csv が見つかりません: {val_pred_path}")

df = pd.read_csv(val_pred_path)
refs = df["ref"].fillna("").astype(str).tolist()
preds = df["pred"].fillna("").astype(str).tolist()

bleu = sacrebleu.corpus_bleu(preds, [refs]).score
chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
gm = math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))
print(f"[val_predictions.csv] bleu={bleu:.4f} chrf={chrf:.4f} gm={gm:.4f}")
```

---

## 8) （任意）test へ推論して submission.csv を作る

> Kaggle の **本番採点は hidden test** なので、ここで出る `dp.validate` は「形式チェック」や「ダミー test」向けです。  
> ただし `submission.csv` を作って提出するところまでは動作確認できます。

```bash
%%bash
set -eux
source /kaggle/working/kaggle_env.sh
cd "$REPO_DIR"

OUT_DIR="/kaggle/working/models/byt5_comp_only_v1"

python -m dp.infer_nmt \
  --config "${NMT_CONFIG}" \
  --ckpt "${OUT_DIR}" \
  --data-dir "${COMP_DATA_DIR}" \
  --norm-variant C \
  --out /kaggle/working/predictions.csv

python -m dp.submit \
  --config "${NMT_CONFIG}" \
  --pred /kaggle/working/predictions.csv \
  --data-dir "${COMP_DATA_DIR}" \
  --out /kaggle/working/submission.csv

# 形式チェック（/kaggle/input の test.csv がダミーの場合、スコア計算は意味がありません）
python -m dp.validate \
  --submission /kaggle/working/submission.csv \
  --data-dir "${COMP_DATA_DIR}"

ls -la /kaggle/working/submission.csv
```

---

## OOM / 時間が厳しい場合の調整

- `configs/nmt_byt5_small.kaggle.override.json` の
  - `per_device_train_batch_size` を **2→1**
  - `gradient_accumulation_steps` を **8→16**
  - `max_source_length` を **384→256**
  - `num_train_epochs` を **5→1**（まず動作確認）
- これらは `実行セル2.md` が想定している「設定を JSON で上書き」方式に沿っています。
