# Kaggle 用 README（Code 形式 / Notebook 実行）

このコンペは **Code 形式** です。ローカルで作成した `submission.csv` を直接アップロードして提出することはできません。Kaggle が **選択した Notebook を再実行**し、`/kaggle/working` に出力された `submission.csv` を提出物として扱います。

---

## 0. Kaggle に載せるリポジトリを用意
まず、Kaggle に載せる「軽量コピー」を作成して zip 化します。

1) 軽量コピーを作成:
```bash
bash scripts/build_kaggle_bundle.sh
```
生成先: `kaggle_bundle/translate-akkadian`

2) zip を作成:
```bash
cd kaggle_bundle
zip -r translate-akkadian.zip translate-akkadian
```

3) Kaggle の **Datasets → New Dataset** で `translate-akkadian.zip` をアップロード（Private 推奨）

---

## 0-B. （オフライン用）ByT5-base のモデルDataset
Kaggle の再実行ではインターネットが使えない場合があるため、**モデルを事前に Dataset 化**します。

```bash
python scripts/export_hf_model.py --model google/byt5-base --out models/byt5-base
```

`models/byt5-base` を zip 化して Kaggle Dataset にアップロードし、Notebook に追加してください。  
（Dataset 名の例: `byt5-base-model`）

---

## 1. Notebook の Data に追加するもの
Kaggle で **New Notebook**（Competition を選択）を作成し、**Data** から以下を追加します。

- コンペ公式データセット（`train.csv`, `test.csv` があるもの）
- リポジトリ Dataset（`translate-akkadian.zip` を含むもの）
- （任意）ByT5-base のモデル Dataset

Data タブで **実際のディレクトリ名**を確認し、以降の `/kaggle/input/...` を合わせてください。

---

## 2. Kaggle 最小ノートブック（セル順）
以下を **上から順番に実行**してください。  
（`%%bash` はシェルセル、それ以外は Python セルです）

---

### セル 1：リポジトリを展開
```bash
%%bash
# Kaggle の Data タブに表示された dataset 名に合わせて修正
REPO_DATASET="translate-akkadian"
ZIP_PATH="/kaggle/input/${REPO_DATASET}/translate-akkadian.zip"

# 展開先
mkdir -p /kaggle/working/repo
unzip -q "${ZIP_PATH}" -d /kaggle/working/repo

# パス確認
ls -la /kaggle/working/repo
ls -la /kaggle/working/repo/translate-akkadian | head
```

> zip ではなく **フォルダで入っている場合**は unzip を省略し、次の `REPO_DIR` を実在パスに合わせてください。  
> 例: `/kaggle/input/translate-akkadian/translate-akkadian`

---

### セル 2：依存関係をインストール
```bash
%%bash
pip -q install -r /kaggle/working/repo/translate-akkadian/requirements.txt
```

---

### セル 3：作業パス（環境変数）をセット
```python
import os
from pathlib import Path

# リポジトリ
REPO_DIR = "/kaggle/working/repo/translate-akkadian"

# コンペデータ（train.csv, test.csv のある場所）
COMP_DATA_DIR = "/kaggle/input/deep-past-initiative-machine-translation"

# （任意）ローカルモデル（無ければ空でOK）
MODEL_DIR = "/kaggle/input/byt5-base-model/byt5-base"

# NMT 設定ファイル（デフォルト）
NMT_CONFIG = "configs/nmt_byt5_small.yaml"

# 学習データ（既定は aligned_train）
TRAIN_PATH = "artifacts/aligned/aligned_train.parquet"

# OCR 追加並列を使うか（0=使わない / 1=使う）
RUN_OCR = "0"

# dp パッケージを import できるようにする
os.environ["REPO_DIR"] = REPO_DIR
os.environ["COMP_DATA_DIR"] = COMP_DATA_DIR
os.environ["MODEL_DIR"] = MODEL_DIR
os.environ["NMT_CONFIG"] = NMT_CONFIG
os.environ["TRAIN_PATH"] = TRAIN_PATH
os.environ["RUN_OCR"] = RUN_OCR
os.environ["PYTHONPATH"] = f"{REPO_DIR}/src"

# %%bash 用に環境変数をファイルに書き出す
env_lines = [
    f'export REPO_DIR="{REPO_DIR}"',
    f'export COMP_DATA_DIR="{COMP_DATA_DIR}"',
    f'export MODEL_DIR="{MODEL_DIR}"',
    f'export NMT_CONFIG="{NMT_CONFIG}"',
    f'export TRAIN_PATH="{TRAIN_PATH}"',
    f'export RUN_OCR="{RUN_OCR}"',
    f'export PYTHONPATH="{REPO_DIR}/src"',
]
Path("/kaggle/working/kaggle_env.sh").write_text("\n".join(env_lines) + "\n")
```

以降の `%%bash` セルは `source /kaggle/working/kaggle_env.sh` を先頭に入れて環境変数を読み込みます。

---

### セル 3-B：学習設定を上書き（任意）
このセルで **学習設定ファイルを上書きして保存**できます。  
以降のセルは `NMT_CONFIG` を参照するため、このセルを実行すると設定が切り替わります。

```python
import json
import os
from pathlib import Path
import sys

# 上書き元の設定（デフォルト: configs/nmt_byt5_small.yaml）
base_config = os.environ.get("NMT_CONFIG", "configs/nmt_byt5_small.yaml")
base_path = Path(base_config)
if not base_path.is_absolute():
    base_path = Path(os.environ["REPO_DIR"]) / base_path

# dp.utils を使って読み込む
sys.path.append(f'{os.environ["REPO_DIR"]}/src')
from dp.utils import load_config

cfg = load_config(str(base_path))

# ここを編集（例）
cfg.update({
    "num_train_epochs": 5,
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.1,
    "warmup_ratio": 0.1,
    "max_source_length": 384,
    "max_target_length": 384,
    "generation_max_length": 320,
    "generation_max_new_tokens": 320,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "bf16": False,
    "use_gloss": True,
    "gloss_max_hints": 6,
    "gloss_max_total_chars": 220,
    "gloss_max_match_len": 4,
    "gloss_max_lemma_freq": 1000000,
    "num_beams": 4,
    "length_penalty": 0.8,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 20,
    "force_single_sentence": True,
    "single_sentence_mode": "merge",
    "data_dir": os.environ["COMP_DATA_DIR"],
})

# ファイル名は任意
out_rel = "configs/nmt_byt5_small.kaggle.override.json"
out_path = Path(os.environ["REPO_DIR"]) / out_rel
out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print("wrote:", out_path)

# 以後はこの設定を使う
os.environ["NMT_CONFIG"] = out_rel

# %%bash 用に環境変数を更新
env_lines = [
    f'export REPO_DIR="{os.environ["REPO_DIR"]}"',
    f'export COMP_DATA_DIR="{os.environ["COMP_DATA_DIR"]}"',
    f'export MODEL_DIR="{os.environ["MODEL_DIR"]}"',
    f'export NMT_CONFIG="{os.environ["NMT_CONFIG"]}"',
    f'export TRAIN_PATH="{os.environ.get("TRAIN_PATH", "artifacts/aligned/aligned_train.parquet")}"',
    f'export RUN_OCR="{os.environ.get("RUN_OCR", "0")}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/kaggle/working/kaggle_env.sh").write_text("\n".join(env_lines) + "\n")
print("NMT_CONFIG:", os.environ["NMT_CONFIG"])
```

---

### セル 3-C：OCR 追加並列を使う場合のフラグ（任意）
OCR を使う場合は `RUN_OCR = "1"` にして実行してください。

```python
import os
from pathlib import Path

RUN_OCR = "1"
os.environ["RUN_OCR"] = RUN_OCR

# %%bash 用に環境変数を更新
env_lines = [
    f'export REPO_DIR="{os.environ["REPO_DIR"]}"',
    f'export COMP_DATA_DIR="{os.environ["COMP_DATA_DIR"]}"',
    f'export MODEL_DIR="{os.environ["MODEL_DIR"]}"',
    f'export NMT_CONFIG="{os.environ["NMT_CONFIG"]}"',
    f'export TRAIN_PATH="{os.environ.get("TRAIN_PATH", "artifacts/aligned/aligned_train.parquet")}"',
    f'export RUN_OCR="{os.environ["RUN_OCR"]}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/kaggle/working/kaggle_env.sh").write_text("\n".join(env_lines) + "\n")
print("RUN_OCR:", os.environ["RUN_OCR"])
```

---

### セル 4：アライン（文ペア作成）
```bash
%%bash
source /kaggle/working/kaggle_env.sh
cd "$REPO_DIR"
python -m dp.align_train --config configs/align.yaml   --data-dir "${COMP_DATA_DIR}"   --variant C   --drop-flagged   --no-oare-sentences
```

---

### セル 4-B：OCR 追加並列（任意）
```bash
%%bash
source /kaggle/working/kaggle_env.sh
cd "$REPO_DIR"

if [ "${RUN_OCR}" != "1" ]; then
  echo "RUN_OCR=0: skip OCR flow"
  exit 0
fi

# publications.csv から OCR 候補を抽出
python -m dp.ocr_candidates --config configs/ocr.yaml --input "${COMP_DATA_DIR}/publications.csv"

# （任意）行番号クリーニングを使う場合は以下に切り替え
# python -m dp.clean_publications --config configs/ocr.yaml --input "${COMP_DATA_DIR}/publications.csv" --out artifacts/ocr/publications_clean.csv
# python -m dp.ocr_candidates --config configs/ocr.yaml --input artifacts/ocr/publications_clean.csv --text-col page_text_clean --overwrite

# 候補 → ペア抽出 → 高品質のみ混合
python -m dp.ocr_pairs --config configs/ocr.yaml --overwrite
python -m dp.mix_ocr --config configs/ocr.yaml --aligned artifacts/aligned/aligned_train.parquet
```

---

### セル 4-C：学習データを OCR 混合に切り替え（任意）
```python
import os
from pathlib import Path

TRAIN_PATH = "artifacts/aligned/aligned_train.parquet"
mixed_path = Path(os.environ["REPO_DIR"]) / "artifacts/ocr_pairs/mixed_train.parquet"
if mixed_path.exists():
    TRAIN_PATH = "artifacts/ocr_pairs/mixed_train.parquet"

os.environ["TRAIN_PATH"] = TRAIN_PATH

# %%bash 用に環境変数を更新
env_lines = [
    f'export REPO_DIR="{os.environ["REPO_DIR"]}"',
    f'export COMP_DATA_DIR="{os.environ["COMP_DATA_DIR"]}"',
    f'export MODEL_DIR="{os.environ["MODEL_DIR"]}"',
    f'export NMT_CONFIG="{os.environ["NMT_CONFIG"]}"',
    f'export TRAIN_PATH="{os.environ["TRAIN_PATH"]}"',
    f'export RUN_OCR="{os.environ.get("RUN_OCR", "0")}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/kaggle/working/kaggle_env.sh").write_text("\n".join(env_lines) + "\n")
print("TRAIN_PATH:", os.environ["TRAIN_PATH"])
```

---

### セル 5：学習（NMT）
```bash
%%bash
source /kaggle/working/kaggle_env.sh
cd "$REPO_DIR"
# ★モデル指定：
# - Datasetにbyt5-baseがあれば：--model-name-or-path "${MODEL_DIR}"
# - 無ければ：configの model_name_or_path（google/byt5-base）を使うので --model-name-or-path は省略OK

MODEL_ARG=""
if [ -d "${MODEL_DIR}" ]; then
  MODEL_ARG="--model-name-or-path ${MODEL_DIR}"
fi

python -m dp.train_nmt   --config "${NMT_CONFIG}"   --data-dir "${COMP_DATA_DIR}"   --train "${TRAIN_PATH}"   --variant C --drop-flagged   --out artifacts/nmt/byt5_small_kaggle   ${MODEL_ARG}   --post-eval-mode full   --save-val-preds --save-val-audit
```

---

### セル 6：推論 → submission.csv を作成
```bash
%%bash
source /kaggle/working/kaggle_env.sh
cd "$REPO_DIR"

# 1) test.csv に対して推論
python -m dp.infer_nmt   --config "${NMT_CONFIG}"   --ckpt artifacts/nmt/byt5_small_kaggle   --data-dir "${COMP_DATA_DIR}"   --out artifacts/predictions.csv

# 2) submission.csv を作成
python -m dp.submit   --pred artifacts/predictions.csv   --data-dir "${COMP_DATA_DIR}"   --out /kaggle/working/submission.csv

# 3) 形式チェック
python -m dp.validate   --submission /kaggle/working/submission.csv   --data-dir "${COMP_DATA_DIR}"
```

---

## 3. 推論のみで提出したい場合（学習済みモデルを使う）
学習済みモデルを Dataset 化して追加した場合は、**セル 5 をスキップ**し、セル 6 の `--ckpt` を差し替えます。

例:
```
/kaggle/input/translate-akkadian-nmt-checkpoint/artifacts/nmt/byt5_small_colab
```

---

## 4. 出力ファイル
- `submission.csv` は **`/kaggle/working/submission.csv`** に出力されます
- Kaggle の「Output」タブから `submission.csv` を選択して提出できます

---

## 5. よくあるトラブル
### Q1. `train.csv not found` / `test.csv not found`
- `COMP_DATA_DIR` が正しいか確認してください
- `ls -la $COMP_DATA_DIR` で `train.csv` / `test.csv` が見える必要があります

### Q2. OOM（CUDA out of memory）
- `per_device_train_batch_size` を下げる
- `max_source_length` / `max_target_length` を下げる
- `gradient_accumulation_steps` を上げて実効バッチを維持する
- `--post-eval-mode quick` にして評価コストを下げる

### Q3. パスが合わない
- `/kaggle/input` 以下の **実ディレクトリ名**に合わせて修正してください

---

## 6. メモ
- 提出時は Kaggle が Notebook を最初から再実行するため、学習セルがあると毎回学習が走ります  
  学習を省きたい場合は「推論のみで提出したい場合」を参照してください
- インターネットが使えない環境では `MODEL_DIR` を必ず指定してください
