# Colab 用 README（最小ノートブック：セットアップ → 学習 → 評価）

このリポジトリ（translate-akkadian）は **Kaggle「Deep Past Challenge（Akkadian → English）」**向けの学習・推論パイプラインです。  
Kaggle Notebook では GPU が T4 / P100 などに限られますが、Google Colab では A100 / H100 などの GPU が選べるため、**学習と検証（CV・val）を Colab 側で高速に回す**ための手順をまとめます。

> ここでの「評価」は、`dp.train_nmt` が学習後に validation に対して generate を行い、**BLEU / chrF++ / gm（幾何平均）**をログ表示する部分を指します。  
> Kaggle の hidden test は Kaggle 上でしか回せません（＝最終提出スコアは Kaggle が再実行して採点）。

---

## Step 0: Oracle Upper Bound（+10 が理論的に可能か）
上位を狙う施策の優先度付けのために、
**1入力あたり多数候補を生成 → gold に最も近い候補を oracle 的に選択 → corpus gm を計算**して、
「理論上どこまで伸び得るか」の上限をまず測ります。

Colab（A100/H100）なら、val に対して beam n-best + sampling で
**200〜500 候補/文**を作っても現実的に回せます。

### 重要：`ID_COL` は「各行で一意」にする
`dp.oracle_eval` は **`ID_COL` ごとに最良候補を 1つ**選ぶため、
`ID_COL` が重複していると評価が壊れます（別行の候補/参照が混ざります）。

たとえば `aligned_train` / `ablation/*/val.parquet` / `evacun_*_val.parquet` は
1文書（`oare_id`）から複数行が生成されるため、`oare_id` が重複しやすいです。
その場合は **oracle 用に `id=0..N-1` の連番を振った val ファイル**を作ってから実行してください。

（例）候補生成 → oracle 評価（`CKPT_DIR` などはあなたの環境に合わせて変更）:

```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
export PYTHONPATH=src

# ===== ここを自分の環境に合わせて変更 =====
CFG="${NMT_CONFIG}"                      # 学習と同じ config を使うのが安全
CKPT_DIR="artifacts/nmt/byt5_small_colab"  # 例（あなたの学習済みモデル）

# 基本は train_nmt の --save-val-preds 出力を使うのが一番楽（src/ref が揃っている）
BASE_VAL="artifacts/nmt/byt5_small_colab/val_predictions.csv"
BASE_SRC_COL="src"
BASE_GOLD_COL="ref"

# val_predictions.csv が無い場合は、gold 付きの val（parquet/csv）を指定して OK
# 例:
# BASE_VAL="artifacts/ablation/variant=C/fold=0/val.parquet"
# BASE_SRC_COL="src_sent"
# BASE_GOLD_COL="tgt_sent"

mkdir -p artifacts/oracle

# oracle 用に「各行で一意な id」を付与した val を作る
export BASE_VAL BASE_SRC_COL BASE_GOLD_COL
python - <<'PY'
import os
from pathlib import Path

import pandas as pd

base = Path(os.environ["BASE_VAL"])
src_col = os.environ["BASE_SRC_COL"]
gold_col = os.environ["BASE_GOLD_COL"]

if not base.exists():
    raise SystemExit(f"ファイルが見つかりません: {base}")

if base.suffix.lower() == ".parquet":
    df = pd.read_parquet(base)
else:
    df = pd.read_csv(base)

missing = [c for c in (src_col, gold_col) if c not in df.columns]
if missing:
    raise SystemExit(f"必要な列が見つかりません: {missing} (columns={list(df.columns)})")

out = Path("artifacts/oracle/val_for_oracle.csv")
out.parent.mkdir(parents=True, exist_ok=True)

df = df.reset_index(drop=True).copy()
df.insert(0, "id", df.index.astype(int))
df = df.rename(columns={src_col: "src", gold_col: "ref"})
df[["id", "src", "ref"]].to_csv(out, index=False)
print("Saved:", out, "rows=", len(df))
PY

VAL_PATH="artifacts/oracle/val_for_oracle.csv"
SRC_COL="src"
ID_COL="id"
GOLD_COL="ref"

# beam 64 本
python -m dp.infer_nmt_nbest \
  --config "${CFG}" \
  --ckpt "${CKPT_DIR}" \
  --test "${VAL_PATH}" \
  --src-col "${SRC_COL}" --id-col "${ID_COL}" \
  --out artifacts/oracle/cands_beam64.csv \
  --k 64 --num-beams 64 --tag beam64

# sampling: 32 本×4 run = 128 本
python -m dp.infer_nmt_nbest \
  --config "${CFG}" \
  --ckpt "${CKPT_DIR}" \
  --test "${VAL_PATH}" \
  --src-col "${SRC_COL}" --id-col "${ID_COL}" \
  --out artifacts/oracle/cands_sample_t0.9_p0.95.csv \
  --do-sample --temperature 0.9 --top-p 0.95 \
  --k 32 --runs 4 --seed 42 --tag sample_t0.9_p0.95

# oracle upper bound
python -m dp.oracle_eval \
  --config "${CFG}" \
  --gold "${VAL_PATH}" \
  --id-col "${ID_COL}" --gold-text-col "${GOLD_COL}" \
  --cands artifacts/oracle/cands_beam64.csv artifacts/oracle/cands_sample_t0.9_p0.95.csv \
  --out artifacts/oracle_eval

cat artifacts/oracle_eval/metrics.json | head
```

出力:
- `artifacts/oracle_eval/metrics.json`
- `artifacts/oracle_eval/oracle_best.csv`

---

## 0. 事前に決めること（おすすめ）

### Colab のランタイム
- Colab の `Runtime → Change runtime type` で **GPU** を選択
- 可能なら **A100 / H100**（bf16 が使えて速い）

### どこにデータを置くか
この手順は **Google Drive に置く前提**です（永続化できるため）。

学習中の `artifacts/`（中間生成物・モデル）は I/O が多いので、**学習中は `/content`（ローカル）に置き、必要なら最後に Drive にコピー**するのが安定です。

---

## 1. Colab 最小ノートブック（セル順）

以下を **Colab ノートブックにそのまま貼り付けて上から実行**してください。  
（`%%bash` で始まるセルはシェル、それ以外は Python セルです）

---

### セル 1：Google Drive をマウント（必須）
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### セル 2：リポジトリを配置（zip をアップロードして展開）
Colab に `translate-akkadian`（リポジトリ本体）が置ければOKです。

例：`translate-akkadian.zip` を Colab にアップロード済みの場合
```bash
%%bash
# 作業ディレクトリ
cd /content

# zip を展開（zip 内に translate-akkadian/ が入っている想定）
unzip -q translate-akkadian.zip -d /content/repo

# パスを確認
ls -la /content/repo
ls -la /content/repo/translate-akkadian | head
```

> zip の中身の構造が違う場合は、`/content/repo/translate-akkadian` の実在パスに合わせて次セルの `REPO_DIR` を調整してください。

---

### セル 3：依存関係をインストール
```bash
%%bash
cd /content/repo/translate-akkadian
pip -q install -r requirements.txt
```

---

### セル 4：データを Kaggle と同じパスで見えるようにする（重要）
Colab では `/kaggle` が読み取り専用になるため、**ここで `/content/kaggle/input` を作って “Kaggle っぽいパス” を再現**します。

#### 4-A) まず `/content/kaggle/input` を作る
```bash
%%bash
mkdir -p /content/kaggle/input
mkdir -p /content/kaggle/working
```

#### 4-B) Drive 上のデータを `/content/kaggle/input/...` に symlink する
以下の **Drive 側のパス**はあなたの配置に合わせて変更してください（例は `MyDrive/kaggle_input/`）。

**Drive 側の配置ルール（重要）**
- Kaggle データセットの中身（`train.csv`/`test.csv`/`sample_submission.csv` など）を **1つのフォルダにまとめて** Drive に置きます
- そのフォルダ名は **`deep-past-initiative-machine-translation`** にするのが最も簡単です  
  （別名でもOKですが、その場合は下の `ln -s` のパスを合わせてください）
- 実際に必須なのは `train.csv` と `test.csv` です  
  ただし Kaggle から落としたフォルダ一式をそのまま置くのが安全です

例（おすすめの構成）:
```
MyDrive/kaggle_input/
  deep-past-initiative-machine-translation/
    train.csv
    test.csv
    sample_submission.csv
    OA_Lexicon_eBL.csv
    Sentences_Oare_FirstWord_LinNum.csv
    bibliography.csv
    eBL_Dictionary.csv
    publications.csv
    published_texts.csv
    resources.csv
    evacun_oracc_parallel_v0.1/
    akkadian_train.txt
    akkadian_validation.txt
    english_train.txt
    english_validation.txt
    transcription_train.txt
    transcription_validation.txt
```

- 公式データ（train.csv / test.csv / sample_submission.csv …）
  - `/content/kaggle/input/deep-past-initiative-machine-translation`
- （任意）EvaCun 追加データ
  - `/content/drive/MyDrive/kaggle_input/evacun_oracc_parallel_v0.1`
- （任意）ByT5（base/large）のローカル保存
  - `/content/kaggle/input/byt5-<variant>-model/byt5-<variant>`

```bash
%%bash
# ===== ここを自分のDrive構成に合わせて編集してください =====
DRIVE_DATA_ROOT="/content/drive/MyDrive/kaggle_input"

# ByT5 の選択（base / large）
BYT5_VARIANT="base"  # "large" も可

# 公式データ（train.csv, test.csv があるディレクトリ）
ln -sf "${DRIVE_DATA_ROOT}/deep-past-initiative-machine-translation"   /content/kaggle/input/deep-past-initiative-machine-translation

# （任意）EvaCun 追加データ
# Drive 直読みなら symlink は不要（EVACUN_DIR を Drive パスにする）
# /content/kaggle/input で読みたい場合のみ symlink を作成:
# ln -sf "${DRIVE_DATA_ROOT}/evacun_oracc_parallel_v0.1"   /content/kaggle/input/evacun_oracc_parallel_v0.1

# 確認
ls -la /content/kaggle/input/deep-past-initiative-machine-translation | head
ls -la "${DRIVE_DATA_ROOT}/evacun_oracc_parallel_v0.1" | head || true
ls -la "/content/kaggle/input/byt5-${BYT5_VARIANT}-model/byt5-${BYT5_VARIANT}" 2>/dev/null | head || true
```

> **ByT5 のモデルを Drive に置いていない場合**でも動きます。  
> その場合は `model_name_or_path=google/byt5-<variant>` を使って Hugging Face から自動取得します（Colab はネット接続できる想定）。  
> 既定のキャッシュ先は `/root/.cache/huggingface` で、セッション終了で消えます。永続化したい場合は以下の「Drive に保存」「キャッシュを Drive に移す」を使ってください。

---

#### 4-C) ByT5 を Drive に保存する（任意・永続化）
すでに ByT5（base/large）を持っている場合は、Drive に以下の構成で置いてください（`<variant>` は `base` / `large`）。

```
MyDrive/kaggle_input/
  byt5-<variant>-model/
    byt5-<variant>/
      config.json
      generation_config.json
      pytorch_model.bin
      special_tokens_map.json
      tokenizer_config.json
      ...
```

まだ持っていない場合は、以下のセルで **Drive に直接ダウンロードして保存**できます（セル 3 のインストール後に実行）。

```bash
%%bash
# ByT5 の選択（base / large）
BYT5_VARIANT="base"  # "large" も可
export BYT5_VARIANT

# Drive に保存する ByT5 の保存先
OUT_DIR="/content/drive/MyDrive/kaggle_input/byt5-${BYT5_VARIANT}-model/byt5-${BYT5_VARIANT}"
mkdir -p "$(dirname "${OUT_DIR}")"

python - <<'PY'
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

byt5_variant = os.environ.get("BYT5_VARIANT", "base").strip()
model_id = f"google/byt5-{byt5_variant}"
out_dir = f"/content/drive/MyDrive/kaggle_input/byt5-{byt5_variant}-model/byt5-{byt5_variant}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer.save_pretrained(out_dir)
model.save_pretrained(out_dir)
print("saved:", out_dir)
PY
```

---

#### 4-D) Hugging Face のキャッシュを Drive に移す（任意・再ダウンロード回避）
`google/byt5-<variant>` を毎回再取得したくない場合は、**キャッシュ先を Drive に変更**できます。  
このセルは **モデル取得前に実行**してください。

```python
import os
from pathlib import Path

# Hugging Face のキャッシュを Drive に置く
HF_HOME = "/content/drive/MyDrive/hf_cache"
Path(HF_HOME).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
```

---

### セル 5：作業パス（環境変数）をセット
```python
import os
from pathlib import Path

# リポジトリ
REPO_DIR = "/content/repo/translate-akkadian"

# コンペデータ（train.csv, test.csv のある場所）
COMP_DATA_DIR = "/content/kaggle/input/deep-past-initiative-machine-translation"

# ===== ByT5 の選択（base / large）=====
BYT5_VARIANT = "base"  # "large" も可
# NOTE: large は OOM しやすいので、必要ならセル 5-B で batch/長さ/grad_ckpt/fp16(bf16) を調整してください

# Hugging Face のモデルID（ネットが使える場合）
MODEL_HF = f"google/byt5-{BYT5_VARIANT}"

# （任意）オフライン/永続化用：Drive→/content/kaggle/input に置いたモデル（無ければ Hugging Face を使う）
MODEL_DATASET = f"byt5-{BYT5_VARIANT}-model"
MODEL_DIR = f"/content/kaggle/input/{MODEL_DATASET}/byt5-{BYT5_VARIANT}"
MODEL_NAME_OR_PATH = MODEL_DIR if Path(MODEL_DIR).exists() else MODEL_HF

# NMT 設定ファイル（デフォルト）
NMT_CONFIG = "configs/nmt_byt5_small.yaml"

# 学習データ（既定は aligned_train）
TRAIN_PATH = "artifacts/aligned/aligned_train.parquet"

# EvaCun 追加データの場所（任意）
EVACUN_DIR = "/content/drive/MyDrive/kaggle_input/evacun_oracc_parallel_v0.1"

# EvaCun 追加データを使うか（0=使わない / 1=使う）
RUN_EVACUN = "0"

# OCR 追加並列を使うか（0=使わない / 1=使う）
RUN_OCR = "0"

# dp パッケージを import できるようにする
os.environ["REPO_DIR"] = REPO_DIR
os.environ["COMP_DATA_DIR"] = COMP_DATA_DIR
os.environ["BYT5_VARIANT"] = BYT5_VARIANT
os.environ["MODEL_DIR"] = MODEL_DIR
os.environ["MODEL_NAME_OR_PATH"] = MODEL_NAME_OR_PATH
os.environ["NMT_CONFIG"] = NMT_CONFIG
os.environ["TRAIN_PATH"] = TRAIN_PATH
os.environ["EVACUN_DIR"] = EVACUN_DIR
os.environ["RUN_EVACUN"] = RUN_EVACUN
os.environ["RUN_OCR"] = RUN_OCR
os.environ["PYTHONPATH"] = f"{REPO_DIR}/src"

# %%bash 用に環境変数をファイルに書き出す
env_lines = [
    f'export REPO_DIR="{REPO_DIR}"',
    f'export COMP_DATA_DIR="{COMP_DATA_DIR}"',
    f'export BYT5_VARIANT="{BYT5_VARIANT}"',
    f'export MODEL_DATASET="{MODEL_DATASET}"',
    f'export MODEL_DIR="{MODEL_DIR}"',
    f'export MODEL_NAME_OR_PATH="{MODEL_NAME_OR_PATH}"',
    f'export NMT_CONFIG="{NMT_CONFIG}"',
    f'export TRAIN_PATH="{TRAIN_PATH}"',
    f'export EVACUN_DIR="{EVACUN_DIR}"',
    f'export RUN_EVACUN="{RUN_EVACUN}"',
    f'export RUN_OCR="{RUN_OCR}"',
    f'export PYTHONPATH="{REPO_DIR}/src"',
]
Path("/content/colab_env.sh").write_text("\n".join(env_lines) + "\n")
```

以降の `%%bash` セルは `source /content/colab_env.sh` を先頭に入れて環境変数を読み込みます。

---

### セル 5-B：学習設定を上書き（任意）
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
    "max_source_length": 384,
    "max_target_length": 384,
    "generation_max_length": 320,
    "generation_max_new_tokens": 320,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size":4,
    "gradient_accumulation_steps": 4,
    "bf16": True,
    "use_gloss": True,
    "gloss_max_hints": 6,
    "gloss_max_total_chars": 220,
    "gloss_max_match_len": 4,
    "gloss_max_lemma_freq": 1000000,
    "repetition_penalty": 1.15,
    "force_single_sentence": True,
    "single_sentence_mode": 'merge', 
    "data_dir": os.environ["COMP_DATA_DIR"],
    "post_eval_max_rows": 300,
})

out_rel = "configs/nmt_byt5_small.colab.override.json"
out_path = Path(os.environ["REPO_DIR"]) / out_rel
out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print("wrote:", out_path)

# 以後はこの設定を使う
os.environ["NMT_CONFIG"] = out_rel

# %%bash 用に環境変数を更新
env_lines = [
    f'export REPO_DIR="{os.environ["REPO_DIR"]}"',
    f'export COMP_DATA_DIR="{os.environ["COMP_DATA_DIR"]}"',
    f'export BYT5_VARIANT="{os.environ.get("BYT5_VARIANT", "base")}"',
    f'export MODEL_DATASET="{os.environ.get("MODEL_DATASET", "")}"',
    f'export MODEL_DIR="{os.environ["MODEL_DIR"]}"',
    f'export MODEL_NAME_OR_PATH="{os.environ.get("MODEL_NAME_OR_PATH", os.environ.get("MODEL_DIR", ""))}"',
    f'export NMT_CONFIG="{os.environ["NMT_CONFIG"]}"',
    f'export TRAIN_PATH="{os.environ.get("TRAIN_PATH", "artifacts/aligned/aligned_train.parquet")}"',
    f'export EVACUN_DIR="{os.environ.get("EVACUN_DIR", "/content/drive/MyDrive/kaggle_input/evacun_oracc_parallel_v0.1")}"',
    f'export RUN_EVACUN="{os.environ.get("RUN_EVACUN", "0")}"',
    f'export RUN_OCR="{os.environ.get("RUN_OCR", "0")}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/content/colab_env.sh").write_text("\n".join(env_lines) + "\n")
print("NMT_CONFIG:", os.environ["NMT_CONFIG"])
```

---

### セル 5-C：辞書グロス（eBL_Dictionary）を使う場合（任意）
`OA_Lexicon_eBL.csv` と `eBL_Dictionary.csv` が **`COMP_DATA_DIR` にあること**が前提です。  
有効化する場合は **学習と推論の両方で同じ設定**を使ってください。

方法A（config で指定）:
```python
# セル 5-B の cfg.update に追記する例
cfg.update({
    "use_gloss": True,
    "gloss_max_hints": 6,
    "gloss_max_total_chars": 220,
    "gloss_max_match_len": 4,
    "gloss_max_lemma_freq": 50,
    "data_dir": os.environ["COMP_DATA_DIR"],
    # 既定と別の場所に置く場合のみ指定（相対パスは data_dir 基準）
    # "oa_lexicon_path": "OA_Lexicon_eBL.csv",
    # "ebl_dictionary_path": "eBL_Dictionary.csv",
})
```

方法B（コマンドで指定）:
```bash
# 学習のコマンドに追加
--use-gloss --gloss-max-hints 8 --gloss-max-total-chars 220

# 推論のコマンドに追加
--use-gloss

# 既定と別の場所に置く場合のみ指定
--oa-lexicon /path/to/OA_Lexicon_eBL.csv
--ebl-dictionary /path/to/eBL_Dictionary.csv
```

---

### セル 5-D：OCR 追加並列を使う場合のフラグ（任意）
Kaggle の `RUN_OCR=True` に相当するフラグです。  
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
    f'export BYT5_VARIANT="{os.environ.get("BYT5_VARIANT", "base")}"',
    f'export MODEL_DATASET="{os.environ.get("MODEL_DATASET", "")}"',
    f'export MODEL_DIR="{os.environ["MODEL_DIR"]}"',
    f'export MODEL_NAME_OR_PATH="{os.environ.get("MODEL_NAME_OR_PATH", os.environ.get("MODEL_DIR", ""))}"',
    f'export NMT_CONFIG="{os.environ["NMT_CONFIG"]}"',
    f'export TRAIN_PATH="{os.environ.get("TRAIN_PATH", "artifacts/aligned/aligned_train.parquet")}"',
    f'export EVACUN_DIR="{os.environ.get("EVACUN_DIR", "/content/drive/MyDrive/kaggle_input/evacun_oracc_parallel_v0.1")}"',
    f'export RUN_EVACUN="{os.environ.get("RUN_EVACUN", "0")}"',
    f'export RUN_OCR="{os.environ["RUN_OCR"]}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/content/colab_env.sh").write_text("\n".join(env_lines) + "\n")
print("RUN_OCR:", os.environ["RUN_OCR"])
```

---

### セル 5-E：EvaCun 追加データを使う場合のフラグ（任意）
EvaCun を使う場合は `RUN_EVACUN = "1"` にして実行してください。

```python
import os
from pathlib import Path

RUN_EVACUN = "1"
os.environ["RUN_EVACUN"] = RUN_EVACUN

# %%bash 用に環境変数を更新
env_lines = [
    f'export REPO_DIR="{os.environ["REPO_DIR"]}"',
    f'export COMP_DATA_DIR="{os.environ["COMP_DATA_DIR"]}"',
    f'export BYT5_VARIANT="{os.environ.get("BYT5_VARIANT", "base")}"',
    f'export MODEL_DATASET="{os.environ.get("MODEL_DATASET", "")}"',
    f'export MODEL_DIR="{os.environ["MODEL_DIR"]}"',
    f'export MODEL_NAME_OR_PATH="{os.environ.get("MODEL_NAME_OR_PATH", os.environ.get("MODEL_DIR", ""))}"',
    f'export NMT_CONFIG="{os.environ["NMT_CONFIG"]}"',
    f'export TRAIN_PATH="{os.environ.get("TRAIN_PATH", "artifacts/aligned/aligned_train.parquet")}"',
    f'export EVACUN_DIR="{os.environ.get("EVACUN_DIR", "/content/drive/MyDrive/kaggle_input/evacun_oracc_parallel_v0.1")}"',
    f'export RUN_EVACUN="{os.environ["RUN_EVACUN"]}"',
    f'export RUN_OCR="{os.environ.get("RUN_OCR", "0")}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/content/colab_env.sh").write_text("\n".join(env_lines) + "\n")
print("RUN_EVACUN:", os.environ["RUN_EVACUN"])
```

---

### セル 6：まずは “超小さく” 動作確認（推奨）
いきなりフル学習に入る前に、**小規模で最後まで動くか**確認します。

```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
# 1) train を少量だけでアライン（--sample は train.csv の先頭N件）
python -m dp.align_train --config configs/align.yaml   --data-dir "${COMP_DATA_DIR}"   --drop-flagged   --sample 30

# 2) 学習も少量だけで回す（max_train_rows / max_val_rows）
python -m dp.train_nmt   --config "${NMT_CONFIG}"   --data-dir "${COMP_DATA_DIR}"   --train "${TRAIN_PATH}"   --variant C --drop-flagged   --out artifacts/nmt/byt5_small_dryrun   --max-train-rows 200 --max-val-rows 50   --post-eval-mode quick
```

---

### セル 7：本番（Colab での学習 → 評価）
#### 7-A) アライン（文ペア作成）
```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
python -m dp.align_train --config configs/align.yaml   --data-dir "${COMP_DATA_DIR}"   --variant C   --drop-flagged    --no-oare-sentences
```

出力: `artifacts/aligned/aligned_train.parquet`

> `Sentences_Oare_FirstWord_LinNum.csv` が `COMP_DATA_DIR` にある場合は **自動で OARE 文アライン補助を使えます**。  
> 明示的に切り替える場合は以下を使ってください。
> - 有効化: `--use-oare-sentences`
> - 無効化: `--no-oare-sentences`
> - 別パス指定: `--oare-sentences-path /path/to/Sentences_Oare_FirstWord_LinNum.csv`
> - デバッグログ: `--oare-debug`（`artifacts/aligned/oare_debug.csv` を出力）

#### 7-A2) OCR 追加並列（任意）
Kaggle の `RUN_OCR=True` 相当です。`セル 5-D` で `RUN_OCR="1"` にしてから実行してください。

```bash
%%bash
source /content/colab_env.sh
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

出力: `artifacts/ocr_pairs/mixed_train.parquet`  
統計ログ:  
- `artifacts/ocr/publications_candidates_stats.json`
- `artifacts/ocr_pairs/summary.json`
- `artifacts/ocr_pairs/mixed_train.stats.json`

#### 7-A2b) EvaCun 追加データの準備（任意）
`セル 5-E` で `RUN_EVACUN="1"` にしてから実行してください。

```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"

if [ "${RUN_EVACUN}" != "1" ]; then
  echo "RUN_EVACUN=0: skip EvaCun flow"
  exit 0
fi

# EvaCun の train/val を整形
python -m dp.prepare_evacun \
  --data-dir "${EVACUN_DIR}" \
  --src-lang transcription \
  --variant C \
  --out-dir artifacts/evacun
# 長文を抑える場合は --max-sentence-endings 1 を追加
```

出力:
- `artifacts/evacun/evacun_transcription_train.parquet`
- `artifacts/evacun/evacun_transcription_val.parquet`

#### 7-A2c) EvaCun 事前学習（任意・推奨）
EvaCun で事前学習し、その後 **コンペの `aligned_train` で微調整**する流れです。

- **事前学習（このセル）**: EvaCun（`transcription → English`）でモデルの翻訳能力/語彙カバレッジを作る
- **微調整（7-B）**: 事前学習済み ckpt を初期値にして、コンペの `aligned_train` で追加学習し「出力の癖/ドメイン」を寄せる
- **EvaCun だけで学習する場合**: このセル（7-A2c）まででOKで、7-B は不要です

`--post-eval-mode quick` は **指標（BLEU/chrF++/gm）だけ**を出す軽量モードです。  
7-B の `full` は **指標 + サンプル/診断（崩壊チェック等）**まで出すため、時間が増えます。  
※ `--save-val-preds` / `--save-val-audit` を使いたい場合は **`--post-eval-mode full` にしてください**（`quick` では保存されません）。

**7-A2c と 7-B の検証データを揃える（任意）**  
7-A2c → 7-B の順で実行すると、7-A2c 時点では `val_doc_ids.json` がまだ無いのが普通です。  
そこで「7-B と同じ split ルール（`NMT_CONFIG` の `val_ratio/seed/val_split_unit/...`）」で **固定 val**（`artifacts/val_fixed.parquet`）を先に作り、7-A2c/7-B の両方で `--val` に渡します。  
このセルは **学習を回さず**、データの分割だけ行います（`--variant` / `--drop-flagged` は 7-B と合わせてください）。

```python
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = os.environ.get("REPO_DIR", "/content/repo/translate-akkadian")
os.chdir(REPO_DIR)
sys.path.append(str(Path(REPO_DIR) / "src"))

from dp.train_nmt import parse_float, parse_int, prepare_df, read_table, split_train_val
from dp.utils import load_config

cfg = load_config(os.environ.get("NMT_CONFIG", "configs/nmt_byt5_small.yaml"))
seed = parse_int(cfg.get("seed"), 42)
val_ratio = parse_float(cfg.get("val_ratio"), 0.0)
val_split_unit = str(cfg.get("val_split_unit", "row")).strip().lower()
val_doc_id_col = str(cfg.get("val_doc_id_col", "oare_id")).strip()
val_source_col = str(cfg.get("val_source_col", "source")).strip()

_ves = cfg.get("val_exclude_sources", ["ocr"])
if _ves is None:
    val_exclude_sources = []
elif isinstance(_ves, str):
    val_exclude_sources = [s.strip() for s in _ves.replace(";", ",").split(",") if s.strip()]
else:
    try:
        val_exclude_sources = [str(s).strip() for s in _ves if str(s).strip()]
    except TypeError:
        val_exclude_sources = [str(_ves).strip()] if str(_ves).strip() else []

val_holdout_all_sources = bool(cfg.get("val_holdout_all_sources", True))

src_col = cfg.get("src_col", "src_sent")
tgt_col = cfg.get("tgt_col", "tgt_sent")
max_src_chars = cfg.get("max_src_chars")
max_tgt_chars = cfg.get("max_tgt_chars")
if max_src_chars is not None:
    max_src_chars = parse_int(max_src_chars, 0) or None
if max_tgt_chars is not None:
    max_tgt_chars = parse_int(max_tgt_chars, 0) or None

# 7-B の CLI と合わせる（必要なら変更してください）
VARIANT = "C"
DROP_FLAGGED = True
NORM_VARIANT = None

train_path = Path(os.environ.get("TRAIN_PATH", "artifacts/aligned/aligned_train.parquet"))
df = read_table(train_path)
df = prepare_df(
    df,
    src_col=src_col,
    tgt_col=tgt_col,
    variant=VARIANT,
    drop_flagged=DROP_FLAGGED,
    norm_variant=NORM_VARIANT,
    max_src_chars=max_src_chars,
    max_tgt_chars=max_tgt_chars,
)
_, val_df = split_train_val(
    df,
    val_ratio=val_ratio,
    seed=seed,
    split_unit=val_split_unit,
    doc_id_col=val_doc_id_col,
    source_col=val_source_col,
    exclude_sources=val_exclude_sources,
    holdout_all_sources=val_holdout_all_sources,
)
if val_df is None or val_df.empty:
    raise RuntimeError("val_df is empty; check val_ratio / train data")

Path("artifacts").mkdir(parents=True, exist_ok=True)
val_df.to_parquet("artifacts/val_fixed.parquet", index=False)
print("Saved: artifacts/val_fixed.parquet")

val_doc_ids = []
if val_doc_id_col in val_df.columns:
    val_doc_ids = sorted(val_df[val_doc_id_col].dropna().astype(str).unique().tolist())
Path("artifacts/val_fixed_doc_ids.json").write_text(
    json.dumps(val_doc_ids, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("Saved: artifacts/val_fixed_doc_ids.json")
```

```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"

MODEL_ARG="--model-name-or-path ${MODEL_NAME_OR_PATH}"
VAL_PATH="artifacts/val_fixed.parquet"
if [ ! -f "${VAL_PATH}" ]; then
  # 固定 val が無ければ EvaCun の val を使う
  VAL_PATH="artifacts/evacun/evacun_transcription_val.parquet"
fi
python -m dp.train_nmt \
  --config "${NMT_CONFIG}" \
  --data-dir "${COMP_DATA_DIR}" \
  --train artifacts/evacun/evacun_transcription_train.parquet \
  --val "${VAL_PATH}" \
  --variant C \     #検証データを揃える場合
  --drop-flagged \  #検証データを揃える場合
  --out artifacts/nmt/byt5_evacun_pre \
  ${MODEL_ARG} \
  --post-eval-mode quick
```

#### 7-A2c-1) 事前学習済みモデルを Drive に保存（任意・再利用向け）
7-A2c は重いので、作成したモデル（`artifacts/nmt/byt5_evacun_pre`）を **Drive に名前付きで保存**しておくと、
セッションを跨いで何度でも 7-B の初期値として再利用できます。

```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"

# ===== 好きな名前/保存先に変えてください =====
PRETRAIN_NAME="byt5_evacun_pre_v1"
DRIVE_PRETRAIN_DIR="/content/drive/MyDrive/translate-akkadian/models/${PRETRAIN_NAME}"

SRC_DIR="${REPO_DIR}/artifacts/nmt/byt5_evacun_pre"
mkdir -p "${DRIVE_PRETRAIN_DIR}"

# rsync があれば高速/安全（無ければ cp -a にフォールバック）
if command -v rsync >/dev/null 2>&1; then
  time rsync -a --info=progress2 "${SRC_DIR}/" "${DRIVE_PRETRAIN_DIR}/"
else
  time cp -a "${SRC_DIR}/." "${DRIVE_PRETRAIN_DIR}/"
fi

du -sh "${DRIVE_PRETRAIN_DIR}" || true
```

時間の目安:
- サイズ（ByT5-base は数GBになりやすい）と Drive の実効速度次第で、**数十秒〜数分**（環境によりそれ以上）です。
- 正確には上の `time rsync ...` の結果で確認してください。

#### 7-A2c-2) Drive のモデルを /content に復元して 7-B で使う（推奨）
Drive 直読みより、`/content` にコピーしてから使う方が **読み込みが安定/高速**になりやすいです。

```bash
%%bash
source /content/colab_env.sh

# ===== 7-A2c-1 で保存したパスに合わせてください =====
PRETRAIN_NAME="byt5_evacun_pre_v1"
DRIVE_PRETRAIN_DIR="/content/drive/MyDrive/translate-akkadian/models/${PRETRAIN_NAME}"

LOCAL_PRETRAIN_DIR="/content/models/${PRETRAIN_NAME}"
mkdir -p "${LOCAL_PRETRAIN_DIR}"

if command -v rsync >/dev/null 2>&1; then
  time rsync -a --info=progress2 "${DRIVE_PRETRAIN_DIR}/" "${LOCAL_PRETRAIN_DIR}/"
else
  time cp -a "${DRIVE_PRETRAIN_DIR}/." "${LOCAL_PRETRAIN_DIR}/"
fi

ls -la "${LOCAL_PRETRAIN_DIR}" | head
```

> **微調整（7-B）では `MODEL_ARG` を事前学習 ckpt に切り替えて実行してください。**  
> - 同一セッション内でそのまま使う: `MODEL_ARG="--model-name-or-path artifacts/nmt/byt5_evacun_pre"`  
> - Drive から復元して使う（推奨）: `MODEL_ARG="--model-name-or-path /content/models/byt5_evacun_pre_v1"`

#### 7-A2d) EvaCun 混合（任意）
OCR 混合と同時には使わず、**どちらか一方を選んでください**。

```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"

if [ "${RUN_EVACUN}" != "1" ]; then
  echo "RUN_EVACUN=0: skip EvaCun mix"
  exit 0
fi

python -m dp.mix_evacun \
  --config "${NMT_CONFIG}" \
  --aligned artifacts/aligned/aligned_train.parquet \
  --evacun artifacts/evacun/evacun_transcription_train.parquet \
  --out artifacts/evacun/mixed_train.parquet \
  --ratio 0.3 --variants C
# 長文を抑える場合は --max-sentence-endings 1 を追加
```

出力:
- `artifacts/evacun/mixed_train.parquet`
- `artifacts/evacun/mixed_train.stats.json`

#### 7-A2e) 学習データを EvaCun 混合に切り替え（任意）
```python
import os
from pathlib import Path

TRAIN_PATH = "artifacts/aligned/aligned_train.parquet"
mixed_path = Path(REPO_DIR) / "artifacts/evacun/mixed_train.parquet"
if mixed_path.exists():
    TRAIN_PATH = "artifacts/evacun/mixed_train.parquet"

os.environ["TRAIN_PATH"] = TRAIN_PATH

# %%bash 用に環境変数を更新
env_lines = [
    f'export REPO_DIR="{os.environ["REPO_DIR"]}"',
    f'export COMP_DATA_DIR="{os.environ["COMP_DATA_DIR"]}"',
    f'export BYT5_VARIANT="{os.environ.get("BYT5_VARIANT", "base")}"',
    f'export MODEL_DATASET="{os.environ.get("MODEL_DATASET", "")}"',
    f'export MODEL_DIR="{os.environ["MODEL_DIR"]}"',
    f'export MODEL_NAME_OR_PATH="{os.environ.get("MODEL_NAME_OR_PATH", os.environ.get("MODEL_DIR", ""))}"',
    f'export NMT_CONFIG="{os.environ["NMT_CONFIG"]}"',
    f'export TRAIN_PATH="{os.environ["TRAIN_PATH"]}"',
    f'export EVACUN_DIR="{os.environ.get("EVACUN_DIR", "/content/drive/MyDrive/kaggle_input/evacun_oracc_parallel_v0.1")}"',
    f'export RUN_EVACUN="{os.environ.get("RUN_EVACUN", "0")}"',
    f'export RUN_OCR="{os.environ.get("RUN_OCR", "0")}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/content/colab_env.sh").write_text("\n".join(env_lines) + "\n")
print("TRAIN_PATH:", os.environ["TRAIN_PATH"])
```

#### 7-A3) 学習データを OCR 混合に切り替え（任意）
OCR を実行した場合は、学習データを `mixed_train.parquet` に切り替えます。

```python
import os
from pathlib import Path

TRAIN_PATH = "artifacts/aligned/aligned_train.parquet"
mixed_path = Path(REPO_DIR) / "artifacts/ocr_pairs/mixed_train.parquet"
if mixed_path.exists():
    TRAIN_PATH = "artifacts/ocr_pairs/mixed_train.parquet"

os.environ["TRAIN_PATH"] = TRAIN_PATH

# %%bash 用に環境変数を更新
env_lines = [
    f'export REPO_DIR="{os.environ["REPO_DIR"]}"',
    f'export COMP_DATA_DIR="{os.environ["COMP_DATA_DIR"]}"',
    f'export BYT5_VARIANT="{os.environ.get("BYT5_VARIANT", "base")}"',
    f'export MODEL_DATASET="{os.environ.get("MODEL_DATASET", "")}"',
    f'export MODEL_DIR="{os.environ["MODEL_DIR"]}"',
    f'export MODEL_NAME_OR_PATH="{os.environ.get("MODEL_NAME_OR_PATH", os.environ.get("MODEL_DIR", ""))}"',
    f'export NMT_CONFIG="{os.environ["NMT_CONFIG"]}"',
    f'export TRAIN_PATH="{os.environ["TRAIN_PATH"]}"',
    f'export EVACUN_DIR="{os.environ.get("EVACUN_DIR", "/content/drive/MyDrive/kaggle_input/evacun_oracc_parallel_v0.1")}"',
    f'export RUN_EVACUN="{os.environ.get("RUN_EVACUN", "0")}"',
    f'export RUN_OCR="{os.environ.get("RUN_OCR", "0")}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/content/colab_env.sh").write_text("\n".join(env_lines) + "\n")
print("TRAIN_PATH:", os.environ["TRAIN_PATH"])
```

**NOTE（validation split の固定化）**

- `dp.train_nmt` は **`--val` を指定しない場合**、config の設定により **validation を OARE 由来の aligned 行だけから作成**します。
- さらに `oare_id` 単位（doc 単位）で split することで、同一ドキュメントの文が train/val に跨らないようにしています。
- OCR 混合（`mixed_train.parquet`）を使っても、**OCR 行は train のみに入り、val には入りません**。
- EvaCun 混合でも **EvaCun 行は train のみに入り、val には入りません**（`val_exclude_sources` に `evacun` を含むため）。
- split は `seed` で決まり、`--val` を指定しない場合は `--out` 配下に `val_doc_ids.json`（val に入った `oare_id` の一覧）が保存されます。
- 固定 val（`artifacts/val_fixed.parquet`）を使う場合は `val_doc_ids.json` は作られないので、doc id が必要なら `artifacts/val_fixed_doc_ids.json` を使ってください。

#### 7-B) 学習（NMT）＋ validation 評価（BLEU / chrF++ / gm）
```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
# ★モデル指定（base/large）:
# - Drive にモデルがあればそれを優先（MODEL_DIR）
# - 無ければ Hugging Face（google/byt5-<variant>）へフォールバック（Colab はネット接続できる想定）

MODEL_ARG="--model-name-or-path ${MODEL_NAME_OR_PATH}"
VAL_PATH="artifacts/val_fixed.parquet"
VAL_ARG=""
if [ -f "${VAL_PATH}" ]; then
  # 7-A2c と同じ固定 val がある場合はそれを使う
  VAL_ARG="--val ${VAL_PATH}"
fi
python -m dp.train_nmt   --config "${NMT_CONFIG}"   --data-dir "${COMP_DATA_DIR}"   --train "${TRAIN_PATH}"   --variant C --drop-flagged   --out artifacts/nmt/byt5_small_colab   ${MODEL_ARG}   ${VAL_ARG}   --post-eval-mode full   --save-val-preds --save-val-audit

# 実行後、以下が `--out` 配下に保存されます:
# - val_predictions.csv : src/ref/pred（監査の元データ）
# - val_audit.csv       : ずれの型分類・診断列付き
```

#### val_todo.csv（監査の「見る順」を作る：任意だがおすすめ）

`val_audit.csv` から「前処理/後処理で直せそうな行」を優先度付きで抽出し、`val_todo.csv` を作れます。

```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
python -m dp.todo_csv \
  --input  artifacts/nmt/byt5_small_colab/val_audit.csv \
  --output artifacts/nmt/byt5_small_colab/val_todo.csv \
  --max-rows 200 --max-rows-per-doc 5
```

- `priority` の高い行から目視して原因を分類し、**「前処理で直るか」**を効率良く見極めるのが目的です。

ログ中に以下のような行が出ます（例）：
- `[eval_metrics] bleu=... chrf=... gm=...`

> 反復を速くしたい場合は `--post-eval-mode quick`（指標のみ）にしてください。

#### 7-C) 監査CSVをざっと見る（ずれの型分類）
```python
import pandas as pd
from pathlib import Path

audit_path = Path(REPO_DIR) / "artifacts/nmt/byt5_small_colab/val_audit.csv"

df = pd.read_csv(audit_path)

# 型の頻度トップ（まずは多い原因から潰す）
display(df["type_primary"].value_counts().head(20))

# スコア（sim_self）が低い順に目視確認
worst = df.sort_values("sim_self").head(30)
cols = [
    c
    for c in [
        "oare_id",
        "src_no_gloss",
        "ref",
        "pred",
        "type_primary",
        "t90_reason",
        "type_secondary",
        "pred_template_count",
        "ref_digit",
        "pred_digit",
        "ref_name_cnt",
        "pred_name_cnt",
        "fix_route",
        "sim_self",
        "best_offset",
        "best_delta",
    ]
    if c in worst.columns
]
display(worst[cols])
```

（任意）Colab から CSV をダウンロードする場合:
```python
from google.colab import files
files.download(str(audit_path))
```

---

## 2. A100/H100 向け：bf16 を有効にする（任意の例）
A100/H100 では bf16 が速く、学習が安定しやすいです。  
**セル 5-B で `bf16: true` を設定済みなら、このセクションは不要**です（同じ上書きを行うため）。

このリポジトリの `dp.train_nmt` は **bf16 を指定しても GPU が対応していなければ自動で無効化**します（例：T4 など）。  
そのため、Colab の GPU が何であっても `bf16: true` を入れて試してOKです。

### 例：runtime config（json）を作って bf16 を上書き
```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
python - <<'PY'
import json
import os
from pathlib import Path

import sys
sys.path.append("src")
from dp.utils import load_config

base_config = os.environ.get("NMT_CONFIG", "configs/nmt_byt5_small.yaml")
cfg = load_config(base_config)

# 上書き（必要に応じて調整）
cfg.update({
    "bf16": True,
    "fp16": False,
    # A100/H100なら batch を上げやすい（OOMしたら下げる）
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,
})

out_rel = "configs/nmt_byt5_small.colab.runtime.json"
out_path = Path(out_rel)
out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print("wrote:", out_path)

# 以後はこの設定を使う
os.environ["NMT_CONFIG"] = out_rel
env_lines = [
    f'export REPO_DIR="{os.environ["REPO_DIR"]}"',
    f'export COMP_DATA_DIR="{os.environ["COMP_DATA_DIR"]}"',
    f'export MODEL_DIR="{os.environ["MODEL_DIR"]}"',
    f'export NMT_CONFIG="{os.environ["NMT_CONFIG"]}"',
    f'export TRAIN_PATH="{os.environ.get("TRAIN_PATH", "artifacts/aligned/aligned_train.parquet")}"',
    f'export EVACUN_DIR="{os.environ.get("EVACUN_DIR", "/content/drive/MyDrive/kaggle_input/evacun_oracc_parallel_v0.1")}"',
    f'export RUN_EVACUN="{os.environ.get("RUN_EVACUN", "0")}"',
    f'export RUN_OCR="{os.environ.get("RUN_OCR", "0")}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/content/colab_env.sh").write_text("\n".join(env_lines) + "\n")
print("NMT_CONFIG:", os.environ["NMT_CONFIG"])
PY
```

以後の学習セルは `NMT_CONFIG` を参照するため、上のセルで bf16 用の設定に切り替わります。
```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
MODEL_ARG="--model-name-or-path ${MODEL_NAME_OR_PATH}"
python -m dp.train_nmt   --config "${NMT_CONFIG}"   --data-dir "${COMP_DATA_DIR}"   --train "${TRAIN_PATH}"   --variant C --drop-flagged   --out artifacts/nmt/byt5_small_colab_bf16   ${MODEL_ARG}   --post-eval-mode quick
```

---

## 3. （任意）推論 → submission.csv 生成（Colab で動作確認）
Kaggle 提出自体は Kaggle が再実行して採点しますが、**パイプラインの動作確認**としては Colab でも submission を作れます。

```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
# 1) test.csv に対して推論
python -m dp.infer_nmt   --config "${NMT_CONFIG}"   --ckpt artifacts/nmt/byt5_small_colab   --data-dir "${COMP_DATA_DIR}"   --out artifacts/predictions.csv

# 2) submission.csv を作成
python -m dp.submit   --pred artifacts/predictions.csv   --data-dir "${COMP_DATA_DIR}"   --out submission.csv

# 3) 形式チェック
python -m dp.validate   --submission submission.csv   --data-dir "${COMP_DATA_DIR}"
```

---

## 3-A. （任意）ログitsアンサンブル（確率平均）で推論する（最終提出向け）

複数の学習済みモデル（例：seed 違い）の **各ステップの出力確率（logits）を平均してからデコード**します。
単純な「文字列の多数決」より安定して伸びやすい一方、**推論時間・VRAM がモデル本数分だけ増えます**。

- まずは **2本アンサンブル**がおすすめ（伸びやすく、コストも現実的）
- 余裕があれば seed を 3〜5 本まわして、良さそうな 2〜3 本を採用…という運用がやりやすいです

**使い方（CLI）**（カンマ区切りで複数 ckpt を指定）
```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
# ckpt のパスは自分の学習結果に合わせて変更
python -m dp.infer_nmt \
  --config "${NMT_CONFIG}" \
  --ckpts artifacts/nmt/byt5_seed42,artifacts/nmt/byt5_seed43 \
  --data-dir "${COMP_DATA_DIR}" \
  --out artifacts/predictions_ens.csv
```

`--ckpts` の代わりに config へ書くこともできます（Notebook 側で編集しやすい）：
```yaml
ensemble_ckpts:
  - artifacts/nmt/byt5_seed42
  - artifacts/nmt/byt5_seed43
```
※ `NMT_CONFIG` が JSON でも同じキー名で配列を持たせれば OK です。

**推論時に [eval_metrics] を出す（ref 列があるデータのみ）**
参照（正解）列を含むデータで `--ref-col` を指定すると BLEU/chrF++/gm を計算して出力します。
学習ログの `[eval_metrics]` と比較するための簡易チェック用途です（test には ref が無いので通常は使いません）。

```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
python -m dp.infer_nmt \
  --config "${NMT_CONFIG}" \
  --ckpts artifacts/nmt/byt5_seed42,artifacts/nmt/byt5_seed43 \
  --test artifacts/aligned/aligned_train.parquet \
  --src-col src_sent --ref-col tgt_sent --id-col oare_id \
  --out artifacts/pred_debug.csv
```
※ `translation` は例です。データの正解列名（例：`target` / `tgt_sent`）に合わせて変更してください。

---

## 3-B. （任意）正規化の効果を val で比較
固定 val（`artifacts/val_fixed.parquet`）がある場合は、それをそのまま使うのが一番簡単です。  
無い場合は `dp.train_nmt` の出力（`val_doc_ids.json`）から val を復元します。

```python
import json
import os
from pathlib import Path

import pandas as pd
os.chdir("/content/repo/translate-akkadian")

val_fixed_path = Path("artifacts/val_fixed.parquet")

if val_fixed_path.exists():
    val_df = pd.read_parquet(val_fixed_path)
else:
    # val の doc id を読み込む（固定 val を作っていれば val_fixed_doc_ids.json がある）
    ids_path = Path("artifacts/val_fixed_doc_ids.json")
    if not ids_path.exists():
        ids_path = Path("artifacts/nmt/byt5_small_colab/val_doc_ids.json")
    val_ids = json.loads(ids_path.read_text())

    # 学習に使ったデータを読み込む
    train_path = os.environ["TRAIN_PATH"]
    if train_path.endswith(".parquet"):
        df = pd.read_parquet(train_path)
    else:
        df = pd.read_csv(train_path)

    val_df = df[df["oare_id"].isin(val_ids)].copy().reset_index(drop=True)
    # source 列がある場合は、デフォルト設定に合わせて OCR/EvaCun を除外（val split と揃える）
    if "source" in val_df.columns:
        val_df = val_df[~val_df["source"].isin(["ocr", "evacun"])].copy().reset_index(drop=True)

# 学習時の列名に合わせて変更（デフォルトは src_sent / tgt_sent）
SRC_COL = "src_sent"
TGT_COL = "tgt_sent"
val_df["id"] = range(len(val_df))

Path("artifacts").mkdir(parents=True, exist_ok=True)
val_df[["id", SRC_COL]].rename(columns={SRC_COL: "transliteration"}).to_csv(
    "artifacts/val_input.csv",
    index=False,
)
val_df[["id", TGT_COL]].rename(columns={TGT_COL: "translation"}).to_csv(
    "artifacts/val_gold.csv",
    index=False,
)
print("Saved: artifacts/val_input.csv / artifacts/val_gold.csv")
```

```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"

# 正規化あり
python -m dp.infer_nmt \
  --config "${NMT_CONFIG}" \
  --ckpt artifacts/nmt/byt5_small_colab \
  --test artifacts/val_input.csv \
  --out artifacts/val_pred_norm.csv \
  --normalize-output

# 正規化なし
python -m dp.infer_nmt \
  --config "${NMT_CONFIG}" \
  --ckpt artifacts/nmt/byt5_small_colab \
  --test artifacts/val_input.csv \
  --out artifacts/val_pred_raw.csv \
  --no-normalize-output

# gm を比較
python -m dp.eval --pred artifacts/val_pred_norm.csv --gold artifacts/val_gold.csv
python -m dp.eval --pred artifacts/val_pred_raw.csv --gold artifacts/val_gold.csv
```

---

## 4. よくあるトラブルと対処

### Q1. `train.csv not found` / `test.csv not found`
- `COMP_DATA_DIR` が正しいか確認してください
- `ls -la $COMP_DATA_DIR` で `train.csv` / `test.csv` が見える必要があります

### Q2. OOM（CUDA out of memory）
- `per_device_train_batch_size` を下げる
- `max_source_length` / `max_target_length` を下げる
- `gradient_accumulation_steps` を上げて実効バッチを維持する
- `--post-eval-mode quick` にして評価コストを下げる

### Q3. Drive 上で学習が遅い
- repo と `artifacts/` は `/content`（ローカル）に置く
- 学習後に必要な成果物だけ Drive にコピーする

### Q4. zipが上書きできません
- 下記で上書きできます
```
%%bash
unzip -q -o /content/translate-akkadian.zip -d /content/repo
```

---

## 5. このREADMEの範囲
- Colab で「学習→評価（BLEU/chrF++/gmログ）」を回すための最小手順を提供します
- Kaggle 提出（hidden test の採点）は Kaggle Notebook 側で行ってください  
  Kaggle 用は `README_KAGGLE.md` を参照してください
