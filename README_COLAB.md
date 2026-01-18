# Colab 用 README（最小ノートブック：セットアップ → 学習 → 評価）

このリポジトリ（translate-akkadian）は **Kaggle「Deep Past Challenge（Akkadian → English）」**向けの学習・推論パイプラインです。  
Kaggle Notebook では GPU が T4 / P100 などに限られますが、Google Colab では A100 / H100 などの GPU が選べるため、**学習と検証（CV・val）を Colab 側で高速に回す**ための手順をまとめます。

> ここでの「評価」は、`dp.train_nmt` が学習後に validation に対して generate を行い、**BLEU / chrF++ / gm（幾何平均）**をログ表示する部分を指します。  
> Kaggle の hidden test は Kaggle 上でしか回せません（＝最終提出スコアは Kaggle が再実行して採点）。

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
```

- 公式データ（train.csv / test.csv / sample_submission.csv …）
  - `/content/kaggle/input/deep-past-initiative-machine-translation`
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

# 確認
ls -la /content/kaggle/input/deep-past-initiative-machine-translation | head
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
    f'export RUN_OCR="{os.environ["RUN_OCR"]}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/content/colab_env.sh").write_text("\n".join(env_lines) + "\n")
print("RUN_OCR:", os.environ["RUN_OCR"])
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
    f'export RUN_OCR="{os.environ.get("RUN_OCR", "0")}"',
    f'export PYTHONPATH="{os.environ["REPO_DIR"]}/src"',
]
Path("/content/colab_env.sh").write_text("\n".join(env_lines) + "\n")
print("TRAIN_PATH:", os.environ["TRAIN_PATH"])
```

**NOTE（validation split の固定化）**

- `dp.train_nmt` は、config の設定により **validation を OARE 由来の aligned 行だけから作成**します。
- さらに `oare_id` 単位（doc 単位）で split することで、同一ドキュメントの文が train/val に跨らないようにしています。
- OCR 混合（`mixed_train.parquet`）を使っても、**OCR 行は train のみに入り、val には入りません**。
- split は `seed` で決まり、`--out` 配下に `val_doc_ids.json`（val に入った `oare_id` の一覧）が保存されます。

#### 7-B) 学習（NMT）＋ validation 評価（BLEU / chrF++ / gm）
```bash
%%bash
source /content/colab_env.sh
cd "$REPO_DIR"
# ★モデル指定（base/large）:
# - Drive にモデルがあればそれを優先（MODEL_DIR）
# - 無ければ Hugging Face（google/byt5-<variant>）へフォールバック（Colab はネット接続できる想定）

MODEL_ARG="--model-name-or-path ${MODEL_NAME_OR_PATH}"
python -m dp.train_nmt   --config "${NMT_CONFIG}"   --data-dir "${COMP_DATA_DIR}"   --train "${TRAIN_PATH}"   --variant C --drop-flagged   --out artifacts/nmt/byt5_small_colab   ${MODEL_ARG}   --post-eval-mode full   --save-val-preds --save-val-audit

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
`dp.train_nmt` 実行後に `artifacts/nmt/byt5_small_colab/val_doc_ids.json` がある前提です。
学習時と同じ `TRAIN_PATH` と列名を使って、val 用の入力/正解を作ります。

```python
import json
import os
from pathlib import Path

import pandas as pd
os.chdir("/content/repo/translate-akkadian")

# val の doc id を読み込む
val_ids = json.loads(Path("artifacts/nmt/byt5_small_colab/val_doc_ids.json").read_text())

# 学習に使ったデータを読み込む
train_path = os.environ["TRAIN_PATH"]
if train_path.endswith(".parquet"):
    df = pd.read_parquet(train_path)
else:
    df = pd.read_csv(train_path)

# 学習時の列名に合わせて変更（デフォルトは src_sent / tgt_sent）
SRC_COL = "src_sent"
TGT_COL = "tgt_sent"

val_df = df[df["oare_id"].isin(val_ids)].copy().reset_index(drop=True)
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
