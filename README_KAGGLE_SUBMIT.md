# Kaggle 提出用 README（学習済み forward / reverse モデル読込 → 推論 / rerank → submission.csv）

この README は、**Google Colab で学習済みの forward モデル（Akkadian→English）**と  
**rerank 用 reverse モデル（English→Akkadian）**を Kaggle Notebook に持ち込み、  
**Kaggle 上で `submission.csv` を生成して提出**できるようにする手順です。

> 注意：このコンペは「Code形式」です。Kaggle は「提出に紐づいた Notebook」を再実行し、  
> `/kaggle/working/submission.csv` を提出物として扱います（ローカルで作った CSV をアップロード提出する形式ではありません）。

---

## 0. 事前に用意する Kaggle Dataset（重要）

Kaggle Notebook の `Data` タブで追加できるよう、以下を **Dataset としてアップロード**してください。

### 0-1) リポジトリ（コード）
- `translate-akkadian.zip`（このリポジトリ一式）
  - 既に `scripts/build_kaggle_bundle.sh` で軽量 bundle を作っているなら、それを推奨（速い＆容量小）

**ローカルで bundle を作る例（任意）**
```bash
# repo 直下で
bash scripts/build_kaggle_bundle.sh
cd kaggle_bundle
zip -r translate-akkadian.zip translate-akkadian
```

### 0-2) 学習済みモデル（必須）
Kaggle の再実行はオフラインになる可能性があるため、**Hugging Face からの自動ダウンロードに頼らず**、
Colab/Drive にある学習済みモデルを **そのまま Kaggle Dataset に入れてください**。

- forward モデル（Akkadian → English）: 例 `fwd_ckpt/`
- reverse モデル（English → Akkadian）: 例 `rev_ckpt/`

> 目安：Hugging Face 形式の checkpoint ディレクトリ（`config.json`, `tokenizer.*`, `pytorch_model*.bin` 等）が揃っていれば OK です。

#### Dataset 内の推奨ディレクトリ構成（例）
```
my-akkadian-models/           (Kaggle Dataset 名)
  fwd/                        (forward ckpt ルート)
    config.json
    tokenizer.json
    pytorch_model.bin
    ...
  rev/                        (reverse ckpt ルート)
    config.json
    tokenizer.json
    pytorch_model.bin
    ...
```

### 0-3) （任意）手元の評価用ファイル（CV/oracle を Kaggle でも回したい場合）
下記を Kaggle Dataset にまとめて入れておくと、Kaggle 上でも `dp.eval_rerank_methods` などで
**ローカルスコア（BLEU/chrF++/gm）**を確認できます。

- `val_for_oracle.csv`
- `cands_beam64.csv`
- `cands_sample_t0.9_p0.95.csv`
- `cands_tmpp_top50.csv`
- `cands_editor.csv`

---

## 1. Kaggle Notebook で追加する Data
Kaggle で **New Notebook** を作り、`Data` から以下を追加します。

- Competition の公式データ（`train.csv`, `test.csv` などが入っているもの）
- 0-1) のリポジトリ Dataset（`translate-akkadian.zip`）
- 0-2) のモデル Dataset（forward/reverse）

> `Data` タブに出る **実ディレクトリ名**が `/kaggle/input/<ここ>` になります。  
> Notebook 内で `!ls /kaggle/input` して確認してください。

---

## 2. Notebook（セル）: セットアップ

### セル 1：リポジトリを展開
```bash
%%bash
set -eux

# Dataタブの dataset 名に合わせて修正
REPO_DATASET="translate-akkadian"   # 例: あなたの repo dataset 名
ZIP_PATH="/kaggle/input/${REPO_DATASET}/translate-akkadian.zip"

mkdir -p /kaggle/working/repo
unzip -q "${ZIP_PATH}" -d /kaggle/working/repo

ls -la /kaggle/working/repo
ls -la /kaggle/working/repo/translate-akkadian | head
```

> zip ではなくフォルダで入っている場合は `unzip` を省略し、`REPO_DIR` を実パスに合わせてください。

### セル 2：共通 env を作る（重要）
`%%bash` セルは「セルごとに別プロセス」なので、毎回パスを打ち直さないよう  
`/kaggle/working/kaggle_env.sh` を作って、各 bash セルで `source` します。

```bash
%%bash
set -eux

cat > /kaggle/working/kaggle_env.sh <<'EOF'
# ===== offline 実行の保険 =====
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_DISABLE_TELEMETRY="1"
export HF_HOME="/kaggle/working/hf"

# ===== リポジトリ =====
export REPO_DIR="/kaggle/working/repo/translate-akkadian"
export PYTHONPATH="${REPO_DIR}/src"

# ===== コンペデータ（train.csv/test.csv のある場所）=====
export COMP_DATA_DIR="/kaggle/input/deep-past-initiative-machine-translation"  # Dataタブに合わせて修正

# ===== 学習済みモデル Dataset =====
export MODEL_DATASET="my-akkadian-models"   # Dataタブに合わせて修正
export FWD_CKPT="/kaggle/input/${MODEL_DATASET}/fwd"   # forward(Akkadian->English)
export REV_CKPT="/kaggle/input/${MODEL_DATASET}/rev"   # reverse(English->Akkadian)

# ===== config =====
export CFG="${REPO_DIR}/configs/nmt_byt5_small.yaml"   # 学習時に使っていた config に合わせて修正
EOF

mkdir -p /kaggle/working/hf
cat /kaggle/working/kaggle_env.sh
```

### セル 3：必要パッケージの確認（※基本は Kaggle 標準で足りる想定）
オフライン提出を想定すると、`pip install` は避けたいです。  
まずは import が通るか確認してください。

```python
import importlib

pkgs = ["pandas","sklearn","pyarrow","torch","transformers","datasets","sentencepiece","sacrebleu"]
missing = []
for p in pkgs:
    try:
        importlib.import_module(p)
    except Exception:
        missing.append(p)

print("missing:", missing)
```

- `missing` が空なら OK  
- もし欠けている場合：
  - 開発中のみ Internet=ON で `pip install -r requirements.txt` して動作確認 →  
    **提出用には Internet=OFF にしても動く状態**にする（Kaggle の再実行で弾かれるケースがあるため）

### セル 4：パス存在チェック（任意）
```bash
%%bash
set -eux
source /kaggle/working/kaggle_env.sh

python - <<'PY'
import os
from pathlib import Path

for k in ["REPO_DIR","COMP_DATA_DIR","FWD_CKPT","REV_CKPT","CFG","HF_HOME"]:
    p = os.environ.get(k,"")
    print(k, p, "exists=", Path(p).exists())
PY
```

---

## 3. 提出ファイル生成（2パターン）

### パターンA：最速（forward 1-best 推論のみ）
まずは **提出が通ること**を確認したい時の最短ルートです。

#### セル A1：推論 → submission.csv
```bash
%%bash
set -eux
source /kaggle/working/kaggle_env.sh
cd "${REPO_DIR}"

python -m dp.infer_nmt \
  --config "${CFG}" \
  --ckpt "${FWD_CKPT}" \
  --data-dir "${COMP_DATA_DIR}" \
  --norm-variant C \
  --out /kaggle/working/predictions.csv

python -m dp.submit \
  --config "${CFG}" \
  --pred /kaggle/working/predictions.csv \
  --data-dir "${COMP_DATA_DIR}" \
  --out /kaggle/working/submission.csv

python -m dp.validate \
  --submission /kaggle/working/submission.csv \
  --data-dir "${COMP_DATA_DIR}"
```

---

### パターンB：rerank（reverse モデル使用 / lambda=0.5）
あなたの前提（`dp.eval_rerank_methods` を使う場合の best lambda=0.5）に合わせて、  
**候補生成 → noisy-channel rerank → submission.csv** を作ります。

#### 重要：Kaggle 実行時間を守るための注意
`beam64` や `sampling 32×4` のような「Oracle 用の大量候補」は、Kaggle の再実行で時間超過しやすいです。  
提出向けには、まず **少ない候補数（例：beam8 + sample4×1）** で動作確認し、必要なら増やしてください。

---

#### セル B1：候補生成（nbest / sampling）
```bash
%%bash
set -eux
source /kaggle/working/kaggle_env.sh
cd "${REPO_DIR}"

mkdir -p /kaggle/working/cands

# 例1) beam n-best（k=8）
python -m dp.infer_nmt_nbest \
  --config "${CFG}" \
  --ckpt "${FWD_CKPT}" \
  --test "${COMP_DATA_DIR}/test.csv" \
  --src-col transliteration --id-col id \
  --norm-variant C \
  --out /kaggle/working/cands/cands_beam8.csv \
  --k 8 --num-beams 8 --tag beam8 \
  --save-forward-score

# 例2) sampling（k=4, runs=1）
python -m dp.infer_nmt_nbest \
  --config "${CFG}" \
  --ckpt "${FWD_CKPT}" \
  --test "${COMP_DATA_DIR}/test.csv" \
  --src-col transliteration --id-col id \
  --norm-variant C \
  --out /kaggle/working/cands/cands_sample_t0.9_p0.95.csv \
  --do-sample --temperature 0.9 --top-p 0.95 \
  --k 4 --runs 1 --seed 42 --tag sample_t0.9_p0.95 \
  --save-forward-score
```

> 追加の候補（`cands_tmpp_top50.csv`, `cands_editor.csv`）を使う場合は、  
> それらの生成（あるいは事前生成済みファイルを Dataset として追加）を行い、  
> 次のセル B2 の `--cands` に並べてください。

---

#### セル B2：noisy-channel rerank（reverse スコア + 0.5 * forward スコア）
`dp.eval_rerank_methods` は本来 validation 用ですが、test 用に「ダミー gold」を作って流用します。  
（gold は空文字で OK。スコアは意味を持ちませんが、`pred_noisy_channel.csv` が得られます）

```bash
%%bash
set -eux
source /kaggle/working/kaggle_env.sh
cd "${REPO_DIR}"

# test.csv から pseudo-val を作る（id, src, dummy_ref）
python - <<'PY'
import os
from pathlib import Path
import pandas as pd

data_dir = Path(os.environ["COMP_DATA_DIR"])
test = pd.read_csv(data_dir / "test.csv")

out = Path("/kaggle/working/pseudo_val.csv")
df = pd.DataFrame({
    "id": test["id"].astype(str),
    "src": test["transliteration"].fillna("").astype(str),
    "dummy_ref": [""] * len(test),
})
df.to_csv(out, index=False)
print("wrote", out, "rows=", len(df))
PY

# rerank 実行
# - lambda=0.5（あなたのベスト値）
# - forward スコア列は infer_nmt_nbest の seq_score を使用
# - reverse スコアの計算量が重い場合は --max-cands-per-id を下げてください
python -m dp.eval_rerank_methods \
  --config "${CFG}" \
  --val /kaggle/working/pseudo_val.csv \
  --id-col id --src-col src --gold-col dummy_ref \
  --cands /kaggle/working/cands/cands_beam8.csv /kaggle/working/cands/cands_sample_t0.9_p0.95.csv \
  --run-noisy \
  --reverse-ckpt "${REV_CKPT}" \
  --lambda-fwd 0.5 \
  --fwd-score-col seq_score \
  --noisy-batch-size 8 \
  --max-cands-per-id 32 \
  --prune-strategy top_fwd \
  --out /kaggle/working/rerank_out
```

出力：
- `/kaggle/working/rerank_out/pred_noisy_channel.csv`（id, translation ほか）
- `/kaggle/working/rerank_out/noisy_scored_candidates.csv`（デバッグ用。候補が多いと巨大化）

---

#### セル B3：submission.csv を作成して validate
```bash
%%bash
set -eux
source /kaggle/working/kaggle_env.sh
cd "${REPO_DIR}"

python -m dp.submit \
  --config "${CFG}" \
  --pred /kaggle/working/rerank_out/pred_noisy_channel.csv \
  --data-dir "${COMP_DATA_DIR}" \
  --out /kaggle/working/submission.csv

python -m dp.validate \
  --submission /kaggle/working/submission.csv \
  --data-dir "${COMP_DATA_DIR}"
```

---

## 4. （任意）Kaggle 上でローカルスコア（gm）を出したい場合
Kaggle の hidden test のスコアは「提出後」にしか分かりません。  
一方で、`val_for_oracle.csv` 等を Dataset として追加しておけば、Kaggle 上でも検証できます。

例（`val_for_oracle.csv` と `cands_*.csv` を追加している前提）：

```bash
%%bash
set -eux
source /kaggle/working/kaggle_env.sh
cd "${REPO_DIR}"

ORACLE_DATASET="my-oracle-files"  # Dataタブの名前に合わせて修正
ORACLE_DIR="/kaggle/input/${ORACLE_DATASET}"

python -m dp.eval_rerank_methods \
  --config "${CFG}" \
  --val "${ORACLE_DIR}/val_for_oracle.csv" \
  --id-col id --src-col src --gold-col ref \
  --cands \
    "${ORACLE_DIR}/cands_beam64.csv" \
    "${ORACLE_DIR}/cands_sample_t0.9_p0.95.csv" \
    "${ORACLE_DIR}/cands_tmpp_top50.csv" \
    "${ORACLE_DIR}/cands_editor.csv" \
  --run-noisy \
  --reverse-ckpt "${REV_CKPT}" \
  --lambda-fwd 0.5 \
  --fwd-score-col seq_score \
  --out /kaggle/working/val_rerank_out

cat /kaggle/working/val_rerank_out/metrics.json | head
```

---

## 5. 提出のやり方
1) Notebook を **Commit & Run**（提出用に再実行）  
2) 実行が終わったら、`/kaggle/working/submission.csv` が Output に出ます  
3) Kaggle UI の Submit から `submission.csv` を提出（スコアは提出後に表示）

---

## 6. 実行時間の目安（超ざっくり）
テストは「約 4,000 文」規模と言及されています（hidden では置き換えあり）。  
このため、候補数がそのまま計算量に効きます。

- パターンA（1-best 推論のみ）
  - GPU（T4/P100 想定）: **数分〜20分**程度
  - CPU: **数十分〜数時間**（非推奨）

- パターンB（候補生成 + reverse rerank）
  - 例の設定（beam8 + sample4×1 → 最大 12 候補/文、さらに top_fwd で 32 までに制限）:
    - GPU: **1〜3時間**程度（候補数・max_length・batch に強く依存）
  - 候補を増やすほど（beam16/32 や sampling runs を増やすほど）**線形に伸びます**。
  - `beam64` や `sampling 32×4` のような Oracle 級は、Kaggle だと **数時間〜時間超過**になりやすいです。

> 速くしたい場合の実用Tips
> - `infer_nmt_nbest` の `k` を減らす（最優先）
> - `eval_rerank_methods` の `--max-cands-per-id` を下げる + `--prune-strategy top_fwd` を使う
> - `--noisy-batch-size` を上げる（GPU メモリに余裕がある範囲で）
> - `max_source_length / generation_max_new_tokens` を短くする

---

## 7. よくある詰まりポイント
- `FileNotFoundError: Model dir not found`  
  → `/kaggle/input/<dataset>/...` の **ディレクトリ名が違う**ことが多いです。`!ls /kaggle/input` で確認。
- `CUDA out of memory`  
  → `--noisy-batch-size` や `infer_nmt_nbest --batch-size` を下げる / `k` を下げる。
- 提出で弾かれる（Internet が原因）  
  → Notebook の Internet を OFF にして Commit し直す（提出用は基本オフライン想定）。

