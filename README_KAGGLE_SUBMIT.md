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

### セル 1：リポジトリをコピー
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
export FWD_CKPT="/kaggle/working/models/byt5_small_colab_v1"   # forward(Akkadian->English)
export REV_CKPT="/kaggle/working/byt5-reverse/byt5_reverse_evacun"   # reverse(English->Akkadian)


# ===== config =====
export NMT_CONFIG="configs/nmt_byt5_small.yaml"   # 学習時に使っていた config に合わせて修正（REPO_DIR からの相対パス）
export CFG="${REPO_DIR}/${NMT_CONFIG}"
EOF

mkdir -p /kaggle/working/hf
cat /kaggle/working/kaggle_env.sh
```

### セル 2.5：config を Notebook 上で上書きする（任意）
**セル 2 の直後、セル 3 の前**で実行してください。  
このセルは先に `kaggle_env.sh` を **Python 側の `os.environ` に読み込んでから** 実行します。  
`%%bash` で `export` した環境変数は Python カーネルには自動では引き継がれないため、この読み込みが必要です。  
また、このセルは `kaggle_env.sh` の `NMT_CONFIG` / `CFG` も更新するので、以後の `%%bash` セルでそのまま反映されます。

```python
import json
import os
import sys
from pathlib import Path

# bash セルで作成した export を Python カーネルへ反映する
env_path = Path("/kaggle/working/kaggle_env.sh")
if not env_path.exists():
    raise FileNotFoundError(f"env file not found: {env_path}")

for raw_line in env_path.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or not line.startswith("export "):
        continue
    key, value = line[len("export ") :].split("=", 1)
    value = value.strip().strip('"').strip("'")
    os.environ[key] = os.path.expandvars(value)

sys.path.append(f'{os.environ["REPO_DIR"]}/src')
from dp.utils import load_config

repo_dir = Path(os.environ["REPO_DIR"])
default_base_rel = "configs/nmt_byt5_small.yaml"
base_rel = os.environ.get("NMT_CONFIG", default_base_rel)
base_path = Path(base_rel)
if not base_path.is_absolute():
    base_path = repo_dir / base_path

# 2.5 を再実行した場合、NMT_CONFIG が override JSON を指したままになることがある。
# そのファイルがまだ無ければ、元の yaml へフォールバックする。
if not base_path.exists():
    fallback_path = repo_dir / default_base_rel
    print("base config missing, fallback to:", fallback_path)
    base_path = fallback_path

cfg = load_config(str(base_path))

# 提出 Notebook で実際に効くのは主に推論・後処理系の設定
cfg.update({
    "max_source_length": 384,
    "max_target_length": 384,
    "generation_max_length": 320,
    "generation_max_new_tokens": 320,
    "infer_batch_size": 4,
    "num_beams": 4,
    "length_penalty": 0.8,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 20,
    "use_gloss": False,
    "gloss_max_hints": 6,
    "gloss_max_total_chars": 220,
    "gloss_max_match_len": 4,
    "gloss_max_lemma_freq": 1000000,
    "force_single_sentence": True,
    "single_sentence_mode": "merge",
    "normalize_output": False,
    "data_dir": os.environ["COMP_DATA_DIR"],
})

out_rel = "configs/nmt_byt5_small.kaggle.override.json"
out_path = repo_dir / out_rel
out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print("wrote:", out_path)

# 現在の Python カーネルでも上書き後の config を参照できるようにする
os.environ["NMT_CONFIG"] = out_rel
os.environ["CFG"] = str(out_path)

# 以後の %%bash セル用に env ファイルも更新する
env_lines = env_path.read_text().splitlines()
env_lines = [
    ln
    for ln in env_lines
    if not ln.startswith("export NMT_CONFIG=") and not ln.startswith("export CFG=")
]
env_lines.append(f'export NMT_CONFIG="{out_rel}"')
env_lines.append('export CFG="${REPO_DIR}/${NMT_CONFIG}"')
env_path.write_text("\n".join(env_lines) + "\n")
print("updated:", env_path)
```

補足:
- 提出 Notebook では、`num_train_epochs` / `learning_rate` / `per_device_train_batch_size` / `gradient_accumulation_steps` / `dropout_rate` / `fp16` / `bf16` のような**学習専用設定は基本的に効きません**。
- 推論で OOM を避けたい場合は `per_device_eval_batch_size` ではなく **`infer_batch_size`** を下げてください。
- rerank の reverse 採点側は config ではなく、セル B2 の `--noisy-batch-size` を下げる方が直接効きます。
- 例のセルでは `--norm-variant C` や `--num-beams 8` のように **CLI で明示指定している値が config より優先**されます。そこも変えたい場合は後続セルの引数も合わせて編集してください。

### セル 3：必要パッケージの確認（※基本は Kaggle 標準で足りる想定）
オフライン提出を想定すると、インターネットに依存する `pip install` は避けたいです。  
まずは import が通るか確認してください（`sacrebleu` は**評価用の任意依存**です）。

```python
import importlib

# 必須（提出ファイル生成に必要）
required = ["pandas","sklearn","pyarrow","torch","transformers","datasets","sentencepiece"]

# 任意（BLEU/chrF++ などのローカル評価ログ用。提出だけなら無くてもOK）
optional = ["sacrebleu"]

missing_required = []
for p in required:
    try:
        importlib.import_module(p)
    except Exception:
        missing_required.append(p)

missing_optional = []
for p in optional:
    try:
        importlib.import_module(p)
    except Exception:
        missing_optional.append(p)

print("missing_required:", missing_required)
print("missing_optional:", missing_optional)
```

- `missing_required` が空なら OK（提出処理は進められます）  
- `missing_optional` に `sacrebleu` だけが入るのは OK（評価指標の表示がスキップされるだけ）  
- `missing_required` がある場合は、提出（Internet=OFF）環境で動くように依存を揃えてください（オンライン `pip install` はできません）


### セル3.5 : モデルのファイル名修正措置
```bash
%%bash
set -eux

SRC_DIR="/kaggle/input/datasets/vashi1000/models/byt5_small_colab_v1"
DST_DIR="/kaggle/working/models/byt5_small_colab_v1"

mkdir -p "$DST_DIR"
cp -a "$SRC_DIR/." "$DST_DIR/"

mv "$DST_DIR/model-001.safetensors" "$DST_DIR/model.safetensors"
ls -la "$DST_DIR" | head

```

```bash
%%bash
set -eux

SRC_DIR="/kaggle/input/datasets/vashi1000/byt5-reverse/byt5_reverse_evacun"
DST_DIR="/kaggle/working/byt5-reverse/byt5_reverse_evacun"

mkdir -p "$DST_DIR"
cp -a "$SRC_DIR/." "$DST_DIR/"

mv "$DST_DIR/model-001.safetensors" "$DST_DIR/model.safetensors"
ls -la "$DST_DIR" | head
```

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
> 重要（gloss あり学習の場合）  
> forward モデルのディレクトリに `train_meta.json` があり、その中の `gloss.enabled=true` になっている場合、  
> そのモデルは **辞書グロス（<LEX>...）付きで学習**されています。  
> この場合は **推論でも同じ gloss 付与**をするのが基本で、`dp.infer_nmt` / `dp.infer_nmt_nbest` は
> `train_meta.json` を見て **自動で gloss を有効化**します（無効化したい場合は `--no-gloss`）。

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

> Kaggle の提出環境では `sacrebleu` が無いことがあります。  
> この場合も **noisy-channel rerank 自体は実行され、metrics は skip されるだけ**です。  
> test 提出用途では `--lambda-fwd 0.5` のように **固定 λ** を使ってください。

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
