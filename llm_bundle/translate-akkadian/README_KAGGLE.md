# Kaggle 提出用 README（Code 形式）

このコンペは **Code 形式** です。ローカルで作成した `submission.csv` を直接アップロードして提出することはできません。Kaggle が **選択した Notebook を hidden test で再実行**し、`/kaggle/working` に生成された出力ファイルを提出物として扱います。

## 0. ローカルで提出用ZIPを作る
まず、Kaggleに載せる「軽量コピー」を作成して zip 化します。

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

## 1. Kaggle Dataset を作成してアップロード
1) Kaggle の **Datasets** から **New Dataset** を作成  
2) 上で作成した `translate-akkadian.zip` をアップロード（Private 推奨）

※ 軽量コピーには `data/` と `artifacts/` の実体は含まれていません。

### モデルDatasetの準備（インターネット不可の場合）
Kaggle の再実行ではインターネットが使えないため、事前にモデルをローカルへ保存して Dataset として追加します。
```bash
python scripts/export_hf_model.py --model google/byt5-base --out models/byt5-base
```
`models/byt5-base` を zip 化して Kaggle Dataset にアップロードし、Notebook に追加してください。
このスクリプトはモデルをロードせずにファイルだけ取得します。

## 2. Notebook を作成して ipynb を読み込む
1) Kaggle で **New Notebook** を作成（Competition を選択）  
2) **Data** から以下を追加:
   - コンペ公式データセット（`train.csv`, `test.csv` があるもの）
   - 先ほど作成した「軽量コピー Dataset」
   - （オフライン実行の場合）ByT5-base のモデルDataset

3) `submit_kaggle.ipynb` を開く:
   - Notebook の **File → Open** から `submit_kaggle.ipynb` を選択  
   - 見つからない場合は、Dataset 内のファイルを開くか、ローカルの `submit_kaggle.ipynb` をアップロード

## 3. Notebook の実行
`submit_kaggle.ipynb` を開いたら **全セル実行**してください。
- Notebook は `/kaggle/input` にあるリポジトリを `/kaggle/working` にコピーし、そこから処理を進めます。
- `python -m dp.*` が動くように Notebook 側で `PYTHONPATH` を設定しています。
- モデルDatasetが1つだけ見つかった場合は `MODEL_DIR` を自動設定します。
- NMTのパラメータは Notebook 内の `overrides` 辞書で上書きできます。`configs/nmt_byt5_small.yaml` の全キーを列挙してあるので、`None` は元設定を維持し、上書きがある場合のみ `configs/nmt_byt5_small.runtime.json` を生成して使います。
- `RUN_TRAIN=False` にすると学習をスキップします。推論に使うモデルは `INFER_CKPT_DIR` か `MODEL_DIR` で指定してください。
- `sacrebleu` が未導入の場合、Notebook は `sacrebleu` だけを追加インストールします（`requirements.txt` 全体は入れません）。
- `sacrebleu` が未導入かつインストール不可の場合、学習ログ内の BLEU/chrF++ はスキップされます（学習自体は継続）。

### 評価を軽く回す（実験を高速化）

`dp.train_nmt` は学習後に validation（val）に対して **生成（generate）** を回し、BLEU / chrF++ / gm を計算します。
フル診断（サンプル表示・崩壊検知・デコード比較）まで行うと時間がかかるため、反復実験では軽量モードの利用を推奨します。

- `--post-eval-mode full` : 指標 + サンプル/診断（従来どおり、デフォルト）
- `--post-eval-mode quick` : **指標のみ**（まずスコア比較したいとき）
- `--post-eval-mode none` : 学習のみ（評価を完全にスキップ）

さらに速くしたい場合は、**val をランダムに間引いて近似スコア**を出せます（※最終比較は full / 全件で）。
- config に `post_eval_max_rows` を追加（例：200〜500）

#### Kaggle Notebook（submit_kaggle.ipynb）での設定例

1) **学習セル**（`python -m dp.train_nmt ...`）の末尾に追記：
```python
run(
    f"python -m dp.train_nmt --config '{config_path}' "
    f"--train '{train_path}' --variant C --drop-flagged "
    f"--out '{ckpt_dir}'" + model_arg +
    " --post-eval-mode quick"
)
```

2) さらに高速化したい場合は、**overrides 辞書**に `post_eval_max_rows` を足します（キーが base config に無くても追加できます）：
```python
overrides = {
    # ... 既存の項目 ...
   'post_eval_mode': 'quick',
   'post_eval_max_rows': 300,  # 例：300行だけで近似評価
}
```

> 注意：`post_eval_max_rows` は近似スコアです。ハイパラ採用の最終判断は、full（全件）で揃えて比較してください。


### パスがうまく解決できない場合
Notebook 内で以下の環境変数を使って明示できます（固定例）。
- `REPO_DIR` : `/kaggle/input/translate-akkadian/translate-akkadian`
- `COMP_DATA_DIR` : `/kaggle/input/deep-past-initiative-machine-translation`
- `MODEL_DIR` : `/kaggle/input/byt5-base-model/byt5-base`

## 4. 実行内容（NMT 版）
- モデル: ByT5-base（Seq2Seq）
- 正規化: **C**

Notebook 内で実行される主な流れ:
1. `dp.align_train` で `aligned_train.parquet` を生成
2. `dp.train_nmt` で学習
3. `dp.infer_nmt` で推論
4. `submission.csv` 生成 → 形式検証

### OCR追加並列を使う場合（任意）
`RUN_OCR=True` にすると、publications.csv から追加並列を作成し、high tier のみ混合します。

実行フロー（RUN_OCR時）:
- `dp.ocr_candidates`（候補抽出）
- `dp.ocr_pairs`（文抽出 → 品質ゲート → tier付与）
- `dp.mix_ocr`（high tier のみ混合）

統計ログ（RUN_OCR時に表示）:
- `artifacts/ocr/publications_candidates_stats.json`
- `artifacts/ocr_pairs/summary.json`
- `artifacts/ocr_pairs/mixed_train.stats.json`

学習データは `artifacts/ocr_pairs/mixed_train.parquet` に切り替わります。  
閾値は `configs/ocr.yaml` で調整できます。

## 5. 出力ファイル
- `submission.csv` は **`/kaggle/working/submission.csv`** に出力されます
- Kaggle の「Output」タブから `submission.csv` を選択して提出できます

## 6. GPUは必要？
推奨です。NMT は GPU の方が高速に学習できます（CPU でも動作しますが時間がかかります）。
ByT5-base はメモリ消費が大きいので、OOM の場合は `per_device_train_batch_size` や
`max_source_length`/`max_target_length` を下げて調整してください。

## 7. 変更したいパラメータ
- Kaggle Notebook では `overrides` 辞書に値を設定してください（`None` は元設定を維持し、上書きがある場合のみ runtime 設定を生成）。
- 推論設定は `dp.infer_nmt` の `--decode-preset`（既定は `cfg`。train_nmt の **beam2_cfg 相当**）、`--num-beams`, `--max-target-length`, `--max-new-tokens`, `--length-penalty`, `--early-stopping`, `--no-repeat-ngram-size`, `--repetition-penalty` で調整できます。
- 1文制約のための後処理は `dp.infer_nmt --no-force-single-sentence` / `--single-sentence-mode merge|truncate` で調整できます（既定は有効）。
- 事前学習モデルをローカルパスから使う場合は `MODEL_DIR` を指定してください（Kaggle Datasetとして追加）。

## 8. Run All と提出の違い
- Run All: 学習→推論→`submission.csv` までを毎回実行します。
- 提出: Kaggle が Notebook を最初から再実行するため、学習セルがあると毎回学習が走ります。
- 学習を省きたい場合は `RUN_TRAIN=False` にし、推論用のモデルを `INFER_CKPT_DIR`（または `MODEL_DIR`）で指定してください。

## 9. Run All から提出まで（固定パス版）
「学習したモデルをDataset化 → 推論のみで提出」までの最短手順です。パスは固定例を使います。

### 9-1. Run All（学習あり）
1) **Datasets** で以下を用意  
   - リポジトリDataset: `translate-akkadian`（`translate-akkadian.zip` をアップロード）  
   - モデルDataset: `byt5-base-model`（`models/byt5-base` を zip でアップロード）  
2) **New Notebook**（Competition を選択）  
3) **Data** に以下を追加  
   - コンペ公式データセット  
   - `translate-akkadian`  
   - `byt5-base-model`  
4) `submit_kaggle.ipynb` を開く  
5) 先頭セルで固定パスを指定（例）:
```python
os.environ['REPO_DIR'] = '/kaggle/input/translate-akkadian/translate-akkadian'
os.environ['COMP_DATA_DIR'] = '/kaggle/input/deep-past-initiative-machine-translation'
os.environ['MODEL_DIR'] = '/kaggle/input/byt5-base-model/byt5-base'
```
6) `RUN_TRAIN=True` のまま **Run All**  
7) 学習済みモデルが `/kaggle/working/artifacts/nmt/byt5_small` に生成されます

### 9-2. 学習済みモデルを Dataset 化
1) Notebook を **Save Version**  
2) Output から **Create Dataset**（または Save Output to Dataset）  
3) Dataset 名を `translate-akkadian-nmt-checkpoint` にする  
4) 生成物に `artifacts/nmt/byt5_small` が含まれていることを確認

### 9-3. 提出（推論のみ）
1) 新しい Notebook を作成  
2) **Data** に以下を追加  
   - コンペ公式データセット  
   - `translate-akkadian`  
   - `translate-akkadian-nmt-checkpoint`  
3) `submit_kaggle.ipynb` を開く  
4) 先頭セルで固定パスを指定（例）:
```python
os.environ['REPO_DIR'] = '/kaggle/input/translate-akkadian/translate-akkadian'
os.environ['COMP_DATA_DIR'] = '/kaggle/input/deep-past-initiative-machine-translation'
```
5) 学習セルで以下を指定:
```python
RUN_TRAIN = False
INFER_CKPT_DIR = '/kaggle/input/translate-akkadian-nmt-checkpoint/artifacts/nmt/byt5_small'
```
6) **Run All** → `submission.csv` が `/kaggle/working/submission.csv` に生成  
7) Output から `submission.csv` を提出

※ パスが合わない場合は Data タブの実際のディレクトリ名に合わせて修正してください。

---

困ったら `submit_kaggle.ipynb` の先頭セルで `REPO_DIR` と `COMP_DATA_DIR`（必要なら `MODEL_DIR`）を手動指定してください。
