# Translate Akkadian (マイルストーン 0)

Deep Past Challenge のための最小限のエンドツーエンドパイプラインです。
このベースラインは意図的にシンプルで、一定の翻訳を出力して
`train -> infer -> submit -> validate` が一通り動くことを確認します。

## 要件
- Python 3.10+
- pandas
- sacrebleu（`dp.eval` のみで使用）
- pyarrow（parquet出力で使用）
- scikit-learn（TF-IDFベースラインで使用）
- torch（NMTで使用）
- transformers（NMTで使用）
- datasets（NMTで使用）
- sentencepiece（NMTで使用）

## ローカルセットアップ（venv）
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python3 -m pip install -U pip
pip install -r requirements.txt
```
※ venv はプログラム側で自動作成/強制しないため、実行前に有効化してください。

## データ
コンペの CSV を `data/` 配下に配置してください。
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`
- `data/published_texts.csv`
- `data/publications.csv`
- `data/bibliography.csv`
- `data/OA_Lexicon_eBL.csv`
- `data/Sentences_Oare_FirstWord_LinNum.csv`（任意: train→文アライン補助。ある場合は自動で利用）
- `data/eBL_Dictionary.csv`（辞書: lemma → 英語定義。任意・NMTの辞書グロスで使用）

## データ品質方針
間違ったデータで学習するくらいなら、そのデータで学習しないことを最優先にします。

## クイックスタート（ローカル）
```bash
# repo ルートで実行
export PYTHONPATH=src

python -m dp.train --config configs/baseline.yaml
python -m dp.infer --config configs/baseline.yaml --ckpt <path_to_model.json>
python -m dp.submit --pred artifacts/predictions.csv --out submission.csv
python -m dp.validate --submission submission.csv --data-dir data
```

任意の評価（gold ファイルが必要）:
```bash
python -m dp.eval --pred submission.csv --gold <gold.csv>
```

## Milestone 1: 文分割＋アライン
```bash
python -m dp.align_train --config configs/align.yaml
```

### OARE 文アライン補助（任意）
`data/Sentences_Oare_FirstWord_LinNum.csv` が存在する場合、該当する train 文書については
OARE 側の「文ごとの英訳」と「先頭語ヒント」から文境界を anchor し、より安定した文ペアを作ります。
（存在しない文書は従来通り length-based DP にフォールバックします。）

#### OARE hints の安全ガード（A/B）
Sentences_Oare_FirstWord_LinNum.csv は便利ですが、
**anchor が見つからない文書**や、**anchor があっても品質ゲートを通過する行が減る文書**では
かえってノイズやデータ欠損が増えることがあります。

そのため `dp.align_train` では、OARE hints がある文書でも以下の doc-level 自動選択を行います。
- **(A) no_anchor / 非anchored**: OARE を使わず baseline（train の translation を split する DP）へフォールバック
- **(B) anchored でも悪化**: OARE 版と baseline 版を両方作り、`--drop-flagged` で残る行数・スコアが悪い方は捨てる

設定（`configs/align.yaml` など）で調整できます:
- `oare_doc_select: true/false`（デフォルト true。false で従来動作に戻す）
- `oare_min_anchors: 1`（デフォルト 1。これ未満なら強制フォールバック）
- `oare_tie_break: "oare" | "dp"`（同点時にどちらを採用するか）

出力 `aligned_train` の `align_meta` には、集計しやすいよう以下の印が入ります:
- `oare_selected`（OARE 版を採用）
- `oare_available; oare_fallback_<reason>`（OARE hints はあったが baseline にフォールバック）

`--oare-debug` を付けると `oare_debug.csv` に `selected_method/selected_reason` などが出力されます。

明示的に使う場合:
```bash
python -m dp.align_train --config configs/align.yaml --use-oare-sentences
```
無効化したい場合（再現性のため／比較用）:
```bash
python -m dp.align_train --config configs/align.yaml --no-oare-sentences
```
別パスに置いた場合:
```bash
python -m dp.align_train --config configs/align.yaml --oare-sentences-path /path/to/Sentences_Oare_FirstWord_LinNum.csv
```
デバッグログを出したい場合:
```bash
python -m dp.align_train --config configs/align.yaml --use-oare-sentences --oare-debug
```
出力: `artifacts/aligned/oare_debug.csv`（文書ごとの anchor/セグメント統計）
出力先: `artifacts/aligned/aligned_train.parquet`
※ 出力には `src_norm_variant` 列があり、A/B/C のバリアントが含まれます。

※ `pyarrow` が無い場合は CSV で出力してください。
```bash
python -m dp.align_train --config configs/align.yaml --format csv
```

### 文ペア品質ゲート（強化）
`configs/align.yaml` のゲート設定で、ノイズの多い文ペアを除外します。
主なチェック項目:
- `len_ratio` / `align_score` / `token_align_score` の下限
- `src/tgt` の記号比率・数字比率・英字比率
- `tgt_sentence_ends`（文末句読点の個数）で複数文を検知
- 末尾引用符（`"` / `'`）の除外（`drop_tgt_trailing_quote`）
- 不均衡な引用符（`"` の数が奇数）の除外（`drop_tgt_odd_quote`）
- `src/tgt` のトークン数/文字数上限
- 重複対応（`src_multi_tgt`, `tgt_multi_src`）

出力される追加列:
`token_align_score`, `src_tokens`, `tgt_tokens`,
`src_alpha_ratio`, `src_digit_ratio`, `src_symbol_ratio`,
`tgt_alpha_ratio`, `tgt_digit_ratio`, `tgt_symbol_ratio`,
`tgt_sentence_ends`

品質ゲートを有効にして出力する場合:
```bash
python -m dp.align_train --config configs/align.yaml --drop-flagged
```
`max_tgt_sentence_endings=1` で1文制約を強めています（略語/小数点はカウントから除外）。
`drop_tgt_trailing_quote=true` で文末の引用符を含む行を除外します（不要なら `false`）。
`drop_tgt_odd_quote=true` で `"` の数が奇数の行を除外します（不要なら `false`）。
aligned_train の品質ログとサンプル確認:
```bash
python -m dp.inspect_aligned --config configs/align.yaml --sample 20
```
flags が空の場合は `flags_top10` は表示されません。
フラグ付きだけ見る場合:
```bash
python -m dp.inspect_aligned --config configs/align.yaml --only-flagged --sample 20
```

## 前処理の健全性チェック（正規化A/B/C）
正規化のログを確認して、`<gap>/<big_gap>` と語区切りの単独 `.`、ALL CAPS、決定詞の変換が
期待通りかをチェックします（`KÙ.BABBAR` など語中ドットは保持）。
※ `dp.inspect_preprocess` は `--config` が必須です。
```bash
python -m dp.inspect_preprocess --config configs/align.yaml --variants A,B,C --max-examples 5
```
特に B/C では `[x]` / `[x?]` が `<gap>`、`[...]` / `…` / `...` と裸の `xx` / `x x` / `x-x` が `<big_gap>` に寄せられることを確認してください（単独の `x` は残す）。
また、`(broken lines)` などの括弧コメントは `<big_gap>` に寄せられること、**それ以外の括弧コメントは中身ごと削除されること（決定詞は保持）**、**`GA2×(ME.EN)` のような記号名は中身を保持すること**、`<< >>` や `< >` の括弧が除去されること、
`1'` などの行番号が消えることも合わせて確認します。
さらに、variant C ではハイフンをスペース化しますが、**先頭大文字トークン（固有名詞シグナル）はハイフンを維持**して
語の分割でシグナルが失われないようにしています。
加えて、CDLI/ORACC の表記（`a2/a₂`, `a3/a₃`, `sz`, `s,`, `t,`, `Xx`, `h/H` など）が
Unicode に統一されていること、**母音の2/3表記はアクセント付き文字へ（例: `a₂ → á`）**、
**それ以外の下付き数字は ASCII（例: `il₅ → il5`）**に寄せられていること、
`gap/big_gap` が山括弧なしで残っていないことも確認します。
ORACC のインライン記法（参照: `Oracc.html`）に基づき、以下も削除/正規化されることを確認してください。
- `{{...}}`（言語グロス）/ `{(...)}`（文書グロス、入れ子含む）を削除
- `%akk` / `%1` などのシフトコードを削除
- `*(...)` / `/(...)` / `:` / `:'` / `:"` / `:.` / `::` の句読点コードを削除
- `*` / `#` フラグを削除（破損/注記フラグ）
- `<<...>>`（本文から外すべき文字）と `(#...#)`（注記）を削除
- `<(...)>` の外側括弧を外して中身のみ残す
- `$AN` / `a$1` などの不確実/近接マーカーを削除
- `~` / `~a`（ロググラム/アログラフ）や `@h` などの修飾子を削除
- `|...|` のパイプを除去（中身は保持）
- `{+...}` の `+` を除去（決定詞の種別マーカー）
- `s'` / `S'` を `ś` / `Ś` に変換
- `;` / `//` / `/` をスペース化（語中の代替読み `KI/DI` も分割）、`+` は `-` に統一
- `<<...>>` 除去で生じる `- -` は `-` に畳み込む
- 決定詞はホワイトリスト（`src/dp/align_train.py` の `_RAW_DETERMINATIVES`）に一致するものだけ `{...}` に統一し、`<sup>...</sup>` の上付きも同様に処理（例: `(TÚG)` も `{TÚG}` に統一）
翻訳文は `<...>` / `[...]` の括弧を外して中身を保持し、行区切りの `/`（数値の分数は除外）だけ削除、通常の句読点（`! ? . :`）は保持します。
文分割後も、末尾に連続する引用符や孤立した引用符（`"` / `'`）は除去されます。
併せて、先頭大文字（固有名詞）と ALL CAPS（ロググラム）が正規化後も残っているか、
`title_tokens_*` / `all_caps_tokens_*` のログで確認してください。

aligned（`src_sent`）側も検査する場合:
```bash
python -m dp.inspect_preprocess \
  --config configs/align.yaml \
  --train artifacts/aligned/aligned_train.parquet \
  --src-col src_sent \
  --use-aligned \
  --variant-col src_norm_variant \
  --max-examples 5
```

不変条件テスト（壊れやすい表記の簡易テスト）:
```bash
python -m unittest tests/test_preprocess_invariants.py
```
英文分割の簡易テスト:
```bash
python -m unittest tests/test_split_english.py
```

## publications.csv 行番号クリーニング（OCR向け）
OCR由来の `page_text` に混入しやすい行番号を除去し、単位判定できない行は落とします。
単位判定には `docs/akkadian_after_number_tokens.txt` を利用します。
図表キャプションや脚注などのノイズ除去には `docs/publications_noise_patterns.txt` を利用します。
超厳格ホワイトリストに合致する行のみ残すことで、誤学習を避けます。

```bash
python -m dp.clean_publications \
  --config configs/align.yaml \
  --unit-list docs/akkadian_after_number_tokens.txt \
  --noise-list docs/publications_noise_patterns.txt \
  --drop-ambiguous \
  --drop-empty
```

出力先: `artifacts/ocr/publications_clean.csv`
※ `--format parquet` を指定すると `artifacts/ocr/` に `*-part-XXXX.parquet` が出力されます。
※ `--keep-ambiguous` で判別不能行を残せます。
※ `has_akkadian == True` の行だけを残すのがデフォルトです（`--no-filter-akkadian` で無効化）。
※ 行番号除去のサマリ（件数）は標準出力に表示されます。
※ `page_text` に `\\n` 形式の改行が入っている場合は実改行に変換してから処理します。
※ 超厳格ホワイトリストに合致しない行は削除するのがデフォルトです（`--no-strict-whitelist` で無効化）。

## OCR 追加並列抽出（候補 → ペア → 混合）
publications.csv から OCR 由来の追加並列を作る流れです。**必ず「候補抽出 → 文抽出 → 品質ゲート」**の順に実行します。
まずは英語っぽい段落のみを対象にします（非英語は別フェーズ）。

### 1) 候補抽出（チャンク処理 + has_akkadian フィルタ）
```bash
python -m dp.ocr_candidates --config configs/ocr.yaml
```
出力:
- `artifacts/ocr/publications_candidates-part-XXXX.parquet`
- `artifacts/ocr/publications_candidates_stats.json`

※ `--out` をディレクトリにすると `part-XXXX.parquet` を直下に出力します。  
※ `--text-col page_text_clean` を指定すれば `dp.clean_publications` の出力を利用できます。  
※ 途中再開は `part` の存在チェックでスキップされます（再計算したい場合は `--overwrite`）。

### 2) ペア生成（文抽出 → 品質ゲート → tier 付与）
```bash
python -m dp.ocr_pairs --config configs/ocr.yaml
```
出力:
- `artifacts/ocr_pairs/part-XXXX.parquet`
- `artifacts/ocr_pairs/summary.json`

`quality_tier`（high/med/low）を付与し、`len_ratio` / `align_score` / 重複率 / flags などの統計を保存します。  
`configs/ocr.yaml` 内の閾値で high/med の基準を明確化しています。

### 3) 学習データに混合（high tier から）
```bash
python -m dp.mix_ocr --config configs/ocr.yaml
```
出力:
- `artifacts/ocr_pairs/mixed_train.parquet`
- `artifacts/ocr_pairs/mixed_train.stats.json`

`ocr_mix_ratio` で混合比率を調整します。CV が悪化する、出力が短文化する、`<gap>` が不自然、
固有名詞が崩れる等の兆候があれば **`ocr_mix_ratio: 0`** または `--disable-ocr` で混合停止できます。

Kaggle Notebook では `RUN_OCR=True` にすると上記フローを実行し、統計ログを出力します。
ログの出力先は `artifacts/ocr/publications_candidates_stats.json`、
`artifacts/ocr_pairs/summary.json`、`artifacts/ocr_pairs/mixed_train.stats.json` です。

## Milestone 2: A/B/C アブレーション
1) CV 分割の作成:
```bash
python -m dp.split_cv --config configs/ablation.yaml
```
出力先: `artifacts/splits/cv_folds_k5.csv`

2) A/B/C の fold 別データ作成:
```bash
python -m dp.prepare_ablation --config configs/ablation.yaml --drop-flagged
```
出力先: `artifacts/ablation/variant=<A|B|C>/fold=<k>/train.parquet` と `val.parquet`

3) 学習/評価の雛形:
```bash
scripts/run_ablation_stub.sh
```
TF-IDFベースラインで学習/評価し、結果を `artifacts/ablation_runs/` に保存します。

4) 結果集約:
```bash
python -m dp.collect_ablation --root artifacts/ablation_runs
```
出力先: `artifacts/ablation_runs/summary.csv`

## 実運用の学習スクリプト（Milestone 2.1）
TF-IDF文字n-gram近傍ベースラインです。
```bash
python -m dp.train_real --config configs/train_real.yaml --train <train.parquet> --val <val.parquet> --out <out_dir>
```
列指定やプレースホルダ復元つきの評価も可能です。
```bash
python -m dp.train_real --config configs/train_real.yaml \
  --train <train.parquet> --val <val.parquet> --out <out_dir> \
  --src-col src_placeholders --tgt-col tgt_placeholders \
  --restore --map-col placeholder_map --restore-target-col tgt_sent
```
推論:
```bash
python -m dp.infer_real --config configs/train_real.yaml --ckpt <out_dir>/model.pkl --out predictions.csv
```
カスタム列を使う場合:
```bash
python -m dp.infer_real --config configs/train_real.yaml \
  --ckpt <out_dir>/model.pkl --test <test.csv> --src-col <src_col> --out predictions.csv
```

## NMT（Seq2Seq）ベースライン
ByT5-base を使った NMT 版の学習/推論です。GPU を推奨します。

1) 文分割＋アライン:
```bash
python -m dp.align_train --config configs/align.yaml --variant C --drop-flagged
```
出力: `artifacts/aligned/aligned_train.parquet`
実行時に align_score / len_ratio / flags / #sentences などの品質ログを出力します。

2) 学習:
```bash
python -m dp.train_nmt --config configs/nmt_byt5_small.yaml \
  --train artifacts/aligned/aligned_train.parquet --variant C --drop-flagged \
  --out artifacts/nmt/byt5_small
```
ByT5-base はメモリ消費が大きいため、`configs/nmt_byt5_small.yaml` では
`per_device_train_batch_size=2` と `gradient_accumulation_steps=8` を推奨値にしています。
OOM の場合は `max_source_length`/`max_target_length` を下げるか、
`per_device_train_batch_size` をさらに下げて調整してください。

### 辞書グロス（eBL_Dictionary）による入力拡張（任意）

`data/eBL_Dictionary.csv`（eBL 辞書: lemma → 英語定義）を追加すると、
**ソース末尾に短い英語グロスを付与**して NMT の語彙カバレッジを補助できます。

付与形式（例）:
`... <LEX> amāru=to see | kaspu=silver </LEX>`

有効化する場合は **学習と推論の両方で同じ設定を使ってください**
（片方だけだと分布がズレて精度が落ちやすいです）。

#### 推奨の安定化設定（ノイズ/長さ対策）

グロスを付与すると入力が長くなり、また機能語（ana/ina/ša/u など）がヒントを埋めてしまうことがあります。
以下のフィルタで **ノイズと入力長をコントロール**できます。

- `gloss_max_hints` : 1文あたりのヒント数（推奨 4〜6、デフォルト 6）
- `gloss_max_match_len` : マッチする最大トークン長（デフォルト 4）
- `gloss_match_types` : OA_Lexicon の type フィルタ（デフォルト `['word']`）
- `gloss_stop_lemmas` / `gloss_stop_lemmas_file` : ストップワード（lemma）
  - デフォルトでいくつかの機能語を除外します。無効化するなら `--gloss-no-default-stop-lemmas`
- `gloss_max_lemma_freq` : 頻度フィルタ（大きいほど緩い）
  - `dp.train_nmt` では train src から lemma 頻度表を作り、`<out_dir>/gloss_lemma_freq.json` として保存します
  - `dp.infer_nmt` は `--gloss-max-lemma-freq` 指定時に、`<ckpt>/gloss_lemma_freq.json` があれば自動で読み込みます（無ければ入力から簡易計算）

学習（例）:
```bash
python -m dp.train_nmt --config configs/nmt_byt5_small.yaml \
  --train artifacts/aligned/aligned_train.parquet --variant C --drop-flagged \
  --out artifacts/nmt/byt5_small \
  --use-gloss \
  --gloss-max-hints 6 --gloss-max-total-chars 220 \
  --gloss-max-match-len 4 \
  --gloss-max-lemma-freq 50
```

推論（例）:
```bash
python -m dp.infer_nmt --config configs/nmt_byt5_small.yaml \
  --ckpt artifacts/nmt/byt5_small --test data/test.csv --norm-variant C \
  --out predictions.csv \
  --use-gloss \
  --gloss-max-hints 6 --gloss-max-total-chars 220 \
  --gloss-max-match-len 4 \
  --gloss-max-lemma-freq 50
```

config で指定する場合（任意）:
```yaml
use_gloss: true
gloss_max_hints: 6
gloss_max_total_chars: 220
gloss_max_match_len: 4
gloss_max_lemma_freq: 50
# gloss_match_types: [word]
# gloss_stop_lemmas: [ana, ina, ša, u, mā, kīma, lā]
# gloss_min_lemma_chars: 2
# oa_lexicon_path: data/OA_Lexicon_eBL.csv
# ebl_dictionary_path: data/eBL_Dictionary.csv
```

※ コメントアウトしている項目も **指定可能**です（CLI オプションと同じ意味）。  
`oa_lexicon_path` / `ebl_dictionary_path` は辞書ファイルの場所を上書きします。  
相対パスは `data_dir` 基準で解決され、絶対パスならそのまま使われます。

各項目の役割:
- `gloss_match_types`: OA_Lexicon の type フィルタ（例: `word`, `PN`, `GN` など）
- `gloss_stop_lemmas`: 除外する lemma のリスト（機能語のヒントを減らす）
- `gloss_max_lemma_freq`: 頻度が高い lemma を除外（`freq > N` を弾く）
- `gloss_min_lemma_chars`: 短すぎる lemma を除外
- `oa_lexicon_path`: `OA_Lexicon_eBL.csv` のパス上書き
- `ebl_dictionary_path`: `eBL_Dictionary.csv` のパス上書き

※ グロスを付与すると入力長が増えるため、必要に応じて `max_source_length` を上げてください。
（ただし上げすぎるとメモリが増えます。まずはヒント数/ストップワード/頻度フィルタで長さを抑えるのがおすすめです。）

#### 学習後の評価を軽く回す（反復実験用）

`dp.train_nmt` は学習後に val で **生成評価**（BLEU / chrF++ / gm）を行います。
反復実験では以下の軽量オプションが使えます。

- `--post-eval-mode full` : 指標 + サンプル/診断（デフォルト）
- `--post-eval-mode quick` : **指標のみ**
- `--post-eval-mode none` : 評価をスキップ（学習のみ）

例（指標のみ）:
```bash
python -m dp.train_nmt --config configs/nmt_byt5_small.yaml \
  --train artifacts/aligned/aligned_train.parquet --variant C --drop-flagged \
  --out artifacts/nmt/byt5_small \
  --post-eval-mode quick
```

さらに高速化したい場合は、config に `post_eval_max_rows` を追加して **val をランダム間引き**できます（例：200〜500）。
※ 近似スコアになるため、採用判断は full（全件）で揃えて比較してください。


3) 推論:
```bash
python -m dp.infer_nmt --config configs/nmt_byt5_small.yaml \
  --ckpt artifacts/nmt/byt5_small --test data/test.csv --norm-variant C \
  --out predictions.csv
```

#### 1文制約の後処理（推奨）
コンペ要件（1文）に合わせ、`dp.infer_nmt` と `dp.submit` は **予測文を1文に強制**できます。
内部の文境界（`. ! ?`）を `;` に置換して **1文に「統合」**する `merge` と、
最初の文だけを残す `truncate` を選べます。

重要: 小数点（例: `3.5`）は文境界として扱わないため、数値が途中で切れる事故を避けます。

設定（config）:
```yaml
force_single_sentence: true
single_sentence_mode: merge  # merge | truncate
```
CLI（推論）:
- `--no-force-single-sentence` : 無効化
- `--single-sentence-mode merge|truncate` : モード変更

#### デコードの既定値（beam2_cfg 相当）
`dp.infer_nmt` のデフォルトは config の `num_beams/length_penalty/no_repeat_ngram_size/repetition_penalty` を使う
`cfg` プリセットで、`dp.train_nmt` のログに出る **beam2_cfg**（= 設定値を反映した decode）と揃います。
最低限の config でも事故りにくいよう、推奨値（beams=2 / length_penalty=0.8 / no_repeat=20 / rep_penalty=1.15）をフォールバックします。

必要ならプリセットで比較できます:
- `--decode-preset cfg|greedy|beam2_free|beam4_free`

4) 提出:
```bash
python -m dp.submit --pred predictions.csv --out submission.csv --data-dir data
```

※ `published_texts.csv` の `AICC_translation` は翻訳文ではなく URL のため、NMTの学習データには使っていません。
※ ByT5 のドロップアウトはモデル全体に共通の値で、層ごとに個別指定はできません。
※ transformers の古い版では一部引数が未対応のため、自動的に互換設定へフォールバックします。

### オフライン用モデル準備（Kaggle向け）
Kaggle の再実行はインターネットが使えないため、事前にモデルをローカルへ保存して Dataset として追加します。
```bash
python scripts/export_hf_model.py --model google/byt5-base --out models/byt5-base
```
`models/byt5-base` を zip 化して Kaggle Dataset にアップロードし、Notebook で `MODEL_DIR` を指定してください。
このスクリプトはモデルを「ロード」せずにファイルを取得するため、torch/numpy の互換問題を回避できます。

### 解析に役立つログ
`dp.train_nmt` はトークン長統計（p50/p90/p95/p99、hit_max/trunc 率）、学習ハイパラを出力します。
学習完了時に validation の BLEU/chrF++/GM と loss を表示し、固定サンプルの src/ref/pred を出力します（`val_ratio` > 0 の場合）。
評価時に `predict_with_generate` が効かず logits が返る場合は、argmax でトークンID化して decode し、失敗時は指標計算やサンプル出力をスキップします。
さらに不変条件の破壊率（`<gap>`/determinatives/ALL CAPS/複数文）と、崩壊検知（distinct率・頻出テンプレ）に加えて、
デコード条件の比較（greedy/beam など）のスコアログを出力します。

加えて、validation の生成直後に `[gen_stop] ...` を 1 行出力します（生成の止まり方の簡易診断）。
- `eos_rate`: EOS が出現したサンプル割合（高いほど EOS で自然に停止できている）
- `hit_limit_rate`: EOS が出ず、返却配列の最大長まで埋まった割合（高いほど長さ上限で切られている可能性が高い）
- `returned_max_len`: Trainer が返した予測配列の幅（padding 後）
- `cfg_limit_total_len`: config（`generation_max_new_tokens` / `generation_max_length`）から計算した想定上限
※ `returned_max_len` と `cfg_limit_total_len` が大きくズレる場合は、generate kwargs / generation_config の反映漏れを疑ってください。

また、デコード比較の `pred_len_tok_*` は `-100` などの負値を pad 扱いに寄せて集計します（負値パディング混入で「長さが同じ値に張り付く」誤検知を避けるため）。

`dp.infer_nmt` は入力/出力の長さ統計に加えて、生成トークン長の分布と上限到達率（`[gen_len]`）、EOS 早期終了率を出力します。
出力が空や繰り返しになる場合は、これらのログを見て `generation_max_new_tokens` や
`no_repeat_ngram_size` を調整してください。

### ターゲット長の分布チェック
学習データのターゲット長（文字数/トークン数）を確認するには `dp.length_stats` を使います。
```bash
python -m dp.length_stats --config configs/nmt_byt5_small.yaml \
  --train artifacts/aligned/aligned_train.parquet --variant C --drop-flagged --use-tokenizer
```
オフライン環境で tokenizer が見つからない場合は `--tokenizer <path>` を指定してください。

主な設定値（`configs/nmt_byt5_small.yaml`）:
- `dropout_rate`, `attention_dropout_rate`
- `val_ratio`
- `learning_rate`, `weight_decay`, `max_grad_norm`
- `per_device_train_batch_size`, `gradient_accumulation_steps`
- `step_decay_steps`, `step_decay_gamma`（0 のとき無効。ステップは optimizer step 単位）
- `max_target_length`
- `generation_max_new_tokens`, `generation_max_length`, `num_beams`, `length_penalty`, `early_stopping`, `no_repeat_ngram_size`, `repetition_penalty`
- `seed`

学習が不安定で出力が空/制御文字になる場合は、`fp16` と `gradient_checkpointing` を無効化し、
`learning_rate` を下げて再学習してください。

## Milestone 3: 固有名詞プレースホルダ（PN/GN）
```bash
python -m dp.prepare_placeholders --config configs/placeholders.yaml --replace-target --strategy pattern
```
出力先: `artifacts/placeholders/aligned_train_placeholders_pattern.parquet`

プレースホルダ戦略の比較:
```bash
python -m dp.compare_placeholders --config configs/placeholders.yaml
```
出力先: `artifacts/placeholders/strategy_summary.csv`

CV検証（Cのみ、復元評価）:
```bash
python -m dp.prepare_ablation --config configs/ablation.yaml \
  --aligned artifacts/placeholders/aligned_train_placeholders_pattern.parquet \
  --variants C --drop-flagged --out-dir artifacts/ablation_placeholders

bash scripts/run_placeholder_cv.sh
```
結果: `artifacts/ablation_placeholder_runs/summary.csv`

## Milestone 5: 自己学習（擬似並列）
published_texts の翻訳なし行から擬似並列を作ります（現データでは 251 行）。

1) 擬似並列の生成:
```bash
python -m dp.pseudo_label --config configs/train_real.yaml \
  --ckpt artifacts/train_real/variant=C/model.pkl \
  --input data/published_texts.csv \
  --require-untranslated --norm-variant C --min-confidence 0.2
```
出力先: `artifacts/pseudo/variant=C/pseudo_train.parquet`

2) CV用データ作成（擬似並列を train に追加）:
```bash
python -m dp.prepare_ablation --config configs/ablation.yaml \
  --aligned artifacts/aligned/aligned_train.parquet \
  --variants C --drop-flagged \
  --out-dir artifacts/ablation_self_train \
  --extra artifacts/pseudo/variant=C/pseudo_train.parquet \
  --extra-mode train
```

3) CV実行:
```bash
bash scripts/run_self_train_cv.sh
```
結果: `artifacts/ablation_self_train_runs/summary.csv`

## Milestone 6: 文脈付き入力（context-conditioned）
1) 文脈データ生成（k_prev=2, k_next=2）:
```bash
python -m dp.prepare_context --config configs/context.yaml
```
出力先:
- `artifacts/context/aligned_train_context.parquet`
- `artifacts/context/test_context.csv`

2) CV用データ作成:
```bash
python -m dp.prepare_ablation --config configs/ablation.yaml \
  --aligned artifacts/context/aligned_train_context.parquet \
  --variants C --drop-flagged --out-dir artifacts/ablation_context
```

3) CV実行:
```bash
bash scripts/run_context_cv.sh
```
結果: `artifacts/ablation_context_runs/summary.csv`

4) 推論で文脈列を使う場合:
```bash
python -m dp.infer_real --config configs/train_real.yaml \
  --ckpt <out_dir>/model.pkl \
  --test artifacts/context/test_context.csv \
  --src-col src_context \
  --out predictions.csv
```

## Milestone 7: n-best + リランキング + averaging
1) n-best を出す:
```bash
python -m dp.infer_nbest --config configs/train_real.yaml \
  --ckpt <out_dir>/model.pkl --out artifacts/nbest.csv --k 5
```

2) リランキングして1本にする:
```bash
python -m dp.rerank --nbest artifacts/nbest.csv \
  --ckpt <out_dir>/model.pkl --out predictions.csv
```

3) n-best リランキングのCV検証:
```bash
python -m dp.eval_nbest --config configs/train_real.yaml \
  --ckpt <model.pkl> --val <val.parquet> --k 5 --out <out_dir>
```
`metrics.json` に rerank 後の指標、`*_base` に top1 の指標が保存されます。

4) 複数モデルの平均化（簡易アンサンブル）:
```bash
python -m dp.ensemble_infer --config configs/train_real.yaml \
  --ckpts model1.pkl,model2.pkl --k 3 --out predictions.csv --rerank
```

## Milestone 8: 異種アンサンブル（最終押し）
TF-IDFの設定違いで簡易的な異種アンサンブルを作ります。

1) CVで検証:
```bash
bash scripts/run_ensemble_cv.sh
```
結果: `artifacts/ablation_ensemble_runs/summary.csv`

2) 本番推論:
```bash
python -m dp.train_real --config configs/train_real.yaml --train <train.parquet> --out artifacts/ensemble/char_3_5
python -m dp.train_real --config configs/train_real_char_2_4.yaml --train <train.parquet> --out artifacts/ensemble/char_2_4
python -m dp.train_real --config configs/train_real_word_1_2.yaml --train <train.parquet> --out artifacts/ensemble/word_1_2

python -m dp.ensemble_infer --config configs/train_real.yaml \
  --ckpts artifacts/ensemble/char_3_5/model.pkl,artifacts/ensemble/char_2_4/model.pkl,artifacts/ensemble/word_1_2/model.pkl \
  --k 3 --out predictions.csv
```

## クイックスタート（Kaggle）
同じコマンドを使いますが、Kaggle の入力ディレクトリを指定します。
```bash
python -m dp.train --config configs/baseline.yaml --data-dir /kaggle/input/<dataset-name>
python -m dp.infer --config configs/baseline.yaml --ckpt <path_to_model.json> --data-dir /kaggle/input/<dataset-name>
python -m dp.submit --pred artifacts/predictions.csv --out submission.csv --data-dir /kaggle/input/<dataset-name>
python -m dp.validate --submission submission.csv --data-dir /kaggle/input/<dataset-name>
```
Kaggle Code 形式の提出は `submit_kaggle.ipynb` を使うのが簡単です。詳細は `README_KAGGLE.md` を参照してください。
リポジトリDatasetのzipにより1階層深いパスになる場合は、Notebook先頭で `REPO_DIR` を明示指定してください。
`submit_kaggle.ipynb` では `PYTHONPATH=src` を自動設定して `python -m dp.*` が動くようにしています。
依存チェックは `sacrebleu` のみ追加インストールします（`requirements.txt` 全体は入れず、Kaggle 標準の依存を前提）。
Notebook内の `overrides` 辞書には `configs/nmt_byt5_small.yaml` の全キーを列挙してあります。`None` は元設定を維持し、上書きがある場合のみ `configs/nmt_byt5_small.runtime.json` を生成してそれを使います。
`submit_kaggle.ipynb` の `RUN_TRAIN` を `False` にすると学習をスキップし、`INFER_CKPT_DIR`（または `MODEL_DIR`）のモデルで推論だけ行います。
提出用の `submission.csv` は `/kaggle/working/submission.csv` に出力されます。
NMT を使う場合、Notebook 内で `MODEL_DIR` を指定するとローカルパスのモデル（Kaggle Dataset）を読み込めます。
`sacrebleu` が未導入の場合、学習ログ内の BLEU/chrF++ はスキップされます（学習自体は継続）。

## Kaggle 提出用の軽量バンドル
Kaggle にアップロードするリポジトリは最小構成にすると楽なので、元のリポジトリで提出用の軽量コピーを作成します。
```bash
bash scripts/build_kaggle_bundle.sh
```
生成先: `kaggle_bundle/translate-akkadian`
含まれるもの: `src/`, `configs/`, `docs/`, `scripts/`, `submit_kaggle.ipynb`, `requirements.txt`
除外: `data/` と `artifacts/` の中身（空ディレクトリのみ作成）

## LLM 分析用の軽量バンドル
他の LLM にコードと仕様を共有するための最小セットを zip 化します。
```bash
bash scripts/build_llm_bundle.sh
```
生成先: `llm_bundle/translate-akkadian` と `llm_bundle/translate-akkadian.zip`
含まれるもの: `src/`, `configs/`, `docs/`, `scripts/`, `tests/`, `submit_kaggle.ipynb`, `README.md`, `README_KAGGLE.md`, `README_COLAB.md`, `requirements.txt`

## 注意点
- このマイルストーンはパイプラインの動作確認のみです。ダミーのベースラインは
  `configs/baseline.yaml` の定数翻訳を返すだけで、スコア向上は目的としていません。
- 設定ファイルのパースは、追加依存なしで読めるフラットな YAML に限定しています。
  高度な YAML 機能が必要なら PyYAML をインストールしてください。

## リポジトリ構成（最小）
- `src/dp/` : CLI エントリポイント（`train`, `infer`, `submit`, `validate`, `eval`）
- `configs/` : 最小限の設定ファイル
- `scripts/` : 実験補助スクリプト（アブレーション実行の雛形など）
- `tests/` : 前処理の不変条件テスト
- `docs/` : 仕様と計画
- `data/` : コンペの CSV ファイル
- `kaggle_bundle/` : Kaggle 提出用の軽量コピー（`scripts/build_kaggle_bundle.sh` で生成）
