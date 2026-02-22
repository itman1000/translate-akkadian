# 改善策A〜E 実装計画書（translate-akkadian / Codex向け）

目的：現状の「NMT候補＋(MBR/noisy-channel)」が飽和しつつある状況から、**候補集合の oracle を押し上げ**、さらに**学習リランカで回収率を上げる**ことで、CV/LB を「小数点」ではなく**数ポイント級**で押し上げる。

本計画は、既存 repo `translate-akkadian/src/dp/` の作法に合わせて、**追加スクリプトは `python -m dp.xxx` で実行できる CLI**として実装する。

---

## 0. 前提（現状パイプラインの整理）

### 現状
1. `dp.infer_nmt_nbest` で候補 CSV を生成（beam + sampling）
2. 候補 CSV を結合して candidate pool を作る
3. `dp.eval_rerank_methods` で MBR / noisy-channel を比較・選択
4. （oracle評価時）`dp.oracle_eval` は **idごとに最良候補を1本**選ぶため、評価用データの `id` は行単位で一意にする

### 評価前提（重要）
- `aligned` / `ablation` 系の検証データで `oare_id` などが重複する場合、そのまま oracle評価に使わない
- oracle評価専用に `id=0..N-1` を再採番したファイルを作ってから、候補生成・評価を実行する

### 今後（A〜E後）
候補生成器を多元化して candidate pool の質を上げる：

- NMT（複数seed/ckpt）候補
- TM++ 候補（A）
- 編集モデル（Prototype Editing）候補（B）
- APE（スタイル整形）候補（C）
- （任意）noisy-channel の選好で生成した候補
- 学習リランカで最終選択（D）
- モデル自体は MRT-lite で目的関数に寄せる（E）

---

## 1. 共通インタフェース（候補CSV仕様）

すべての候補生成器は long CSV（1行=1候補）を出す。  
ただし現行実装との互換性のため、列要件を「最小必須」と「推奨」で分ける。

**最小必須列（現行 `dp.eval_rerank_methods` 互換）**
- `id` : サンプルID（string化して扱う）
- `translation` : 英文候補

**推奨共通列（運用上ほぼ必須）**
- `tag` : 候補生成器識別（例 `beam64`, `tmpp`, `editor`, `ape`）
- `run` : 生成run（beam=1, samplingは1..runs）
- `rank` : run内順位（1..k）
- `model` : 生成モデル識別（例 `byt5_large`, `editor_byt5_base`）

**スコア列（現行 `dp.eval_rerank_methods` で利用可能）**
- forward系は `gen_score` / `seq_score` / `sequence_score` / `score` のいずれかを受理
- noisy-channel で `--lambda-fwd != 0` または `--prune-strategy top_fwd` を使う場合、forwardスコア列は必須
- `score` は生成logprobではなく類似度の意味で使われる場合があるため、必要に応じて `--fwd-score-col` で明示

**拡張列（A/Dで使う特徴）**
- `tm_sim`
- `slot_match_num`, `slot_match_pn`, `slot_match_unit`
- `rev_score`（事前計算する場合）

> NOTE: 現行の `dp.eval_rerank_methods.load_candidates()` は上記拡張列を保持しない。  
> D（学習リランカ）で使う場合は **新しい loader** を用意するか既存 loader を拡張する。

---

## 2. 改善策A：TM++（スロット化検索）を候補生成器として最大化

### A-1. 目的
- “検索して終わり”ではなく、**参照に近い定型句**を候補集合へ直接注入し、BLEU（=表現一致）を大きく回復させる。

### A-2. 成果物（新規モジュール）
- `src/dp/tm_slotify.py` : スロット化＆復元
- `src/dp/tm_index.py` : TF-IDF char n-gram の索引作成/保存
- `src/dp/tm_retrieve.py` : クエリ（src）から topK 検索
- `src/dp/tm_generate_candidates.py` : 候補CSV出力（TMそのまま候補）

### A-3. データ
索引に入れるペア（可能な限り多いほど良い、ただし品質ゲート必須）：
- コンペ `train.csv`（aligned版があるならそれ優先）
- EvaCun / aligned コーパス（既存 `dp.prepare_evacun` / `dp.align_train` の出力）
- `published_texts` など外部は**規約に従う**（公開・利用可の範囲のみ）

CV評価時のリーク防止：
- foldごとに **そのfoldの train データだけ** で index を作る（`artifacts/ablation/.../train.parquet`）
- 同foldの val/test 側サンプルは index 側に混ぜない

### A-4. スロット化仕様（必須）
- 数字 → `<NUM>`
- 欠損 → `<gap>` / `<big_gap>`（表記統一）
- 人名/地名 → `<PN1> <PN2> ...`（順序保持、同一入力内で安定番号付け）
- 単位（shekel/mina/talent 等）→ `<UNIT>`

実装は「抽出→置換→マッピング保持」の2段：
- `slotify_src(text) -> (slot_text, slot_map)`
- `apply_slot_map(tgt_text, slot_map) -> restored_tgt`

### A-5. 近傍検索（TF-IDF char n-gram）
- `TfidfVectorizer(analyzer="char", ngram_range=(3,5), lowercase=False)` を基本
- コーパス側：slotified_src を fit/transform
- クエリ側：slotified_src を transform
- 類似度：cosine（sparse dot）

保存形式：`joblib.dump({vectorizer, X, meta_df}, path)`

### A-6. 候補生成（複線化の入口）
まずは “TMそのまま” を候補として出す：
- topK 取得（例：50）
- それぞれの `tm_tgt` を **スロット復元**して `translation` にする
- `tm_sim` と `slot_match_*` を列に入れる
- 評価用 `id` は行単位で一意にする（重複する `oare_id` はそのまま使わない）

**CLI例**
```bash
python -m dp.tm_index \
  --pairs artifacts/ablation/variant=C/fold=0/train.parquet \
  --src-col src_sent --tgt-col tgt_sent \
  --out artifacts/tm/variant=C/fold=0/index.joblib

python -m dp.tm_generate_candidates \
  --index artifacts/tm/variant=C/fold=0/index.joblib \
  --queries artifacts/oracle/val_for_oracle.csv \
  --id-col id --src-col src \
  --out artifacts/oracle/cands_tmpp.csv \
  --topk 50 --tag tmpp
```

### A-7. Definition of Done（DoD）
- `cands_tmpp.csv` が long形式で出る
- `translation` は1文に強制（既存 `enforce_single_sentence` を再利用）
- `tm_sim` が入り、topK がちゃんと多様に出ている
- `dp.eval_rerank_methods` に `--cands ... cands_tmpp.csv` を足すと MBR/noisy で改善が見える

---

## 3. 改善策B：編集モデル（Prototype Editing）

### B-1. 目的
TMは “近いが微妙に違う” ケースでズレる。編集モデルで **TM候補を勝てる答えへ補正**し、BLEU/chrF++を同時に上げる。

### B-2. 成果物（新規モジュール）
- `src/dp/prepare_editor_data.py` : 学習データ生成（TM検索→プロトタイプ付与）
- `src/dp/train_editor.py` : 編集モデル学習（seq2seq）
- `src/dp/infer_editor_nbest.py` : 推論（TM候補を編集して候補を増やす）

### B-3. 入出力フォーマット
入力（文字列）：
```
<SRC> {src}
<TM_SRC> {tm_src}
<TM_TGT> {tm_tgt}
```
出力：`tgt`（gold）

Tokenizerに special tokens 追加（推奨）：
`["<SRC>", "<TM_SRC>", "<TM_TGT>"]`

### B-4. 学習データ生成
train の各 (src, tgt) について：
- TM++ index で topK（例：10）取得（ただし self-match 排除）
- 上位M（例：2〜3）をプロトタイプとして採用
- それぞれを1例として書き出す

品質ゲート（最低限）：
- `tm_sim` が閾値以上（例 0.2〜0.3）
- 極端な長さ比は捨てる

### B-5. 推論（候補生成器）
val/test の各 src について：
- TM++で topK 検索（例 10）
- 各 TM につき editor で beam/sampling 生成（例 beam=4）
- 生成結果を候補CSVで出す（tag=`editor`）

### B-6. DoD
- editor 候補を candidate pool に加えると oracle が上がる
- MBR/noisy/学習リランカで editor が採用されるケースが増える

---

## 4. 改善策C：APE（自動後編集）をスタイル整形器として分離

### C-1. 目的
「意味を変えずに」参照スタイルへ寄せる（句読点・大文字・反復・<gap>整形）。
chrF++ と BLEU の両方に効く可能性がある。

### C-2. 成果物（新規モジュール）
- `src/dp/prepare_ape_data.py` : (src, pred)->gold を作る
- `src/dp/train_ape.py` : APEモデル学習
- `src/dp/apply_ape.py` : 任意候補にAPEを適用し、候補集合を増やす

### C-3. 学習データの作り方（リーク回避が重要）
CV fold 単位で：
- foldの学習データで NMT を学習
- その NMT で foldの検証データに推論し pred を作る
- (src, pred)->gold をAPE学習データとして蓄積

簡易版（まず動かす）：
- 既存ベースモデルで train 全体に推論→APE学習（ただし若干リーク）

### C-4. 追加：ルールベース整形（安全装置）
モデルAPEの前後に、確実に安全なルールを適用：
- 重複トークン抑制
- `<gap>`/`<big_gap>` の表記統一
- 1文制約（既存 `enforce_single_sentence`）

### C-5. DoD
- APEを候補に追加すると、採用時に chrF++ が改善する
- “意味破壊”（長さ激減、固有名詞消失）が監視で検出されない

---

## 5. 改善策D：学習リランカ（gm最大化器）

### D-1. 目的
MBR/noisy は教師なしなので限界がある。候補集合が強くなった時ほど、**学習リランカ**が効く。
狙い：候補内に正解があるなら拾う（oracle gap を埋める）。

### D-2. 成果物
- `src/dp/rerank_features.py` : 特徴抽出
- `src/dp/train_reranker.py` : 学習（CV、group=id）
- `src/dp/rerank_learned.py` : 推論（最終選択）
- （必要なら）`src/dp/score_reverse.py` : reverseモデルで全候補に `rev_score` を付与して保存

### D-3. ラベル
train/val では gold があるので、候補ごとに sentence-level 指標を計算：
- `sentence_bleu`（SacreBLEU）
- `sentence_chrf`（chrF++）
- `sentence_gm = sqrt(bleu * chrf)`（0-100スケール）

### D-4. 特徴量（軽量＆強い）
- `seq_score`（forward）
- `rev_score`（reverse）
- `tm_sim`
- スロット整合：NUM/PN/UNIT の一致率
- 文字数・単語数・長さ比（対 src / 対 TM）
- 句読点数、括弧、ハイフン、コロン
- `<GAP>/<BIG_GAP>` の個数
- 禁止パターン（例：同一ngram反復、数字消失、PN消失、"and and" など）

### D-5. モデル（依存を増やさず scikit-learn で）
まずは堅い順：
1) `HistGradientBoostingRegressor`（速い、強い）
2) `RandomForestRegressor`（頑丈）
3) `Ridge`（ベースライン）

学習は `GroupKFold(n_splits=K)` で id 単位リーク防止。

### D-6. 推論
- 候補CSV（A/B/C/NMTを全部結合）
- 必要なら reverse scoring を追加で付与
- 特徴量を計算して `y_hat` を出し、idごとに最大を選ぶ

### D-7. DoD
- CVで MBR/noisy を **明確に**超える
- 特に “TM/editor/APE を足したとき” の伸びが大きい

---

## 6. 改善策E：MRT-lite（Reward-weighted CE）

### E-1. 目的
目的関数ミスマッチ（NLL最適化≠GM最大化）を縮める。
フルMRT/RLは沼りやすいので、**Reward-weighted CE** を先に入れる。

### E-2. 成果物
- `src/dp/prepare_rwc_data.py` : 候補生成＋報酬計算
- `src/dp/train_rwc.py` : Reward-weighted CE で微調整
- （任意）`src/dp/monitor_decode.py` : 崩壊監視（distinct_pred_rate等）

### E-3. 学習データ生成
train の各 src から現行モデルで K 候補生成（例 8〜16）：
- beam または sampling
- 各候補の reward = sentence_gm(gold, cand) を計算
- 重み `w = softmax(reward / tau)`（tauは温度、例 2〜5）
- 学習例として (src, cand, weight) を保存

### E-4. 学習（teacher forcing）
通常の CE を重み付きにする：
`loss = Σ_i w_i * NLL(cand_i | src)`

安全策：
- 小さい学習率
- 早期終了
- 反復率・長さ分布の監視（短文化/同一出力化を検知）

### E-5. DoD
- 同一候補生成条件でも BLEU が上がる（少なくともCVで数ポイント級の可能性）
- 出力崩壊（短文化、定型文連発）が起きない

---

## 7. 実装順序（最短で“数ポイント級”を狙う）

推奨順（リターン/工数比）：
1) **A: TM++ 候補生成**（候補集合の質を変える）
2) **B: 編集モデル**（TMを勝てる答えにする）
3) **D: 学習リランカ**（候補集合から当てる）
4) **C: APE**（整形で取りこぼし回収）
5) **E: MRT-lite**（モデル自体を押し上げる、工数は重め）

---

## 8. コーディング規約（Codex向け注意）

- すべて `python -m dp.xxx` で動く CLI を実装
- 既存の `dp.utils.load_config`, `clean_text`, `enforce_single_sentence` を再利用
- 依存追加は最小（requirementsにある `scikit-learn`, `pandas`, `torch`, `transformers`, `sacrebleu` の範囲）
- 生成物は `artifacts/` 配下に集約
- すべてのスクリプトで `--help` が通ること
