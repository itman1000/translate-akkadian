# 02_ALIGNMENT_SPEC.md
train（文書）→ 文レベル並列データ生成（分割＋アライン）仕様

## 0. 目的
- train.csv の `transliteration`（文書）と `translation`（文書）を、**文レベルのペア**へ変換する
- 生成物をそのまま学習に使える品質にする（ノイズ混入を抑える）

## 1. 生成物（中間データのフォーマット）
`artifacts/aligned/aligned_train.parquet`（推奨）
- `oare_id`
- `src_sent`（正規化済みtransliteration文）
- `tgt_sent`（英語文）
- `src_norm_variant`（A/B/C）
- `align_score`（0..1）
- `len_ratio`（src/tgtまたは文字数比）
- `flags`（除外理由など）

## 2. 文分割（Segmentation）
### 2.1 source（transliteration）
優先順位：
1) 明示的な行境界がある場合のみ利用（`\n` など）。train には基本的に存在しない
2) 数字は数量の可能性が高く、行番号としては使わない（train では `1'` 系もほぼ出ない）
3) 欠損 `<gap>/<big_gap>` は文境界の候補になり得るが、基本は保持しつつ文を極端に短くしない

実装方針（現実的なもの）
- train は行番号/改行がないため、**target 文数に合わせて source を長さDPで分割**する
- source は空白トークン列として扱い、候補境界は「トークン境界のみ」
- 目的関数は `len_ratio` と境界ペナルティ（極端に短い断片を避ける）で最小化
- determinatives やロググラム（例：`KÙ.BABBAR`）を分断しないよう、境界ペナルティを調整

### 2.2 target（英訳）
- `.` `?` `!` を基本に分割、`;` は弱境界として扱う（過分割を避ける）
- `:` は列挙や説明に使われやすいので、単独では分割トリガにしない
- 下限長が短すぎる文は結合（例：`e.g.` や括弧終わり等で短文が出る場合）

## 3. アライン（Alignment）
段階的に精密化する（いきなり最適化しない）

## 3.x OARE 文アライン補助（任意）
- `data/Sentences_Oare_FirstWord_LinNum.csv` がある文書は、OARE 側の sentence translations と「先頭語ヒント」から anchor を探し、文境界を安定化できる。
- ただし **anchor が見つからない（no_anchor）/ 非anchored** な場合や、**anchored でも品質ゲートを通過する行が減る**場合はノイズになり得るため、doc-level で baseline DP と比較して自動的にフォールバックする（A/B safety）。
- `configs/align.yaml` の `oare_doc_select / oare_min_anchors / oare_tie_break` で挙動を調整できる。
- 出力 `aligned_train` の `align_meta` には `oare_selected` / `oare_available; oare_fallback_*` が入るので、後段で集計しやすい。

### 3.1 粗合わせ（length-based）
- Gale–Church風：文字数（またはトークン数）の近さでDP整列
- 許容：1-to-1 / 1-to-2 / 2-to-1 まで（それ以上はノイズになりやすい）
- 1-to-2 / 2-to-1 は品質ゲートで基本除外（1文制約の学習を壊す）

### 3.2 精密化（類似度）
- 文字n-gram（chrFの発想）で src と tgt の対応の妥当性をスコア化
- lexiconヒント：固有名詞（PN/GN）らしき要素が tgt 側にも反映されているか（大文字パターン等）

### 3.3 品質ゲート（捨てる条件）
- `len_ratio` が極端（例：>4 または <0.25）
- `tgt_sent` が英語として不自然（極端に記号だらけ、ほぼ数字のみ等）
- 1-to-2 / 2-to-1 の結合で tgt が複数文になった（1文制約の学習を壊す）
- 同一の `src_sent` に複数 `tgt_sent` が高頻度で紐づく（重複ノイズ）

## 4. 追加データ（OCR）と整合させるための拡張
### 4.1 publications.csv（OCR）処理は必ず「候補抽出 → 文抽出 → 品質ゲート」の順
実装手順（厳守）：

1) **候補抽出（最初に必ずやる）**
- publications.csv をチャンクで読み、必要な列だけ読む（usecols相当）
- `has_akkadian == True` を最初の高速フィルタとして使う
- ルールで候補行だけを抽出して `publications_candidates.parquet` に保存
- 抽出ルール例：
  - `oare_id` / 書誌キー / aliases 等で一致するもの
  - 英語っぽい段落（アルファベット比率、文末句点の頻度、極端な記号の少なさ）
  - Old Assyrian 転写っぽい記号・パターン（ハイフン、{det}/(det)、KÙ.BABBAR 等）

2) **文抽出（候補からだけ）**
- candidates から「翻訳らしい段落」「転写らしい段落」を切り出し、文分割
- `page_text` に改行/ハイフネーションがあるため、文抽出前に正規化してから分割
- 非英語は一旦除外（後フェーズで扱う）

3) **品質ゲート（厳しめ）**
- 長さ比、重複、記号過多、英語崩れ、複数文化などで除外
- 品質階層（高/中/低）を付与して保存

4) **分割出力（再開可能）**
- `artifacts/ocr_pairs/part-XXXX.parquet` に追記
- すでに処理済みのpartはスキップできる設計にする

### 4.2
- `published_texts.csv` の識別子（aliases等）で publications から該当箇所を特定
- OCR由来の `page_text` はノイズが多いので、まずは
  - 翻訳らしい段落抽出 → 文分割
  - 言語判定（英語/非英語）→ 非英語は別フェーズで英語化
- 追加並列は必ず「品質階層（高/中/低）」を付ける

## 5. テスト（最低限）
- `aligned_train.parquet` が作れる
- `align_score` の分布が極端でない（高品質が一定数ある）
- サンプルを10件プリントして、人間が見ても致命的にズレていない
