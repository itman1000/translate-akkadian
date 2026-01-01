# 01_DATA_SPEC.md
データ仕様・前処理の不変条件（Codex実装用）

## 1. 入力ファイル（Kaggle dataset）
- `train.csv`
  - `oare_id` : 文書ID（OARE）
  - `transliteration` : アッカド語転写表記（文書レベル）
  - `translation` : 対応する英訳（文書レベル）
- `test.csv`
  - `id` : 文ID（提出キー）
  - `text_id` : 文書ID（同一文書内で複数文）
  - `line_start`, `line_end` : 文の範囲（文字列型：`1`, `1'`, `1''` 等）
  - `transliteration` : 入力（文レベル）
- `published_texts.csv`（翻訳なし）
  - `oare_id`, `transliteration_orig`, `transliteration`（クリーン版）など
  - 注意：列名にスペースあり（例：`online transcript`）。読み込み時は列名をそのまま扱う
- `publications.csv`（OCR由来のページテキスト）
  - `pdf_name`, `page`, `page_text`, `has_akkadian`
- `bibliography.csv`
- `OA_Lexicon_eBL.csv`（語彙・固有名詞ヒント）

## 1.1 実データ差分メモ（重要）
- train の `transliteration` は改行/行番号がほぼ無い。数字は数量の可能性が高く、境界検出に使えない
- determinatives は **括弧表記 `(d)` `(ki)`** が多く、`published_texts` の `{d}` `{ki}` と表記が異なる
- 欠損は train では `[...]` / `…` / `xxx` 系で表現され、`published_texts` は `<gap>/<big_gap>` を使用
- test には `„`（ダブルロークォート）が混入するため、正規化で一貫した扱いにする
- test のサンプルは `line_start/line_end` が数値だが、本番は文字列（`1'` など）を想定して扱う

## 2. 前処理のゴール
- train（文書レベル）を **文レベルの並列データ** に再構成し、test形式に寄せる
- 表記ゆれ/ノイズに強い入力を作る一方で、翻訳に効く情報（固有名詞・ロググラム等）を壊さない

## 3. 前処理の“不変条件”（壊したら負けるルール）
以下は基本的に**保持**する（削る場合はA/Bテストして勝つときだけ）
- 先頭大文字：固有名詞シグナル（可能な限り維持）
- ALL CAPS：ロググラム（可能な限り維持）
- determinatives `{...}` / `(...)`：分類子（波括弧/括弧自体も含め維持、可能なら表記を統一）
- 欠損表現：`<gap>`, `<big_gap>`（最終形として必ず維持）
- `.`（ロググラム区切り）や下付き数字（例：`₄`）は情報なので保持
- ハイフン：音節境界情報（安易に全削除しない。A/Bで検証）

## 4. 正規化（Normalization）
### 4.1 Unicode統一（必須）
- ダッシュ類：`–` `—` `−` → `-`
- アポストロフィ類：`’` `‘` → `'`
- 引用符類：`“` `”` `„` → `"`（または一括で `'` に寄せる。どちらかに固定）
- 全角/半角：必要なら統一（CSVの実体に合わせる）

### 4.2 Dataset Instructions準拠の削除/置換（基本形）
※ここは競技の指示に沿うが、**過度に削らない**のが方針。A/B/Cで比較する。
- 削除（現代注記）
  - `!`, `?`, `/`
  - 語区切り `:` は「扱いを統一」（削除 or スペース化のどちらかに固定）
  - `.` はロググラム区切りや小数に使われるため、**一律削除はしない**
  - `< >` は括弧自体を除去し、中身は残す
  - `˹ ˺` は括弧を除去（中身は残す/除くは指示に従う）
  - `[...]` は括弧を除去し中身は残す（例：`[KÙ.BABBAR] → KÙ.BABBAR`）
- 置換（欠損）
  - `[x] → <gap>`
  - `…` や `[… …] → <big_gap>`
  - `xxx/xxxx/xxxxx` は欠損記号として `<gap>` / `<big_gap>` に寄せる
  - `transliteration_orig` の `{large break}` / `{broken area}` は `<big_gap>` に寄せる

## 5. 正規化バリアント（A/B/C）
Codexは最初から1本に決め打ちせず、3系統を用意してCVで勝つものを採用。
- A（最小）：Unicode統一＋明らかな現代注記のみ除去、他はほぼ維持
- B（標準）：A＋語区切りの統一（削除 or スペース化）、欠損置換、determinatives 表記の統一を実施
- C（強）：B＋ハイフン簡約（ハイフンをスペース化）、残存する括弧の除去
  - ロググラム区切りの `.` は維持する

## 6. 提出（出力）側の不変条件
- `translation` は **必ず1文**（改行を含めない）
- CSVはUTF-8、ヘッダ `id,translation`
- ダブルクォート/カンマ含有時はCSVとして壊れないようエスケープ

## 7. publications.csv から追加並列を作る際の前提
- publications.csv は重いので、**必ずストリーミング（チャンク）で読む**
- まず「候補抽出」を行い、以後の処理対象を数％に絞る（例：IDキー一致、翻訳らしい段落の存在、転写っぽいパターン等）
- 候補抽出で生成するファイル：
  - `artifacts/ocr/publications_candidates.parquet`
- 追加並列（文ペア）は最終的に以下へ出す：
  - `artifacts/ocr_pairs/part-0000.parquet`（分割出力）
  - 最後に結合して `artifacts/ocr_pairs/all.parquet`
