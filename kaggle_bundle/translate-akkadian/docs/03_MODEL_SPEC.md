# 03_MODEL_SPEC.md
モデル方針（byte/char主力＋subword副＋文脈導入）

## 0. 方針（評価指標に寄せる）
- 指標が BLEU×chrF++（幾何平均）なので、**文字レベルの一致（chrF++）**を強化しやすい設計を優先
- ただし意味崩壊を避けるため、subword系も併用してアンサンブル/リランキングで補完

## 1. モデルの系統
### 1.1 主力：byte/char系 seq2seq
- 目的：表記ノイズ・希少語・欠損表現に強く、chrF++を伸ばしやすい
- 実装：Hugging FaceのByT5系相当（利用可能な範囲で）

### 1.2 副：subword系 seq2seq
- 目的：文法・語順・意味の一貫性を補完（BLEU側にも効く）
- 実装：mT5/mBART等（利用可能な範囲で）

## 2. 入力正規化の前提（実データ対応）
- determinatives 表記は `(d)` `(ki)` と `{d}` `{ki}` が混在するため、**どちらかに統一**して学習する
- 欠損は `[...]` / `…` / `xxx` 系を `<gap>/<big_gap>` に揃える（train と published_texts を統一）
- `.` はロググラム区切りとして重要なので削除しない
- 変音記号や下付き数字（例：`š`, `ḫ`, `₄`）は語形情報のため保持
- test には `„` が混入するため、`"` などに正規化して表記ゆれを抑える

## 3. 入力フォーマット
### 3.1 文単体（baseline）
- 入力は「正規化済みのtransliteration文」をそのまま渡す（最初は余計な加工をしない）
```text
<SRC> {transliteration_sentence} </SRC>
```

### 3.2 文脈付き（context-conditioned）
- `text_id` ごとに並べた前後文を付与し、ターゲット文をマークする
- 出力は <T> に対応する1文のみ
- 例（k=2）
```text
<CTX>
  <P2> ... </P2>
  <P1> ... </P1>
  <T>  target sentence here  </T>
  <N1> ... </N1>
  <N2> ... </N2>
</CTX>
```

## 4. 固有名詞プレースホルダ（optionalだが強い）
- Lexicon（OA_Lexicon_eBL.csv）の`type`を利用し、PN/GNを検出
- 学習時：A-mur-{d}UTU → <PN_1> のように置換し、復元マップを保持
- 推論後：復元して表記を整える（大文字規則やハイフンはルールで固定）
- 重要：置換・復元は 完全に可逆であること（復元漏れゼロ）

## 5. 学習設定（初期値の例）
- 最大長：contextありは要調整（max_source_lenがネック）
- 最適化：AdamW
- 正則化：label smoothing（例0.1）
- 早期終了：CVで監視（BLEU/chrF++）
- クリップ：gradient clipping
- 混合データ：品質階層（高/中/低）でサンプリング比率を持つ

### 5.1 OCR追加並列の混合ルール（publications由来）
- OCR由来の追加並列はノイズ源になりやすいので、必ず `quality_tier` を持たせる（high/med/low）
- 学習では最初は **high のみ** を混ぜる（baselineを壊さない）
- CVで改善が確認できた場合のみ、med を少量混ぜる（比率を探索）
- low は原則使わない（使うなら自己学習や特別な正則化が整った後）

## 6. デコード（推論）
- beam search（beam=4〜10程度を候補）
- length penalty を探索（1文制約を壊さない）
- n-best を吐き、後段でリランキングできるようにする

## 7. アンサンブル方針
- 異種（byte/char + subword）＋seed違い
- リランキング：
  - 1文制約
  - 数値・固有名詞整合
  - 文字一致（chrF寄り）を加点
- 計算制約が厳しければ、最後に軽量化（量子化など）を検討
