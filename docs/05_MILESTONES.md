# 05_MILESTONES.md
実装マイルストーン（Codex用：順番に完了させる）

## Milestone 0：最小で動く（最優先）
Done条件：
- baselineモデルで学習→推論→submission.csv生成ができる
- dp.validate がPASS
- ローカルで指標が計算できる（CVはまだ粗くてOK）

## Milestone 1：train（文書）→文の再構成（分割＋アライン）
Done条件：
- `artifacts/aligned/aligned_train.parquet` を生成
- 品質ゲート後の文ペア数が十分（例：数千以上）
- サンプル目視で破綻が少ない
- train に行番号/改行が無い前提で、target 文数に合わせた長さDP分割が動く

## Milestone 2：正規化A/B/CのAblation
Done条件：
- A/B/Cで同一学習条件のCV比較ができる
- 勝つ正規化を1つ決める（理由つき）
- `(d)/(ki)` と `{d}/{ki}` の統一、`[...]`/`…`/`xxx` の欠損正規化が含まれる

## Milestone 2.1：実運用の学習スクリプト（必須）
Done条件：
- `aligned_train`（A/B/C）の入力から学習できる
- fold別の学習/評価が回せる（A/B/Cアブレーションが実行可能）
- 学習ログ・評価指標・設定が保存される（再現性確保）

## Milestone 3：固有名詞プレースホルダ（PN/GN）
Done条件：
- 置換→復元が自動で完結
- 固有名詞崩壊がCVで改善する（少なくとも悪化しない）
- `aligned_train` からプレースホルダ適用済みデータを生成できる
現状判断：
- pattern戦略 + C でCVを実施したが score がわずかに悪化したため、プレースホルダは現時点で採用見送り

## Milestone 4：追加並列（OCR / publications.csv）高品質抽出（改訂）
Done条件：
- publications.csv を **チャンク処理**で読み、最初に候補抽出して `publications_candidates.parquet` を作れる
- `has_akkadian == True` を一次フィルタとして使える
- candidates から文抽出→品質ゲートで `artifacts/ocr_pairs/part-XXXX.parquet` を分割出力できる
- `page_text` の改行/ハイフネーションを正規化してから文抽出する
- 「high tierのみ混合」でCVが悪化しない（改善なら次へ、悪化ならゲート/抽出ルールを改善）
- 途中停止/再開が可能（partの存在チェック等）

## Milestone 5：自己学習（翻訳なし8k）
Done条件：
- 擬似並列を生成し、信頼度フィルタで絞れる
- 追加してCVが改善する（過学習していない）
補足：
- `published_texts.csv` は 7,953 行、`AICC_translation` あり 7,702 行、翻訳なしは 251 行のみ
- 現状の自己学習は翻訳なし 251 行から擬似並列を作って検証する
現状判断：
- min_confidence=0.3 の擬似並列 210 行を追加したCVで score が +0.20 改善（fold 1つだけ微減）

## Milestone 6：文脈付き入力（context-conditioned）
Done条件：
- text_id単位で前後k文を組み立てる
- CVで改善が確認できる（特に固有名詞/照応）
現状判断：
- k_prev=2, k_next=2 の文脈入力は TF-IDF baseline のCVで大幅悪化（score -5.20）したため現時点では採用見送り

## Milestone 7：n-best + リランキング + checkpoint averaging
Done条件：
- n-bestが出せる
- ルール/指標近似のリランキングができる
- averagingで安定化が確認できる
現状判断：
- n-best + ルールベースのリランキングはCVで悪化（score -1.83）したため現時点では採用見送り
- averaging は実装のみで安定化は未検証

## Milestone 8：異種アンサンブル（最終押し）
Done条件：
- byte/char＋subword＋seed違いを統合
- 推論時間がKaggle制約内で完走
- 提出事故がゼロ
現状判断：
- TF-IDF（char 3-5 + char 2-4 + word 1-2）の簡易アンサンブルはCVで悪化（score -1.59）したため現時点では採用見送り
