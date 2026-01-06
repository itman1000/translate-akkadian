# val_audit.csv 列定義（簡易）

このファイルは、validation の `src/ref/pred` に対して **ずれの原因を診断**するための監査CSVです。
- **先頭9列（`oare_id`〜`pred`）は `val_predictions.csv` と同じ**で、行対応が取れるようになっています。
- `*_n0/n1/n2` は「分類のための正規化」テキストです（提出テキストを強く正規化する意図ではありません）。

## 正規化レベル（n0/n1/n2）

- **n0**: Unicode統一・空白/改行の統一など最小限（情報を極力保持）
- **n1**: 句読点前後の空白など“表記ゆれ”を追加で吸収
- **n2**: 分類専用の強め正規化（小文字化・広めの句読点除去など）。ただし `<gap>/<big_gap>` は保持

## 列一覧

| column | 説明 |
|---|---|
| `oare_id` | 文書ID（OARE）。doc分割・doc内位置判定に使う。 |
| `src_norm_variant` | source正規化のバリアント名（A/B/Cなど）。 |
| `align_method` | 文書→文（または文境界）のアライン方法（例: dp=Dynamic Programming）。 |
| `align_meta` | アラインの追加メタ情報（JSON想定。空の場合あり）。 |
| `flags` | 前処理・生成時のフラグ等（文字列/JSON想定。空の場合あり）。 |
| `src` | モデル入力に使ったソース文字列（gloss入りのことがある）。 |
| `ref` | 参照訳（正解）。 |
| `src_no_gloss` | gloss（<LEX>…</LEX>）を除去したソース。 |
| `pred` | モデル予測（提出候補）。 |
| `ref_raw` | 参照訳のraw（監査用にrefをそのまま保持）。 |
| `pred_raw` | 予測のraw（監査用にpredをそのまま保持）。 |
| `ref_n0` | refの正規化n0（Unicode/空白など最小限を統一）。 |
| `pred_n0` | predの正規化n0。 |
| `ref_n1` | refの正規化n1（句読点前後の空白など評価に効きやすい揺れを吸収）。 |
| `pred_n1` | predの正規化n1。 |
| `ref_n2` | refの正規化n2（分類専用の強め正規化：小文字化/広めに句読点除去 等。ただし<gap>保持）。 |
| `pred_n2` | predの正規化n2。 |
| `sim_self` | pred_n2 と ref_n2 の類似度（0〜1、Levenshtein正規化類似度）。 |
| `len_ref` | ref_n0 の文字長（簡易）。 |
| `len_pred` | pred_n0 の文字長（簡易）。 |
| `len_ratio` | (len_pred+1)/(len_ref+1) の長さ比。極端だと短文化/冗長化の兆候。 |
| `ref_gap` | ref_n0 に含まれる <gap> 数。 |
| `ref_big_gap` | ref_n0 に含まれる <big_gap> 数。 |
| `pred_gap` | pred_n0 に含まれる <gap> 数。 |
| `pred_big_gap` | pred_n0 に含まれる <big_gap> 数。 |
| `pred_has_modern` | predに現代編集記号（[ ]、<< >>、< >など ※<gap>除外）が残っていそうならTrue。 |
| `pred_multi_sentence` | predが複数文っぽい（文境界が複数）ならTrue（force_single_sentence検知）。 |
| `pred_repetition` | 反復ループっぽさのスコア（大きいほど怪しい）。 |
| `ref_digit` | refに含まれる数字（0-9）の個数。 |
| `pred_digit` | predに含まれる数字（0-9）の個数。 |
| `pred_gloss_leak` | predにglossトークン（<LEX>等）が漏れた疑いがあればTrue。 |
| `pred_translit_leak` | predに転写表記っぽいトークン（記号/ダイアクリティカル/ハイフンなど）が混入した疑いがあればTrue。 |
| `ref_name_cnt` | ref内の“名前っぽい語”の検出数（固有名詞ヒューリスティック）。 |
| `pred_name_cnt` | pred内の“名前っぽい語”の検出数。 |
| `ref_name_sample` | refで検出した名前っぽい語のサンプル（;区切り、上限あり）。 |
| `pred_name_sample` | predで検出した名前っぽい語のサンプル。 |
| `pred_template_count` | 同一（正規化済み）predテンプレの出現回数（val内）。テンプレ崩壊検知用。 |
| `pred_template_rank` | predテンプレの頻度順位（1=最頻）。 |
| `doc_pos` | 文書内の位置（0始まり想定）。doc内比較（前後文）に使う。 |
| `sim_prev` | pred_n2 と（同一doc内の）前の ref_n2 の類似度。 |
| `sim_next` | pred_n2 と（同一doc内の）次の ref_n2 の類似度。 |
| `sim_merge_next` | pred_n2 と（ref_n2 + 次ref_n2）の連結文の類似度（“結合ずれ”検知）。 |
| `sim_merge_prev` | pred_n2 と（前ref_n2 + ref_n2）の連結文の類似度。 |
| `best_offset` | ±k行シフト比較で最も近かったオフセット（例: -1, +2）。0は自行refが最良。 |
| `best_sim` | best_offset のときの類似度。 |
| `best_delta` | best_sim - sim_self（シフトの方がどれだけ良いか）。 |
| `type_primary` | 主因のずれ分類コード（例: T00_EXACT, T03_CASE..., T21_TRUNCATION, T90...）。 |
| `type_secondary` | 副因タグ（TAG_*）を | で連結。複数可。 |
| `fix_route` | 修正方針の推奨（例: OK / OUTPUT_PP / DECODE / ALIGN / MODEL）。 |
| `t90_reason` | type_primary=T90のときの二次分類（T90_TEMPLATE_COLLAPSE / T90_NUMERIC / T90_NAME / T90_COPY_SRC / T90_OTHER）。 |


## コードの意味

表記・形式系（前処理/後処理で一気に直せる）
| code                        | ずれの特徴                      | 自動検出の目安                                    | 対処（fix_route）                       |
| --------------------------- | -------------------------- | ------------------------------------------ | ----------------------------------- |
| **T01_UNICODE_OR_WS**       | 空白/改行/Unicode記号の差だけ        | `pred_n0 == ref_n0`                        | `OUTPUT_PP`（最小正規化を提出直前に適用）          |
| **T02_PUNCT_SPACING**       | 句読点の有無・句読点前後の空白            | `pred_n1 == ref_n1`                        | `OUTPUT_PP`（スペース規則統一）               |
| **T03_CASE_OR_MINOR_STYLE** | 大文字小文字・軽微なスタイル             | `pred_n2 == ref_n2` かつ `pred_n1 != ref_n1` | まずは**壊してないか確認**：固有名詞/ALL CAPS保持が前提  |
| **T04_NUM_FORMAT**          | `1,000` vs `1000` / 単位表記揺れ | 数字周りだけdiff（正規表現）                           | `OUTPUT_PP`（数値ルール統一）                |
| **T05_QUOTE_DASH**          | `„` や “ ”、–/— の揺れ          | 特定Unicode文字の出現                             | `INPUT_PP`+`OUTPUT_PP`（両側で統一）       |

欠損・編集記号系（コンペ特有：ここを外すと精度が落ちる）
| code                        | ずれの特徴                                           | 自動検出の目安                                    | 対処（fix_route）                                |
| --------------------------- | ----------------------------------------------- | ------------------------------------------ | -------------------------------------------- |
| **T10_GAP_MARKER**          | `<gap>/<big_gap>` の抜け・種類違い                      | `n_gap/n_big_gap` の差が大、または `[x]`/`…` が残ってる | `INPUT_PP`（`[x]→<gap>`, `…→<big_gap>` 等に統一）  |
| **T11_MODERN_NOTATION**     | `! ? / : < > << >> ˹ ˺ [ ] ( )` 等の現代注記が混入/除去しすぎ | これらの文字が `src/pred/ref` に残る                 | `INPUT_PP`（削除/保持ルールを固定）                      |
| **T13_DETERMINATIVE_BRACE** | `{d}` `{ki}` 等の波括弧が消える/増える                      | `{` `}` 数の差、または `ki` の扱い揺れ                 | `INPUT_PP`（限定詞は情報なので保持前提）                    |
| **T14_SUBSCRIPT**           | `il₅` vs `il5` など下付き数字                          | Unicode下付きが残る                              | `INPUT_PP`（下付き→通常数字）                         |

生成制約・デコード事故系（後処理 or デコード調整）
| code                   | ずれの特徴               | 自動検出の目安                | 対処（fix_route）                      |
| ---------------------- | ------------------- | ---------------------- | ---------------------------------- |
| **T20_MULTI_SENTENCE** | 複数文生成 / 改行混入（提出は1文） | `n_newline>0`、句点が複数等   | `DECODE` + `OUTPUT_PP`（改行禁止・1文化）   |
| **T21_TRUNCATION**     | 途中で途切れる/極端に短い       | `len_ratio` が極端、末尾が不自然 | `DECODE`（max_len/length_penalty）   |
| **T22_REPETITION**     | 同語反復ループ             | 同一n-gram連発             | `DECODE`（repetition penalty相当、禁止語） |

アライン・文分割系（“前処理で最も伸びる”ことが多い）
| code                      | ずれの特徴              | 自動検出の目安                                            | 対処（fix_route）        |
| ------------------------- | ------------------ | -------------------------------------------------- | -------------------- |
| **T30_ALIGN_SHIFT**       | 1つ前/後の答えと似ている（行ズレ） | `sim(pred, ref[i±1]) > sim(pred, ref[i]) + margin` | `ALIGN`（分割/対応付け再構成）  |
| **T31_ALIGN_MERGE_SPLIT** | 2行分が結合 / 1行が分割されてる | `sim(pred, ref[i]+ref[i+1])` が高い等                  | `ALIGN`（DP等で文境界最適化）  |
| **T32_BAD_PAIR**          | 原文と答えがそもそも別物っぽい    | どの近傍とも似ない/特徴量異常                                    | `DROP`（品質ゲートで除外）     |

“前処理ではない”領域（分類しておくと迷走しない）
| code                      | ずれの特徴              | 自動検出の目安          | 対処（fix_route）                           |
| ------------------------- | ------------------ | ---------------- | --------------------------------------- |
| **T90_SEMANTIC_OR_MODEL** | 意味が違う（語彙選択/省略/誤解釈） | n2でも一致しない、内容語が違う | `MODEL`（学習/特徴/lexicon/文脈）               |
| **T91_OCR_NOISE_REF**     | 答え側が明らかにOCRノイズ     | refに異常文字、単語崩れ    | 原則`DROP`か、学習時にロバスト化（ただし提出側はrefに寄せる必要あり） |

### T90 の二次分類を追加(type_primary=T90_SEMANTIC_OR_MODEL の中を、t90_reason で粗く分類)

T90_TEMPLATE_COLLAPSE（同一テンプレ予測の多発）
T90_NUMERIC（数字の落ち/幻覚）
T90_NAME（固有名詞の落ち/幻覚）
T90_COPY_SRC（転写・原文混入っぽい）
T90_OTHER