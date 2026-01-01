# Deep Past Challenge 日本語訳（統合版）

_古アッシリア語（Old Assyrian）楔形文字文書の機械翻訳コンペ説明・データセット説明の日本語訳_

## コンペ概要

Deep Past Challenge は、大胆な問いを投げかけます。
AIは4,000年前のビジネス記録を解読できるのか？
このコンペでは、古代アッシリア商人たちの日常的な商取引記録の解読を手助けします。
8,000点の楔形文字文書からなるデータを使い、あなたの目標は古アッシリア語（Old Assyrian）の翻訳システムを構築することです。世界中の博物館の引き出しには、同じような文書がさらに何千点も未解読のまま眠っています。
あなたの取り組みは、彼らの声を人類の物語へ取り戻す助けになります。

## 説明

4,000年前、アッシリア商人たちは、日常生活と商業活動に関する世界でも最も豊かな一次資料群のひとつを残しました。数万枚に及ぶ粘土板には、債務の決済、隊商の派遣、そして日々の家族のやり取りといった出来事が記録されています。ところが現在、これらの粘土板の半分は沈黙したままです。損傷しているからではなく、粘土に刻まれた言語を読める人があまりに少ないためです。多くは100年以上、博物館の引き出しで未翻訳のまま保管されてきました。
Deep Past Challenge は、この古代の謎を現代の機械学習課題へと変換し、参加者に「古代世界最大級の未翻訳アーカイブ」を解き明かす手助けを求めます。私たちは、古アッシリア語の楔形文字粘土板――青銅器時代の文書で、博物館のコレクションで1世紀以上未読のまま残ってきたもの――を対象に、翻訳モデルを構築してくれる参加者を募集します。粘土板に使われた方言である古アッシリア語は、アッカド語（Akkadian）の初期形であり、アッカド語は記録が残る最古のセム語です。
メソポタミアとアナトリアを結んだ古アッシリア交易ネットワークを示す粘土板は、約23,000枚が現存しています。しかし翻訳されたのは半分に留まり、残りを読める専門家は世界でも十数名未満です。
これらはギリシャ・ローマの「古典」のように、後世の写字生が選別し、写し続けた洗練された作品ではありません。書いた人々から直接届く、加工のない記録です。古代商人とその家族が粘土に残した手紙、請求書、契約書――いわば青銅器時代の「インスタストーリー」です。平凡で即時的で、驚くほど生々しい。
あなたの課題は、転写表記（transliteration）されたアッカド語を英語へ変換するニューラル機械翻訳モデルを作ることです。難しさは、アッカド語が低リソースかつ形態論的に複雑であり、英語なら複数語で表す内容が、単語1つに符号化され得る点にあります。現代のデータ豊富な言語向けに作られた標準的アーキテクチャは、ここではうまく機能しません。この問題を突破できれば、1万枚以上の未翻訳粘土板に声を与えられます。そして過去を蘇らせるだけでなく、AI時代がまだ十分に到達できていない、古代・現代を問わず消滅危機や見過ごされてきた何千もの言語を翻訳するための青写真づくりにも貢献できます。
運営チーム、Deep Past Initiative、そして追加の背景資料については、案内されているウェブサイトを参照してください。
https://www.deeppast.org/

## 評価

提出物は、BLEU スコアと chrF++ スコアの 幾何平均（Geometric Mean）で評価されます。各スコアの十分統計量はコーパス全体で集計され（つまり各スコアはマイクロ平均として計算されます）。
実装の詳細については SacreBLEU ライブラリを参照できます。また、Kaggle 上でこの評価指標を実装したノートブックは次にあります：「Geometric Mean of BLEU and chrF++」。

```
"""Metric for Deep Past Initiative 1 (121150): Geometric Mean of BLEU and CHRF++ scores."""

import math

import pandas as pd
import pandas.api.types
import sacrebleu


class ParticipantVisibleError(Exception):

    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    text_column_name: str,
) -> float:
    """Calculates the geometric average of BLEU and CHRF++ scores.

    This metric expects the solution and submission dataframes to contain text columns.

    The score is calculated as: sqrt(BLEU * CHRF++)
    Both BLEU and CHRF++ are on a 0-100 scale, so the result will be on a 0-100 scale.

    Parameters
    ----------
    solution : pd.DataFrame
        A DataFrame containing the ground truth text.

    submission : pd.DataFrame
        A DataFrame containing the predicted text.

    row_id_column_name : str
        The name of the column containing the row IDs. This column is removed
        before scoring.

    text_column_name : str
        The name of the column containing the text to be evaluated.

    Returns
    -------
    float
        The geometric mean of the BLEU and CHRF++ scores.


    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> text_column_name = "text"
    >>> solution = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["The dog bit the man.", "It was not a cat."]
    ... })

    Case: Perfect match
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["The dog bit the man.", "It was not a cat."]
    ... })
    >>> s = score(solution.copy(), submission.copy(), row_id_column_name, text_column_name)
    >>> print(f"{s:.1f}")
    100.0

    Case: Complete mismatch
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["Completely different.", "Nothing alike."]
    ... })
    >>> s = score(solution.copy(), submission.copy(), row_id_column_name, text_column_name)
    >>> print(f"{s:.1f}")
    0.0

    Case: Partial match
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["The dog bit the man.", "It was a cat."]
    ... })
    >>> s = score(solution.copy(), submission.copy(), row_id_column_name, text_column_name)
    >>> print(f"{s:.1f}")
    75.7
    """

    if row_id_column_name in solution.columns:
        del solution[row_id_column_name]
    if row_id_column_name in submission.columns:
        del submission[row_id_column_name]


    # Validate submission column type
    if not (
        pandas.api.types.is_string_dtype(submission[text_column_name])
        or pandas.api.types.is_object_dtype(submission[text_column_name])
    ):
        raise ParticipantVisibleError(
            f"Submission column '{text_column_name}' must be of string type."
        )

    # Extract lists of strings
    references = solution[text_column_name].astype(str).tolist()
    hypotheses = submission[text_column_name].astype(str).tolist()

    # Calculate BLEU
    # corpus_bleu expects lists of references (list of lists)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    # Calculate CHRF++ (word_order=2)
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    return math.sqrt(bleu.score * chrf.score)

```

## 提出ファイル

テストセット内の各 id について、対応するアッカド語の転写（翻字）に対する 英語訳を予測してください。各翻訳は 1文で構成されている必要があります。提出ファイルには ヘッダー行を含め、次の形式にしてください。
id,translation
0,Thus Kanesh, say to the -payers, our messenger, every single colony, and the...
1,In the letter of the City (it is written): From this day on, whoever buys meteoric...
2,As soon as you have heard our letter, who(ever) over there has either sold it to...
3,Send a copy of (this) letter of ours to every single colony and to all the trading...
...

## データセットの注意事項

アッカド語／古アッシリア語テキストを扱ううえで、圧倒的に最大の課題は書式（フォーマット）の問題に対処することです。よく言われるように「ゴミを入れればゴミが出る（garbage in, garbage out）」ので、残念ながら翻字（transliteration）されたテキストの形式は、トークナイズから変換・埋め込み処理に至るまで、機械学習ワークフローのあらゆる段階で課題を引き起こします。
そこで、こうした問題を軽減するために、翻字テキストと翻訳テキストの両方におけるさまざまな書式上の課題への対処方法について、以下の情報と提案を提供します。

## 転写表記テキスト（Texts in Transliteration）

主な書式上の課題：ハイフンでつながれた音節を含む標準的な翻字形式に加えて、写字生による追加記入によって、本文には上付き文字・下付き文字、そしてアッシリア学の専門家にしか意味の分からない句読点が付され、テキストがさらに扱いにくくなっています（下記の「完全な翻字変換ガイド」参照）。
https://oracc.museum.upenn.edu/doc/help/editinginatf/primer/inlinetutorial/index.html#Compound

また、大文字・小文字の使い分けも課題です。これは2通りの意味を符号化しているからです。単語の先頭文字が大文字の場合、その語が人名または地名（＝固有名詞）であることを示します。一方、単語がすべて大文字（ALL CAPS）の場合、それはシュメール語のロゴグラム（表語文字）であり、写字上の簡略化のためにアッカド語の音節綴りの代わりに書かれていることを示します。

限定詞（determinatives）は、アッカド語で名詞や固有名詞に対する一種の分類子として用いられます。これらの記号は通常、分類対象となる名詞の隣に上付きで印字されます。限定詞記号を語の一部として誤読してしまう可能性を避けるため、私たちは標準的な翻字ガイドに従い、限定詞を波括弧（{}）で囲んだ表記を保持しています。これは機械学習上の課題になり得ますが、翻字における波括弧の用法はこれだけである点に注意してください（例：a-lim{ki}, A-mur-{d}UTU）。

粘土板上の欠損テキスト：これらは古代文書であるため、欠損や判読不能部分（lacunae）が多数含まれます。欠損の表記を標準化するため、私たちはマーカーを2種類だけ使うことを提案します。単一の記号が欠けている小さな欠損には <gap> を、複数記号以上から大きな欠損までには <big_gap> を用います。

本チャレンジの目的のため、これらの書式上の問題を最適に扱う方法について、以下に提案を示します。

## 翻訳テキスト（Texts in Translation）

現在、古代の楔形文字文書の翻訳について、完全または大規模なデータベースは存在していません。これは特に古アッシリア語テキストに当てはまります。そのため私たちは、古アッシリア語テキストの翻訳と注釈を含む書籍や論文を集め、OCRでデジタル化し、さらにLLMを用いて修正を行いました。それだけの作業を経ても、翻訳文にはなお複数の書式上の問題が残っており、機械翻訳を成功させるうえで、この点が本チャレンジの中心的な要素になっています。
翻訳では通常、固有名詞の大文字表記が原文同様に維持されます。そして一般に、こうした固有名詞こそが、多くの機械学習タスクで性能が出にくい部分です。これらの問題に対応するため、私たちはデータセット内に、専門家が出版物向けに正規化した形での固有名詞をすべて収録した語彙集（レキシコン）を含めています。

## 現代の編集記号（Modern Scribal Notations）

最後に重要な点として、翻字（transliteration）と翻訳（translation）には、現代の編集者による写字上の注記が付随していることに注意してください。まず挙げられるのが行番号です。行番号は通常 1、5、10、15…のように振られます。ところが欠損している行がある場合、行番号の直後にアポストロフィ（’）が付き、欠損行のまとまりが2つある場合はアポストロフィが2つ（’’）付きます。これらは引用符ではなく、出版物で編集者がときどき用いる写字上の慣習です。

### 追加の写字上の注記：

- 難しい記号の読みが確実だと研究者が判断した場合の感嘆符：!
- 難しい記号の読みが不確かだと研究者が判断した場合の疑問符：?
- その行に属する記号が行の下に見つかる場合のスラッシュ：/
- 古アッシリア語の語区切り記号を示すコロン：:
- 欠損や削除（消し跡）に関するコメントを丸括弧で示す：( )
- 訂正が行われた際の写字生の挿入を山括弧で示す：< >
- 迷入・誤記とみなされる記号の範囲を二重山括弧で示す：<< >>
- 部分的に破損した記号を半括弧で示す：˹ ˺
- 明確に破損している記号や行を角括弧で示す：[ ]
- 限定詞（後述）を波括弧で示す：{ }

## 転写表記・翻訳の整形提案（Formatting Suggestions）

### 削除（現代の編集記号を除去）

- !（読みが確実）
- ?（読みが疑わしい）
- /（行の区切り）
- : または .（語の区切り）
- < >（写字生の挿入。ただし、翻字／翻訳の本文テキスト自体は残す）
- ˹ ˺（部分的に欠けた記号。翻字からは削除する）
- [ ]（文書レベルの翻字から削除。例：[KÙ.BABBAR] → KÙ.BABBAR）

### 置換（欠損・ギャップ・上付き/下付き）

[x] → <gap>
… → <big_gap>
[… …] → <big_gap>
ki → {ki}（下の完全な一覧を参照）
il₅ → il5（下付き数字は他も同様）

## 追加の文字と形式（遭遇する可能性があります）

下図は、転写表記で遭遇し得る追加文字と、CDLI/ORACC 表記・Unicode の対応例です。

（図：追加の文字と形式（CDLI / ORACC / Unicode 対応表）— 元PDF p.9）

![Additional Characters & Formats](assets/page_9.png)

（上の図をMarkdown表として再掲）

| Character | CDLI | ORACC | Unicode |
|---|---|---|---|
| á | a2 | a₂ | |
| à | a3 | a₃ | |
| é | e2 | e₂ | |
| è | e3 | e₃ | |
| í | i2 | i₂ | |
| ì | i3 | i₃ | |
| ú | u2 | u₂ | |
| ù | u3 | u₃ | |
| š | sz | š | U+161 |
| Š | SZ | Š | U+160 |
| ṣ | s, | ṣ | U+1E63 |
| Ṣ | S, | Ṣ | U+1E62 |
| Ṣ	| S, | Ṣ | U+1E62 |
| ṭ	| t, | ṭ | U+1E6D |
| Ṭ	| T, | Ṭ | U+1E6C |
| ʾ	| '	 | ʾ | U+02BE |
|₀-₉|0-9 |subscript₀-₉| U+2080-U+2089 |
| ₓ	| Xx | subscriptₓ | U+208A |
| ḫ	| h	 | h | U+1E2B |
| Ḫ	| H	 | H | U+1E2A |

図：追加の文字と形式（CDLI / ORACC / Unicode 対応表）

## 波括弧 { } によるアッカド語の限定符（Akkadian determinatives）

1. {d} = dingir ‘god, deity’ — 人間ではない神的存在の前に付く d
2. {mul} = ‘stars’ — 天体や星座の前に付く MUL
3. {ki} = ‘earth’ — 地名・場所名の後ろに付く KI
4. {lu₂} = LÚ 人物や職業名の前に付く
5. {e₂} = {É} 神殿や宮殿など、建物・施設（機関）の前に付く
6. {uru} = (URU) 村・町・都市など、集落名の前に付く
7. {kur} = (KUR) 土地・領域、および山の前に付く
8. {mi} = munus (f) 女性の人名の前に付く
9. {m} = (1 or m) 男性の人名の前に付く
10. {geš} / {ĝeš) = (GIŠ) 樹木や木製品の前に付く
11. {tug₂} = (TÚG) 織物やその他の編まれた物の前に付く
12. {dub} = (DUB) 粘土板、ひいては文書・法的記録の前に付く
13. {id₂} = (ÍD) (a ligature of A and ENGUR, transliterated: A.ENGUR) 運河や川の名称の前に付く／単独で書かれる場合は神格化された川を指す
14. {mušen} = (MUŠEN) 鳥の前に付く
15. {na₄} = (na4) 石の前に付く
16. {kuš} = (kuš) （動物の）皮、羊毛、革などの前に付く
17. {u₂} = (Ú) 植物の前に付く

## 引用（Citation）

Abdulla, F., Agarwal, R., Anderson, A., Barjamovic, G., Lassen, A., Ryan Holbrook, and María Cruz. Deep Past Challenge - Translate Akkadian to English. Kaggle, 2025.https://kaggle.com/competitions/deep-past-initiative-machine-translation

## データセット説明

コンペのデータは、8,000点を超える古アッシリア語の楔形文字テキストの翻字と、包括的なメタデータで構成されています。これらのうち一部については、対応付けされた英語訳（アライン済み翻訳）を提供します。さらに、約900本の学術出版物に含まれる未加工テキストも提供しており、そこには追加の翻訳が含まれているため、そこから追加の学習データを作成することも試みられます。
なお、このコンペは「コードコンペ」です。test.csv に含まれるデータは、解法を作成するためのダミーデータにすぎません。提出物が採点される際には、この例のテストデータは完全なテストセットに置き換えられます。

## ファイルとフィールド情報（File and Field Information）

<train.csv>
発掘された原資料の粘土板にもとづく古アッシリア語テキストの翻字が約1,500件収録されており、各テキストには英訳が付いています。
- oare_id：Old Assyrian Research Environment（OARE）データベース上の識別子(URL: https://oare.byu.edu/?utm_source=chatgpt.com)。各テキストを一意に特定します。
- transliteration：原資料の粘土板テキストをアッカド語として翻字したもの。
- translation：対応する英語訳。

<test.csv>
テストデータを代表する小規模な例示用データセットです。提出物が採点される際には、この例示用テストデータは完全なテストセットに置き換えられます。テストデータには、約400件の一意な文書から抽出された約4,000文が含まれます。学習データは「文書レベル」で翻訳が対応付けられている一方、テストデータは「文レベル」で翻訳が対応付けられている点に注意してください。
- id：各文の一意な識別子。
- text_id：各文書の一意な識別子。
- line_start, line_end：原資料（粘土板）内での文の範囲（境界）を示します。文書内での文の順序も表します。なお、このフィールドは文字列型で、1、1'、1'' のような値を取ります。行番号に関する注記は、データセット手順内の「現代の写字上の注記（Modern Scribal Notations）」の項を参照してください。
- transliteration：原資料の粘土板テキストをアッカド語として翻字したもの。あなたの目標は、対応する翻訳（英語）を生成することです。

<sample_submission.csv>
正しい形式の提出ファイル例です。詳細は Evaluation（評価）ページを参照してください。

## 補助データ（Supplemental Data）

<published_texts.csv>
OARE データベースで公開されている、古アッシリア語テキストの翻字約8,000件と、データベース／博物館記録に基づくメタデータ項目およびカタログ情報を収録しています。これらの識別子を使って、リンク先のウェブサイトから追加情報を取得できます。なお、これらの翻字には翻訳は付いていません。
- oare_id：train.csv と同様の OARE データベースの識別子。
- online transcript：DPI ウェブサイト上にある翻字トランスクリプトの URL(https://oare.byu.edu/?utm_source=chatgpt.com)。
- cdli_id：CDLI ウェブサイトの識別子(URL: https://cdli.earth/)。複数ある場合は |（パイプ）で区切られます。
- aliases：そのテキストに対する他の公開ラベル（例：出版番号、博物館 ID など）。複数は | で区切られます。
- label：テキストの主要な呼称（ラベル）。
- publication_catalog：出版物や博物館記録に見られるテキストのラベル。複数は | で区切られます。
- description：テキストの基本的な説明。
- genre_label：テキストに付与された基本ジャンル。すべてのテキストにあるわけではありません。
- inventory_position：博物館での所蔵ラベル。複数は | で区切られます。
- online_catalog：ヤール大学（Yale）のコレクション（CC-0 のメタデータと画像を含む）の URL。
- note：注釈や翻訳のために専門家が付したメモ。
- interlinear_commentary：特定の行について議論している出版物への参照。
- online_information：大英博物館（British Museum）における当該テキストの URL（画像の著作権は大英博物館にあり、CC ではありません）。すべてのテキストにあるわけではありません。
- excavation_no：発掘時にそのテキストへ割り当てられた識別子。
- oatp_key：Old Assyrian Text Project が割り当てた識別子。
- eBL_id：eBL ウェブサイトの識別子(URL: https://www.ebl.lmu.de/library/)。
- AICC_translation：最初に公開されたオンライン機械翻訳への URL(https://aicuneiform.com/search?q=P361099)。なお、これらの翻訳の多くは品質が非常に低い点に注意してください。
- transliteration_orig：OARE データベース由来の元の翻字テキスト。
- transliteration：上記の## データセットの注意事項の書式提案に基づいて整形した翻字テキスト（クリーン版）。

<publications.csv>
古アッシリア語から複数の現代語へ翻訳した内容を含む、約880本の学術出版物の「生テキスト」を収録しています。テキストは OCR により作成され、LLM による後処理が施されています。これらの翻訳を抽出し、published_texts.csv 内の翻字と対応付け（アライン）することを試みることができます。なお、翻訳は英語以外の言語で与えられていることも多い点に注意してください。
- pdf_name：テキスト抽出元の PDF ファイル名。
- page：該当テキストが出現するページ番号。
- page_text：論文（記事）本文のテキスト。
- has_akkadian：テキストにアッカド語の翻字が含まれるかどうか。

<bibliography.csv>
publications.csv に含まれるテキストの書誌情報データです。
- pdf_name：publications.csv の pdf_name に対応する ID。
- title, author, author_place, journal, volume, year, pages：一般的な書誌情報。

<OA_Lexicon_eBL.csv>
古アッシリア語の単語（翻字表記）をすべて列挙し、その語彙上の対応形（＝辞書に載っている形）を付したファイルです。リンクは、LMU がホストするオンラインのアッカド語辞書（電子バビロニア図書館 eBL, URL: https://www.ebl.lmu.de/dictionary）を参照します。
- type：語の種類（例：word、PN＝人名、GN＝地名）。
- form：翻字に現れるそのままの文字列。
- norm：ハイフンを除去し、母音長の表示を含む正規化形。
- lexeme：辞書に載る見出し形（レンマ）。
- eBL：電子バビロニア図書館（eBL）のオンライン辞書 URL。
- I_IV：同音異義レンマのローマ数字区分。eBL にある Concise Dictionary of Akkadian（CDA）に対応。
- A_D：同音異義レンマのアルファベット区分。Chicago Assyrian Dictionary に対応。URL: https://isac.uchicago.edu/research/publications/chicago-assyrian-dictionary
- Female(f)：女性性（女性）を示す区分。
- Alt_lex：代替の正規化形。

<eBL_Dictionary.csv>
eBL データベースにあるアッカド語辞書の完全版です。OA_Lexicon_eBL.csv 内の eBL（URL）で提供されるデータを集約しています。

<resources.csv>
追加データとして利用できそうなリソース一覧です。

<Sentences_Oare_FirstWord_LinNum.csv>
train.csv のデータを文単位で翻訳と対応付け（アライン）するための補助ファイルです。各文の先頭語と、その粘土板上での位置を示します。

## 追加の学習データを作るための推奨ワークフロー

publications.csv には、約900本のPDFから得られたOCR出力が含まれており、そこから翻訳文を抽出することが最初の重要なステップになります。機械学習を始める前に、学習データを再構成し、対応付け（アライン）する必要があります。以下は、取り組みやすい手順の一例です。

1.各テキストと翻訳を特定する
文書識別子（ID、別名、博物館番号など）を使って、翻字（transliteration）と、OCR出力内の対応する翻訳を突き合わせます。
2.翻訳をすべて英語に統一する
翻訳元は複数言語（例：英語、フランス語、ドイツ語、トルコ語）で書かれている可能性があります。一貫性のため、すべて英語に変換します。
3.文単位でアラインを作る
アッカド語の翻字と、対応する英訳の両方を文に分割し、文どうしをペアで対応付けます。この「文レベルの対応表」が、機械翻訳モデルの学習・評価に最も有用な形式です。

これらの手順が完了すれば、機械学習に投入できる、整ったアライン済みデータセットが得られます。

## 参考文献（Bibliography）

参考文献リストは、本チャレンジの翻訳を取得するために使用した二次資料を反映しています。著作権の扱いが資料ごとに異なるため、機械翻訳の生成に利用した場合は、各文献を引用することを推奨します。
一次資料に関する追加の参考文献（引用情報）は、以下で確認できます。

- https://cdli.earth/publications
- https://cdli.ox.ac.uk/wiki/abbreviations_for_assyriology