# 00_REPO_BLUEPRINT.md
Deep Past Challenge（Akkadian/Old Assyrian transliteration → English）
Codex（Cursor）実装用：リポジトリ設計図（最小で動く → 強化の順で拡張）

## 0. 目的（このリポジトリで達成すること）
- Kaggle Code Competition向けに、**学習→推論→submission.csv生成**までを再現可能にする
- train（文書レベル）↔ test（文レベル）のギャップを埋め、トップ帯を狙える強いNMTを構築する
- 主要アウトプット
  - `submission.csv`（`id,translation` / 1行=1文）
  - 学習済みモデル（複数：byte/char系主力＋subword系副＋seed違い）
  - 中間生成物（文分割・文アライン済み学習データ、品質スコア、辞書・置換ログ）

## 1. 推奨リポジトリ構造
 translate-akkadian/
  docs/
    translate_akkadian.md      # コンペとデータセットの説明
    top_model_plan.md          # コンペでトップを狙う戦略
    00_*.md〜05_*.md           # top_model_plan.pdfの詳細
  data/                        # 入力データ（Kaggleの /kaggle/input を想定）
  artifacts/                   # 中間生成物（アライン済み、疑似並列、品質スコア）
  models/                      # 学習済み重み（必要に応じて）
  src/
    dp/                        # パッケージ本体（Python package）
      __init__.py              # これを置いて dp をパッケージ化する
      data/                    # 前処理、文分割、アライン、辞書
        __init__.py
      models/                  # モデル定義（HF/自作ラッパ）
        __init__.py
      utils/                   # ログ、設定、seed、I/O
        __init__.py
      train.py                 # 学習入口（python -m dp.train）
      infer.py                 # 推論入口（python -m dp.infer）
      submit.py                # submission生成入口（python -m dp.submit）
      eval.py                  # 指標計算（python -m dp.eval）
      validate.py              # 提出事故防止（python -m dp.validate）
  configs/
    baseline_byt5.yaml
    baseline_subword.yaml
    context.yaml
  tests/
    test_validate_submission.py
    test_preprocess_invariants.py
  scripts/
    run_cv.sh
    run_train.sh
    run_infer.sh
  README.md                    # プログラムの使用方法や各ファイルの役割を記載

## 2. 入口コマンド（CLI）
**最初に動く最小構成**（Milestone 0）として以下を揃える。
- `python -m dp.train --config configs/baseline_byt5.yaml`
- `python -m dp.infer --config configs/baseline_byt5.yaml --ckpt <path>`
- `python -m dp.submit --pred <pred_path> --out submission.csv`
- `python -m dp.eval --pred submission.csv --gold <gold.csv>`（ローカル検証用）
- `python -m dp.validate --submission submission.csv`（提出事故防止）

## 3. ルール/運用上の前提（コードコンペ対応）
- ネットワーク無効前提（Kaggle実行） → 事前に必要なモデル/辞書はデータセットとして同梱、またはKaggleで提供されるモデルを利用
- 乱数固定：Python/NumPy/PyTorch/Transformers（seed）
- 推論時間・メモリは最初から計測し、最後に“入らない”を防ぐ

## 3.1 ローカル開発環境（venv）
ローカルでは venv（.venv）で実行する。

### セットアップ
- Python は Kaggle に寄せる（例：3.10 〜 3.11）
- 仮想環境作成
  - `python -m venv .venv`
- 有効化
  - macOS/Linux: `source .venv/bin/activate`
  - Windows: `.venv\\Scripts\\activate`
- 依存インストール
  - `python -m pip install -U pip`
  - `pip install -r requirements.txt`
- 以後は仮想環境上で実行（`python -m dp.train ...` など）
  - プログラム側で venv を自動作成/強制はしないため、実行前に必ず有効化する

### 依存管理の方針（重要）
- 依存は `requirements.txt` を単一の正（source of truth）とする
- Kaggle/Colab でも同じ `requirements.txt` を `pip install -r requirements.txt` で入れる
- `.venv/` は Git 管理しない（.gitignore に追加）

### srcレイアウトの扱い
- `src/dp` を import できるように、どちらかで統一する
  - 推奨：`pip install -e .`（pyproject.toml を用意して editable install）
  - 代替：環境変数 `PYTHONPATH=src` を設定して実行

## 3.2 大規模CSV（publications.csv）の処理方針（必須）
publications.csv（約20万行以上）は全行をメモリに載せて処理しない。必ず次の方針で実装する。

- **事前分割してLLMで処理する方針は採用しない**（再現性・コスト・管理性が悪い）
- 代わりに **コード側でストリーミング（チャンク処理）**し、かつ **先に候補抽出で母集団を激減**させる
- `has_akkadian` を一次フィルタに使い、page_text の改行/ハイフネーションを正規化してから文抽出する
- 中間生成物はCSVではなく **Parquet** を基本とし、途中再開できるよう **part-XXXX.parquet** に分割出力する

Done条件：
- publications.csv をメモリ全読み込みせずに処理できる（chunksize / lazy / duckdb）
- 途中で落ちても再開できる（part出力＋再処理防止の仕組み）

## 4. Done条件（作業完了の定義）
各タスクは「成果物 + テスト + 計測」で完了とする。
- 例：前処理変更 → `tests/test_preprocess_invariants.py` がPASSし、CVが改善 or 退化理由が説明可能
- 例：提出生成 → `dp.validate` がPASS（1文・改行なし・全idあり・空行なし）

## 5. ログ・成果物命名規則（再現性）
- 実験ID：`YYYYMMDD_HHMM_<short_tag>`
- 保存先：`artifacts/exp/<exp_id>/...`
- すべての実験で保存
  - config（実行時に確定したyaml）
  - git commit hash
  - seed
  - 指標（fold別 + 平均）
  - 主要ハイパラ（max_len, lr, batch, context_k 等）
