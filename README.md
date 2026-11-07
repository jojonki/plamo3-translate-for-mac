# PLaMo 3 Translate for Mac

Preferred Networks の翻訳特化 LLM [PLaMo 3 Translate](https://huggingface.co/collections/pfnet/plamo-3-translate) を、Apple silicon の Mac 上だけで動かす翻訳 UI です。起動後の翻訳はネットワークを使いません。

## 使い方

Finder で `PLaMo Translate.command` をダブルクリックするか、次を実行します。

```bash
./run.sh
```

初回はモデル（約 6 GB）をダウンロードします。準備ができるとブラウザで http://127.0.0.1:8765/ が開きます。終了はターミナルで Ctrl+C か、ウィンドウを閉じます。

必要なもの: Apple silicon の Mac と [uv](https://docs.astral.sh/uv/)（`brew install uv`）

## モデルのダウンロード

モデルはこのリポジトリに含まれていません。各ユーザーの Mac に、初回起動時に自動でダウンロードされます。

- 取得元は PFN の Hugging Face リポジトリ（`pfnet/plamo-3-translate-8b-MLX-4bit`）で、このプロジェクトを経由しません。
- 保存先は `models/` です。重みのほか、モデル定義のコード、トークナイザー、推論サーバー（`serve.py` など）も含まれます。このディレクトリは Git の管理対象外なので、コミットしないでください。
- 取得するリビジョンは `app.py` の `MODEL_REVISION` で固定しています。
- 2 回目以降はダウンロードせずに起動します。
- モデルの利用には [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license) が適用され、ライセンスは各ユーザーと PFN の間で成立します。同意しない場合は、ダウンロードの案内が出た時点で Ctrl+C で中止してください。

## 機能

- 日本語⇔英語の自動切り替え、53 言語の指定
- 訳文のストリーミング表示と途中停止（⌘↵ で翻訳）
- 日本語の文体指定（です・ます / だ・である）
- コンテキストと用語集
- 長文は段落ごとに分割し、直前の訳文を参照しながら翻訳

## モデルの選び方

既定は `plamo-3-translate-8b-MLX-4bit` です。Mac mini (M6, 24 GB) での実測は、ピークメモリ約 7.5 GB、短文の翻訳が 1 秒前後、長文（英語 1.1 万字）が 142 秒でした。

| サイズ | 重み | 向いている環境 |
| --- | ---: | --- |
| 0.8B | 1 GB | メモリ 8 GB |
| 2B | 2.4 GB | メモリ 8〜16 GB |
| **8B** | 6 GB | メモリ 16 GB 以上 |
| 31B | 18.7 GB | メモリ 32 GB 以上を推奨（未検証） |

8B と 31B の精度差はわずかです（PFTB 0.6102 / 0.6222、モデルカードより）。上の「向いている環境」は目安で、実測しているのは 8B だけです。別のサイズを使うときは、ダウンロードしてディレクトリを指定します。

```bash
.venv/bin/hf download pfnet/plamo-3-translate-2b-MLX-4bit --local-dir models/plamo-3-translate-2b-MLX-4bit
./run.sh --model models/plamo-3-translate-2b-MLX-4bit
```

モデルに同梱のコードが実行される（`trust_remote_code`）点に注意してください。既定モデルは内容を確認したリビジョンに固定しています。

## 構成

- `app.py`: UI と翻訳 API を `127.0.0.1` で提供し、モデル同梱の `serve.py`（mlx-lm サーバー）を子プロセスとして管理する
- `static/index.html`: UI
- `tests/`: `uv run pytest` で実行
- `logs/mlx-server.log`: 推論サーバーのログ

## ライセンス

このリポジトリのコードは [MIT License](LICENSE) です。モデルは含まれていません。

**Built with PLaMo.** モデルは初回起動時に Hugging Face からダウンロードされ、[PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license) が適用されます。MIT License はモデルには適用されません。

- 個人利用と非商用利用は無料で、登録も不要です。
- 商用利用には PFN への登録または個別契約が必要です（前年度の売上で条件が変わります）。組織内部での利用も商業目的に含まれます。
- 翻訳結果を公表するときは、このモデルの生成物であることを明示する必要があります。

詳しくはライセンス本文を確認してください。

このプロジェクトは非公式のもので、Preferred Networks とは関係ありません。
