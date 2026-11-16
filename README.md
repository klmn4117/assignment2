# 自然言語処理 第2回演習：日本語投稿のポジ／ネガ判定

自作の日本語投稿10件を、単語リストとローカルLLMで分類する。公開データは使わない。実測結果は**単語リスト5/10（50%）、LLM9/10（90%）**。

- [レポート：タスク・手法・設定・結果・誤り考察](report.md)
- [正解ラベル付き投稿10件](data/posts.csv)：positive 5件、negative 5件。10件とも文脈によって判断に迷う表現を含む。
- [全件の予測結果](results/predictions.csv)、[集計・設定・入力ハッシュ](results/summary.json)
- [生成AI利用申告](AI_USAGE.md)、[実装方針](PLAN.md)

## 環境と依存関係
Python 3.10以上。確認環境はPython 3.11.9 / Windows、外部Pythonパッケージ不要。requirements.txtはその旨を記載している。
LLMの再推論に限り、Qwen3.5-122BのFP8モデルを起動したvLLMのOpenAI互換サーバーが必要。モデルの導入・ダウンロードはこのリポジトリに含めない。

## 保存済み結果の再評価（サーバー不要）
リポジトリのフォルダーで実行する。

```console
python evaluate.py --llm results/llm_response.json --metadata results/llm_metadata.json
python -m unittest discover -s tests -v
```

10件制約、ラベル、重複・欠落した回答、入力ハッシュを検証して採点する。LLMの回答順序は問わずIDで対応付ける。未実行は0点に換算しない。既定の入出力パスはスクリプトのあるフォルダーを基準にする。

ルールだけを実行し、LLM用のラベルなし入力を生成する場合：

```console
python evaluate.py --out results/rule_only
```

ルールは肯定7語・否定7語の部分文字列出現回数を合計し、肯定点が否定点を上回ればpositive、それ以外はnegative。該当語なし・同点もnegative。否定・引用・皮肉への特別処理はしない。

## ローカルLLMを再実行する場合
モデルのAPI識別子は `qwen3.5-122b-artifact`。サーバーが別のIDを使う場合は実際のIDを指定する。認証キーは環境変数 `LOCAL_LLM_API_KEY` に設定する。キーの値をファイルやGitに保存しない。

```console
python run_llm.py --base-url http://localhost:8005/v1 --model qwen3.5-122b-artifact
```

別名の環境変数は `--api-key-env 変数名` で指定できる。サーバーはlocalhost/ループバックのみ許可する。
各実行は独立した1件のuserメッセージとして、固定プロンプトとid・textだけの10投稿を一括送信する。正解ラベル・判断理由・ルール出力は送らない。
設定はtemperature=0、top_p=1、seed=42、max_tokens=4096、stream=false。指定していないパラメーターはサーバー既定値。設定と入力は推論前に保存する。失敗時に自動で再推論せず、結果の良い回だけを選別しない。

`results/runs/<UTC日時>/` に送信本文、モデル一覧、APIの生応答、回答JSON、実行条件、個別結果・集計を保存する。認証ヘッダーは保存しない。有効な回答を検証後に `results/` の採点結果を更新する。
コードフェンスがあれば外側だけ除去し、回答のラベルは変更しない。サーバーが返した補助的なreasoningフィールドは生応答に残るが、採点はcontent内のラベルのみを使う。

再採点は保存回答から再現できる。再推論はモデルの配布元リビジョン、vLLMのバージョン、ハードウェア、サーバー既定値等にも依存するため完全一致は保証しない。FP8形式とvLLM利用は実行環境を用意した本人の申告であり、正確な配布元リポジトリ・リビジョン・バージョンは未記録。

## 構成
```text
data/          投稿・正解ラベル・理由、単語リスト
prompts/       固定プロンプト
results/       全件予測、集計、LLM回答・実行条件、実行記録
tests/        入力検証と模擬HTTPサーバーによる結合テスト
evaluate.py   データ検証・ルール分類・保存回答の採点
run_llm.py    ローカルLLMへの実リクエスト・記録・採点
report.md     提出用レポート
AI_USAGE.md   生成AI利用申告
```

## 提出
GitHubリポジトリのURLをClassroomへ提出する。公開リポジトリは招待・ログインなしでcloneできる。氏名・学籍番号等が授業で別途必要ならClassroom側に記入する。講義PDF、認証情報、キャッシュはGit対象外。
