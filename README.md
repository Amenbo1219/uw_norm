# kabeuchi-agent-template

`p40` サーバ（[kabeuchi-llm](https://github.com/Amenbo1219/kabeuchi-llm) のスタック）の
ローカルLLMをバックエンドに **Claude Code を動かすためのプロジェクト雛形**。
GitHub の Template repository として使う。

- モデル: LiteLLM 経由のローカル `local-kabeuchi`（gemma4-26b-a4b / Tesla P40）
- ツール: Obsidian Vault を読み書きする MCP サーバ

---

## ローカルに向くのは `./bin/claude-local` を打ったときだけ

**接続先の環境変数を `.claude/settings.json` にも `~/.bashrc` の `export` にも
常設していない。** 常設すると、画面上は普通の Claude Code と見分けが付かないまま
ローカルモデルに繋がってしまい、取り違える。

| 打つもの | 繋がる先 |
|---|---|
| `claude` | 普段どおりの Claude Code（サブスク）。このディレクトリでも変わらない |
| `./bin/claude-local` | ローカルの `local-kabeuchi`（引数はそのまま `claude` に渡る） |

起動後の見分けはステータスラインで付く。ローカルのときだけ赤く出る:

```
◆ LOCAL local-kabeuchi @127.0.0.1:4000 · myproject · main
```

サブスクのときは淡色で `Claude (subscription)` と出る。エイリアスと違って
**後からそのウィンドウを見ても判別できる**のが利点。

Obsidian MCP と `/learn` は**どちらのモードでも使える**（`.mcp.json` は
接続先と無関係のため）。

---

## 使い方

```bash
gh repo create <新プロジェクト名> --template Amenbo1219/kabeuchi-agent-template --private --clone
cd <新プロジェクト名>
./bin/kabeuchi-doctor      # 疎通確認（キー→モデル→/v1/messages→MCP）
claude                     # 初回だけ trust ダイアログを承認する（MCPの有効化に必要）
./bin/claude-local         # ローカルモデルで作業
```

### 手元のマシンから使う場合

サーバー内・LAN・外部で別々のアドレスを使うと設定が3本に増えるので、
**どこからでも `127.0.0.1:4000` を指す**ように SSH トンネルで揃える。

`~/.ssh/config`:

```
Host p40
  HostName 10.10.10.40      # 外部からはグローバル側の到達経路に置き換える
  User amembo
  LocalForward 4000 127.0.0.1:4000
```

```bash
ssh p40                              # トンネルを張ったまま維持する
export KABEUCHI_SSH_HOST=p40         # Obsidian MCP を ssh 越しに起動させる
./bin/claude-local
```

`KABEUCHI_SSH_HOST` を設定しない場合、Obsidian MCP はローカルの docker を探して
失敗する（vault の実体はサーバ上のコンテナ内にしか無いため）。

### APIキー

`bin/claude-local` がこの順で `LITELLM_MASTER_KEY` を探し、
`ANTHROPIC_AUTH_TOKEN` として渡す:

1. `$KABEUCHI_KEY_FILE`（未設定なら `~/.config/kabeuchi/litellm-key`）
2. `~/kabeuchi-llm/.env` の `LITELLM_MASTER_KEY=`

サーバー内では 2 が効くので何もしなくてよい。手元のマシンには `.env` が無いので、
1 を一度だけ作る:

```bash
mkdir -p ~/.config/kabeuchi
printf '%s' '<LITELLM_MASTER_KEY の値>' > ~/.config/kabeuchi/litellm-key
chmod 600 ~/.config/kabeuchi/litellm-key
```

**キーはリポジトリに入れないこと。** この雛形にキーを直書きすると、
テンプレートから作る全リポジトリにコピーされる。

---

## 中身

| ファイル | 役割 |
|---|---|
| `bin/claude-local` | ローカルモデルで `claude` を起動（接続先・モデル・64kを渡す） |
| `bin/statusline` | 今どちらに繋がっているかを常時表示 |
| `bin/obsidian-mcp` | vault への stdio ブリッジ（コンテナ内の obsidian-mcp を借りる） |
| `bin/kabeuchi-doctor` | 疎通確認を上から順に |
| `.claude/settings.json` | statusline・MCP自動承認・読み取り系ツールの許可**のみ**（接続先は入れない） |
| `.mcp.json` | `obsidian` MCP サーバの登録 |
| `.claude/commands/learn.md` | `/learn` — 会話の知見を Vault にノート化する |
| `CLAUDE.md` | プロジェクト説明。**書き換えて使う**（「動作環境」節だけ残す） |

### モデル指定について

`bin/claude-local` は `ANTHROPIC_MODEL` に加えて `ANTHROPIC_DEFAULT_HAIKU_MODEL` /
`ANTHROPIC_SMALL_FAST_MODEL` も `local-kabeuchi` にしている。Claude Code は要約や
補助的な処理を小型モデルに投げるが、LiteLLM 側の Claude 系モデルは
`ANTHROPIC_API_KEY` がプレースホルダのままなので、指定しないとそこで 401 になる。

`CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536` は llama.cpp の `-c`（`.env` の `CTX_SIZE`）と
揃えてある。Claude Code は未知のモデル名に既定で 200k を仮定するので、
指定しないと auto-compact が効かず窓を超える。**サーバ側の `CTX_SIZE` を変えたら
ここも変える。**

### Obsidian の経路

Open WebUI 用の `mcpo`（:8001, OpenAPI変換）とは**別経路**。Claude Code は MCP を
直接喋れるので変換を挟まず、稼働中の `kb-mcpo-obsidian` コンテナの中で
`obsidian-mcp` をもう一つ stdio で起動して使う。ホスト側に Node は不要。

読み取り系（`list_vaults` / `search_vault` / `read_note`）だけ `.claude/settings.json`
で自動許可してある。作成・編集・削除は毎回確認が出る。

---

## 期待値

**実用性は期待しないこと。** P40 + gemma4-26b-a4b は生成 約52 t/s・コンテキスト 64k。
Claude Code は長いシステムプロンプトとツール呼び出しを何往復もさせるので、
動きはするが待ち時間とツール呼び出しの失敗率が高い。タダで回せる実験環境と
割り切る用途。本気でコードを書かせるなら `claude`（Pro/Maxサブスク）をそのまま
使う方が速くて安い。

## 困ったとき

まず `./bin/kabeuchi-doctor`。それでも分からない場合:

- **応答が空で返る** — gemma は思考部を `reasoning_content` に分離するため、
  `max_tokens` が小さいと推論の途中で打ち切られて本文が空になる。
- **`Prompt is too long`** — サーバ側の `CTX_SIZE` が足りない。Claude Code は
  システムプロンプト+組み込みツールだけで 32k をほぼ使い切る（64k 必要）。
- **Obsidian が空を返す / MCP が落ちる** — コンテナではなく rclone マウントを疑う。
  サーバ側で `systemctl --user restart rclone-obsidian.service` の後、
  **必ず** `docker compose restart mcpo-obsidian`（順序を守らないと空の vault を掴む）。
- **モデル一覧に `local-kabeuchi` が無い** — 手元のマシンならトンネル切れ。
