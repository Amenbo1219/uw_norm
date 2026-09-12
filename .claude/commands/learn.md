---
description: この会話で分かったことを Obsidian Vault にノート化する（MCP経由）
argument-hint: [トピック名（省略可）]
---

このプロジェクトでの作業から得られた知見を Obsidian Vault にノートとして残してください。
対象: $ARGUMENTS

Vault へのアクセスは `obsidian` MCP サーバ経由（vault id は `notes`）。
ローカルのファイル操作ツールでは vault に触れない。

手順:

1. `obsidian_read_note` で `CLAUDE.md` を読む。**ノートの書式・frontmatter スキーマ・
   フォルダの意味はすべてそこが唯一の正**。ここには複製しない。
2. `obsidian_search_vault` で既存ノートを探す。あれば `obsidian_edit_note` で
   追記・更新し、`updated` を今日の日付に書き換える。**重複ノートを作らない。**
3. 無ければ `obsidian_create_note` で 1概念1ノートに分けて `00_Inbox/` に作る。
4. frontmatter の `via` は `claude-code` とする。未検証の推測には
   `status: draft` と警告コールアウトを付ける。
5. 関連する既存ノートへ `[[リンク]]` を張り、リンク先にも逆リンクを追記する。

コードそのものではなく、**次に同じ問題に当たったときに効く知識**を書くこと
（詰まった原因、選択の理由、効いた手順）。リポジトリを見れば分かることは書かない。

作成・更新したノートのパスを一覧で報告してください。
