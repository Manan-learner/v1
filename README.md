# v1 — Multi-LLM Proactive Context Finder & Google Drive Memory Backup

A lightweight, **zero-dependency** toolkit giving AI coding agents persistent, portable memory across sessions, machines, and LLM harnesses:

1. **Proactive Context Finder & Recall (`v1-context`, `v1-recall`)** — Scans past conversations across **Gemini Antigravity CLI**, **Claude Code**, **OpenAI Codex CLI**, and custom **JSONL/Markdown** logs using local SQLite FTS5 + domain-token overlap (`>= 25%` cutoff). Automatically checks new conversation starts and **every 5 prompts** as a tumbling topic-pivot batch, presents the **top 3 summaries**, and requires **explicit user permission** before loading full transcripts.
2. **Non-Destructive Cloud Memory Backup (`v1-sync`, `v1-status`)** — Incrementally mirrors conversation histories, transcripts, SQLite databases, skills, and hook configs to Google Drive (via a **mounted Google Drive Desktop folder** OR **`rclone`**) using WAL-safe SQLite `.backup()` snapshots.
3. **Universal MCP Stdio Server (`v1-mcp`)** — Exposes `find_context`, `recall_conversation`, `sync_backup`, `backup_status`, and `reindex_providers` to **any** MCP-compatible client with zero third-party pip packages.

---

## Why It Exists & Why It Works

| Challenge | How `v1` Solves It |
| :--- | :--- |
| Reinstalling an OS or switching laptops wipes local agent chats & skills | Incremental non-destructive delta sync preserves cloud history (`never deletes old Drive snapshots`) + one-command `bootstrap` restores across Windows/macOS/Linux. |
| Active SQLite conversations hold `-wal` locks that corrupt plain file copies | `sqlite_safe_snapshot()` uses SQLite's online `.backup()` API so every live DB is copied atomically. |
| Windows Smart App Control / enterprise policies block unsigned binaries like `rclone.exe` | Dual-backend storage: automatically uses a signed **Google Drive for Desktop** mount (`G:\My Drive\...`) natively in Python, falling back to `rclone` on Linux/macOS/unrestricted Windows. |
| Running LLM embeddings or an LLM classifier on every prompt adds cost & latency | Pure local SQLite FTS5 + curated coding-stopword domain matching finishes in **~115–250 ms** offline at **$0 extra API cost**. |
| Mid-chat proactive suggestions get noisy if run on every single turn | Auto-checks run **once on turn 1** (deferring if turn 1 is a short greeting) and then **every 5 user prompts** using a **tumbling 5-prompt batch** (`pending_prompts` vs `prev_window_tokens`) + instant call words (`@context`, `check context`, etc.). |
| Auto-injecting huge transcripts wastes context tokens and leaks unintended context | Injects **only top-3 titles + timestamps + brief summaries** with a strict **permission gate** before calling `v1-recall --show`. |

---

## Quick Start (3 Minutes)

### 1. Clone & Install (Python 3.10+, zero external dependencies)
```bash
git clone https://github.com/Manan-learner/v1.git
cd v1
pip install -e .
```

### 2. Index Local Non-Antigravity Sessions & Test Context Search
```bash
# Indexes Claude Code (~/.claude/projects), Codex CLI (~/.codex/sessions), and custom dirs
v1-context --reindex

# Preview top 3 matches without modifying hook state
v1-context --test "STM32H7 FDCAN bootloader timeout"

# Retrieve full turns after choosing a conversation ID
v1-recall --show "<conversation_id>"
```

### 3. Back Up All Providers to Google Drive
* **Windows (with Google Drive for Desktop on `G:\My Drive`):**
  ```powershell
  v1-sync --dry-run
  v1-sync
  v1-status
  ```
* **macOS / Linux / `rclone`:**
  ```bash
  v1-sync --remote gdrive:AI-Agent-Memory
  ```

### 4. Wire Into Your Favorite LLM Harness
* **Gemini Antigravity CLI:** Copy [`examples/antigravity_hooks.json`](examples/antigravity_hooks.json) into `~/.gemini/config/hooks.json`.
* **Claude Code:** Add [`examples/claude_settings_hook.json`](examples/claude_settings_hook.json) into `~/.claude/settings.json` (`UserPromptSubmit`).
* **Cursor / Windsurf / Cline / Codex / Claude Desktop / Any MCP Client:** Register [`examples/mcp_config.json`](examples/mcp_config.json) (`v1-mcp`).

---

## Documentation
* [Architecture & Design Rationale](docs/architecture.md) — Why each threshold, tumbling window, FTS5 fallback, and WAL snapshot works.
* [Getting Started & Cross-Device Restore](docs/getting-started.md) — Daily Windows Task Scheduler / cron setup and new-laptop bootstrap.
* [Multi-LLM Integration Guide](docs/multi-llm-adapters.md) — Exact setup for Antigravity, Claude Code, Codex CLI, MCP clients, and custom scripts.

## Repository Layout
```
src/v1_memory/
  text_utils.py      # Tag stripper, coding stopwords, safe alphanumeric splitter
  providers.py       # Antigravity, Claude Code, Codex CLI, Markdown/JSONL parsers
  index_store.py     # Portable FTS5 index + unindexed parent fallback scanner
  context_finder.py  # Multi-format hook engine (25% cutoff, 5-prompt tumbling window)
  recall.py          # Cross-provider search & transcript turn viewer
  cloud_sync.py      # Non-destructive WAL-safe Drive folder + rclone delta sync
  mcp_server.py      # Zero-dependency MCP stdio JSON-RPC server
examples/            # Ready-to-copy hook & MCP configs
scripts/             # Cross-OS bootstrap + daily scheduler scripts
tests/               # Offline unit & integration test suite
```
