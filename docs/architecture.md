# Architecture & Why It Works

## 1. Proactive Context Matching Pipeline

```
Incoming prompt (Antigravity PreInvocation / Claude UserPromptSubmit / MCP / CLI)
   │
   ├─► Clean agent XML wrapper tags (<ADDITIONAL_METADATA>, <system-reminder>, etc.)
   ├─► Check explicit call word (@context, #context, check context, recall check, find context)
   │     ├─ If bare call word -> pull recent 1-3 user turns from transcriptPath / recent_prompts
   │     └─ Manual lookup sets exclude_ids=set() so older dismissed topics can be re-found
   ├─► Otherwise gate automatic triggers:
   │     ├─ Turn 1 (initialNumSteps <= 4; defers to Turn 2 if Turn 1 is a short greeting)
   │     └─ Every 5th user prompt (tumbling batch: pending_prompts vs prev_window_tokens)
   │
   ▼
FTS5 Search + Fallback (~115–250 ms, $0 LLM calls)
   ├─► Native Antigravity DB: conversations_fts (LIMIT 250 chunks) + unindexed parent fallback
   ├─► Portable multi-LLM DB (~/.v1-memory/index.db): Claude Code, Codex CLI, custom JSONL/MD
   └─► Score: len(matched) / min(len(q_tokens), len(doc_tokens)) >= 25% & matched >= 2
   │
   ▼
Top-3 Summary + Permission Gate
   └─► Assistant prints top 3 titles, dates, % overlap, matched tokens, brief summary,
       and MUST ask permission before invoking v1-recall --show <id>
```

### Why `25%` + Curated Coding Stopwords Works Better Than `10%`
With `denom = min(len(query_tokens), len(doc_tokens))`, past conversations usually have 100–500+ unique tokens while user prompts have 3–15 domain tokens. That makes `denom` effectively equal to `len(query_tokens)`.
- At **`10%`**, a long 18–20 token prompt only needed 2 incidental words (like `script` and `yesterday`) to hit `10%`.
- Adding generic software/time stopwords (`bug`, `code`, `script`, `python`, `system`, `repo`, `yesterday`) and keeping mixed chip/board codes whole (`stm32` stays `stm32` instead of splitting out noisy `'32'`) removes accidental overlaps.
- Raising the default threshold to **`25%`** preserves all specific queries (`2/5 = 40%`, `2/8 = 25%`, `3/10 = 30%`) while rejecting stray coincidental overlaps.

### Why a Tumbling 5-Prompt Window Beats Whole-Convo Re-Checks
Evaluating every single prompt interrupts active work. Evaluating only prompt `#5` misses a topic shift that started on prompt `#3` followed by short follow-ups (`"why timeout?"`, `"try again"`).
The **tumbling window** accumulates turns in `pending_prompts` between checks:
1. On prompt `#5`, `detect_batch_pivot()` compares only the new 5-prompt batch (and trailing sub-shifts inside that batch) against `prev_window_tokens`.
2. It rotates `prev_window_tokens` to the new baseline and clears `pending_prompts = []`.
This prevents re-searching old turns and focuses strictly on newly introduced work.

---

## 2. WAL-Safe Non-Destructive Cloud Memory Sync

### Atomic Snapshots First
Agent runtimes keep SQLite databases open in WAL (`-wal` / `-shm`) mode. Copying a `.db` file mid-write skips uncommitted pages in `-wal`. `sqlite_safe_snapshot()` opens the source DB read-only (`?mode=ro`) and runs SQLite's online `.backup()` API into a temporary staging directory before syncing.

### Dual Storage Backend
- **Mounted Drive Folder (`drive_folder`):** Directly copies incremental deltas using Python's `shutil.copy2` whenever `mtime` or `size` changes. Never blocked by application-control policies targeting unsigned third-party CLI binaries.
- **`rclone` (`gdrive:AI-Agent-Memory`):** Ideal on headless Linux servers, WSL, or macOS machines without Google Drive for Desktop.
- **Strictly Non-Destructive:** Neither backend runs mirror-delete (`--delete`). Historical conversations removed or rotated locally remain safely preserved in Google Drive.
