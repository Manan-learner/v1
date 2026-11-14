# Getting Started & Cross-Device Restore

## Prerequisites
- **Python 3.10+** (standard library only — zero pip dependencies)
- **Cloud destination (pick one):**
  1. **Windows / macOS Desktop:** Google Drive for Desktop mounted on `G:\My Drive` (or `H:\My Drive` / `~/Google Drive`), OR
  2. **Linux / macOS / Windows (`rclone`):** `rclone` configured with a Google Drive remote (default: `gdrive:AI-Agent-Memory`).

---

## Fresh Install

```bash
git clone https://github.com/Manan-learner/v1.git
cd v1
```

### Windows (PowerShell)
```powershell
.\scripts\bootstrap.ps1
```
This installs `v1-context`, `v1-recall`, `v1-sync`, `v1-status`, and `v1-mcp`, initializes `~/.v1-memory/context_finder_config.json`, indexes local Claude/Codex sessions, and registers a daily 9:00 PM Windows Scheduled Task (`V1AgentMemoryDailySync`) with `-StartWhenAvailable` catch-up on wake.

### macOS / Linux
```bash
bash ./scripts/bootstrap.sh
```
Installs the CLI entrypoints and adds a daily `21:00` cron entry for `v1-sync`.

---

## Tuning Context Finder (`~/.v1-memory/context_finder_config.json`)

Copy or customize [`examples/context_finder_config.json`](../examples/context_finder_config.json):

```json
{
  "enabled": true,
  "threshold_pct": 40.0,
  "min_matched_tokens": 2,
  "max_matches": 3,
  "start_of_convo_only": true,
  "max_initial_steps_for_start": 4,
  "auto_pivot_every_n_prompts": 5,
  "call_words": ["@context", "#context", "check context", "recall check", "find context"],
  "hybrid_rag": true,
  "semantic_weight": 0.45,
  "synonyms": {
    "custom_topic": ["alias1", "alias2"]
  }
}
```

- **`"hybrid_rag": true`** (default): Enables v2 Hybrid-RAG combining Sparse FTS5 BM25, built-in technical concept clusters + custom `"synonyms"`, 2-tier 256-dim L2 semantic vectors, and Reciprocal Rank Fusion (RRF). Set to `false` for strict exact-token sparse matching only.
- **`"semantic_weight": 0.45`**: Controls the balance between dense semantic cosine similarity (`0.45`) and concept coverage (`0.55`) when scoring conceptual/synonym matches.
- **`"synonyms": {}`**: Add project-specific domain clusters (`"cluster_name": ["term1", "term2", ...]`) so related project terms match even when prompts use different words.
- **Optional Neural Embeddings (`fastembed`)**: The default 256-dim stdlib feature-hashed vector requires zero dependencies. To blend `BAAI/bge-small-en-v1.5` neural embeddings, install `pip install -e ".[rag]"` and set `V1_USE_FASTEMBED=1`.
- **Incremental Archive Caching**: Both `v1-context --reindex` and unindexed Antigravity parent transcript scans automatically cache digests and 256-dim vectors in `~/.v1-memory/index.db` keyed by `(last_modified_time, step_count)`, avoiding repeated disk reads on unchanged sessions.
- Set `"auto_pivot_every_n_prompts": 0` to disable periodic mid-chat checks and rely solely on Turn 1 + explicit call words.
- Increase `"threshold_pct"` to `30.0`–`35.0` if you prefer stricter triggering.

---

## Moving to a New Laptop
1. Sign into Google Drive on the new machine (or run `rclone config` for `gdrive:`).
2. Clone this repo and run:
   - Windows: `.\scripts\bootstrap.ps1 -Restore`
   - macOS / Linux: `bash ./scripts/bootstrap.sh gdrive:AI-Agent-Memory --restore`
3. Verify restored transcripts and schedule:
   ```bash
   v1-status
   v1-recall "your previous project keyword"
   ```
