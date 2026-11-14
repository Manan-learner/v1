# Architecture & Why It Works

## 1. Proactive Hybrid-RAG Context Matching Pipeline (`v2`)

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
Hybrid-RAG Search + Incremental Cache (~50–200 ms, $0 LLM calls)
   ├─► Concept Expansion: Built-in technical clusters (rf_telemetry, can_embedded, k8s_containers,
   │   auth_security, database_storage, cloud_backup_sync) + user "synonyms" expand query recall
   ├─► Sparse Leg: Native Antigravity FTS5 (LIMIT 250 chunks) + Portable ~/.v1-memory/index.db FTS5
   ├─► Dense / Semantic Leg: 2-tier 256-dim L2-normalized semantic vector
   │     ├─ Tier 1 (optional): fastembed (BAAI/bge-small-en-v1.5) when V1_USE_FASTEMBED=1
   │     └─ Tier 2 (default zero-dep): Deterministic CRC32 feature-hashed L2 vector combining
   │        domain tokens (1.0x), canonical concept IDs (2.5x), and char 4-grams (0.5x)
   ├─► Incremental Archive Cache:
   │     ├─ conversations table caches content_hash + vector_json (skips unchanged mtime + step_count)
   │     └─ antigravity_unindexed_cache table caches unindexed parent digests + vectors (0 disk re-reads)
   └─► Reciprocal Rank Fusion (RRF): Fuses sparse + dense ranks & scores hybrid_pct >= 40%
   │
   ▼
Top-3 Summary + Permission Gate
   └─► Assistant prints top 3 titles, dates, % overlap, matched tokens/concepts, brief summary,
       and MUST ask permission before invoking v1-recall --show <id>
```

### Why v2 Hybrid-RAG Beats Keyword-Only FTS5
Pure lexical matching fails when a user describes a problem with different terminology than the original session (for example, asking about `"wireless radio rssi link dropouts"` when the earlier conversation used `"915MHz SX1262 transceiver packet framing"`).
1. **Concept & Synonym Expansion (`expand_with_concepts`)**: Maps domain vocabulary into canonical concept IDs (`concept:rf_telemetry`, `concept:can_embedded`, `concept:k8s_containers`, `concept:auth_security`, `concept:database_storage`, `concept:cloud_backup_sync`, plus user-defined `"synonyms"` in `context_finder_config.json`).
2. **2-Tier 256-Dim L2 Semantic Vectors (`compute_semantic_vector`)**:
   - **Zero-Dependency Default**: Uses deterministic CRC32 feature hashing with signed projections across domain tokens (`1.0x`), multi-hash canonical concept IDs (`2.5x`), and character 4-grams (`0.5x`) for sub-word/part-number resilience (`sx1262` vs `sx1276`, `stm32h7` vs `stm32f4`).
   - **Optional Neural Tier**: If `pip install -e ".[rag]"` (`fastembed>=0.3.0`) is installed and `V1_USE_FASTEMBED=1` is set, blends `BAAI/bge-small-en-v1.5` neural embeddings into the 256-dim L2 vector.
3. **Reciprocal Rank Fusion (`reciprocal_rank_fusion`)**: Combines sparse token-overlap ranking and dense cosine/concept ranking via `1 / (k + rank)` (`k = 60`), surfacing both exact part-number hits and zero-word-overlap conceptual matches at `>= 40%` while keeping generic coding prompts (`"can you help me fix a bug in this python script I wrote yesterday"`) at `0` matches.

### Why Incremental Archive Caching Eliminates Disk Bottlenecks
As conversation history grows across providers, re-parsing `.jsonl` transcripts or re-computing vectors on every run wastes I/O:
- **Incremental `rebuild_portable_index()`**: Stores `content_hash` and `vector_json` alongside `(last_modified_time, step_count)` in `conversations`. Unchanged sessions are skipped in sub-millisecond time (`get_last_rebuild_stats()` tracks `total`, `updated`, and `skipped`).
- **Cached Unindexed Antigravity Parents (`antigravity_unindexed_cache`)**: Some parent conversations in Antigravity's `conversation_summaries.db` are absent from `conversations_fts`. On the first encounter, `index_store.py` extracts a `32KB + 32KB` head/tail preview from `transcript.jsonl`, computes the 256-dim semantic vector, and persists the row in `antigravity_unindexed_cache`. Subsequent runs match `(last_modified_time, step_count)` and reuse the cached digest and vector with **zero disk reads** (`get_last_unindexed_cache_stats()` reports `hits` vs `misses`).

### Why `40%` + Hybrid-RAG + v2.1 Anti-Noise Scoring Works
With `denom_weight = min(sum_idf(q_active), sum_idf(doc_active))`, past conversations usually have 100–500+ unique tokens while user prompts have 3–15 domain tokens. That makes `denom_weight` effectively equal to the query's total IDF weight.
- **Expanded Stoplist & Safe Tokenization (`v2.1`):** Generic software/workflow/diagnostic boilerplate (`bug`, `code`, `script`, `python`, `problem`, `expected`, `located`, `concern`, `demonstration`, `verification`, `implementation`, `table`, `index`), contraction fragments (`don`, `won`, `didn`), single-char dotted abbreviations (`e.g.`, `i.e.`), and bare 1–2 digit numbers (`10`, `12`) are dropped before matching, while mixed codes stay whole (`stm32` stays `stm32`; `915mhz` -> `915mhz`, `915`, `mhz`).
- **Dynamic Corpus High-DF Suppression & BM25-IDF Weighting (`v2.1`):** When `N_corpus >= 5`, any term appearing in `>= 55%` of corpus conversations is dynamically suppressed (`IDF = 0`), and all remaining active tokens are weighted by `ln(1 + (N - df + 0.5) / (df + 0.5))` so rare anchors (`sx1262`, `fdcan`, `oauth2`) dominate over common words.
- **Strict Concept Cluster Gating (`v2.1`):** Concept clusters (`rf_telemetry`, `can_embedded`, `k8s_containers`, `auth_security`, `database_storage`, `rag_search`, `cloud_backup_sync`) qualify only when both the active query and active document share `>= 2` tokens in the same cluster (never on a single token). Genuine matches score in the `50%–100%` band while conversational boilerplate stays silent.

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
