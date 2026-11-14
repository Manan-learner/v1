# Architecture & Why It Works

## 1. Proactive Hybrid-RAG Context Matching Pipeline (`v2.2`)

```
Incoming prompt (Antigravity PreInvocation / Claude UserPromptSubmit / MCP / CLI)
   │
   ├─► Ensure UTF-8 stdio (cp1252-safe) & clean XML wrapper tags (<ADDITIONAL_METADATA>, <system-reminder>)
   ├─► Word-boundary call-word check (@context, #context, check context, recall check, find context)
   │     ├─ Strictly bounded regex (?<![a-z0-9_])...(?![a-z0-9_-]) avoids @context7 / checkpoint false triggers
   │     ├─ If bare call word -> bounded tail read (131 KB) from transcriptPath / recent_prompts
   │     └─ Manual lookup sets exclude_ids=set() and locks start_checked=True
   ├─► Otherwise gate automatic triggers (before short-prompt filtering):
   │     ├─ Turn 1 (initialNumSteps <= 4; defers to Turn 2 if Turn 1 is a short greeting; keeps 2-token tech prompts)
   │     └─ Every 5th user prompt (tumbling batch: pending_prompts vs prev_window_tokens, even on short Turn 5)
   │
   ▼
Hybrid-RAG Search + Incremental Cache (< 500 ms at 1,000 convs, $0 LLM calls)
   ├─► Concept Expansion: 8 canonical technical clusters (rf_telemetry, mcu_embedded, uav_mavlink,
   │   rag_search, k8s_containers, auth_security, database_storage, cloud_backup_sync) + user "synonyms"
   ├─► Sparse Leg: Core query + Antigravity FTS5 (LIMIT 350, 6-chunk/conv cap, anti-crowding) + Portable index.db
   │   └─ fts5_terms_for_tokens() splits hyphen/dot tokens cleanly (v1-context, sub-ghz, node.js)
   ├─► Dense / Semantic Leg: 2-tier 256-dim L2-normalized semantic vector
   │     ├─ Tier 1 (optional): fastembed (BAAI/bge-small-en-v1.5) when V1_USE_FASTEMBED=1
   │     └─ Tier 2 (default zero-dep, exact 1.000 parity with standalone hook): Deterministic sorted-token
   │        CRC32 feature-hashed L2 vector: tokens (1.0x), stems (0.5x), >=2-stem qualified concepts
   │        (1.6x, 4 projections), and sorted char 4-grams (0.5x) with 1/sqrt(N) group normalization
   ├─► Incremental Archive & FTS Caches (keyed by compute_rag_signature()):
   │     ├─ conversations table caches content_hash, tokens_json, vector_json, cache_sig
   │     ├─ antigravity_unindexed_cache (bounded 500 newest; skips incomplete deadline-expired scans)
   │     └─ antigravity_fts_cache (bounded 1,000 newest; eliminates per-query re-tokenization)
   └─► Scoring & Reciprocal Rank Fusion (RRF, k=60):
         ├─ Short-doc denominator protection + >=2 distinct-stem concept gate + dense fallback (>= 0.42)
         └─ Final ranking by rank_score (hybrid_pct + title bonus) and RRF score
   │
   ▼
Top-3 Summary + Permission Gate
   └─► Assistant prints top 3 titles, dates, % overlap, matched tokens/concepts, brief summary,
       and MUST ask permission before invoking v1-recall --show <id>
```

### Why v2.2 Hybrid-RAG Beats Keyword-Only FTS5
Pure lexical matching fails when a user describes a problem with different terminology than the original session (for example, asking about `"wireless radio rssi link dropouts"` when the earlier conversation used `"915MHz SX1262 transceiver packet framing"`).
1. **Canonical Concept & Synonym Expansion (`expand_with_concepts`)**: Maps domain vocabulary into 8 clean canonical concept clusters (`concept:rf_telemetry`, `concept:mcu_embedded`, `concept:uav_mavlink`, `concept:rag_search`, `concept:k8s_containers`, `concept:auth_security`, `concept:database_storage`, `concept:cloud_backup_sync`, plus user-defined `"synonyms"` in `context_finder_config.json`). Overlapping noise terms (`token`, `session`, `frame`, `retry`, `timeout`, `signal`, `packet`, `table`, `index`) are excluded from clusters so CSS design tokens or Playwright timeouts never hijack Auth or Embedded matches.
2. **Deterministic, Unbiased 2-Tier 256-Dim L2 Semantic Vectors (`compute_semantic_vector`)**:
   - **Zero-Dependency Default**: Deduplicates and sorts tokens before projecting (`PYTHONHASHSEED`-invariant) via deterministic `zlib.crc32` across domain tokens (`1.0x`), plural-normalized stems (`0.5x`), `>= 2`-stem qualified canonical concept IDs (`1.6x`, 4 projections), and sorted character 4-grams (`0.5x`) with `1/sqrt(N)` group normalization. Both `v1_memory.hybrid_rag` and standalone `context_match_hook.py` produce exact `1.000` cosine parity.
   - **Optional Neural Tier**: If `pip install -e ".[rag]"` (`fastembed>=0.3.0`) is installed and `V1_USE_FASTEMBED=1` is set, blends `BAAI/bge-small-en-v1.5` neural embeddings into the 256-dim L2 vector.
3. **Pure-Semantic Dense Fallback & Reciprocal Rank Fusion (`reciprocal_rank_fusion`)**: Combines sparse BM25-IDF ranking and dense cosine ranking via `1 / (k + rank)` (`k = 60`), topped by `rank_score = hybrid_pct + min(25.0, title_hits * 10.0)`. When a query falls outside the 8 built-in clusters (e.g., `"pytorch transformers optimizers dataloaders gradients"`), the dense fallback (`cos_sim >= 0.42` with `>= 2` active query tokens) still surfaces strong semantic matches while keeping generic coding prompts (`"can you help me fix a bug in this python script I wrote yesterday"`, `"fixed issues on tuesday"`) at `0` matches.

### Why Incremental Archive & FTS Caching Eliminates Disk Bottlenecks
As conversation history grows past 1,000+ sessions across providers, re-parsing `.jsonl` transcripts or re-tokenizing 120 KB chunks on every hook invocation blows past hook timeouts:
- **Pre-Tokenized `rebuild_portable_index()`**: Stores `content_hash`, `tokens_json`, `vector_json`, and `cache_sig` alongside `(last_modified_time, step_count)` in `conversations` using bounded `80 KB` head+tail transcript slices. Unchanged sessions are skipped cleanly (`get_last_rebuild_stats()`), and candidate queries read lightweight cached metadata without loading raw transcript bodies.
- **Cached Antigravity Unindexed & FTS Digests (`antigravity_unindexed_cache`, `antigravity_fts_cache`)**: Caches pre-tokenized lists, summaries, and 256-dim vectors keyed by `compute_rag_signature()` (`v2.2` + `STOPWORDS` + clusters + `synonyms`), keeping newest entries (500 unindexed / 1,000 FTS) and refusing to persist degraded `title + preview` rows if a cold transcript scan hits its time deadline.

### Why `40%` + Hybrid-RAG + v2.2 Anti-Noise Scoring Works
- **Short-Doc Denominator Protection (`v2.2`):** `denom_weight = max(0.75 * q_idf_sum, min(q_idf_sum, doc_idf_sum))` ensures short 2-word stubs (like `"docker nginx"`) cannot invert the denominator to `100%` and outrank rich 10-of-12 matches (`"kubernetes helm ingress ..."`).
- **Balanced Stoplist & Safe FTS5 Splitting (`v2.2`):** Conversational noise (`bug`, `fixed`, `issues`, `tuesday`..`sunday`, `hi`, `hey`, `hello`, `problem`, `expected`, `located`, `demonstration`, `verification`) and contraction suffixes (`we're`, `you'll`, `don't`) are filtered, while real technical tokens (`table`, `index`, `cli`, `json`, `branch`, `sqlite`, `git`, `docker`) stay searchable and hyphenated/dotted tokens (`v1-context`, `sub-ghz`, `node.js`) split into FTS5-safe terms via `fts5_terms_for_tokens()`.
- **Corpus-Aware High-DF Suppression & BM25-IDF Weighting (`v2.2`):** When `N_corpus >= 5`, tokens appearing in `>= 55%` of the corpus are dynamically suppressed (`IDF = 0`) using a bounded sample denominator (`eff_denom = n_corpus if n_corpus < 50 else min(n_corpus, max(sample_n, 50))`) so suppression scales cleanly past `N = 1,000` without self-suppressing rare query terms on small corpora (`N = 5..15`).
- **Lemmatized Concept Cluster Gating (`v2.2`):** `qualifying_shared_concepts()` requires `>= 2` **distinct stemmed** (`stem_token()`) hits in BOTH active query and active document, eliminating singular/plural double-counting (`token`/`tokens`) and generic collision false positives.

### Why a Tumbling 5-Prompt Window Beats Whole-Convo Re-Checks
Evaluating every single prompt interrupts active work. Evaluating only prompt `#5` misses a topic shift that started on prompt `#3` followed by short follow-ups (`"why timeout?"`, `"go ahead"`).
The **tumbling window** accumulates turns in `pending_prompts` between checks:
1. Even if Turn `#5` is short (`"go ahead"`), `is_nth_prompt` runs before short-prompt filtering so `detect_batch_pivot()` always evaluates the 5-turn batch against `prev_window_tokens` using stem and concept alignment (returning `False` on same-topic continuations and isolating mid-batch trailing shifts via `split_idx`).
2. It rotates `prev_window_tokens` to the new baseline and clears `pending_prompts = []`.
This prevents re-searching old turns and focuses strictly on newly introduced work, while `save_seen_state()` persists state atomically via `tempfile` + `os.replace()`.

---

## 2. WAL-Safe Non-Destructive Cloud Memory Sync & Path Security

### URI-Safe Read-Only Opens, Temp-File Safety & Atomic Snapshots
Agent runtimes keep SQLite databases open in WAL (`-wal` / `-shm`) mode:
- **Queries (`open_readonly_snapshot`):** First opens read-only via `Path.resolve().as_uri() + "?mode=ro"` (`PRAGMA query_only = ON`) and reuses a single connection per query. Only when locked does it fall back to SQLite's online `.backup()`, guaranteeing temporary snapshot unlinking in `except:` if a database is corrupt or unreadable.
- **Backup (`sqlite_safe_snapshot` & `stage_wal_snapshots`):** Stages transactionally consistent WAL backups keyed by full relative path (`<remote_sub>/<rel_posix>`) so `.db` + `.db-wal` pairs never snapshot twice and same-basename files in separate directories are never excluded.
- **Path-Traversal Guard (`recall.py` / `recall_query.py`):** Strictly validates conversation IDs (`..`, `/`, `\`, `:`, `\x00` rejected) and verifies `.resolve().is_relative_to(brain_dir.resolve())` before reading any transcript.

### Dual Storage Backend
- **Mounted Drive Folder (`drive_folder`):** Directly copies incremental deltas using Python's `shutil.copy2` whenever `mtime` or `size` changes. Never blocked by application-control policies targeting unsigned third-party CLI binaries.
- **`rclone` (`gdrive:AI-Agent-Memory`):** Ideal on headless Linux servers, WSL, or macOS machines without Google Drive for Desktop.
- **Strictly Non-Destructive:** Neither backend runs mirror-delete (`--delete`). Historical conversations removed or rotated locally remain safely preserved in Google Drive.
