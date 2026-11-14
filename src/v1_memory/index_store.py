"""Portable WAL-safe SQLite + FTS5 + Hybrid-RAG index across Antigravity CLI, Claude Code, Codex CLI, and custom logs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Iterable

from v1_memory.hybrid_rag import (
    compute_semantic_vector,
    cosine_similarity,
    expand_with_concepts,
    group_tokens_by_concept,
    reciprocal_rank_fusion,
)
from v1_memory.providers import iter_provider_conversations
from v1_memory.text_utils import build_summary_line, clean_snippet, tokenize_domain

DEFAULT_STATE_HOME = Path.home() / ".v1-memory"
VECTOR_CACHE_VERSION = "2.1"
HIGH_DF_SUPPRESSION_RATIO = 0.55
MIN_CORPUS_FOR_IDF = 5

LAST_REBUILD_STATS: dict[str, int] = {"total": 0, "updated": 0, "skipped": 0}
LAST_UNINDEXED_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0}


def get_last_rebuild_stats() -> dict[str, int]:
    """Return telemetry from the most recent `rebuild_portable_index()` call."""
    return dict(LAST_REBUILD_STATS)


def get_last_unindexed_cache_stats() -> dict[str, int]:
    """Return telemetry from the most recent Antigravity unindexed cache scan."""
    return dict(LAST_UNINDEXED_CACHE_STATS)


def default_state_home(state_home: Path | None = None) -> Path:
    env_home = os.environ.get("V1_MEMORY_HOME")
    if state_home is not None:
        return state_home
    if env_home:
        return Path(env_home).expanduser()
    return DEFAULT_STATE_HOME


def open_readonly_snapshot(db_path: Path) -> tuple[sqlite3.Connection, str | None]:
    """Open read-only cleanly; if a WAL sidecar exists, copy atomically via SQLite .backup()."""
    wal_path = Path(str(db_path) + "-wal")
    if wal_path.exists():
        tmp = tempfile.NamedTemporaryFile(prefix="v1_mem_", suffix=".db", delete=False)
        tmp.close()
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            dst = sqlite3.connect(tmp.name)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        return sqlite3.connect(tmp.name), tmp.name
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0), None


def ensure_portable_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            title TEXT NOT NULL,
            last_modified_time TEXT NOT NULL,
            step_count INTEGER NOT NULL,
            workspace TEXT NOT NULL,
            transcript_path TEXT NOT NULL,
            summary TEXT NOT NULL,
            user_text TEXT NOT NULL,
            agent_text TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            vector_json TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    if "content_hash" not in existing_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
    if "vector_json" not in existing_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN vector_json TEXT NOT NULL DEFAULT '[]'")

    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
            conversation_id UNINDEXED,
            provider UNINDEXED,
            title,
            user_text,
            agent_text,
            tokenize='porter unicode61'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS antigravity_unindexed_cache (
            conversation_id TEXT PRIMARY KEY,
            last_modified_time TEXT NOT NULL,
            step_count INTEGER NOT NULL,
            title TEXT NOT NULL,
            workspace TEXT NOT NULL,
            summary TEXT NOT NULL,
            combined_text TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            cache_version TEXT NOT NULL DEFAULT ''
        )
        """
    )
    unindexed_cols = {row[1] for row in conn.execute("PRAGMA table_info(antigravity_unindexed_cache)").fetchall()}
    if "cache_version" not in unindexed_cols:
        conn.execute("ALTER TABLE antigravity_unindexed_cache ADD COLUMN cache_version TEXT NOT NULL DEFAULT ''")
    conn.commit()


def rebuild_portable_index(
    index_db: Path,
    providers: Iterable[str] | None = None,
    home: Path | None = None,
    custom_dirs: Iterable[Path] | None = None,
    custom_synonyms: dict[str, list[str]] | None = None,
) -> int:
    """Incrementally index Claude Code, Codex CLI, and custom .jsonl/.md conversations into ~/.v1-memory/index.db.

    Skips re-indexing and re-embedding conversations whose `(last_modified_time, step_count)` have not changed.
    """
    global LAST_REBUILD_STATS
    index_db.parent.mkdir(parents=True, exist_ok=True)
    provider_set = set(providers) if providers is not None else None
    records = iter_provider_conversations(providers=provider_set, home=home, custom_dirs=custom_dirs)
    conn = sqlite3.connect(str(index_db), timeout=10.0)
    try:
        ensure_portable_schema(conn)
        conn.row_factory = sqlite3.Row
        existing_rows = conn.execute(
            "SELECT conversation_id, provider, last_modified_time, step_count, content_hash, vector_json FROM conversations"
        ).fetchall()
        existing_by_id = {r["conversation_id"]: r for r in existing_rows}
        fts_cids = {
            r[0] for r in conn.execute("SELECT DISTINCT conversation_id FROM conversations_fts").fetchall()
        }

        seen_cids: set[str] = set()
        updated_count = 0
        skipped_count = 0

        for rec in records:
            seen_cids.add(rec.conversation_id)
            prev = existing_by_id.get(rec.conversation_id)
            if (
                prev is not None
                and prev["last_modified_time"] == rec.last_modified_time
                and int(prev["step_count"]) == int(rec.step_count)
                and str(prev["content_hash"]).startswith("v21_")
                and prev["vector_json"]
                and prev["vector_json"] != "[]"
                and rec.conversation_id in fts_cids
            ):
                skipped_count += 1
                continue

            user_joined = "\n".join(rec.user_turns)[:80000]
            agent_joined = "\n".join(rec.agent_turns)[:80000]
            first_u = rec.user_turns[0] if rec.user_turns else rec.title
            last_a = rec.agent_turns[-1] if rec.agent_turns else ""
            summary = build_summary_line(first_u, last_a, rec.title)
            combined_payload = f"{rec.title}\n{user_joined}\n{agent_joined}"
            content_hash = "v21_" + hashlib.sha1(combined_payload.encode("utf-8", errors="ignore")).hexdigest()[:12]
            vec = compute_semantic_vector(combined_payload, custom_synonyms=custom_synonyms)
            vector_json = json.dumps([round(v, 6) for v in vec])

            conn.execute(
                """
                INSERT OR REPLACE INTO conversations (
                    conversation_id, provider, title, last_modified_time,
                    step_count, workspace, transcript_path, summary,
                    user_text, agent_text, content_hash, vector_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.conversation_id,
                    rec.provider,
                    rec.title,
                    rec.last_modified_time,
                    rec.step_count,
                    rec.workspace,
                    rec.transcript_path,
                    summary,
                    user_joined,
                    agent_joined,
                    content_hash,
                    vector_json,
                ),
            )
            conn.execute("DELETE FROM conversations_fts WHERE conversation_id = ?", (rec.conversation_id,))
            conn.execute(
                "INSERT INTO conversations_fts (conversation_id, provider, title, user_text, agent_text) VALUES (?, ?, ?, ?, ?)",
                (rec.conversation_id, rec.provider, rec.title, user_joined, agent_joined),
            )
            updated_count += 1

        # Clean up conversations deleted from disk for the active provider scope
        for old_cid, old_row in existing_by_id.items():
            if old_cid not in seen_cids:
                if provider_set is None or old_row["provider"] in provider_set:
                    conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (old_cid,))
                    conn.execute("DELETE FROM conversations_fts WHERE conversation_id = ?", (old_cid,))

        conn.commit()
        LAST_REBUILD_STATS = {
            "total": len(records),
            "updated": updated_count,
            "skipped": skipped_count,
        }
        return len(records)
    finally:
        conn.close()


def _extract_antigravity_transcript_preview(tfile: Path) -> tuple[list[str], list[str]]:
    users: list[str] = []
    agents: list[str] = []
    try:
        size = tfile.stat().st_size
        chunks: list[str] = []
        with open(tfile, "rb") as f:
            chunks.append(f.read(32768).decode("utf-8", errors="replace"))
            if size > 32768:
                f.seek(max(0, size - 32768))
                chunks.append(f.read(32768).decode("utf-8", errors="replace"))
        for line in ("\n".join(chunks)).splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            stype = obj.get("type", "")
            src = obj.get("source", "")
            content = (obj.get("content") or "").strip()
            if not content:
                continue
            if stype == "USER_INPUT" and src == "USER_EXPLICIT":
                users.append(content)
            elif stype == "PLANNER_RESPONSE" and not obj.get("tool_calls"):
                agents.append(content)
    except Exception:
        pass
    return users, agents


def _load_unindexed_cache(index_db: Path | None) -> dict[str, dict]:
    if index_db is None or not index_db.exists():
        return {}
    conn, tmp_path = open_readonly_snapshot(index_db)
    try:
        conn.row_factory = sqlite3.Row
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='antigravity_unindexed_cache'"
            ).fetchall()
        }
        if "antigravity_unindexed_cache" not in tables:
            return {}
        cols = {row[1] for row in conn.execute("PRAGMA table_info(antigravity_unindexed_cache)").fetchall()}
        if "cache_version" not in cols:
            return {}
        rows = conn.execute(
            """
            SELECT conversation_id, last_modified_time, step_count, title, workspace, summary, combined_text, vector_json, cache_version
            FROM antigravity_unindexed_cache
            """
        ).fetchall()
        return {r["conversation_id"]: dict(r) for r in rows}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _save_unindexed_cache_rows(index_db: Path | None, rows: list[tuple]) -> None:
    if index_db is None or not rows:
        return
    try:
        index_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(index_db), timeout=5.0)
        try:
            ensure_portable_schema(conn)
            conn.executemany(
                """
                INSERT OR REPLACE INTO antigravity_unindexed_cache (
                    conversation_id, last_modified_time, step_count,
                    title, workspace, summary, combined_text, vector_json, cache_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def _collect_antigravity_candidates(
    agy_db: Path,
    fts_query: str,
    current_conv_id: str,
    exclude_ids: set[str],
    index_db: Path | None = None,
    custom_synonyms: dict[str, list[str]] | None = None,
) -> tuple[dict[str, dict], int]:
    global LAST_UNINDEXED_CACHE_STATS
    docs: dict[str, dict] = {}
    if not agy_db.exists():
        return docs, 0

    conn, tmp_path = open_readonly_snapshot(agy_db)
    total_count = 0
    try:
        conn.row_factory = sqlite3.Row
        ranked_chunks = conn.execute(
            """
            SELECT
                f.conversation_id,
                cs.title,
                cs.preview,
                cs.last_modified_time,
                cs.step_count,
                cs.workspace_uris,
                snippet(conversations_fts, 13, '', '', ' ... ', 26) AS user_snip,
                snippet(conversations_fts, 14, '', '', ' ... ', 26) AS agent_snip,
                COALESCE(f.title, '') || ' ' || COALESCE(f.user_body, '') || ' ' || COALESCE(f.agent_body, '') AS chunk_text
            FROM conversations_fts f
            JOIN conversation_summaries cs
              ON cs.conversation_id = f.conversation_id
            WHERE conversations_fts MATCH ?
              AND f.conversation_id != ?
              AND COALESCE(cs.parent_conversation_id, '') = ''
            ORDER BY bm25(conversations_fts) ASC
            LIMIT 250
            """,
            (fts_query, current_conv_id),
        ).fetchall()

        unindexed = conn.execute(
            """
            SELECT conversation_id, title, preview, last_modified_time, step_count, workspace_uris
            FROM conversation_summaries
            WHERE conversation_id != ?
              AND COALESCE(parent_conversation_id, '') = ''
              AND conversation_id NOT IN (SELECT DISTINCT conversation_id FROM conversations_fts)
            ORDER BY last_modified_time DESC
            LIMIT 50
            """,
            (current_conv_id,),
        ).fetchall()

        cnt_row = conn.execute(
            """
            SELECT COUNT(*)
            FROM conversation_summaries
            WHERE conversation_id != ?
              AND COALESCE(parent_conversation_id, '') = ''
            """,
            (current_conv_id,),
        ).fetchone()
        total_count = int(cnt_row[0]) if cnt_row else 0
    except sqlite3.OperationalError:
        return docs, 0
    finally:
        conn.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    for r in ranked_chunks:
        cid = r["conversation_id"]
        if cid in exclude_ids:
            continue
        if cid not in docs:
            title = (r["title"] or "").strip() or clean_snippet(r["preview"] or "", 80) or "(Untitled Conversation)"
            docs[cid] = {
                "conversation_id": cid,
                "provider": "antigravity",
                "title": title,
                "last_modified_time": (r["last_modified_time"] or "")[:19],
                "step_count": r["step_count"],
                "workspace": r["workspace_uris"],
                "summary": build_summary_line(r["user_snip"], r["agent_snip"], r["preview"] or title),
                "parts": [title],
                "vector_json": "[]",
            }
        docs[cid]["parts"].append(r["chunk_text"])

    cached_unindexed = _load_unindexed_cache(index_db)
    new_cache_rows: list[tuple] = []
    cache_hits = 0
    cache_misses = 0

    brain_dir = agy_db.parent / "brain"
    deadline = time.perf_counter() + 0.25
    for r in unindexed:
        cid = r["conversation_id"]
        if cid in exclude_ids or cid in docs:
            continue
        mtime = (r["last_modified_time"] or "")[:19]
        step_count = int(r["step_count"] or 0)
        cached = cached_unindexed.get(cid)
        if (
            cached
            and cached.get("cache_version") == VECTOR_CACHE_VERSION
            and cached.get("last_modified_time") == mtime
            and int(cached.get("step_count", -1)) == step_count
        ):
            cache_hits += 1
            docs[cid] = {
                "conversation_id": cid,
                "provider": "antigravity",
                "title": cached["title"],
                "last_modified_time": mtime,
                "step_count": step_count,
                "workspace": cached["workspace"],
                "summary": cached["summary"],
                "parts": [cached["title"], cached["combined_text"]],
                "vector_json": cached.get("vector_json", "[]"),
            }
            continue

        cache_misses += 1
        title = (r["title"] or "").strip()
        preview = (r["preview"] or "").strip()
        display_title = title or clean_snippet(preview, 80) or "(Untitled Conversation)"
        parts = [title, preview]
        users: list[str] = []
        agents: list[str] = []
        if time.perf_counter() < deadline:
            tp = brain_dir / cid / ".system_generated" / "logs" / "transcript.jsonl"
            if tp.exists():
                users, agents = _extract_antigravity_transcript_preview(tp)
                parts.extend(users[:3] + users[-3:] + agents[:2] + agents[-2:])
        first_u = users[0] if users else preview
        last_a = agents[-1] if agents else ""
        summary = build_summary_line(first_u, last_a, preview or display_title)
        clean_parts = [p for p in parts if p]
        combined_text = " ".join(clean_parts)[:120000]
        vec = compute_semantic_vector(combined_text, custom_synonyms=custom_synonyms)
        vec_json = json.dumps([round(v, 6) for v in vec])
        workspace_str = r["workspace_uris"] or ""

        docs[cid] = {
            "conversation_id": cid,
            "provider": "antigravity",
            "title": display_title,
            "last_modified_time": mtime,
            "step_count": step_count,
            "workspace": workspace_str,
            "summary": summary,
            "parts": clean_parts,
            "vector_json": vec_json,
        }
        new_cache_rows.append(
            (cid, mtime, step_count, display_title, workspace_str, summary, combined_text, vec_json, VECTOR_CACHE_VERSION)
        )

    if new_cache_rows:
        _save_unindexed_cache_rows(index_db, new_cache_rows)
    LAST_UNINDEXED_CACHE_STATS = {"hits": cache_hits, "misses": cache_misses}
    return docs, total_count


def _collect_portable_candidates(
    index_db: Path,
    fts_query: str,
    current_conv_id: str,
    exclude_ids: set[str],
    provider_filter: str | None = None,
    include_dense_scan: bool = True,
) -> tuple[dict[str, dict], int]:
    docs: dict[str, dict] = {}
    if not index_db.exists():
        return docs, 0

    conn, tmp_path = open_readonly_snapshot(index_db)
    total_count = 0
    try:
        conn.row_factory = sqlite3.Row
        cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
        vec_expr = "c.vector_json" if "vector_json" in cols else "'[]' AS vector_json"

        base_where = ["c.conversation_id != ?"]
        base_params: list = [current_conv_id]
        if provider_filter and provider_filter not in ("auto", "all"):
            base_where.append("c.provider = ?")
            base_params.append(provider_filter)

        cnt_row = conn.execute(
            f"SELECT COUNT(*) FROM conversations c WHERE {' AND '.join(base_where)}",
            base_params,
        ).fetchone()
        total_count = int(cnt_row[0]) if cnt_row else 0

        where = ["conversations_fts MATCH ?"] + base_where
        params: list = [fts_query] + base_params

        rows = conn.execute(
            f"""
            SELECT
                c.conversation_id,
                c.provider,
                c.title,
                c.last_modified_time,
                c.step_count,
                c.workspace,
                c.summary,
                c.user_text,
                c.agent_text,
                {vec_expr}
            FROM conversations_fts f
            JOIN conversations c
              ON c.conversation_id = f.conversation_id
            WHERE {' AND '.join(where)}
            ORDER BY bm25(conversations_fts) ASC
            LIMIT 100
            """,
            params,
        ).fetchall()

        dense_rows = []
        if include_dense_scan:
            dense_rows = conn.execute(
                f"""
                SELECT
                    c.conversation_id,
                    c.provider,
                    c.title,
                    c.last_modified_time,
                    c.step_count,
                    c.workspace,
                    c.summary,
                    c.user_text,
                    c.agent_text,
                    {vec_expr}
                FROM conversations c
                WHERE {' AND '.join(base_where)}
                ORDER BY c.last_modified_time DESC
                LIMIT 150
                """,
                base_params,
            ).fetchall()
    except sqlite3.OperationalError:
        return docs, 0
    finally:
        conn.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    for idx_rank, r in enumerate(rows, start=1):
        cid = r["conversation_id"]
        if cid in exclude_ids or cid in docs:
            continue
        docs[cid] = {
            "conversation_id": cid,
            "provider": r["provider"],
            "title": r["title"],
            "last_modified_time": (r["last_modified_time"] or "")[:19],
            "step_count": r["step_count"],
            "workspace": r["workspace"],
            "summary": r["summary"],
            "parts": [r["title"], r["user_text"], r["agent_text"]],
            "vector_json": r["vector_json"] or "[]",
            "fts_rank": idx_rank,
        }

    for r in dense_rows:
        cid = r["conversation_id"]
        if cid in exclude_ids or cid in docs:
            continue
        docs[cid] = {
            "conversation_id": cid,
            "provider": r["provider"],
            "title": r["title"],
            "last_modified_time": (r["last_modified_time"] or "")[:19],
            "step_count": r["step_count"],
            "workspace": r["workspace"],
            "summary": r["summary"],
            "parts": [r["title"], r["user_text"], r["agent_text"]],
            "vector_json": r["vector_json"] or "[]",
            "fts_rank": 999,
        }
    return docs, total_count


def _load_configured_synonyms(state_home: Path) -> dict[str, list[str]]:
    for cand in [
        state_home / "context_finder_config.json",
        Path.home() / ".gemini" / "antigravity-cli" / "context_finder_config.json",
    ]:
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                syns = data.get("synonyms")
                if isinstance(syns, dict):
                    return syns
            except Exception:
                pass
    return {}


def _compute_corpus_stats(
    doc_tokens_by_cid: dict[str, set[str]],
    total_db_parent_count: int,
    high_df_ratio: float = HIGH_DF_SUPPRESSION_RATIO,
    min_corpus_size: int = MIN_CORPUS_FOR_IDF,
) -> tuple[int, dict[str, int], set[str]]:
    """Compute corpus document count, per-token document frequencies, and high-DF suppress list."""
    n_corpus = max(len(doc_tokens_by_cid), total_db_parent_count)
    df: dict[str, int] = {}
    for d_toks in doc_tokens_by_cid.values():
        for tok in d_toks:
            df[tok] = df.get(tok, 0) + 1
    if n_corpus < min_corpus_size:
        return n_corpus, df, set()
    high_df = {tok for tok, count in df.items() if (count / n_corpus) >= high_df_ratio}
    return n_corpus, df, high_df


def _bm25_idf(tok: str, n_corpus: int, df: dict[str, int], high_df_tokens: set[str]) -> float:
    """Compute BM25-style IDF weight with N < 5 small-corpus guard and df=0 clamping."""
    if n_corpus < MIN_CORPUS_FOR_IDF:
        return 1.0
    if tok in high_df_tokens:
        return 0.0
    df_eff = max(1, df.get(tok, 0))
    return math.log(1.0 + (n_corpus - df_eff + 0.5) / (df_eff + 0.5))


def find_context_matches(
    prompt_text: str,
    current_conv_id: str = "",
    threshold_pct: float = 40.0,
    min_matched_tokens: int = 2,
    max_matches: int = 3,
    exclude_ids: set[str] | None = None,
    provider_filter: str = "all",
    home: Path | None = None,
    state_home: Path | None = None,
    agy_db_path: Path | None = None,
    index_db_path: Path | None = None,
    hybrid_rag: bool = True,
    semantic_weight: float = 0.45,
    custom_synonyms: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[dict]]:
    """Score past conversations via Hybrid-RAG (Sparse FTS5 BM25 + Dense Semantic/Concept Vectors + RRF)."""
    q_tokens = tokenize_domain(prompt_text)
    if len(q_tokens) < min_matched_tokens:
        return q_tokens, []

    ex = exclude_ids or set()
    user_home = home or Path.home()
    resolved_state = default_state_home(state_home)
    agy_db = agy_db_path or (user_home / ".gemini" / "antigravity-cli" / "conversation_summaries.db")
    port_db = index_db_path or (resolved_state / "index.db")

    syns = custom_synonyms if custom_synonyms is not None else _load_configured_synonyms(resolved_state)
    if hybrid_rag:
        expanded_q_tokens, _ = expand_with_concepts(q_tokens, syns)
    else:
        expanded_q_tokens = list(q_tokens)

    fts_terms = []
    for t in expanded_q_tokens[:60]:
        safe = re.sub(r"[^a-z0-9_]", "", t)
        if len(safe) >= 2:
            fts_terms.append(f'"{safe}"')
    if not fts_terms:
        return q_tokens, []

    fts_query = " OR ".join(dict.fromkeys(fts_terms))

    docs_by_id: dict[str, dict] = {}
    total_db_parent_count = 0
    if provider_filter in ("all", "auto", "antigravity"):
        agy_docs, agy_count = _collect_antigravity_candidates(
            agy_db,
            fts_query,
            current_conv_id,
            ex,
            index_db=port_db,
            custom_synonyms=syns,
        )
        docs_by_id.update(agy_docs)
        total_db_parent_count += agy_count
    if provider_filter != "antigravity":
        port_docs, port_count = _collect_portable_candidates(
            port_db,
            fts_query,
            current_conv_id,
            ex,
            provider_filter=provider_filter,
            include_dense_scan=hybrid_rag,
        )
        docs_by_id.update(port_docs)
        total_db_parent_count += port_count

    if not docs_by_id:
        return q_tokens, []

    combined_by_cid: dict[str, str] = {}
    title_tokens_by_cid: dict[str, set[str]] = {}
    doc_tokens_by_cid: dict[str, set[str]] = {}
    for cid, info in docs_by_id.items():
        combined = " ".join(p for p in info["parts"] if p)[:120000]
        title_tokens = set(tokenize_domain(info["title"]))
        doc_tokens = set(tokenize_domain(combined)) | title_tokens
        if not doc_tokens:
            continue
        combined_by_cid[cid] = combined
        title_tokens_by_cid[cid] = title_tokens
        doc_tokens_by_cid[cid] = doc_tokens

    if not doc_tokens_by_cid:
        return q_tokens, []

    n_corpus, df, high_df_tokens = _compute_corpus_stats(doc_tokens_by_cid, total_db_parent_count)
    q_active = [t for t in q_tokens if t not in high_df_tokens]
    q_active_set = set(q_active)
    if len(q_active_set) < min_matched_tokens:
        return q_tokens, []

    q_idf_sum = sum(_bm25_idf(t, n_corpus, df, high_df_tokens) for t in q_active_set)
    q_concept_hits = group_tokens_by_concept(q_active, syns) if hybrid_rag else {}
    if hybrid_rag:
        q_vec_input = q_active if (high_df_tokens & set(q_tokens)) else prompt_text
        q_vec = compute_semantic_vector(q_vec_input, custom_synonyms=syns)
    else:
        q_vec = []

    w_sem = max(0.0, min(1.0, float(semantic_weight)))
    w_con = 1.0 - w_sem

    candidate_evals: dict[str, dict] = {}
    sparse_rank_tuples: list[tuple[tuple, str]] = []
    dense_rank_tuples: list[tuple[tuple, str]] = []

    for cid, doc_tokens in doc_tokens_by_cid.items():
        info = docs_by_id[cid]
        doc_active = doc_tokens - high_df_tokens
        if not doc_active:
            continue

        exact_matched = sorted(q_active_set & doc_active)
        title_hits = len(q_active_set & (title_tokens_by_cid[cid] - high_df_tokens))
        doc_idf_sum = sum(_bm25_idf(t, n_corpus, df, high_df_tokens) for t in doc_active)
        denom_weight = min(q_idf_sum, doc_idf_sum)
        exact_weight = sum(_bm25_idf(t, n_corpus, df, high_df_tokens) for t in exact_matched)
        exact_pct = round(min(100.0, (exact_weight / denom_weight) * 100.0), 1) if denom_weight > 1e-9 else 0.0

        cos_sim = 0.0
        qualifying_concepts: list[str] = []
        covered_q_tokens = set(exact_matched)
        concept_doc_hits: list[str] = []

        if hybrid_rag:
            doc_vec: list[float] = []
            raw_vjson = info.get("vector_json") or "[]"
            if raw_vjson and raw_vjson != "[]":
                try:
                    parsed_v = json.loads(raw_vjson)
                    if isinstance(parsed_v, list) and parsed_v:
                        doc_vec = [float(x) for x in parsed_v]
                except Exception:
                    doc_vec = []
            if not doc_vec:
                doc_vec_input = sorted(doc_active) if (high_df_tokens & doc_tokens) else combined_by_cid[cid]
                doc_vec = compute_semantic_vector(doc_vec_input, custom_synonyms=syns)

            cos_sim = max(0.0, cosine_similarity(q_vec, doc_vec))
            doc_concept_hits = group_tokens_by_concept(doc_active, syns)
            doc_hit_pool: set[str] = set()
            for concept_id, q_c_hits in q_concept_hits.items():
                d_c_hits = doc_concept_hits.get(concept_id, [])
                if len(q_c_hits) >= 2 and len(d_c_hits) >= 2:
                    qualifying_concepts.append(concept_id)
                    covered_q_tokens.update(q_c_hits)
                    doc_hit_pool.update(t for t in d_c_hits if t not in q_active_set)
            concept_doc_hits = sorted(doc_hit_pool)

        if len(covered_q_tokens) < min_matched_tokens:
            continue

        covered_weight = sum(_bm25_idf(t, n_corpus, df, high_df_tokens) for t in covered_q_tokens)
        concept_coverage_pct = round(min(100.0, (covered_weight / denom_weight) * 100.0), 1) if denom_weight > 1e-9 else 0.0

        if hybrid_rag and qualifying_concepts:
            concept_and_cosine_pct = round(min(100.0, w_con * concept_coverage_pct + w_sem * (cos_sim * 100.0)), 1)
            hybrid_pct = round(max(exact_pct, concept_and_cosine_pct), 1)
        else:
            concept_and_cosine_pct = 0.0
            hybrid_pct = exact_pct

        if hybrid_pct < threshold_pct:
            continue

        matched_out = list(exact_matched)
        if qualifying_concepts and (len(exact_matched) < len(q_active_set) or concept_and_cosine_pct >= exact_pct):
            for sc in qualifying_concepts:
                if sc not in matched_out:
                    matched_out.append(sc)
            for dh in concept_doc_hits[:5]:
                if dh not in matched_out:
                    matched_out.append(dh)

        candidate_evals[cid] = {
            "info": info,
            "hybrid_pct": hybrid_pct,
            "exact_pct": exact_pct,
            "semantic_score": round(cos_sim, 4),
            "matched_tokens": matched_out,
            "title_hits": title_hits,
            "retrieval_mode": "hybrid_rrf" if hybrid_rag else "sparse",
        }
        sparse_rank_tuples.append(
            ((exact_pct, len(exact_matched), len(concept_doc_hits), title_hits, info["last_modified_time"]), cid)
        )
        if hybrid_rag:
            dense_rank_tuples.append(
                ((cos_sim, len(qualifying_concepts), concept_coverage_pct, info["last_modified_time"]), cid)
            )

    if not candidate_evals:
        return q_tokens, []

    sparse_rank_tuples.sort(key=lambda item: item[0], reverse=True)
    dense_rank_tuples.sort(key=lambda item: item[0], reverse=True)
    sparse_ids = [cid for _, cid in sparse_rank_tuples]
    dense_ids = [cid for _, cid in dense_rank_tuples]
    rrf_scores = reciprocal_rank_fusion(sparse_ids, dense_ids, k=60) if hybrid_rag else {
        cid: 1.0 / (60 + r) for r, cid in enumerate(sparse_ids, start=1)
    }

    scored: list[tuple[tuple, dict]] = []
    for cid, ev in candidate_evals.items():
        info = ev["info"]
        pct = ev["hybrid_pct"]
        title_hits = ev["title_hits"]
        rrf_val = round(rrf_scores.get(cid, 0.0), 6)
        rank_score = pct + min(25.0, title_hits * 10.0)
        obj = {
            "conversation_id": cid,
            "provider": info["provider"],
            "title": info["title"],
            "last_modified_time": info["last_modified_time"],
            "step_count": info["step_count"],
            "workspace": info["workspace"],
            "match_pct": pct,
            "matched_tokens": ev["matched_tokens"],
            "summary": info["summary"],
            "retrieval_mode": ev["retrieval_mode"],
            "semantic_score": ev["semantic_score"],
            "rrf_score": rrf_val,
        }
        scored.append(
            (
                (rrf_val, rank_score, title_hits, pct, ev["semantic_score"], info["last_modified_time"]),
                obj,
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return q_tokens, [item[1] for item in scored[:max_matches]]
