"""Portable WAL-safe SQLite + FTS5 + Hybrid-RAG index across Antigravity CLI, Claude Code, Codex CLI, and custom logs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence

from v1_memory.hybrid_rag import (
    VECTOR_CACHE_VERSION,
    build_concept_lookup,
    compute_rag_signature,
    compute_semantic_vector,
    cosine_similarity,
    expand_with_concepts,
    group_tokens_by_concept,
    qualifying_shared_concepts,
    reciprocal_rank_fusion,
)
from v1_memory.providers import iter_provider_conversations
from v1_memory.text_utils import (
    build_summary_line,
    clean_snippet,
    fts5_terms_for_tokens,
    stem_token,
    tokenize_domain,
)

DEFAULT_STATE_HOME = Path.home() / ".v1-memory"
HIGH_DF_SUPPRESSION_RATIO = 0.55
MIN_CORPUS_FOR_IDF = 5
DENSE_COSINE_FLOOR = 0.42

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
    """Open read-only cleanly via URI; fall back to an atomic temp .backup() snapshot if WAL is locked.

    Guarantees cleanup of temporary snapshot files if connection or backup fails.
    """
    resolved = Path(db_path).resolve()
    uri = f"{resolved.as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            conn.execute("PRAGMA query_only = ON")
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
            return conn, None
        except Exception:
            conn.close()
            raise
    except sqlite3.Error:
        tmp = tempfile.NamedTemporaryFile(prefix="v1_mem_", suffix=".db", delete=False)
        tmp_name = tmp.name
        tmp.close()
        src: sqlite3.Connection | None = None
        dst: sqlite3.Connection | None = None
        try:
            src = sqlite3.connect(uri, uri=True, timeout=5.0)
            dst = sqlite3.connect(tmp_name)
            src.backup(dst)
            dst.close()
            dst = None
            src.close()
            src = None
            snap_conn = sqlite3.connect(tmp_name)
            snap_conn.execute("PRAGMA query_only = ON")
            return snap_conn, tmp_name
        except Exception:
            if dst is not None:
                try:
                    dst.close()
                except Exception:
                    pass
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
            raise


def _bounded_head_tail_text(text: str, max_chars: int = 80000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


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
            vector_json TEXT NOT NULL DEFAULT '[]',
            tokens_json TEXT NOT NULL DEFAULT '[]',
            cache_sig TEXT NOT NULL DEFAULT ''
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    if "content_hash" not in existing_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
    if "vector_json" not in existing_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN vector_json TEXT NOT NULL DEFAULT '[]'")
    if "tokens_json" not in existing_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN tokens_json TEXT NOT NULL DEFAULT '[]'")
    if "cache_sig" not in existing_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN cache_sig TEXT NOT NULL DEFAULT ''")

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
            cache_version TEXT NOT NULL DEFAULT '',
            tokens_json TEXT NOT NULL DEFAULT '[]',
            cache_sig TEXT NOT NULL DEFAULT ''
        )
        """
    )
    unindexed_cols = {row[1] for row in conn.execute("PRAGMA table_info(antigravity_unindexed_cache)").fetchall()}
    if "cache_version" not in unindexed_cols:
        conn.execute("ALTER TABLE antigravity_unindexed_cache ADD COLUMN cache_version TEXT NOT NULL DEFAULT ''")
    if "tokens_json" not in unindexed_cols:
        conn.execute("ALTER TABLE antigravity_unindexed_cache ADD COLUMN tokens_json TEXT NOT NULL DEFAULT '[]'")
    if "cache_sig" not in unindexed_cols:
        conn.execute("ALTER TABLE antigravity_unindexed_cache ADD COLUMN cache_sig TEXT NOT NULL DEFAULT ''")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS antigravity_fts_cache (
            conversation_id TEXT PRIMARY KEY,
            last_modified_time TEXT NOT NULL,
            step_count INTEGER NOT NULL,
            title TEXT NOT NULL,
            workspace TEXT NOT NULL,
            summary TEXT NOT NULL,
            tokens_json TEXT NOT NULL DEFAULT '[]',
            vector_json TEXT NOT NULL DEFAULT '[]',
            cache_sig TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.commit()


def rebuild_portable_index(
    index_db: Path,
    providers: Iterable[str] | None = None,
    home: Path | None = None,
    custom_dirs: Iterable[Path] | None = None,
    custom_synonyms: dict[str, list[str]] | None = None,
) -> int:
    """Incrementally index Claude Code, Codex CLI, and custom .jsonl/.md conversations into ~/.v1-memory/index.db.

    Skips re-indexing and re-embedding conversations whose `(last_modified_time, step_count, content_hash, cache_sig)`
    have not changed.
    """
    global LAST_REBUILD_STATS
    index_db.parent.mkdir(parents=True, exist_ok=True)
    provider_set = set(providers) if providers is not None else None
    records = iter_provider_conversations(providers=provider_set, home=home, custom_dirs=custom_dirs)
    syns = custom_synonyms if custom_synonyms is not None else _load_configured_synonyms(index_db.parent)
    cache_sig = compute_rag_signature(syns)

    conn = sqlite3.connect(str(index_db), timeout=10.0)
    try:
        ensure_portable_schema(conn)
        conn.row_factory = sqlite3.Row
        existing_rows = conn.execute(
            """
            SELECT conversation_id, provider, last_modified_time, step_count, content_hash, vector_json, tokens_json, cache_sig
            FROM conversations
            """
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
                and prev["cache_sig"] == cache_sig
                and str(prev["content_hash"]).startswith(f"v{VECTOR_CACHE_VERSION}_")
                and prev["vector_json"]
                and prev["vector_json"] != "[]"
                and prev["tokens_json"]
                and prev["tokens_json"] != "[]"
                and rec.conversation_id in fts_cids
            ):
                skipped_count += 1
                continue

            user_joined = _bounded_head_tail_text("\n".join(rec.user_turns), 80000)
            agent_joined = _bounded_head_tail_text("\n".join(rec.agent_turns), 80000)
            first_u = rec.user_turns[0] if rec.user_turns else rec.title
            last_a = rec.agent_turns[-1] if rec.agent_turns else ""
            summary = build_summary_line(first_u, last_a, rec.title)
            combined_payload = f"{rec.title}\n{user_joined}\n{agent_joined}"
            payload_sha = hashlib.sha1(combined_payload.encode("utf-8", errors="ignore")).hexdigest()[:12]
            content_hash = f"v{VECTOR_CACHE_VERSION}_{cache_sig}_{payload_sha}"

            if (
                prev is not None
                and prev["content_hash"] == content_hash
                and prev["cache_sig"] == cache_sig
                and prev["vector_json"]
                and prev["vector_json"] != "[]"
                and prev["tokens_json"]
                and prev["tokens_json"] != "[]"
                and rec.conversation_id in fts_cids
            ):
                skipped_count += 1
                continue

            doc_tokens = sorted(tokenize_domain(combined_payload))
            tokens_json = json.dumps(doc_tokens)
            vec = compute_semantic_vector(doc_tokens, custom_synonyms=syns)
            vector_json = json.dumps([round(v, 6) for v in vec])

            conn.execute(
                """
                INSERT OR REPLACE INTO conversations (
                    conversation_id, provider, title, last_modified_time,
                    step_count, workspace, transcript_path, summary,
                    user_text, agent_text, content_hash, vector_json,
                    tokens_json, cache_sig
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    tokens_json,
                    cache_sig,
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
        with open(tfile, "rb") as f:
            if size > 1048576:
                raw_bytes = f.read(262144) + b"\n"
                f.seek(max(0, size - 786432))
                raw_bytes += f.read(786432)
            else:
                raw_bytes = f.read()
        for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
            if '"USER_INPUT"' not in line and '"PLANNER_RESPONSE"' not in line:
                continue
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
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


def _load_antigravity_caches_from_conn(
    conn: sqlite3.Connection | None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    unindexed_map: dict[str, dict] = {}
    fts_map: dict[str, dict] = {}
    if conn is None:
        return unindexed_map, fts_map
    try:
        conn.row_factory = sqlite3.Row
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('antigravity_unindexed_cache', 'antigravity_fts_cache')"
            ).fetchall()
        }
        if "antigravity_unindexed_cache" in tables:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(antigravity_unindexed_cache)").fetchall()}
            if {"cache_version", "tokens_json", "cache_sig"}.issubset(cols):
                rows = conn.execute(
                    """
                    SELECT conversation_id, last_modified_time, step_count, title, workspace, summary,
                           vector_json, cache_version, tokens_json, cache_sig
                    FROM antigravity_unindexed_cache
                    """
                ).fetchall()
                unindexed_map = {r["conversation_id"]: dict(r) for r in rows}
        if "antigravity_fts_cache" in tables:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(antigravity_fts_cache)").fetchall()}
            if {"tokens_json", "vector_json", "cache_sig"}.issubset(cols):
                rows = conn.execute(
                    """
                    SELECT conversation_id, last_modified_time, step_count, title, workspace, summary,
                           tokens_json, vector_json, cache_sig
                    FROM antigravity_fts_cache
                    """
                ).fetchall()
                fts_map = {r["conversation_id"]: dict(r) for r in rows}
    except sqlite3.Error:
        pass
    return unindexed_map, fts_map


def _save_antigravity_caches(
    index_db: Path | None,
    unindexed_rows: list[tuple],
    fts_rows: list[tuple],
    indexed_cids: set[str] | None = None,
    valid_unindexed_cids: set[str] | None = None,
) -> None:
    if index_db is None or (not unindexed_rows and not fts_rows and not indexed_cids and valid_unindexed_cids is None):
        return
    try:
        index_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(index_db), timeout=5.0)
        try:
            ensure_portable_schema(conn)
            if unindexed_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO antigravity_unindexed_cache (
                        conversation_id, last_modified_time, step_count,
                        title, workspace, summary, combined_text, vector_json,
                        cache_version, tokens_json, cache_sig
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    unindexed_rows,
                )
            if indexed_cids:
                placeholders = ",".join("?" for _ in indexed_cids)
                conn.execute(
                    f"DELETE FROM antigravity_unindexed_cache WHERE conversation_id IN ({placeholders})",
                    list(indexed_cids),
                )
            if valid_unindexed_cids is not None:
                if valid_unindexed_cids:
                    placeholders = ",".join("?" for _ in valid_unindexed_cids)
                    conn.execute(
                        f"DELETE FROM antigravity_unindexed_cache WHERE conversation_id NOT IN ({placeholders})",
                        list(valid_unindexed_cids),
                    )
                else:
                    conn.execute("DELETE FROM antigravity_unindexed_cache")
            conn.execute(
                """
                DELETE FROM antigravity_unindexed_cache
                WHERE conversation_id NOT IN (
                    SELECT conversation_id FROM antigravity_unindexed_cache
                    ORDER BY last_modified_time DESC
                    LIMIT 500
                )
                """
            )
            if fts_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO antigravity_fts_cache (
                        conversation_id, last_modified_time, step_count,
                        title, workspace, summary, tokens_json, vector_json, cache_sig
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    fts_rows,
                )
                conn.execute(
                    """
                    DELETE FROM antigravity_fts_cache
                    WHERE conversation_id NOT IN (
                        SELECT conversation_id FROM antigravity_fts_cache
                        ORDER BY last_modified_time DESC
                        LIMIT 1000
                    )
                    """
                )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:
        pass


def _parse_json_list(raw: str | None) -> list:
    if not raw or raw == "[]":
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except Exception:
        return []


def _collect_antigravity_candidates(
    agy_db: Path,
    fts_query: str,
    current_conv_id: str,
    exclude_ids: set[str],
    index_db: Path | None = None,
    custom_synonyms: dict[str, list[str]] | None = None,
    core_fts_query: str | None = None,
    hybrid_rag: bool = True,
    port_conn: sqlite3.Connection | None = None,
) -> tuple[dict[str, dict], int]:
    global LAST_UNINDEXED_CACHE_STATS
    docs: dict[str, dict] = {}
    if not agy_db.exists():
        return docs, 0

    cache_sig = compute_rag_signature(custom_synonyms)
    close_port = False
    port_tmp: str | None = None
    if port_conn is None and index_db is not None and index_db.exists():
        try:
            port_conn, port_tmp = open_readonly_snapshot(index_db)
            close_port = True
        except sqlite3.Error:
            port_conn = None

    try:
        cached_unindexed, cached_fts = _load_antigravity_caches_from_conn(port_conn)
    finally:
        if close_port and port_conn is not None:
            port_conn.close()
            if port_tmp and os.path.exists(port_tmp):
                try:
                    os.unlink(port_tmp)
                except OSError:
                    pass

    try:
        conn, tmp_path = open_readonly_snapshot(agy_db)
    except sqlite3.Error:
        return docs, 0

    total_count = 0
    ranked_chunks: list[sqlite3.Row] = []
    unindexed: list[sqlite3.Row] = []
    recent_indexed_fallback: list[sqlite3.Row] = []

    try:
        conn.row_factory = sqlite3.Row
        fts_cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations_fts)").fetchall()}
        u_col = "user_body" if "user_body" in fts_cols else ("user_text" if "user_text" in fts_cols else None)
        a_col = "agent_body" if "agent_body" in fts_cols else ("agent_text" if "agent_text" in fts_cols else None)
        t_col = "title" if "title" in fts_cols else None
        u_expr = f"COALESCE(f.{u_col}, '')" if u_col else "''"
        a_expr = f"COALESCE(f.{a_col}, '')" if a_col else "''"
        t_expr = f"COALESCE(f.{t_col}, '')" if t_col else "''"

        base_fts_sql = f"""
            SELECT
                f.conversation_id,
                cs.title,
                cs.preview,
                cs.last_modified_time,
                cs.step_count,
                cs.workspace_uris,
                {u_expr} AS user_snip,
                {a_expr} AS agent_snip,
                {t_expr} || ' ' || {u_expr} || ' ' || {a_expr} AS chunk_text
            FROM conversations_fts f
            JOIN conversation_summaries cs
              ON cs.conversation_id = f.conversation_id
            WHERE conversations_fts MATCH ?
              AND f.conversation_id != ?
              AND COALESCE(cs.parent_conversation_id, '') = ''
            ORDER BY bm25(conversations_fts) ASC
            LIMIT ?
        """

        if fts_cols:
            if core_fts_query and core_fts_query != fts_query:
                ranked_chunks.extend(conn.execute(base_fts_sql, (core_fts_query, current_conv_id, 120)).fetchall())
            main_chunks = conn.execute(base_fts_sql, (fts_query, current_conv_id, 350)).fetchall()
            ranked_chunks.extend(main_chunks)

            found_cids = {r["conversation_id"] for r in ranked_chunks}
            if len(main_chunks) >= 350 and len(found_cids) < 80 and found_cids:
                placeholders = ",".join("?" for _ in found_cids)
                supp_sql = f"""
                    SELECT
                        f.conversation_id,
                        cs.title,
                        cs.preview,
                        cs.last_modified_time,
                        cs.step_count,
                        cs.workspace_uris,
                        {u_expr} AS user_snip,
                        {a_expr} AS agent_snip,
                        {t_expr} || ' ' || {u_expr} || ' ' || {a_expr} AS chunk_text
                    FROM conversations_fts f
                    JOIN conversation_summaries cs
                      ON cs.conversation_id = f.conversation_id
                    WHERE conversations_fts MATCH ?
                      AND f.conversation_id != ?
                      AND COALESCE(cs.parent_conversation_id, '') = ''
                      AND f.conversation_id NOT IN ({placeholders})
                    ORDER BY bm25(conversations_fts) ASC
                    LIMIT 250
                """
                ranked_chunks.extend(
                    conn.execute(supp_sql, [fts_query, current_conv_id, *found_cids]).fetchall()
                )
                found_cids = {r["conversation_id"] for r in ranked_chunks}

            if hybrid_rag and len(found_cids) < 25:
                recent_indexed_fallback = conn.execute(
                    f"""
                    SELECT
                        f.conversation_id,
                        cs.title,
                        cs.preview,
                        cs.last_modified_time,
                        cs.step_count,
                        cs.workspace_uris,
                        {u_expr} AS user_snip,
                        {a_expr} AS agent_snip,
                        {t_expr} || ' ' || {u_expr} || ' ' || {a_expr} AS chunk_text
                    FROM conversation_summaries cs
                    JOIN conversations_fts f
                      ON f.conversation_id = cs.conversation_id
                    WHERE cs.conversation_id != ?
                      AND COALESCE(cs.parent_conversation_id, '') = ''
                    ORDER BY cs.last_modified_time DESC
                    LIMIT 60
                    """,
                    (current_conv_id,),
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
    except sqlite3.Error:
        return docs, 0
    finally:
        conn.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    chunk_counts: dict[str, int] = {}
    for r in ranked_chunks + recent_indexed_fallback:
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
                "step_count": int(r["step_count"] or 0),
                "workspace": r["workspace_uris"] or "",
                "summary": build_summary_line(r["user_snip"], r["agent_snip"], r["preview"] or title),
                "parts": [title, r["preview"] or ""],
                "vector_json": "[]",
                "tokens": None,
                "cache_sig": "",
            }
            chunk_counts[cid] = 0
        if chunk_counts[cid] < 6:
            docs[cid]["parts"].append(r["chunk_text"])
            chunk_counts[cid] += 1

    new_fts_cache_rows: list[tuple] = []
    fts_cids_seen = set(docs.keys())
    for cid, d_info in docs.items():
        mtime = d_info["last_modified_time"]
        sc = d_info["step_count"]
        c_fts = cached_fts.get(cid)
        if (
            c_fts is not None
            and c_fts.get("cache_sig") == cache_sig
            and c_fts.get("last_modified_time") == mtime
            and int(c_fts.get("step_count", -1)) == sc
        ):
            tok_list = _parse_json_list(c_fts.get("tokens_json"))
            if tok_list and c_fts.get("vector_json") and c_fts.get("vector_json") != "[]":
                d_info["tokens"] = [str(x) for x in tok_list]
                d_info["vector_json"] = c_fts["vector_json"]
                d_info["cache_sig"] = cache_sig
                continue

        combined = " ".join(p for p in d_info["parts"] if p)[:120000]
        tok_list = sorted(set(tokenize_domain(combined)) | set(tokenize_domain(d_info["title"])))
        tok_json = json.dumps(tok_list)
        vec = compute_semantic_vector(tok_list, custom_synonyms=custom_synonyms)
        vec_json = json.dumps([round(v, 6) for v in vec])
        d_info["tokens"] = tok_list
        d_info["vector_json"] = vec_json
        d_info["cache_sig"] = cache_sig
        new_fts_cache_rows.append(
            (cid, mtime, sc, d_info["title"], d_info["workspace"], d_info["summary"], tok_json, vec_json, cache_sig)
        )

    new_unindexed_cache_rows: list[tuple] = []
    cache_hits = 0
    cache_misses = 0

    brain_dir = agy_db.parent / "brain"
    deadline = time.perf_counter() + 0.25
    all_unindexed_cids: set[str] = {r["conversation_id"] for r in unindexed}

    for r in unindexed:
        cid = r["conversation_id"]
        if cid in exclude_ids or cid in docs:
            continue
        mtime = (r["last_modified_time"] or "")[:19]
        step_count = int(r["step_count"] or 0)
        cached = cached_unindexed.get(cid)
        if (
            cached is not None
            and cached.get("cache_version") == VECTOR_CACHE_VERSION
            and cached.get("cache_sig") == cache_sig
            and cached.get("last_modified_time") == mtime
            and int(cached.get("step_count", -1)) == step_count
        ):
            tok_list = _parse_json_list(cached.get("tokens_json"))
            if tok_list:
                cache_hits += 1
                docs[cid] = {
                    "conversation_id": cid,
                    "provider": "antigravity",
                    "title": cached["title"],
                    "last_modified_time": mtime,
                    "step_count": step_count,
                    "workspace": cached["workspace"],
                    "summary": cached["summary"],
                    "parts": [cached["title"]],
                    "tokens": [str(x) for x in tok_list],
                    "vector_json": cached.get("vector_json", "[]"),
                    "cache_sig": cache_sig,
                }
                continue

        cache_misses += 1
        title = (r["title"] or "").strip()
        preview = (r["preview"] or "").strip()
        display_title = title or clean_snippet(preview, 80) or "(Untitled Conversation)"
        parts = [title, preview]
        users: list[str] = []
        agents: list[str] = []
        tp = brain_dir / cid / ".system_generated" / "logs" / "transcript.jsonl"
        has_tp = tp.exists()
        transcript_complete = False
        if time.perf_counter() < deadline:
            if has_tp:
                users, agents = _extract_antigravity_transcript_preview(tp)
                parts.extend(users + agents)
            transcript_complete = True

        first_u = users[0] if users else preview
        last_a = agents[-1] if agents else ""
        summary = build_summary_line(first_u, last_a, preview or display_title)
        clean_parts = [p for p in parts if p]
        combined_text = _bounded_head_tail_text(" ".join(clean_parts), 120000)
        doc_tokens = sorted(set(tokenize_domain(combined_text)) | set(tokenize_domain(display_title)))
        tokens_json = json.dumps(doc_tokens)
        vec = compute_semantic_vector(doc_tokens, custom_synonyms=custom_synonyms)
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
            "tokens": doc_tokens,
            "vector_json": vec_json,
            "cache_sig": cache_sig,
        }
        # Only persist to cache when transcript inspection completed before the deadline.
        if transcript_complete:
            new_unindexed_cache_rows.append(
                (
                    cid,
                    mtime,
                    step_count,
                    display_title,
                    workspace_str,
                    summary,
                    "",
                    vec_json,
                    VECTOR_CACHE_VERSION,
                    tokens_json,
                    cache_sig,
                )
            )

    valid_prune_set = all_unindexed_cids if len(unindexed) < 50 else None
    if new_unindexed_cache_rows or new_fts_cache_rows or fts_cids_seen:
        _save_antigravity_caches(
            index_db,
            new_unindexed_cache_rows,
            new_fts_cache_rows,
            indexed_cids=fts_cids_seen,
            valid_unindexed_cids=valid_prune_set,
        )
    LAST_UNINDEXED_CACHE_STATS = {"hits": cache_hits, "misses": cache_misses}
    return docs, total_count


def _collect_portable_candidates(
    index_db: Path,
    fts_query: str,
    current_conv_id: str,
    exclude_ids: set[str],
    provider_filter: str | None = None,
    include_dense_scan: bool = True,
    port_conn: sqlite3.Connection | None = None,
) -> tuple[dict[str, dict], int]:
    docs: dict[str, dict] = {}
    if not index_db.exists() and port_conn is None:
        return docs, 0

    close_conn = False
    tmp_path: str | None = None
    conn = port_conn
    if conn is None:
        try:
            conn, tmp_path = open_readonly_snapshot(index_db)
            close_conn = True
        except sqlite3.Error:
            return docs, 0

    total_count = 0
    rows: list[sqlite3.Row] = []
    dense_rows: list[sqlite3.Row] = []
    legacy_missing_tokens_cids: list[str] = []
    legacy_texts: dict[str, tuple[str, str]] = {}

    try:
        conn.row_factory = sqlite3.Row
        cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
        if not cols:
            return docs, 0
        vec_expr = "c.vector_json" if "vector_json" in cols else "'[]' AS vector_json"
        tok_expr = "c.tokens_json" if "tokens_json" in cols else "'[]' AS tokens_json"
        sig_expr = "c.cache_sig" if "cache_sig" in cols else "'' AS cache_sig"

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
                {vec_expr},
                {tok_expr},
                {sig_expr}
            FROM conversations_fts f
            JOIN conversations c
              ON c.conversation_id = f.conversation_id
            WHERE {' AND '.join(where)}
            ORDER BY bm25(conversations_fts) ASC
            LIMIT 100
            """,
            params,
        ).fetchall()

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
                    {vec_expr},
                    {tok_expr},
                    {sig_expr}
                FROM conversations c
                WHERE {' AND '.join(base_where)}
                ORDER BY c.last_modified_time DESC
                LIMIT 150
                """,
                base_params,
            ).fetchall()

        for r in list(rows) + list(dense_rows):
            cid = r["conversation_id"]
            if cid in exclude_ids:
                continue
            if not r["tokens_json"] or r["tokens_json"] == "[]":
                legacy_missing_tokens_cids.append(cid)

        if legacy_missing_tokens_cids:
            dedup_legacy = list(dict.fromkeys(legacy_missing_tokens_cids))
            placeholders = ",".join("?" for _ in dedup_legacy)
            for lr in conn.execute(
                f"SELECT conversation_id, user_text, agent_text FROM conversations WHERE conversation_id IN ({placeholders})",
                dedup_legacy,
            ).fetchall():
                legacy_texts[lr["conversation_id"]] = (lr["user_text"] or "", lr["agent_text"] or "")
    except sqlite3.Error:
        return docs, 0
    finally:
        if close_conn:
            conn.close()
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    for idx_rank, r in enumerate(rows, start=1):
        cid = r["conversation_id"]
        if cid in exclude_ids or cid in docs:
            continue
        tok_list = _parse_json_list(r["tokens_json"])
        u_txt, a_txt = legacy_texts.get(cid, ("", ""))
        docs[cid] = {
            "conversation_id": cid,
            "provider": r["provider"],
            "title": r["title"],
            "last_modified_time": (r["last_modified_time"] or "")[:19],
            "step_count": r["step_count"],
            "workspace": r["workspace"],
            "summary": r["summary"],
            "parts": [r["title"], u_txt, a_txt] if not tok_list else [r["title"]],
            "tokens": [str(x) for x in tok_list] if tok_list else None,
            "vector_json": r["vector_json"] or "[]",
            "cache_sig": r["cache_sig"] or "",
            "fts_rank": idx_rank,
        }

    for r in dense_rows:
        cid = r["conversation_id"]
        if cid in exclude_ids or cid in docs:
            continue
        tok_list = _parse_json_list(r["tokens_json"])
        u_txt, a_txt = legacy_texts.get(cid, ("", ""))
        docs[cid] = {
            "conversation_id": cid,
            "provider": r["provider"],
            "title": r["title"],
            "last_modified_time": (r["last_modified_time"] or "")[:19],
            "step_count": r["step_count"],
            "workspace": r["workspace"],
            "summary": r["summary"],
            "parts": [r["title"], u_txt, a_txt] if not tok_list else [r["title"]],
            "tokens": [str(x) for x in tok_list] if tok_list else None,
            "vector_json": r["vector_json"] or "[]",
            "cache_sig": r["cache_sig"] or "",
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


def _build_prioritized_fts_terms(
    q_tokens: Sequence[str],
    custom_synonyms: dict[str, list[str]] | None = None,
    hybrid_rag: bool = True,
    max_terms: int = 60,
) -> list[str]:
    """Build FTS5-safe terms with raw query tokens first and round-robin concept terms."""
    if not hybrid_rag:
        return fts5_terms_for_tokens(q_tokens, max_terms=max_terms)

    clusters, _ = build_concept_lookup(custom_synonyms)
    concept_hits = group_tokens_by_concept(q_tokens, custom_synonyms)
    cluster_lists: list[list[str]] = []
    for concept_id in concept_hits:
        raw_cid = concept_id.split(":", 1)[1]
        c_terms = clusters.get(raw_cid, [])
        if c_terms:
            cluster_lists.append(c_terms)

    prioritized: list[str] = list(q_tokens)
    if cluster_lists:
        max_len = max(len(lst) for lst in cluster_lists)
        for idx in range(max_len):
            for lst in cluster_lists:
                if idx < len(lst):
                    prioritized.append(lst[idx])

    return fts5_terms_for_tokens(prioritized, max_terms=max_terms)


def _compute_corpus_stats(
    doc_tokens_by_cid: dict[str, set[str]],
    total_db_parent_count: int,
    high_df_ratio: float = HIGH_DF_SUPPRESSION_RATIO,
    min_corpus_size: int = MIN_CORPUS_FOR_IDF,
) -> tuple[int, dict[str, int], set[str]]:
    """Compute corpus document count, per-token document frequencies, and high-DF suppress list.

    Uses `eff_denom` bounded safely so:
    - On small corpora (`N < 50`), full `n_corpus` prevents FTS selectivity from self-suppressing rare query terms.
    - On large corpora (`N >= 50`, e.g., `N = 1000` where candidate retrieval is capped at ~150-300 docs),
      high-DF ratio and IDF scale against the candidate sample denominator `max(len(doc_tokens_by_cid), 50)`.
    """
    sample_n = len(doc_tokens_by_cid)
    n_corpus = max(sample_n, total_db_parent_count)
    df: dict[str, int] = {}
    for d_toks in doc_tokens_by_cid.values():
        for tok in d_toks:
            df[tok] = df.get(tok, 0) + 1
    if n_corpus < min_corpus_size:
        return n_corpus, df, set()

    eff_denom = n_corpus if n_corpus < 50 else min(n_corpus, max(sample_n, 50))
    high_df = {tok for tok, count in df.items() if (count / eff_denom) >= high_df_ratio}
    if n_corpus > eff_denom and eff_denom > 0:
        scale = n_corpus / eff_denom
        df = {tok: min(n_corpus, max(1, int(round(count * scale)))) for tok, count in df.items()}
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
    cache_sig = compute_rag_signature(syns)

    fts_raw_terms = _build_prioritized_fts_terms(q_tokens, syns, hybrid_rag=hybrid_rag, max_terms=60)
    if not fts_raw_terms:
        return q_tokens, []
    fts_query = " OR ".join(f'"{t}"' for t in fts_raw_terms)

    core_raw_terms = fts5_terms_for_tokens(q_tokens, max_terms=20)
    core_fts_query = " OR ".join(f'"{t}"' for t in core_raw_terms) if core_raw_terms else fts_query

    # Open a single read-only connection to port_db to share across cache loads and portable candidate search
    port_conn: sqlite3.Connection | None = None
    port_tmp: str | None = None
    if port_db.exists():
        try:
            port_conn, port_tmp = open_readonly_snapshot(port_db)
        except sqlite3.Error:
            port_conn = None

    docs_by_id: dict[str, dict] = {}
    total_db_parent_count = 0
    try:
        if provider_filter in ("all", "auto", "antigravity"):
            agy_docs, agy_count = _collect_antigravity_candidates(
                agy_db,
                fts_query,
                current_conv_id,
                ex,
                index_db=port_db,
                custom_synonyms=syns,
                core_fts_query=core_fts_query,
                hybrid_rag=hybrid_rag,
                port_conn=port_conn,
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
                port_conn=port_conn,
            )
            docs_by_id.update(port_docs)
            total_db_parent_count += port_count
    finally:
        if port_conn is not None:
            port_conn.close()
            if port_tmp and os.path.exists(port_tmp):
                try:
                    os.unlink(port_tmp)
                except OSError:
                    pass

    if not docs_by_id:
        return q_tokens, []

    title_tokens_by_cid: dict[str, set[str]] = {}
    doc_tokens_by_cid: dict[str, set[str]] = {}
    for cid, info in docs_by_id.items():
        title_tokens = set(tokenize_domain(info["title"]))
        pre_toks = info.get("tokens")
        if isinstance(pre_toks, list) and pre_toks:
            doc_tokens = set(pre_toks) | title_tokens
        else:
            combined = " ".join(p for p in info.get("parts", []) if p)[:120000]
            doc_tokens = set(tokenize_domain(combined)) | title_tokens
        if not doc_tokens:
            continue
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
    q_vec = compute_semantic_vector(q_active, custom_synonyms=syns) if hybrid_rag else []

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
        title_active = title_tokens_by_cid[cid] - high_df_tokens
        title_stems = {stem_token(t) for t in title_active}
        title_hits = sum(1 for qt in q_active_set if qt in title_active or stem_token(qt) in title_stems)

        doc_idf_sum = sum(_bm25_idf(t, n_corpus, df, high_df_tokens) for t in doc_active)
        denom_weight = max(0.75 * q_idf_sum, min(q_idf_sum, doc_idf_sum))
        exact_weight = sum(_bm25_idf(t, n_corpus, df, high_df_tokens) for t in exact_matched)
        exact_pct = round(min(100.0, (exact_weight / denom_weight) * 100.0), 1) if denom_weight > 1e-9 else 0.0

        cos_sim = 0.0
        qualifying_concepts: list[str] = []
        covered_q_tokens = set(exact_matched)
        concept_doc_hits: list[str] = []
        stem_matched_q: list[str] = []

        if hybrid_rag:
            doc_vec: list[float] = []
            raw_vjson = info.get("vector_json") or "[]"
            if raw_vjson and raw_vjson != "[]" and (not info.get("cache_sig") or info.get("cache_sig") == cache_sig):
                parsed_v = _parse_json_list(raw_vjson)
                if parsed_v:
                    doc_vec = [float(x) for x in parsed_v]
            if not doc_vec:
                doc_vec = compute_semantic_vector(sorted(doc_active), custom_synonyms=syns)

            cos_sim = max(0.0, cosine_similarity(q_vec, doc_vec))
            doc_concept_hits = group_tokens_by_concept(doc_active, syns)
            shared_concepts = qualifying_shared_concepts(q_concept_hits, doc_concept_hits, min_distinct_stems=2)
            doc_hit_pool: set[str] = set()
            for concept_id, (q_c_hits, d_c_hits) in shared_concepts.items():
                qualifying_concepts.append(concept_id)
                covered_q_tokens.update(q_c_hits)
                doc_hit_pool.update(t for t in d_c_hits if t not in q_active_set)

            doc_stem_to_tok: dict[str, str] = {}
            for dt in sorted(doc_active):
                doc_stem_to_tok.setdefault(stem_token(dt), dt)
            for qt in sorted(q_active_set):
                st = stem_token(qt)
                if st in doc_stem_to_tok:
                    stem_matched_q.append(qt)
                    if qt not in exact_matched:
                        doc_hit_pool.add(doc_stem_to_tok[st])
            if len(stem_matched_q) >= min_matched_tokens:
                covered_q_tokens.update(stem_matched_q)

            concept_doc_hits = sorted(doc_hit_pool)

        dense_fallback = bool(
            hybrid_rag and len(q_active_set) >= min_matched_tokens and cos_sim >= DENSE_COSINE_FLOOR
        )
        if len(covered_q_tokens) < min_matched_tokens and not dense_fallback:
            continue

        covered_weight = sum(_bm25_idf(t, n_corpus, df, high_df_tokens) for t in covered_q_tokens)
        concept_coverage_pct = (
            round(min(100.0, (covered_weight / denom_weight) * 100.0), 1) if denom_weight > 1e-9 else 0.0
        )

        has_extra_stem_coverage = len(stem_matched_q) > len(exact_matched) and len(stem_matched_q) >= min_matched_tokens
        if hybrid_rag and (qualifying_concepts or has_extra_stem_coverage):
            concept_and_cosine_pct = round(min(100.0, w_con * concept_coverage_pct + w_sem * (cos_sim * 100.0)), 1)
            hybrid_pct = round(max(exact_pct, concept_and_cosine_pct), 1)
        elif dense_fallback and len(exact_matched) < min_matched_tokens:
            concept_and_cosine_pct = round(
                min(100.0, max(w_con * concept_coverage_pct + w_sem * (cos_sim * 100.0), cos_sim * 100.0)),
                1,
            )
            hybrid_pct = round(max(exact_pct, concept_and_cosine_pct), 1)
        else:
            concept_and_cosine_pct = 0.0
            hybrid_pct = exact_pct

        if hybrid_pct < threshold_pct:
            continue

        matched_out = list(exact_matched)
        if hybrid_rag and (
            qualifying_concepts
            or has_extra_stem_coverage
            or (dense_fallback and len(exact_matched) < min_matched_tokens)
        ):
            if len(exact_matched) < len(q_active_set) or concept_and_cosine_pct >= exact_pct:
                for sc in qualifying_concepts:
                    if sc not in matched_out:
                        matched_out.append(sc)
                for dh in concept_doc_hits[:5]:
                    if dh not in matched_out:
                        matched_out.append(dh)
                if not matched_out and dense_fallback:
                    matched_out.append("semantic:dense_vector")

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
                ((round(cos_sim, 6), len(qualifying_concepts), concept_coverage_pct, info["last_modified_time"]), cid)
            )

    if not candidate_evals:
        return q_tokens, []

    sparse_rank_tuples.sort(key=lambda item: item[0], reverse=True)
    dense_rank_tuples.sort(key=lambda item: item[0], reverse=True)
    sparse_ids = [cid for _, cid in sparse_rank_tuples]
    dense_ids = [cid for _, cid in dense_rank_tuples]
    rrf_scores = (
        reciprocal_rank_fusion(sparse_ids, dense_ids, k=60)
        if hybrid_rag
        else {cid: 1.0 / (60 + r) for r, cid in enumerate(sparse_ids, start=1)}
    )

    scored: list[tuple[tuple, dict]] = []
    for cid, ev in candidate_evals.items():
        info = ev["info"]
        pct = ev["hybrid_pct"]
        title_hits = ev["title_hits"]
        rrf_val = round(rrf_scores.get(cid, 0.0), 6)
        rank_score = round(pct + min(25.0, title_hits * 10.0), 2)
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
                (rank_score, rrf_val, title_hits, pct, ev["semantic_score"], info["last_modified_time"]),
                obj,
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return q_tokens, [item[1] for item in scored[:max_matches]]
