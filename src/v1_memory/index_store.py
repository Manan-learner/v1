"""Portable WAL-safe SQLite + FTS5 index across Antigravity CLI, Claude Code, Codex CLI, and custom logs."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Iterable

from v1_memory.providers import ConversationRecord, iter_provider_conversations
from v1_memory.text_utils import build_summary_line, clean_snippet, tokenize_domain

DEFAULT_STATE_HOME = Path.home() / ".v1-memory"


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
            agent_text TEXT NOT NULL
        )
        """
    )
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
    conn.commit()


def rebuild_portable_index(
    index_db: Path,
    providers: Iterable[str] | None = None,
    home: Path | None = None,
    custom_dirs: Iterable[Path] | None = None,
) -> int:
    """Index Claude Code, Codex CLI, and custom .jsonl/.md conversations into ~/.v1-memory/index.db."""
    index_db.parent.mkdir(parents=True, exist_ok=True)
    records = iter_provider_conversations(providers=providers, home=home, custom_dirs=custom_dirs)
    conn = sqlite3.connect(str(index_db), timeout=10.0)
    try:
        ensure_portable_schema(conn)
        conn.execute("DELETE FROM conversations")
        conn.execute("DELETE FROM conversations_fts")
        for rec in records:
            user_joined = "\n".join(rec.user_turns)[:80000]
            agent_joined = "\n".join(rec.agent_turns)[:80000]
            first_u = rec.user_turns[0] if rec.user_turns else rec.title
            last_a = rec.agent_turns[-1] if rec.agent_turns else ""
            summary = build_summary_line(first_u, last_a, rec.title)
            conn.execute(
                """
                INSERT OR REPLACE INTO conversations (
                    conversation_id, provider, title, last_modified_time,
                    step_count, workspace, transcript_path, summary, user_text, agent_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            conn.execute(
                "INSERT INTO conversations_fts (conversation_id, provider, title, user_text, agent_text) VALUES (?, ?, ?, ?, ?)",
                (rec.conversation_id, rec.provider, rec.title, user_joined, agent_joined),
            )
        conn.commit()
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


def _collect_antigravity_candidates(
    agy_db: Path,
    fts_query: str,
    current_conv_id: str,
    exclude_ids: set[str],
) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    if not agy_db.exists():
        return docs

    conn, tmp_path = open_readonly_snapshot(agy_db)
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
    except sqlite3.OperationalError:
        return docs
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
            }
        docs[cid]["parts"].append(r["chunk_text"])

    brain_dir = agy_db.parent / "brain"
    deadline = time.perf_counter() + 0.25
    for r in unindexed:
        cid = r["conversation_id"]
        if cid in exclude_ids or cid in docs:
            continue
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
        docs[cid] = {
            "conversation_id": cid,
            "provider": "antigravity",
            "title": display_title,
            "last_modified_time": (r["last_modified_time"] or "")[:19],
            "step_count": r["step_count"],
            "workspace": r["workspace_uris"],
            "summary": build_summary_line(first_u, last_a, preview or display_title),
            "parts": [p for p in parts if p],
        }
    return docs


def _collect_portable_candidates(
    index_db: Path,
    fts_query: str,
    current_conv_id: str,
    exclude_ids: set[str],
    provider_filter: str | None = None,
) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    if not index_db.exists():
        return docs

    conn, tmp_path = open_readonly_snapshot(index_db)
    try:
        conn.row_factory = sqlite3.Row
        where = ["conversations_fts MATCH ?", "c.conversation_id != ?"]
        params: list = [fts_query, current_conv_id]
        if provider_filter and provider_filter not in ("auto", "all"):
            where.append("c.provider = ?")
            params.append(provider_filter)

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
                c.agent_text
            FROM conversations_fts f
            JOIN conversations c
              ON c.conversation_id = f.conversation_id
            WHERE {' AND '.join(where)}
            ORDER BY bm25(conversations_fts) ASC
            LIMIT 100
            """,
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return docs
    finally:
        conn.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    for r in rows:
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
        }
    return docs


def find_context_matches(
    prompt_text: str,
    current_conv_id: str = "",
    threshold_pct: float = 25.0,
    min_matched_tokens: int = 2,
    max_matches: int = 3,
    exclude_ids: set[str] | None = None,
    provider_filter: str = "all",
    home: Path | None = None,
    state_home: Path | None = None,
    agy_db_path: Path | None = None,
    index_db_path: Path | None = None,
) -> tuple[list[str], list[dict]]:
    """Score past conversations across Antigravity DB and portable multi-LLM index."""
    q_tokens = tokenize_domain(prompt_text)
    if len(q_tokens) < min_matched_tokens:
        return q_tokens, []

    fts_terms = []
    for t in q_tokens[:25]:
        safe = re.sub(r"[^a-z0-9_]", "", t)
        if len(safe) >= 2:
            fts_terms.append(f'"{safe}"')
    if not fts_terms:
        return q_tokens, []

    fts_query = " OR ".join(dict.fromkeys(fts_terms))
    ex = exclude_ids or set()
    user_home = home or Path.home()
    resolved_state = default_state_home(state_home)
    agy_db = agy_db_path or (user_home / ".gemini" / "antigravity-cli" / "conversation_summaries.db")
    port_db = index_db_path or (resolved_state / "index.db")

    docs_by_id: dict[str, dict] = {}
    if provider_filter in ("all", "auto", "antigravity"):
        docs_by_id.update(_collect_antigravity_candidates(agy_db, fts_query, current_conv_id, ex))
    if provider_filter != "antigravity":
        docs_by_id.update(_collect_portable_candidates(port_db, fts_query, current_conv_id, ex, provider_filter))

    q_set = set(q_tokens)
    scored: list[tuple[tuple, dict]] = []
    for cid, info in docs_by_id.items():
        combined = " ".join(p for p in info["parts"] if p)[:120000]
        title_tokens = set(tokenize_domain(info["title"]))
        doc_tokens = set(tokenize_domain(combined)) | title_tokens
        if not doc_tokens:
            continue
        matched = sorted(q_set & doc_tokens)
        if len(matched) < min_matched_tokens:
            continue
        denom = min(len(q_set), len(doc_tokens))
        pct = round((len(matched) / denom) * 100.0, 1) if denom else 0.0
        if pct >= threshold_pct:
            title_hits = len(q_set & title_tokens)
            rank_score = pct + min(25.0, title_hits * 10.0)
            obj = {
                "conversation_id": cid,
                "provider": info["provider"],
                "title": info["title"],
                "last_modified_time": info["last_modified_time"],
                "step_count": info["step_count"],
                "workspace": info["workspace"],
                "match_pct": pct,
                "matched_tokens": matched,
                "summary": info["summary"],
            }
            scored.append(((rank_score, title_hits, pct, len(matched), info["last_modified_time"]), obj))

    scored.sort(key=lambda item: item[0], reverse=True)
    return q_tokens, [item[1] for item in scored[:max_matches]]
