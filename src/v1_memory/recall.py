"""Multi-LLM FTS5 history search and transcript viewer."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from v1_memory.index_store import default_state_home, open_readonly_snapshot, rebuild_portable_index
from v1_memory.providers import parse_generic_jsonl, parse_markdown_chat
from v1_memory.text_utils import clean_snippet, ensure_utf8_stdio, fts5_terms_for_tokens

_BARE_PUNCT_TOKEN_RE = re.compile(r'(?<!["\w])([A-Za-z0-9_]+(?:[-.][A-Za-z0-9_]+)+)(?!["\w])')


def _prepare_fts5_match_query(query: str) -> str:
    """Normalize dotted/hyphenated tokens (e.g. v1-context, sub-ghz, node.js) into FTS5-safe phrase clauses."""
    raw = (query or "").strip()
    if not raw:
        return ""

    def _replace_punct_token(m: re.Match[str]) -> str:
        parts = [p for p in re.split(r"[^A-Za-z0-9_]+", m.group(1)) if p]
        if not parts:
            return '""'
        if len(parts) == 1:
            return f'"{parts[0]}"'
        return f'"{" ".join(parts)}"'

    normalized = _BARE_PUNCT_TOKEN_RE.sub(_replace_punct_token, raw)
    return normalized


def _fallback_fts5_or_query(query: str) -> str:
    terms = fts5_terms_for_tokens([query], max_terms=30)
    if not terms:
        parts = [p for p in re.split(r"[^a-z0-9_]+", (query or "").lower()) if len(p) >= 2]
        terms = list(dict.fromkeys(parts))[:30]
    return " OR ".join(f'"{t}"' for t in terms)


def _fts_column_index(conn: sqlite3.Connection, table_name: str, candidates: tuple[str, ...], default_idx: int) -> int:
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        name_to_idx = {str(r[1]).lower(): int(r[0]) for r in rows}
        for cand in candidates:
            if cand.lower() in name_to_idx:
                return name_to_idx[cand.lower()]
        if rows and default_idx >= len(rows):
            return -1
    except sqlite3.Error:
        return -1
    return default_idx


def search_conversations(
    query: str,
    days: int | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 8,
    provider: str = "all",
    include_subagents: bool = False,
    home: Path | None = None,
    state_home: Path | None = None,
) -> list[dict]:
    if not query or not query.strip():
        return []

    user_home = home or Path.home()
    resolved_state = default_state_home(state_home)
    results: list[dict] = []

    fts_query = _prepare_fts5_match_query(query)
    fallback_query = _fallback_fts5_or_query(query)

    date_where: list[str] = []
    date_params: list[str] = []
    if days is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        date_where.append("last_modified_time >= ?")
        date_params.append(cutoff)
    if after:
        date_where.append("last_modified_time >= ?")
        date_params.append(after)
    if before:
        date_where.append("last_modified_time <= ?")
        date_params.append(before + " 23:59:59")

    agy_db = user_home / ".gemini" / "antigravity-cli" / "conversation_summaries.db"
    if provider in ("all", "antigravity") and agy_db.exists():
        try:
            conn, tmp = open_readonly_snapshot(agy_db)
        except sqlite3.Error as exc:
            print(f"[v1-recall] warning: unable to open {agy_db}: {exc}", file=sys.stderr)
            conn, tmp = None, None
        if conn is not None:
            try:
                conn.row_factory = sqlite3.Row
                u_idx = _fts_column_index(conn, "conversations_fts", ("user_body", "user_text"), 13)
                a_idx = _fts_column_index(conn, "conversations_fts", ("agent_body", "agent_text"), 14)
                where = ["conversations_fts MATCH ?"]
                if not include_subagents:
                    where.append("COALESCE(cs.parent_conversation_id, '') = ''")
                where.extend([f"cs.{clause}" for clause in date_where])
                sql = f"""
                    SELECT
                        f.conversation_id,
                        'antigravity' AS provider,
                        cs.title,
                        cs.last_modified_time,
                        cs.step_count,
                        cs.workspace_uris AS workspace,
                        snippet(conversations_fts, {u_idx}, '[', ']', '...', 18) AS user_hit,
                        snippet(conversations_fts, {a_idx}, '[', ']', '...', 18) AS agent_hit,
                        bm25(conversations_fts) AS rank_score
                    FROM conversations_fts f
                    JOIN conversation_summaries cs ON cs.conversation_id = f.conversation_id
                    WHERE {' AND '.join(where)}
                    ORDER BY rank_score ASC, cs.last_modified_time DESC
                    LIMIT ?
                """
                rows = []
                try:
                    rows = conn.execute(sql, [fts_query, *date_params, limit * 4]).fetchall()
                except sqlite3.OperationalError:
                    if fallback_query:
                        rows = conn.execute(sql, [fallback_query, *date_params, limit * 4]).fetchall()
                seen: set[str] = set()
                for r in rows:
                    cid = r["conversation_id"]
                    if cid in seen:
                        continue
                    seen.add(cid)
                    results.append(dict(r))
            except sqlite3.Error as exc:
                print(f"[v1-recall] warning: Antigravity FTS query error: {exc}", file=sys.stderr)
            finally:
                conn.close()
                if tmp and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

    port_db = resolved_state / "index.db"
    if provider != "antigravity" and port_db.exists():
        try:
            conn, tmp = open_readonly_snapshot(port_db)
        except sqlite3.Error as exc:
            print(f"[v1-recall] warning: unable to open {port_db}: {exc}", file=sys.stderr)
            conn, tmp = None, None
        if conn is not None:
            try:
                conn.row_factory = sqlite3.Row
                u_idx = _fts_column_index(conn, "conversations_fts", ("user_text", "user_body"), 3)
                a_idx = _fts_column_index(conn, "conversations_fts", ("agent_text", "agent_body"), 4)
                where = ["conversations_fts MATCH ?"]
                extra_params: list = []
                if provider not in ("all", "auto"):
                    where.append("c.provider = ?")
                    extra_params.append(provider)
                where.extend([f"c.{clause}" for clause in date_where])
                extra_params.extend(date_params)
                extra_params.append(limit * 2)
                sql = f"""
                    SELECT
                        c.conversation_id,
                        c.provider,
                        c.title,
                        c.last_modified_time,
                        c.step_count,
                        c.workspace,
                        snippet(conversations_fts, {u_idx}, '[', ']', '...', 18) AS user_hit,
                        snippet(conversations_fts, {a_idx}, '[', ']', '...', 18) AS agent_hit,
                        bm25(conversations_fts) AS rank_score
                    FROM conversations_fts f
                    JOIN conversations c ON c.conversation_id = f.conversation_id
                    WHERE {' AND '.join(where)}
                    ORDER BY rank_score ASC, c.last_modified_time DESC
                    LIMIT ?
                """
                rows = []
                try:
                    rows = conn.execute(sql, [fts_query, *extra_params]).fetchall()
                except sqlite3.OperationalError:
                    if fallback_query:
                        rows = conn.execute(sql, [fallback_query, *extra_params]).fetchall()
                seen_port: set[str] = set()
                for r in rows:
                    cid = r["conversation_id"]
                    if cid in seen_port:
                        continue
                    seen_port.add(cid)
                    results.append(dict(r))
            except sqlite3.Error as exc:
                print(f"[v1-recall] warning: Portable FTS query error: {exc}", file=sys.stderr)
            finally:
                conn.close()
                if tmp and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

    results.sort(key=lambda item: str(item.get("last_modified_time") or ""), reverse=True)
    results.sort(key=lambda item: float(item.get("rank_score", 0.0)))
    return results[:limit]


def sqlite3_row_factory(conn):
    return sqlite3.Row


def _is_safe_conversation_id(conv_id: str) -> bool:
    if not conv_id or not isinstance(conv_id, str):
        return False
    if ".." in conv_id or "/" in conv_id or "\\" in conv_id or "\x00" in conv_id:
        return False
    # Reject drive-letter paths like C:foo or leading/trailing colons
    if conv_id.startswith(":") or conv_id.endswith(":") or (len(conv_id) >= 2 and conv_id[1] == ":"):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)?", conv_id))


def load_conversation_turns(
    conv_id: str,
    max_turns: int = 20,
    home: Path | None = None,
    state_home: Path | None = None,
) -> tuple[str, list[tuple[str, str, str]]] | None:
    if not _is_safe_conversation_id(conv_id):
        return None

    user_home = home or Path.home()
    resolved_state = default_state_home(state_home)

    # Antigravity brain directory lookup (only when conv_id is a single slug/UUID without provider colon)
    if ":" not in conv_id and re.fullmatch(r"[A-Za-z0-9._-]+", conv_id):
        try:
            brain_dir = (user_home / ".gemini" / "antigravity-cli" / "brain").resolve()
            agy_tfile = (brain_dir / conv_id / ".system_generated" / "logs" / "transcript.jsonl").resolve()
            if agy_tfile.is_relative_to(brain_dir) and agy_tfile.is_file():
                rec = parse_generic_jsonl(agy_tfile, provider="antigravity", conv_id=conv_id)
                return rec.title, rec.raw_turns[-max_turns:]
        except OSError:
            pass

    port_db = resolved_state / "index.db"
    if port_db.exists():
        try:
            conn, tmp = open_readonly_snapshot(port_db)
        except sqlite3.Error:
            return None
        try:
            row = conn.execute(
                "SELECT provider, title, transcript_path FROM conversations WHERE conversation_id = ?",
                (conv_id,),
            ).fetchone()
            if row and row[2]:
                tpath = Path(row[2])
                if tpath.exists() and tpath.is_file():
                    rec = (
                        parse_markdown_chat(tpath, row[0])
                        if tpath.suffix.lower() == ".md"
                        else parse_generic_jsonl(tpath, row[0], conv_id)
                    )
                    return row[1] or rec.title, rec.raw_turns[-max_turns:]
        except sqlite3.Error:
            return None
        finally:
            conn.close()
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    return None


def main() -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Search and recall past conversations across AI coding CLIs.")
    parser.add_argument("query", nargs="?", help="FTS5 search query")
    parser.add_argument("--provider", default="all", help="all, antigravity, claude, codex, generic")
    parser.add_argument("--days", type=int, help="Only match conversations modified in the last N days")
    parser.add_argument("--after", help="Filter last_modified_time >= YYYY-MM-DD")
    parser.add_argument("--before", help="Filter last_modified_time <= YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=8, help="Max results (default: 8)")
    parser.add_argument("--include-subagents", action="store_true")
    parser.add_argument("--show", metavar="CONVERSATION_ID", help="Print key turns from a conversation ID")
    parser.add_argument("--max-turns", type=int, default=20, help="Max turns to display for --show")
    parser.add_argument("--reindex", action="store_true", help="Rebuild portable FTS index for non-Antigravity CLIs")
    args = parser.parse_args()

    state_home = default_state_home()
    if args.reindex:
        n = rebuild_portable_index(state_home / "index.db")
        print(f"Indexed {n} non-Antigravity conversation(s).", file=sys.stderr)

    if args.show:
        loaded = load_conversation_turns(args.show, max_turns=args.max_turns, state_home=state_home)
        if not loaded:
            print(f"Transcript not found for conversation ID: {args.show}", file=sys.stderr)
            return 1
        title, turns = loaded
        print(f"=== Transcript Digest: {title} ({args.show}, showing {len(turns)} turn(s)) ===")
        for role, ts, text in turns:
            print(f"\n[{role}] {ts}\n{text}")
        return 0

    if not args.query:
        parser.print_help()
        return 1

    hits = search_conversations(
        args.query,
        days=args.days,
        after=args.after,
        before=args.before,
        limit=args.limit,
        provider=args.provider,
        include_subagents=args.include_subagents,
        state_home=state_home,
    )
    print(json.dumps(hits, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
