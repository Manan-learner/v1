"""Multi-LLM FTS5 history search and transcript viewer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from v1_memory.index_store import default_state_home, open_readonly_snapshot, rebuild_portable_index
from v1_memory.providers import parse_generic_jsonl, parse_markdown_chat


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
    user_home = home or Path.home()
    resolved_state = default_state_home(state_home)
    results: list[dict] = []

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
        conn, tmp = open_readonly_snapshot(agy_db)
        try:
            conn.row_factory = sqlite3_row_factory(conn)
            where = ["conversations_fts MATCH ?"]
            params: list = [query]
            if not include_subagents:
                where.append("COALESCE(cs.parent_conversation_id, '') = ''")
            where.extend([f"cs.{clause}" for clause in date_where])
            params.extend(date_params)
            params.append(limit * 4)
            sql = f"""
                SELECT
                    f.conversation_id,
                    'antigravity' AS provider,
                    cs.title,
                    cs.last_modified_time,
                    cs.step_count,
                    cs.workspace_uris AS workspace,
                    snippet(conversations_fts, 13, '[', ']', '...', 18) AS user_hit,
                    snippet(conversations_fts, 14, '[', ']', '...', 18) AS agent_hit,
                    bm25(conversations_fts) AS rank_score
                FROM conversations_fts f
                JOIN conversation_summaries cs ON cs.conversation_id = f.conversation_id
                WHERE {' AND '.join(where)}
                ORDER BY rank_score ASC, cs.last_modified_time DESC
                LIMIT ?
            """
            seen: set[str] = set()
            for r in conn.execute(sql, params).fetchall():
                cid = r["conversation_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                results.append(dict(r))
        except Exception:
            pass
        finally:
            conn.close()
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    port_db = resolved_state / "index.db"
    if provider != "antigravity" and port_db.exists():
        conn, tmp = open_readonly_snapshot(port_db)
        try:
            conn.row_factory = sqlite3_row_factory(conn)
            where = ["conversations_fts MATCH ?"]
            params = [query]
            if provider not in ("all", "auto"):
                where.append("c.provider = ?")
                params.append(provider)
            where.extend([f"c.{clause}" for clause in date_where])
            params.extend(date_params)
            params.append(limit * 2)
            sql = f"""
                SELECT
                    c.conversation_id,
                    c.provider,
                    c.title,
                    c.last_modified_time,
                    c.step_count,
                    c.workspace,
                    snippet(conversations_fts, 3, '[', ']', '...', 18) AS user_hit,
                    snippet(conversations_fts, 4, '[', ']', '...', 18) AS agent_hit,
                    bm25(conversations_fts) AS rank_score
                FROM conversations_fts f
                JOIN conversations c ON c.conversation_id = f.conversation_id
                WHERE {' AND '.join(where)}
                ORDER BY rank_score ASC, c.last_modified_time DESC
                LIMIT ?
            """
            seen_port: set[str] = set()
            for r in conn.execute(sql, params).fetchall():
                cid = r["conversation_id"]
                if cid in seen_port:
                    continue
                seen_port.add(cid)
                results.append(dict(r))
        except Exception:
            pass
        finally:
            conn.close()
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    results.sort(key=lambda item: (float(item.get("rank_score", 0.0)), -(len(str(item.get("last_modified_time", ""))))))
    return results[:limit]


def sqlite3_row_factory(conn):
    import sqlite3

    return sqlite3.Row


def load_conversation_turns(
    conv_id: str,
    max_turns: int = 20,
    home: Path | None = None,
    state_home: Path | None = None,
) -> tuple[str, list[tuple[str, str, str]]] | None:
    user_home = home or Path.home()
    resolved_state = default_state_home(state_home)

    agy_tfile = user_home / ".gemini" / "antigravity-cli" / "brain" / conv_id / ".system_generated" / "logs" / "transcript.jsonl"
    if agy_tfile.exists():
        rec = parse_generic_jsonl(agy_tfile, provider="antigravity", conv_id=conv_id)
        return rec.title, rec.raw_turns[-max_turns:]

    port_db = resolved_state / "index.db"
    if port_db.exists():
        conn, tmp = open_readonly_snapshot(port_db)
        try:
            row = conn.execute("SELECT provider, title, transcript_path FROM conversations WHERE conversation_id = ?", (conv_id,)).fetchone()
            if row and row[2] and Path(row[2]).exists():
                tpath = Path(row[2])
                rec = parse_markdown_chat(tpath, row[0]) if tpath.suffix.lower() == ".md" else parse_generic_jsonl(tpath, row[0], conv_id)
                return row[1] or rec.title, rec.raw_turns[-max_turns:]
        finally:
            conn.close()
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
    return None


def main() -> int:
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
