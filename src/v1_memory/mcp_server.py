"""Zero-dependency stdlib MCP stdio server for ANY MCP-capable LLM (Claude, Cursor, Windsurf, Codex, Gemini)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from v1_memory.cloud_sync import get_sync_status, run_sync
from v1_memory.index_store import default_state_home, find_context_matches, rebuild_portable_index
from v1_memory.recall import load_conversation_turns

TOOLS = [
    {
        "name": "find_context",
        "description": (
            "Search past conversations across Antigravity CLI, Claude Code, Codex CLI, and custom logs by domain token "
            "overlap (default >= 40%). Returns top 3 matches with brief summaries so the assistant can ask user permission "
            "BEFORE calling recall_conversation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User prompt or topic keywords"},
                "threshold_pct": {"type": "number", "default": 40.0},
                "provider": {"type": "string", "default": "all"},
                "current_conversation_id": {"type": "string", "default": ""},
            },
            "required": ["query"],
        },
    },
    {
        "name": "recall_conversation",
        "description": "Retrieve full recent turns for a confirmed conversation_id returned by find_context (ask user permission first).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {"type": "string"},
                "max_turns": {"type": "integer", "default": 20},
            },
            "required": ["conversation_id"],
        },
    },
    {
        "name": "backup_status",
        "description": "Check incremental Google Drive / rclone backup status across configured AI CLI providers.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "sync_backup",
        "description": "Run a non-destructive WAL-safe incremental backup across configured AI CLI providers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean", "default": False},
                "force": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "reindex_providers",
        "description": "Refresh ~/.v1-memory/index.db from local Claude Code, Codex CLI, and custom JSONL/Markdown transcripts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def handle_rpc(req: dict[str, Any], home: Path | None = None, state_home: Path | None = None) -> dict[str, Any] | None:
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params") or {}
    resolved = default_state_home(state_home)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "v1-agent-memory", "version": "0.1.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "find_context":
            tokens, matches = find_context_matches(
                args.get("query", ""),
                current_conv_id=args.get("current_conversation_id", ""),
                threshold_pct=float(args.get("threshold_pct", 40.0)),
                provider_filter=args.get("provider", "all"),
                home=home,
                state_home=resolved,
            )
            payload = {"query_tokens": tokens, "matches": matches}
        elif name == "recall_conversation":
            loaded = load_conversation_turns(args["conversation_id"], int(args.get("max_turns", 20)), home=home, state_home=resolved)
            if not loaded:
                payload = {"error": f"Conversation not found: {args['conversation_id']}"}
            else:
                title, turns = loaded
                payload = {
                    "conversation_id": args["conversation_id"],
                    "title": title,
                    "turns": [{"role": r, "timestamp": ts, "content": txt} for r, ts, txt in turns],
                }
        elif name == "backup_status":
            payload = get_sync_status(state_home=resolved)
        elif name == "sync_backup":
            payload = run_sync(dry_run=bool(args.get("dry_run", False)), force=bool(args.get("force", False)), state_home=resolved)
        elif name == "reindex_providers":
            payload = {"indexed_conversations": rebuild_portable_index(resolved / "index.db")}
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {name}"}}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]},
        }

    if req_id is None:
        return None
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        resp = handle_rpc(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
