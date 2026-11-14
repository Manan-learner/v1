"""Zero-dependency stdlib MCP stdio server for ANY MCP-capable LLM (Claude, Cursor, Windsurf, Codex, Gemini)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from v1_memory.cloud_sync import get_sync_status, run_sync
from v1_memory.context_finder import load_config
from v1_memory.index_store import default_state_home, find_context_matches, rebuild_portable_index
from v1_memory.recall import load_conversation_turns
from v1_memory.text_utils import ensure_utf8_stdio

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
    if not isinstance(req, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}

    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params") if isinstance(req.get("params"), dict) else {}
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
        raw_args = params.get("arguments")
        if raw_args is None:
            args: dict[str, Any] = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params: 'arguments' must be an object"},
            }

        cfg = load_config(resolved)
        syns = cfg.get("synonyms") if isinstance(cfg.get("synonyms"), dict) else None

        try:
            if name == "find_context":
                if "query" not in args or not isinstance(args.get("query"), str):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Invalid params: missing required string 'query'"},
                    }
                eff_threshold = (
                    float(args["threshold_pct"])
                    if args.get("threshold_pct") is not None
                    else float(cfg.get("threshold_pct", 40.0))
                )
                eff_min_matched = (
                    int(args["min_matched_tokens"])
                    if args.get("min_matched_tokens") is not None
                    else int(cfg.get("min_matched_tokens", 2))
                )
                eff_max_matches = (
                    int(args["max_matches"])
                    if args.get("max_matches") is not None
                    else int(cfg.get("max_matches", 3))
                )
                eff_hybrid = (
                    bool(args["hybrid_rag"])
                    if args.get("hybrid_rag") is not None
                    else bool(cfg.get("hybrid_rag", True))
                )
                eff_sem_weight = (
                    float(args["semantic_weight"])
                    if args.get("semantic_weight") is not None
                    else float(cfg.get("semantic_weight", 0.45))
                )
                tokens, matches = find_context_matches(
                    args["query"],
                    current_conv_id=str(args.get("current_conversation_id") or ""),
                    threshold_pct=eff_threshold,
                    min_matched_tokens=eff_min_matched,
                    max_matches=eff_max_matches,
                    provider_filter=str(args.get("provider") or "all"),
                    home=home,
                    state_home=resolved,
                    hybrid_rag=eff_hybrid,
                    semantic_weight=eff_sem_weight,
                    custom_synonyms=syns,
                )
                payload: dict[str, Any] = {"query_tokens": tokens, "matches": matches}
            elif name == "recall_conversation":
                conv_id = args.get("conversation_id")
                if not isinstance(conv_id, str) or not conv_id.strip():
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Invalid params: missing required 'conversation_id'"},
                    }
                max_turns = int(args["max_turns"]) if args.get("max_turns") is not None else 20
                loaded = load_conversation_turns(conv_id, max_turns=max_turns, home=home, state_home=resolved)
                if not loaded:
                    payload = {"error": f"Conversation not found: {conv_id}"}
                else:
                    title, turns = loaded
                    payload = {
                        "conversation_id": conv_id,
                        "title": title,
                        "turns": [{"role": r, "timestamp": ts, "content": txt} for r, ts, txt in turns],
                    }
            elif name == "backup_status":
                payload = get_sync_status(home=home, state_home=resolved)
            elif name == "sync_backup":
                payload = run_sync(
                    dry_run=bool(args.get("dry_run", False)),
                    force=bool(args.get("force", False)),
                    home=home,
                    state_home=resolved,
                )
            elif name == "reindex_providers":
                payload = {
                    "indexed_conversations": rebuild_portable_index(
                        resolved / "index.db",
                        home=home,
                        custom_synonyms=syns,
                    )
                }
            else:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {name}"}}
        except (KeyError, TypeError, ValueError) as exc:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Invalid params: {exc}"}}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": f"Tool execution error: {exc}"}}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]},
        }

    if req_id is None:
        return None
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main() -> int:
    ensure_utf8_stdio()
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
