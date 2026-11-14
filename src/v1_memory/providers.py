"""Multi-LLM provider discovery & transcript extraction (Antigravity, Claude Code, Codex CLI, Generic)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class ConversationRecord:
    conversation_id: str
    provider: str
    title: str
    last_modified_time: str
    step_count: int
    workspace: str
    user_turns: list[str] = field(default_factory=list)
    agent_turns: list[str] = field(default_factory=list)
    transcript_path: str = ""
    raw_turns: list[tuple[str, str, str]] = field(default_factory=list)


def default_provider_roots(home: Path | None = None) -> dict[str, Path]:
    base = home or Path.home()
    return {
        "antigravity": base / ".gemini",
        "claude": base / ".claude",
        "codex": base / ".codex",
    }


def detect_active_providers(home: Path | None = None) -> list[str]:
    roots = default_provider_roots(home)
    return [name for name, root in roots.items() if root.exists()]


def _iso_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return ""


def _flatten_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("input_text") or item.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(p.strip() for p in parts if p and p.strip())
    if isinstance(content, dict):
        txt = content.get("text") or content.get("content")
        if isinstance(txt, str):
            return txt.strip()
    return ""


def parse_generic_jsonl(path: Path, provider: str = "generic", conv_id: str | None = None) -> ConversationRecord:
    """Parse Antigravity, Claude Code, Codex CLI, or generic role/content JSONL files cleanly."""
    users: list[str] = []
    agents: list[str] = []
    raw_turns: list[tuple[str, str, str]] = []
    title = ""
    workspace = ""
    total_lines = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            ts = str(obj.get("created_at") or obj.get("timestamp") or "")
            if not workspace and isinstance(obj.get("cwd"), str):
                workspace = obj["cwd"]
            if not title and isinstance(obj.get("title"), str):
                title = obj["title"].strip()

            # Format 1: Antigravity CLI transcript.jsonl
            stype = obj.get("type", "")
            src = obj.get("source", "")
            content = _flatten_content(obj.get("content", ""))
            if stype == "USER_INPUT" and src == "USER_EXPLICIT" and content:
                users.append(content)
                raw_turns.append(("USER", ts, content))
                continue
            if stype == "PLANNER_RESPONSE" and content and not obj.get("tool_calls"):
                agents.append(content)
                raw_turns.append(("ASSISTANT", ts, content[:1500]))
                continue

            # Format 2: Claude Code JSONL (`type`: `user`/`assistant`, nested `message.content`)
            if stype in ("user", "assistant") and isinstance(obj.get("message"), dict):
                msg = obj["message"]
                text = _flatten_content(msg.get("content", ""))
                role = msg.get("role") or stype
                if text:
                    if role == "user":
                        users.append(text)
                        raw_turns.append(("USER", ts, text))
                    elif role == "assistant":
                        agents.append(text)
                        raw_turns.append(("ASSISTANT", ts, text[:1500]))
                continue

            # Format 3: Codex CLI (`type`: `response_item` / `event_msg`) or OpenAI `role`/`content`
            payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
            role = str(payload.get("role") or obj.get("role") or "").lower()
            if stype == "event_msg" and isinstance(payload, dict) and payload.get("type") == "user_message":
                txt = str(payload.get("message") or "").strip()
                if txt:
                    users.append(txt)
                    raw_turns.append(("USER", ts, txt))
                continue
            if role in ("user", "human"):
                txt = _flatten_content(payload.get("content") or payload.get("text") or "")
                if txt:
                    users.append(txt)
                    raw_turns.append(("USER", ts, txt))
            elif role in ("assistant", "model"):
                txt = _flatten_content(payload.get("content") or payload.get("text") or "")
                if txt:
                    agents.append(txt)
                    raw_turns.append(("ASSISTANT", ts, txt[:1500]))

    if not title:
        title = (users[0][:80].replace("\n", " ").strip() if users else path.stem)
    cid = conv_id or f"{provider}:{path.stem}"
    return ConversationRecord(
        conversation_id=cid,
        provider=provider,
        title=title or "(Untitled Conversation)",
        last_modified_time=_iso_mtime(path),
        step_count=max(total_lines, len(raw_turns)),
        workspace=workspace,
        user_turns=users,
        agent_turns=agents,
        transcript_path=str(path),
        raw_turns=raw_turns,
    )


def parse_markdown_chat(path: Path, provider: str = "generic") -> ConversationRecord:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    first_heading = ""
    for line in text.splitlines():
        if line.strip().startswith("#"):
            first_heading = line.strip().lstrip("#").strip()
            break
    cid = f"{provider}:{path.stem}"
    return ConversationRecord(
        conversation_id=cid,
        provider=provider,
        title=first_heading or path.stem,
        last_modified_time=_iso_mtime(path),
        step_count=max(1, len(text.splitlines())),
        workspace=str(path.parent),
        user_turns=[text[:4000]] if text else [],
        agent_turns=[],
        transcript_path=str(path),
        raw_turns=[("NOTE", _iso_mtime(path), text[:4000])] if text else [],
    )


def iter_provider_conversations(
    providers: Iterable[str] | None = None,
    home: Path | None = None,
    custom_dirs: Iterable[Path] | None = None,
) -> list[ConversationRecord]:
    """Discover transcripts across Claude Code, Codex CLI, Antigravity transcripts, and custom directories."""
    roots = default_provider_roots(home)
    selected = list(providers) if providers else detect_active_providers(home)
    records: list[ConversationRecord] = []

    if "claude" in selected:
        proj_dir = roots["claude"] / "projects"
        if proj_dir.exists():
            for jfile in sorted(proj_dir.rglob("*.jsonl")):
                records.append(parse_generic_jsonl(jfile, provider="claude", conv_id=f"claude:{jfile.stem}"))

    if "codex" in selected:
        sess_dir = roots["codex"] / "sessions"
        if sess_dir.exists():
            for jfile in sorted(sess_dir.rglob("*.jsonl")):
                records.append(parse_generic_jsonl(jfile, provider="codex", conv_id=f"codex:{jfile.stem}"))

    for cdir in custom_dirs or []:
        cpath = Path(cdir).expanduser()
        if not cpath.exists():
            continue
        for p in sorted(cpath.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() == ".jsonl":
                records.append(parse_generic_jsonl(p, provider="generic", conv_id=f"generic:{p.stem}"))
            elif p.suffix.lower() == ".md":
                records.append(parse_markdown_chat(p, provider="generic"))

    return records
