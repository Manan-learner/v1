"""Model-agnostic proactive context matcher (Antigravity PreInvocation, Claude UserPromptSubmit, Codex/Generic CLI)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from v1_memory.index_store import default_state_home, find_context_matches, rebuild_portable_index
from v1_memory.providers import parse_generic_jsonl
from v1_memory.text_utils import clean_snippet, clean_user_prompt, tokenize_domain

DEFAULT_CONFIG = {
    "enabled": True,
    "threshold_pct": 40.0,
    "min_matched_tokens": 2,
    "max_matches": 3,
    "start_of_convo_only": True,
    "max_initial_steps_for_start": 4,
    "auto_pivot_every_n_prompts": 5,
    "call_words": ["@context", "#context", "check context", "recall check", "find context"],
    "hybrid_rag": True,
    "semantic_weight": 0.45,
    "synonyms": {},
}


def load_config(state_home: Path | None = None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    resolved = default_state_home(state_home)
    for cand in [
        resolved / "context_finder_config.json",
        Path.home() / ".gemini" / "antigravity-cli" / "context_finder_config.json",
    ]:
        if cand.exists():
            try:
                cfg.update(json.loads(cand.read_text(encoding="utf-8")))
                break
            except Exception:
                pass
    return cfg


def extract_call_word(prompt: str, call_words: list[str]) -> tuple[bool, str]:
    lowered = prompt.lower()
    triggered = False
    cleaned = prompt
    for cw in call_words:
        if cw.lower() in lowered:
            triggered = True
            cleaned = re.sub(re.escape(cw), " ", cleaned, flags=re.IGNORECASE)
            lowered = cleaned.lower()
    return triggered, cleaned.strip()


def load_seen_state(seen_path: Path, conv_id: str) -> dict:
    default = {
        "ids": set(),
        "last_prompt_hash": "",
        "start_checked": False,
        "prompt_count": 0,
        "recent_prompts": [],
        "pending_prompts": [],
        "prev_window_tokens": [],
    }
    if not seen_path.exists():
        return default
    try:
        data = json.loads(seen_path.read_text(encoding="utf-8"))
        entry = data.get(conv_id, {})
        if isinstance(entry, list):
            return {
                "ids": set(entry),
                "last_prompt_hash": "",
                "start_checked": True,
                "prompt_count": 1,
                "recent_prompts": [],
                "pending_prompts": [],
                "prev_window_tokens": [],
            }
        return {
            "ids": set(entry.get("ids", [])),
            "last_prompt_hash": str(entry.get("last_prompt_hash", "")),
            "start_checked": bool(entry.get("start_checked", False)),
            "prompt_count": int(entry.get("prompt_count", 0)),
            "recent_prompts": [p for p in entry.get("recent_prompts", []) if isinstance(p, str)][-5:],
            "pending_prompts": [p for p in entry.get("pending_prompts", []) if isinstance(p, str)][-10:],
            "prev_window_tokens": [t for t in entry.get("prev_window_tokens", []) if isinstance(t, str)][-120:],
        }
    except Exception:
        return default


def save_seen_state(
    seen_path: Path,
    conv_id: str,
    new_ids: list[str],
    prompt_hash: str,
    start_checked: bool,
    prompt_count: int,
    recent_prompts: list[str],
    pending_prompts: list[str],
    prev_window_tokens: list[str],
) -> None:
    data: dict = {}
    if seen_path.exists():
        try:
            data = json.loads(seen_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    entry = data.get(conv_id, {})
    existing = set(entry if isinstance(entry, list) else entry.get("ids", []))
    prev_checked = False if isinstance(entry, list) else bool(entry.get("start_checked", False))
    existing.update(new_ids)
    if conv_id in data:
        del data[conv_id]
    data[conv_id] = {
        "ids": sorted(existing),
        "last_prompt_hash": prompt_hash,
        "start_checked": prev_checked or start_checked,
        "prompt_count": int(prompt_count),
        "recent_prompts": recent_prompts[-5:],
        "pending_prompts": pending_prompts[-10:],
        "prev_window_tokens": prev_window_tokens[-120:],
    }
    if len(data) > 50:
        for old_k in list(data.keys())[:-50]:
            del data[old_k]
    seen_path.parent.mkdir(parents=True, exist_ok=True)
    seen_path.write_text(json.dumps(data), encoding="utf-8")


def detect_batch_pivot(
    pending_prompts: list[str],
    prev_window_tokens: list[str],
    min_matched_tokens: int,
) -> tuple[bool, str, list[str]]:
    """Inspect ONLY the new tumbling batch of prompts against the prior baseline window."""
    batch_text = " ".join(p for p in pending_prompts if p.strip()).strip()
    batch_tokens = tokenize_domain(batch_text)
    if len(batch_tokens) < min_matched_tokens:
        return False, "", batch_tokens[-120:]

    prev_set = set(prev_window_tokens)
    if not prev_set:
        return True, batch_text, batch_tokens[-120:]

    batch_set = set(batch_tokens)
    overlap_ratio = len(batch_set & prev_set) / len(batch_set)
    novel_tokens = [t for t in batch_tokens if t not in prev_set]
    if (overlap_ratio < 0.35 and len(novel_tokens) >= min_matched_tokens) or len(novel_tokens) >= 4:
        query = " ".join(novel_tokens) if len(novel_tokens) >= min_matched_tokens else batch_text
        return True, query, batch_tokens[-120:]

    # Isolate a late shift (e.g., Turn 3 pivot + short Turn 4-5 follow-ups) inside the 5-prompt batch
    for split_idx in range(1, len(pending_prompts)):
        tail_text = " ".join(pending_prompts[split_idx:]).strip()
        tail_tokens = tokenize_domain(tail_text)
        if len(tail_tokens) < min_matched_tokens:
            continue
        head_set = set(tokenize_domain(" ".join(pending_prompts[:split_idx]))) | prev_set
        tail_set = set(tail_tokens)
        tail_novel = [t for t in tail_tokens if t not in head_set]
        tail_overlap = len(tail_set & head_set) / len(tail_set)
        if tail_overlap < 0.30 and len(tail_novel) >= min_matched_tokens:
            return True, " ".join(tail_novel), batch_tokens[-120:]

    return False, "", batch_tokens[-120:]


def format_match_banner(matches: list[dict], trigger_reason: str, threshold_pct: float) -> str:
    lines = [
        f"[Context-Finder Match >= {threshold_pct:g}% Detected ({trigger_reason})]",
        f"Top matching older conversation(s) (>= {threshold_pct:g}% domain token overlap):",
    ]
    for idx, m in enumerate(matches, 1):
        lines.append(
            f"{idx}. **{m['match_pct']}% match** | \"{m['title']}\" [{m['provider']}] ({m['last_modified_time']}, {m['step_count']} steps) | ID: `{m['conversation_id']}`\n"
            f"   - Matched tokens: {', '.join(m['matched_tokens'])}\n"
            f"   - Brief summary: {m['summary']}"
        )
    lines.extend(
        [
            "",
            "REQUIRED PERMISSION RULE: Before retrieving any older conversation's full transcript, briefly present the top matches above (including the brief summary for each so the user knows what they are accepting/denying) and ASK the user which one(s), if any, they want retrieved. Only run `v1-recall --show <conversation_id>` (or MCP `recall_conversation`) after explicit confirmation.",
        ]
    )
    return "\n".join(lines)


def detect_hook_format(payload: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    if "hookEventName" in payload and payload.get("hookEventName") == "PreInvocation":
        return "antigravity"
    if payload.get("hook_event_name") == "UserPromptSubmit" or ("prompt" in payload and "session_id" in payload):
        return "claude"
    if "lastUserInput" in payload:
        return "antigravity"
    return "generic"


def emit_hook_output(message: str | None, fmt: str) -> None:
    if not message:
        print("{}")
        return
    if fmt == "claude":
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": message,
                    }
                }
            )
        )
    elif fmt == "antigravity":
        print(json.dumps({"injectSteps": [{"systemMessage": {"systemMessage": message}}]}))
    elif fmt == "plain":
        print(message)
    else:
        print(json.dumps({"systemMessage": message}))


def run_hook_payload(
    payload: dict,
    fmt: str = "auto",
    provider_filter: str = "all",
    home: Path | None = None,
    state_home: Path | None = None,
    agy_db_path: Path | None = None,
    index_db_path: Path | None = None,
) -> tuple[str, str | None, list[dict]]:
    detected_fmt = detect_hook_format(payload, fmt)
    if int(payload.get("invocationNum", 1)) != 1:
        return detected_fmt, None, []

    raw_input = payload.get("lastUserInput") or payload.get("prompt") or ""
    prompt_raw = clean_user_prompt(str(raw_input)).strip()
    if not prompt_raw:
        return detected_fmt, None, []

    resolved_state = default_state_home(state_home)
    cfg = load_config(resolved_state)
    if not cfg.get("enabled", True):
        return detected_fmt, None, []

    conv_id = str(payload.get("conversationId") or payload.get("session_id") or "default-session")
    initial_steps = int(payload.get("initialNumSteps", 0))
    call_words = cfg.get("call_words", DEFAULT_CONFIG["call_words"])
    called_explicitly, prompt = extract_call_word(prompt_raw, call_words)

    seen_path = resolved_state / "context_finder_seen.json"
    state = load_seen_state(seen_path, conv_id)
    prompt_hash = hashlib.sha1(prompt_raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
    if prompt_hash == state["last_prompt_hash"]:
        return detected_fmt, None, []

    min_matched = int(cfg.get("min_matched_tokens", 2))
    threshold_pct = float(cfg.get("threshold_pct", 40.0))
    max_matches = int(cfg.get("max_matches", 3))
    auto_pivot_n = int(cfg.get("auto_pivot_every_n_prompts", 5))
    start_only = bool(cfg.get("start_of_convo_only", True))
    max_start_steps = int(cfg.get("max_initial_steps_for_start", 4))
    hybrid_rag = bool(cfg.get("hybrid_rag", True))
    semantic_weight = float(cfg.get("semantic_weight", 0.45))
    custom_synonyms = cfg.get("synonyms") if isinstance(cfg.get("synonyms"), dict) else {}

    new_prompt_count = state["prompt_count"] + 1
    curr_tokens = tokenize_domain(prompt)
    cleaned_entry = clean_snippet(prompt, 300)
    updated_recent = (state["recent_prompts"] + ([cleaned_entry] if cleaned_entry else []))[-5:]
    updated_pending = (state["pending_prompts"] + ([cleaned_entry] if cleaned_entry else []))[-10:]

    allowed_start_steps = max(max_start_steps, 6) if state["prompt_count"] >= 1 else max_start_steps
    is_start_of_convo = (not state["start_checked"]) and (initial_steps <= allowed_start_steps)
    is_nth_prompt = (auto_pivot_n > 0) and (new_prompt_count % auto_pivot_n == 0)

    if len(prompt_raw) < 12 and not called_explicitly:
        save_seen_state(
            seen_path, conv_id, [], prompt_hash, False, new_prompt_count, updated_recent, updated_pending, state["prev_window_tokens"]
        )
        return detected_fmt, None, []

    search_query = prompt
    trigger_reason = ""
    exclude_ids = set(state["ids"])
    lock_start = False
    new_baseline = list(state["prev_window_tokens"])
    reset_pending = False

    if called_explicitly:
        trigger_reason = "Call word triggered"
        exclude_ids = set()
        if len(curr_tokens) < min_matched:
            candidates: list[str] = []
            tpath = str(payload.get("transcriptPath") or payload.get("transcript_path") or "").strip()
            if tpath and Path(tpath).exists():
                rec = parse_generic_jsonl(Path(tpath), provider="transcript")
                for ut in rec.user_turns[-3:]:
                    _, stripped = extract_call_word(ut, call_words)
                    if stripped:
                        candidates.append(stripped)
            for rp in state["recent_prompts"][-3:]:
                _, stripped = extract_call_word(rp, call_words)
                if stripped:
                    candidates.append(stripped)
            fallback_text = " ".join(dict.fromkeys(candidates)).strip()
            if len(tokenize_domain(fallback_text)) < min_matched:
                save_seen_state(
                    seen_path, conv_id, [], prompt_hash, False, new_prompt_count, updated_recent, updated_pending, new_baseline
                )
                return (
                    detected_fmt,
                    "[Context-Finder Bare Call Word]\n"
                    "The user triggered context-finder without topic keywords, and no prior turns had enough domain tokens. "
                    "Briefly ask what topic or keywords they want to look up before running a search.",
                    [],
                )
            search_query = fallback_text
            trigger_reason = "Call word triggered (recent context)"
    elif is_start_of_convo:
        if len(curr_tokens) < min_matched:
            # Keep start_checked False so a turn-1 greeting ("hi there") defers to turn 2.
            save_seen_state(
                seen_path, conv_id, [], prompt_hash, False, new_prompt_count, updated_recent, updated_pending, new_baseline
            )
            return detected_fmt, None, []
        trigger_reason = "New conversation start"
        lock_start = True
        new_baseline = curr_tokens[-120:]
        reset_pending = True
    elif is_nth_prompt:
        is_pivot, pivot_query, rotated_baseline = detect_batch_pivot(updated_pending, state["prev_window_tokens"], min_matched)
        new_baseline = rotated_baseline
        reset_pending = True
        if not is_pivot:
            save_seen_state(
                seen_path, conv_id, [], prompt_hash, False, new_prompt_count, updated_recent, [], new_baseline
            )
            return detected_fmt, None, []
        search_query = pivot_query
        trigger_reason = f"Auto topic-pivot check (prompt #{new_prompt_count})"
    elif not start_only:
        trigger_reason = "Mid-conversation check"
    else:
        save_seen_state(
            seen_path, conv_id, [], prompt_hash, False, new_prompt_count, updated_recent, updated_pending, new_baseline
        )
        return detected_fmt, None, []

    _, matches = find_context_matches(
        search_query,
        current_conv_id=conv_id,
        threshold_pct=threshold_pct,
        min_matched_tokens=min_matched,
        max_matches=max_matches,
        exclude_ids=exclude_ids,
        provider_filter=provider_filter,
        home=home,
        state_home=resolved_state,
        agy_db_path=agy_db_path,
        index_db_path=index_db_path,
        hybrid_rag=hybrid_rag,
        semantic_weight=semantic_weight,
        custom_synonyms=custom_synonyms,
    )

    next_pending = [] if reset_pending else updated_pending
    if not matches:
        save_seen_state(
            seen_path, conv_id, [], prompt_hash, lock_start, new_prompt_count, updated_recent, next_pending, new_baseline
        )
        return detected_fmt, None, []

    save_seen_state(
        seen_path,
        conv_id,
        [m["conversation_id"] for m in matches],
        prompt_hash,
        lock_start,
        new_prompt_count,
        updated_recent,
        next_pending,
        new_baseline,
    )
    return detected_fmt, format_match_banner(matches, trigger_reason, threshold_pct), matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-LLM proactive conversation context matcher.")
    parser.add_argument("--test", metavar="PROMPT", help="Evaluate a prompt without writing hook state")
    parser.add_argument("--pivot", metavar="TOPIC", help="Check a mid-conversation topic pivot cleanly")
    parser.add_argument("--prompt", metavar="PROMPT", help="Direct non-hook prompt check (returns JSON)")
    parser.add_argument("--conv", default="", help="Current conversation/session ID to exclude")
    parser.add_argument("--threshold", type=float, default=40.0, help="Match percentage threshold (default: 40.0)")
    parser.add_argument("--provider", default="all", help="all, antigravity, claude, codex, generic")
    parser.add_argument("--format", default="auto", choices=["auto", "antigravity", "claude", "generic", "plain"])
    parser.add_argument("--reindex", action="store_true", help="Refresh ~/.v1-memory/index.db for non-Antigravity transcripts")
    args = parser.parse_args()

    state_home = default_state_home()
    cfg = load_config(state_home)
    syns = cfg.get("synonyms") if isinstance(cfg.get("synonyms"), dict) else {}
    if args.reindex:
        count = rebuild_portable_index(state_home / "index.db", custom_synonyms=syns)
        print(f"Indexed {count} non-Antigravity conversation(s) -> {state_home / 'index.db'}", file=sys.stderr)

    query_text = args.pivot or args.test or args.prompt
    if query_text is not None:
        tokens, matches = find_context_matches(
            query_text,
            current_conv_id=args.conv,
            threshold_pct=args.threshold,
            provider_filter=args.provider,
            state_home=state_home,
            hybrid_rag=bool(cfg.get("hybrid_rag", True)),
            semantic_weight=float(cfg.get("semantic_weight", 0.45)),
            custom_synonyms=syns,
        )
        print(json.dumps({"query_tokens": tokens, "matches": matches}, indent=2))
        return 0

    raw = sys.stdin.read()
    if not raw.strip():
        print("{}")
        return 0
    try:
        payload = json.loads(raw)
    except Exception:
        print("{}")
        return 0

    fmt, msg, _ = run_hook_payload(payload, fmt=args.format, provider_filter=args.provider, state_home=state_home)
    emit_hook_output(msg, fmt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
