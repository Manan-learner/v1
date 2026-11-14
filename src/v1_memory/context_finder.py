"""Model-agnostic proactive context matcher (Antigravity PreInvocation, Claude UserPromptSubmit, Codex/Generic CLI)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from v1_memory.hybrid_rag import build_concept_lookup, group_tokens_by_concept
from v1_memory.index_store import default_state_home, find_context_matches, rebuild_portable_index
from v1_memory.providers import _flatten_content
from v1_memory.text_utils import (
    clean_snippet,
    clean_user_prompt,
    ensure_utf8_stdio,
    stem_token,
    tokenize_domain,
)

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
                loaded = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    cfg.update(loaded)
                    break
            except Exception:
                pass
    return cfg


def extract_call_word(prompt: str, call_words: list[str]) -> tuple[bool, str]:
    """Match configured call words on strict token boundaries and strip them from prompt."""
    triggered = False
    cleaned = prompt or ""
    for cw in call_words or []:
        cw_clean = str(cw).strip()
        if not cw_clean:
            continue
        pattern = rf"(?<![a-z0-9_]){re.escape(cw_clean)}(?![a-z0-9_-])"
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            triggered = True
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return triggered, cleaned


def _extract_recent_user_prompts_from_transcript(
    tpath: Path,
    max_prompts: int = 3,
    tail_bytes: int = 131072,
) -> list[str]:
    """Read at most the last `tail_bytes` of a JSONL transcript to extract recent user prompts."""
    prompts: list[str] = []
    try:
        if not tpath.exists() or not tpath.is_file():
            return prompts
        size = tpath.stat().st_size
        with open(tpath, "rb") as f:
            if size > tail_bytes:
                f.seek(max(0, size - tail_bytes))
            raw_chunk = f.read(tail_bytes).decode("utf-8", errors="replace")
        for line in raw_chunk.splitlines():
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
            content = _flatten_content(obj.get("content", ""))
            if stype == "USER_INPUT" and src == "USER_EXPLICIT" and content:
                prompts.append(content)
                continue
            if stype == "user" and isinstance(obj.get("message"), dict):
                txt = _flatten_content(obj["message"].get("content", ""))
                if txt:
                    prompts.append(txt)
                continue
            payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
            if stype == "event_msg" and isinstance(payload, dict) and payload.get("type") == "user_message":
                txt = str(payload.get("message") or "").strip()
                if txt:
                    prompts.append(txt)
                continue
            role = str((payload.get("role") if isinstance(payload, dict) else None) or obj.get("role") or "").lower()
            if role in ("user", "human") and isinstance(payload, dict):
                txt = _flatten_content(payload.get("content") or payload.get("text") or "")
                if txt:
                    prompts.append(txt)
    except Exception:
        pass
    return prompts[-max_prompts:]


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
        if not isinstance(data, dict):
            return default
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
        if not isinstance(entry, dict):
            return default
        return {
            "ids": set(entry.get("ids", [])) if isinstance(entry.get("ids"), list) else set(),
            "last_prompt_hash": str(entry.get("last_prompt_hash", "")),
            "start_checked": bool(entry.get("start_checked", False)),
            "prompt_count": int(entry.get("prompt_count", 0) or 0),
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
    seen_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if seen_path.exists():
        for attempt in range(3):
            try:
                raw_text = seen_path.read_text(encoding="utf-8")
                if raw_text.strip():
                    loaded = json.loads(raw_text)
                    if isinstance(loaded, dict):
                        data = loaded
                break
            except Exception:
                if attempt < 2:
                    time.sleep(0.01)

    entry = data.get(conv_id, {})
    if isinstance(entry, list):
        existing = set(entry)
        prev_checked = False
    elif isinstance(entry, dict):
        existing = set(entry.get("ids", [])) if isinstance(entry.get("ids"), list) else set()
        prev_checked = bool(entry.get("start_checked", False))
    else:
        existing = set()
        prev_checked = False

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

    payload_str = json.dumps(data)
    tmp_name: str | None = None
    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(seen_path.parent),
            prefix="context_seen_",
            suffix=".tmp",
            delete=False,
        )
        tmp_name = tmp.name
        try:
            tmp.write(payload_str)
            tmp.flush()
        finally:
            tmp.close()

        for attempt in range(3):
            try:
                os.replace(tmp_name, seen_path)
                tmp_name = None
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.01)
                else:
                    raise
    except Exception:
        pass
    finally:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def detect_batch_pivot(
    pending_prompts: list[str],
    prev_window_tokens: list[str],
    min_matched_tokens: int,
    custom_synonyms: dict[str, list[str]] | None = None,
) -> tuple[bool, str, list[str]]:
    """Inspect ONLY the new tumbling batch of prompts against the prior baseline window."""
    prompt_token_lists = [tokenize_domain(p) for p in pending_prompts if isinstance(p, str) and p.strip()]
    prompt_token_lists = [toks for toks in prompt_token_lists if toks]

    seen_bt: dict[str, None] = {}
    for toks in prompt_token_lists:
        for t in toks:
            seen_bt[t] = None
    batch_tokens = list(seen_bt.keys())
    next_baseline = batch_tokens[-120:] if batch_tokens else list(prev_window_tokens[-120:])
    if len(batch_tokens) < min_matched_tokens:
        return False, "", next_baseline

    prev_set = set(prev_window_tokens)
    if not prev_set:
        return True, " ".join(batch_tokens), next_baseline

    prev_stems = {stem_token(t) for t in prev_set}
    prev_c_hits = group_tokens_by_concept(prev_set, custom_synonyms)
    prev_concepts = {
        cid for cid, hits in prev_c_hits.items() if len({stem_token(t) for t in hits}) >= 2
    }
    _, term_to_concepts = build_concept_lookup(custom_synonyms)

    def _is_aligned(tok: str, extra_stems: set[str], extra_concepts: set[str]) -> bool:
        st = stem_token(tok)
        if tok in prev_set or st in prev_stems or st in extra_stems:
            return True
        t_cids = term_to_concepts.get(tok, set()) | term_to_concepts.get(st, set())
        active_concepts = prev_concepts | extra_concepts
        return any(f"concept:{c}" in active_concepts for c in t_cids)

    # 1. Isolate a mid-batch shift (e.g., Turn 2 continues Turn 1 topic, then Turn 3-5 pivots to a new topic)
    for split_idx in range(1, len(prompt_token_lists)):
        head_tokens = list(dict.fromkeys(t for toks in prompt_token_lists[:split_idx] for t in toks))
        tail_tokens = list(dict.fromkeys(t for toks in prompt_token_lists[split_idx:] for t in toks))
        if len(tail_tokens) < min_matched_tokens or not head_tokens:
            continue
        head_aligned = sum(1 for t in head_tokens if _is_aligned(t, set(), set()))
        if (head_aligned / len(head_tokens)) >= 0.25:
            head_stems = {stem_token(t) for t in head_tokens}
            head_c_hits = group_tokens_by_concept(head_tokens, custom_synonyms)
            head_concepts = {
                cid for cid, hits in head_c_hits.items() if len({stem_token(t) for t in hits}) >= 2
            }
            tail_aligned = sum(1 for t in tail_tokens if _is_aligned(t, head_stems, head_concepts))
            tail_novel = [t for t in tail_tokens if not _is_aligned(t, head_stems, head_concepts)]
            if (tail_aligned / len(tail_tokens)) < 0.30 and len(tail_novel) >= min_matched_tokens:
                return True, " ".join(tail_novel), tail_novel[-120:]

    # 2. Whole-batch pivot check
    aligned_count = sum(1 for t in batch_tokens if _is_aligned(t, set(), set()))
    overlap_ratio = aligned_count / len(batch_tokens)
    novel_tokens = [t for t in batch_tokens if not _is_aligned(t, set(), set())]
    if overlap_ratio < 0.35 and len(novel_tokens) >= min_matched_tokens:
        return True, " ".join(novel_tokens), novel_tokens[-120:]

    return False, "", next_baseline


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
    if not isinstance(payload, dict):
        return "generic"
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
    if not isinstance(payload, dict):
        return ("generic" if fmt == "auto" else fmt), None, []

    detected_fmt = detect_hook_format(payload, fmt)
    raw_inv = payload.get("invocationNum", 1)
    try:
        inv_num = 1 if raw_inv is None else int(raw_inv)
    except (ValueError, TypeError):
        inv_num = 1
    if inv_num != 1:
        return detected_fmt, None, []

    raw_input = payload.get("lastUserInput") or payload.get("prompt") or ""
    prompt_raw = clean_user_prompt(str(raw_input)).strip()
    if not prompt_raw:
        return detected_fmt, None, []

    resolved_state = default_state_home(state_home)
    cfg = load_config(resolved_state)
    if not cfg.get("enabled", True):
        return detected_fmt, None, []

    conv_id = str(payload.get("conversationId") or payload.get("session_id") or "default-session").strip() or "default-session"
    raw_steps = payload.get("initialNumSteps", 0)
    try:
        initial_steps = 0 if raw_steps is None else int(raw_steps)
    except (ValueError, TypeError):
        initial_steps = 0

    call_words = cfg.get("call_words", DEFAULT_CONFIG["call_words"])
    if not isinstance(call_words, list):
        call_words = list(DEFAULT_CONFIG["call_words"])
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

    # Only skip short prompts when they are not call-word triggers, not Nth-prompt batch checks,
    # and have fewer than `min_matched` domain tokens (so 2-token prompts like "stm32 uart" still run).
    if (
        len(prompt_raw) < 12
        and len(curr_tokens) < min_matched
        and not called_explicitly
        and not is_nth_prompt
    ):
        save_seen_state(
            seen_path,
            conv_id,
            [],
            prompt_hash,
            False,
            new_prompt_count,
            updated_recent,
            updated_pending,
            state["prev_window_tokens"],
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
        lock_start = True
        if len(curr_tokens) < min_matched:
            candidates: list[str] = []
            tpath_str = str(payload.get("transcriptPath") or payload.get("transcript_path") or "").strip()
            if tpath_str:
                for ut in _extract_recent_user_prompts_from_transcript(Path(tpath_str), max_prompts=3, tail_bytes=131072):
                    _, stripped = extract_call_word(ut, call_words)
                    if stripped:
                        candidates.append(stripped)
            for rp in state["recent_prompts"][-3:]:
                _, stripped = extract_call_word(rp, call_words)
                if stripped:
                    candidates.append(stripped)
            fallback_text = " ".join(dict.fromkeys(candidates)).strip()
            fb_tokens = tokenize_domain(fallback_text)
            if len(fb_tokens) < min_matched:
                save_seen_state(
                    seen_path,
                    conv_id,
                    [],
                    prompt_hash,
                    True,
                    new_prompt_count,
                    updated_recent,
                    updated_pending,
                    new_baseline,
                )
                return (
                    detected_fmt,
                    "[Context-Finder Bare Call Word]\n"
                    "The user triggered context-finder without topic keywords, and no prior turns had enough domain tokens. "
                    "Briefly ask what topic or keywords they want to look up before running a search.",
                    [],
                )
            search_query = fallback_text
            if not new_baseline:
                new_baseline = fb_tokens[-120:]
            trigger_reason = "Call word triggered (recent context)"
        else:
            if not new_baseline:
                new_baseline = curr_tokens[-120:]
    elif is_start_of_convo:
        if len(curr_tokens) < min_matched:
            # Keep start_checked False so a turn-1 greeting ("hi there") defers to turn 2.
            save_seen_state(
                seen_path,
                conv_id,
                [],
                prompt_hash,
                False,
                new_prompt_count,
                updated_recent,
                updated_pending,
                new_baseline,
            )
            return detected_fmt, None, []
        trigger_reason = "New conversation start"
        lock_start = True
        new_baseline = curr_tokens[-120:]
        reset_pending = True
    elif is_nth_prompt:
        is_pivot, pivot_query, rotated_baseline = detect_batch_pivot(
            updated_pending,
            state["prev_window_tokens"],
            min_matched,
            custom_synonyms=custom_synonyms,
        )
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
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Multi-LLM proactive conversation context matcher.")
    parser.add_argument("--test", metavar="PROMPT", help="Evaluate a prompt without writing hook state")
    parser.add_argument("--pivot", metavar="TOPIC", help="Check a mid-conversation topic pivot cleanly")
    parser.add_argument("--prompt", metavar="PROMPT", help="Direct non-hook prompt check (returns JSON)")
    parser.add_argument("--conv", default="", help="Current conversation/session ID to exclude")
    parser.add_argument("--threshold", type=float, default=None, help="Match percentage threshold (default: from config or 40.0)")
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
        eff_threshold = float(args.threshold) if args.threshold is not None else float(cfg.get("threshold_pct", 40.0))
        tokens, matches = find_context_matches(
            query_text,
            current_conv_id=args.conv,
            threshold_pct=eff_threshold,
            min_matched_tokens=int(cfg.get("min_matched_tokens", 2)),
            max_matches=int(cfg.get("max_matches", 3)),
            provider_filter=args.provider,
            state_home=state_home,
            hybrid_rag=bool(cfg.get("hybrid_rag", True)),
            semantic_weight=float(cfg.get("semantic_weight", 0.45)),
            custom_synonyms=syns,
        )
        print(json.dumps({"query_tokens": tokens, "matches": matches}, indent=2))
        return 0

    try:
        raw = sys.stdin.read()
    except Exception:
        print("{}")
        return 0
    if not raw.strip():
        print("{}")
        return 0
    try:
        payload = json.loads(raw)
    except Exception:
        print("{}")
        return 0

    try:
        fmt, msg, _ = run_hook_payload(payload, fmt=args.format, provider_filter=args.provider, state_home=state_home)
        emit_hook_output(msg, fmt)
    except Exception:
        print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
