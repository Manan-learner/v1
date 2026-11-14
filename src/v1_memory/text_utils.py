"""Shared domain tokenization, tag stripping, stopword filtering, and snippet helpers."""

from __future__ import annotations

import re
import sys
from typing import Sequence

STOPWORDS = {
    # Core English glue, pronouns, auxiliaries, prepositions, time & generic verbs
    "a", "about", "above", "accept", "accepted", "accepting", "actual", "actually",
    "add", "added", "adding", "after", "again", "against", "agent", "agents", "all",
    "almost", "already", "also", "always", "am", "an", "and", "another", "any",
    "anything", "app", "apps", "are", "around", "as", "ask", "asked",
    "asking", "at", "auto", "basically", "be", "because", "been", "before", "being",
    "below", "best", "better", "between", "both", "bug",
    "bugs", "build", "built", "but", "by", "call", "called", "calling", "can",
    "change", "changed", "changes", "chat", "check", "checked", "checking", "choose",
    "code", "coding", "come", "config", "context", "control", "convo", "convos",
    "conversation", "conversations", "could", "create", "created", "current",
    "data", "day", "days", "debug", "debugging", "default", "denied", "deny", "denying",
    "dev", "did", "different", "dir", "do", "doc", "docs", "does", "doing", "done",
    "down", "during", "each", "edit", "else", "enough", "error", "etc", "even",
    "every", "everything", "example", "explain", "feature", "feel", "file", "files",
    "find", "fix", "fixed", "flag", "flags", "folder", "for", "friday", "from",
    "further", "gave", "get", "getting", "give", "given", "giving", "go", "going",
    "good", "got", "great", "had", "has", "have", "having", "he", "hello", "help",
    "her", "here", "hers", "hey", "hi", "him", "his", "how", "i", "idea", "ideas",
    "if", "in", "include", "included", "includes", "including", "info", "input",
    "into", "is", "issue", "issues", "it", "its", "job", "jobs", "just", "keep",
    "kind", "know", "larger", "last", "least", "less", "let", "lets", "like",
    "likely", "limit", "limited", "line", "lines", "list", "little", "llm", "llms",
    "look", "looking", "looks", "lot", "made", "major", "make", "making", "manage",
    "many", "match", "matched", "matches", "matching", "may", "maybe", "me", "might",
    "mine", "minor", "mode", "model", "models", "monday", "month", "months", "more",
    "most", "much", "must", "my", "name", "need", "needed", "needs", "never", "new",
    "next", "nice", "no", "not", "nothing", "now", "number", "of", "off", "ok",
    "okay", "old", "older", "on", "once", "one", "only", "option", "options", "or",
    "orchestrator", "other", "our", "out", "output", "over", "own", "part", "past",
    "path", "perhaps", "permission", "person", "pivot", "please", "point",
    "previous", "probably", "project", "prompt", "prompts", "put", "python",
    "question", "quick", "quite", "rather", "read", "really", "recent", "related",
    "remember", "repo", "repos", "run", "running", "said", "same", "saturday", "say",
    "saying", "script", "scripts", "see", "seeing", "seems", "seen", "sees",
    "select", "selection", "selector", "send", "sense", "sensing", "sent", "set",
    "several", "she", "should", "show", "showing", "shows", "similar", "simple",
    "since", "skill", "skills", "so", "some", "someone", "something", "sometimes",
    "sorry", "sort", "start", "started", "step", "steps", "still", "subagent",
    "subagents", "such", "sunday", "sure", "switched", "system", "systems", "take",
    "taking", "talk", "talked", "talking", "tell", "telling", "terms", "test",
    "testing", "text", "than", "thank", "thanks", "that", "the", "their", "them",
    "then", "there", "these", "they", "thing", "things", "think", "thinking", "this",
    "those", "though", "thought", "three", "through", "thursday", "time", "times",
    "to", "today", "together", "told", "tomorrow", "too", "tool", "tools", "topic",
    "tried", "try", "trying", "tuesday", "two", "under", "understand", "until", "up",
    "update", "updated", "us", "use", "used", "useful", "using", "user", "users",
    "usually", "very", "want", "wanted", "wanting", "wants", "was", "way", "ways",
    "we", "wednesday", "week", "weeks", "well", "were", "what", "whatever", "when",
    "where", "whether", "which", "while", "who", "whom", "why", "will", "with",
    "within", "without", "work", "worked", "working", "works", "would", "write",
    "writing", "written", "wrote", "yeah", "yes", "yesterday", "yet", "you", "your",
    "yours",

    # Contraction & auxiliary fragments
    "ain", "aren", "cant", "couldn", "didn", "doesn", "don", "dont", "hadn", "hasn",
    "haven", "isn", "mustn", "needn", "shan", "shouldn", "ve",
    "wasn", "weren", "won", "wont", "wouldn",

    # Diagnostic, meta-conversational & evaluation words
    "problem", "problems", "expect", "expected", "expecting", "expects",
    "locate", "located", "locating", "location", "locations",
    "concern", "concerns", "concerned", "demonstrate", "demonstration",
    "demonstrations", "demo", "demos", "verify", "verified", "verifying",
    "verification", "ready", "complete", "completed", "completing",
    "completely", "completion", "exact", "exactly", "direct", "directly",
    "inside", "across", "instead", "immediate", "immediately", "automatic",
    "automatically", "implement", "implemented", "implementing",
    "implementation", "implementations", "status", "statuses", "summary",
    "summaries", "summarize", "overview", "detail", "details", "detailed",
    "result", "results", "resulting", "response", "responses", "respond",
    "note", "notes", "noted", "via", "vs", "versus", "e.g", "i.e", "eg", "ie",
    "behavior", "behaviour", "behaviors", "behaviours", "review", "reviewed",
    "reviewing", "reviews", "provide", "provided", "provides", "providing",
    "content", "contents", "document", "documents", "documentation",

    # High-DF generic software-engineering, markdown & workflow boilerplate
    "tests", "tested", "full", "fully", "uses", "runs", "ran", "local", "locally",
    "back", "live", "state", "states", "type", "types", "typed", "updates",
    "updating", "zero", "top", "bottom", "left", "right", "command",
    "commands", "src", "source", "sources", "latest", "connect", "connected",
    "connecting", "clean", "cleaned", "cleaning", "standard", "standards",
    "pass", "passed", "passing", "passes", "global", "globally", "true",
    "false", "task", "tasks", "checks", "core", "click", "clicked", "100",
    "open", "opened", "opening", "opens", "miss", "missed", "missing",
    "base", "based", "whenever", "separate", "separated", "separately",
    "apply", "applied", "applying", "applies", "block", "blocked",
    "blocking", "blocks", "target", "targets", "targeted", "root", "entire",
    "entirely", "small", "smaller", "smallest", "single", "multiple",
    "folders", "directory", "directories", "paths", "value",
    "values", "item", "items", "entry", "entries", "level", "levels",
    "support", "supported", "supporting", "supports", "allow", "allowed",
    "allowing", "allows", "ensure", "ensured", "ensuring", "ensures",
    "section", "sections", "page", "pages", "total", "count", "counts",
    "order", "ordered", "first", "second", "third", "final", "finally",
    "initial", "initially", "original", "originally", "properly", "correctly",
    "success", "successfully", "failed", "failing", "failure", "failures",
    "errors", "warning", "warnings", "reason", "reasons", "cause", "caused",
    "causes", "case", "cases", "place", "placed", "places", "points",
    "pointed", "side", "sides", "end", "ended", "ending", "ends", "begin",
    "began", "beginning", "continue", "continued", "continuing", "stop",
    "stopped", "stopping", "wait", "waited", "waiting", "move", "moved",
    "moving", "turn", "turned", "turning", "turns", "return", "returned",
    "returning", "returns", "handle", "handled", "handles", "handling",
    "process", "processed", "processes", "processing", "exist", "existed",
    "existing", "exists", "available", "require", "required", "requires",
    "requiring", "specific", "specifically", "general", "generally",
    "common", "commonly", "normal", "normally", "currently", "previously",
    "prior", "active", "actively", "clear", "clearly", "easy", "easily",
    "fast", "faster", "slow", "high", "higher", "low", "lower", "large",
    "short", "long", "bad", "fine", "wrong", "correct", "possible", "able",
    "unable", "certain", "real", "simply", "basic", "main", "mainly",
    "primary", "empty", "extra", "additional", "various", "particular",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{1,}", re.IGNORECASE)
ALNUM_SPLIT_RE = re.compile(r"([a-z]+)(\d+)|(\d+)([a-z]+)", re.IGNORECASE)
_CONTRACTION_SUFFIX_RE = re.compile(r"(?<=[a-z])['\u2019](?:re|ll|ve)\b", re.IGNORECASE)
_NO_STEM_TOKENS = frozenset(
    {
        "rtos",
        "freertos",
        "https",
        "postgres",
        "kubernetes",
        "prometheus",
        "macos",
        "bios",
        "cmos",
        "ddos",
        "nixos",
        "centos",
        "devops",
        "mlops",
        "gitops",
        "keras",
        "pandas",
    }
)


def ensure_utf8_stdio() -> None:
    """Safely reconfigure stdio streams to UTF-8 with replacement on non-UTF8 consoles."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def stem_token(tok: str) -> str:
    """Normalize basic English plural inflections while keeping short codes and tech terms intact."""
    t = tok.lower().strip("._-")
    if len(t) <= 3 or t in _NO_STEM_TOKENS:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("ches", "shes", "xes", "zes", "sses")) and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def clean_user_prompt(text: str) -> str:
    """Strip common XML/system wrapper tags emitted by agent harnesses."""
    text = re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", " ", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<system-reminder>.*?</system-reminder>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?(?:USER_REQUEST|USER_SETTINGS_CHANGE)>", " ", text, flags=re.IGNORECASE)
    return text


def split_alnum_parts(token: str) -> list[str]:
    """Split mixed alpha+digit runs inside a token when both halves have length >= 3."""
    parts: list[str] = []
    for subpart in re.split(r"[_.-]+", token):
        for m in ALNUM_SPLIT_RE.finditer(subpart):
            g1, g2, g3, g4 = m.groups()
            pair = (g1, g2) if g1 is not None else (g3, g4)
            if pair[0] and pair[1] and len(pair[0]) >= 3 and len(pair[1]) >= 3:
                parts.extend(pair)
    return parts


def tokenize_domain(text: str) -> list[str]:
    """Extract ordered unique domain tokens; drop stopwords, <=2-digit bare numbers, and single-char abbreviations."""
    cleaned_text = _CONTRACTION_SUFFIX_RE.sub("", clean_user_prompt(text))
    seen: dict[str, None] = {}
    for raw in TOKEN_RE.findall(cleaned_text.lower()):
        cleaned = raw.strip("._-")
        if not cleaned or len(cleaned) < 2:
            continue
        if cleaned.isdigit() and len(cleaned) <= 2:
            continue
        if "." in cleaned and all(len(p) <= 1 for p in cleaned.split(".") if p):
            continue
        if cleaned not in STOPWORDS:
            seen[cleaned] = None
        if "-" in cleaned or "." in cleaned:
            for sub in re.split(r"[-.]+", cleaned):
                sub = sub.strip("_")
                if len(sub) < 2 or (sub.isdigit() and len(sub) <= 2):
                    continue
                if sub not in STOPWORDS:
                    seen[sub] = None
        # Only split combined tokens when both halves are informative (e.g. 915mhz -> 915, mhz; keep stm32 whole).
        for grp in split_alnum_parts(cleaned):
            if grp not in STOPWORDS:
                seen[grp] = None
    return list(seen.keys())


def fts5_terms_for_tokens(tokens: Sequence[str], max_terms: int = 60) -> list[str]:
    """Split domain tokens on non-identifier delimiters into deduplicated FTS5-safe terms."""
    out: dict[str, None] = {}
    for tok in tokens:
        t = str(tok).lower().strip("._-")
        if not t:
            continue
        for part in re.split(r"[^a-z0-9_]+", t):
            if len(part) < 2 or part in STOPWORDS:
                continue
            if part.isdigit() and len(part) <= 2:
                continue
            out[part] = None
            if len(out) >= max_terms:
                return list(out.keys())
    return list(out.keys())


def clean_snippet(raw: str, max_len: int = 180) -> str:
    txt = clean_user_prompt(raw or "")
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    if len(txt) > max_len:
        return txt[: max_len - 3].rstrip() + "..."
    return txt


def build_summary_line(user_part: str, agent_part: str, fallback: str = "") -> str:
    u_snip = clean_snippet(user_part, 110)
    a_snip = clean_snippet(agent_part, 110)
    if u_snip and a_snip and u_snip != a_snip:
        return f'User asked: "{u_snip}" | Outcome: "{a_snip}"'
    return u_snip or a_snip or clean_snippet(fallback, 160)
