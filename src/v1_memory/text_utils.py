"""Shared domain tokenization, tag stripping, stopword filtering, and snippet helpers."""

from __future__ import annotations

import re

STOPWORDS = {
    "a", "about", "above", "accept", "accepted", "accepting", "actual", "actually",
    "add", "added", "adding", "after", "again", "against", "agent", "agents", "all",
    "almost", "already", "also", "always", "am", "an", "and", "another", "any",
    "anything", "app", "apps", "are", "aren", "around", "as", "ask", "asked",
    "asking", "at", "basically", "be", "because", "been", "before", "being",
    "below", "best", "better", "between", "both", "branch", "branches", "bug",
    "bugs", "build", "but", "by", "call", "called", "calling", "can", "cant",
    "change", "changed", "changes", "chat", "check", "checked", "checking", "cli",
    "code", "coding", "come", "config", "context", "convo", "convos",
    "conversation", "conversations", "could", "create", "created", "current",
    "data", "day", "days", "debug", "debugging", "denied", "deny", "denying", "dev", "did", "different",
    "dir", "do", "does", "doing", "done", "dont", "down", "during", "each", "edit",
    "else", "enough", "error", "etc", "even", "every", "everything", "example",
    "explain", "feature", "feel", "file", "files", "find", "first", "fix", "flag",
    "flags", "folder", "for", "from", "further", "gave", "get", "getting", "give",
    "given", "giving", "go", "going", "good", "got", "great", "had", "has", "have",
    "having", "he", "help", "her", "here", "hers", "him", "his", "how", "i",
    "idea", "ideas", "if", "in", "include", "included", "includes", "including",
    "info", "input", "into", "is", "issue", "it", "its", "job", "jobs", "json",
    "just", "keep", "kind", "know", "last", "least", "less", "let", "lets",
    "like", "likely", "limit", "limited", "line", "lines", "list", "little",
    "look", "looking", "looks", "lot", "made", "major", "make", "making", "many",
    "match", "matched", "matches", "matching", "may", "maybe", "me", "might",
    "mine", "minor", "mode", "monday", "month", "months", "more", "most", "much",
    "must", "my", "name", "need", "needed", "needs", "never", "new", "next",
    "nice", "no", "not", "nothing", "now", "number", "of", "off", "ok", "okay",
    "old", "older", "on", "once", "one", "only", "option", "options", "or",
    "other", "our", "out", "output", "over", "own", "part", "past", "path",
    "perhaps", "person", "pivot", "please", "point", "previous", "probably",
    "problem", "project", "put", "python", "question", "quick", "quite", "rather",
    "read", "really", "recent", "related", "remember", "repo", "repos", "right",
    "run", "running", "said", "same", "say", "saying", "script", "scripts", "see",
    "seeing", "seems", "seen", "sees", "send", "sense", "sensing", "sent", "set",
    "several", "she", "should", "show", "showing", "shows", "similar", "simple",
    "since", "skill", "skills", "so", "some", "someone", "something", "sometimes",
    "sorry", "sort", "start", "started", "step", "steps", "still", "such",
    "summaries", "summary", "sure", "system", "systems", "take", "taking", "talk",
    "talked", "talking", "tell", "telling", "test", "testing", "text", "than",
    "thank", "thanks", "that", "the", "their", "them", "then", "there", "these",
    "they", "thing", "things", "think", "thinking", "this", "those", "though",
    "thought", "three", "through", "time", "times", "to", "today", "together",
    "told", "tomorrow", "too", "tool", "tools", "topic", "tried", "try", "trying",
    "two", "under", "understand", "until", "up", "update", "updated", "us", "use",
    "used", "useful", "using", "user", "users", "usually", "very", "want",
    "wanted", "wanting", "wants", "was", "way", "ways", "we", "week", "weeks",
    "well", "were", "what", "whatever", "when", "where", "whether", "which",
    "while", "who", "whom", "why", "will", "with", "within", "without", "work",
    "worked", "working", "works", "would", "write", "writing", "written", "wrote",
    "yeah", "yes", "yesterday", "yet", "you", "your", "yours",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{1,}", re.IGNORECASE)
ALNUM_SPLIT_RE = re.compile(r"^([a-z]+)(\d+)$|^(\d+)([a-z]+)$", re.IGNORECASE)


def clean_user_prompt(text: str) -> str:
    """Strip common XML/system wrapper tags emitted by agent harnesses."""
    text = re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", " ", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<system-reminder>.*?</system-reminder>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?(?:USER_REQUEST|USER_SETTINGS_CHANGE)>", " ", text, flags=re.IGNORECASE)
    return text


def tokenize_domain(text: str) -> list[str]:
    """Extract ordered unique domain tokens; avoid bare <=2-digit numeric split noise."""
    cleaned_text = clean_user_prompt(text)
    seen: dict[str, None] = {}
    for raw in TOKEN_RE.findall(cleaned_text.lower()):
        cleaned = raw.strip("._-")
        if not cleaned or len(cleaned) < 2:
            continue
        if cleaned not in STOPWORDS:
            seen[cleaned] = None
        m = ALNUM_SPLIT_RE.match(cleaned)
        if m:
            alpha = m.group(1) or m.group(4) or ""
            digits = m.group(2) or m.group(3) or ""
            # Only split combined tokens when both halves are informative (e.g. 915mhz -> 915, mhz; keep stm32 whole).
            if len(alpha) >= 3 and len(digits) >= 3:
                if digits not in STOPWORDS:
                    seen[digits] = None
                if alpha not in STOPWORDS:
                    seen[alpha] = None
    return list(seen.keys())


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
