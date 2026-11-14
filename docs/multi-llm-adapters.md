# Multi-LLM Integration Guide

`v1` is built around a shared local FTS5 index (`~/.v1-memory/index.db`) and provider adapters so you can search across conversations created by **any** supported LLM CLI or MCP client.

---

## 1. Gemini Antigravity CLI
1. Add [`examples/antigravity_hooks.json`](../examples/antigravity_hooks.json) to `~/.gemini/config/hooks.json`:
   ```json
   {
     "context-finder": {
       "enabled": true,
       "PreInvocation": [
         {
           "type": "command",
           "command": "v1-context --format antigravity",
           "timeout": 8
         }
       ]
     }
   }
   ```
2. Automatic search queries both Antigravity's native `conversation_summaries.db` (including unindexed parent fallback) and sessions from Claude/Codex stored in `~/.v1-memory/index.db`.

---

## 2. Claude Code (`~/.claude`)
1. Hook setup — merge [`examples/claude_settings_hook.json`](../examples/claude_settings_hook.json) into `~/.claude/settings.json`:
   ```json
   {
     "hooks": {
       "UserPromptSubmit": [
         {
           "hooks": [
             {
               "type": "command",
               "command": "v1-context --format claude",
               "timeout": 8
             }
           ]
         }
       ]
     }
   }
   ```
   `v1-context --format claude` natively parses Claude's stdin JSON (`prompt`, `session_id`, `transcript_path`) and emits `hookSpecificOutput.additionalContext`.
2. Transcript discovery: `v1-context --reindex` and `v1-sync` automatically scan `~/.claude/projects/**/*.jsonl` and back up `~/.claude`.

---

## 3. OpenAI Codex CLI (`~/.codex`)
- **Transcript Indexing & Cloud Sync:** `v1-context --reindex` and `v1-sync` automatically index and back up `~/.codex/sessions/**/*.jsonl`.
- **Recall Inside Codex:** Register `v1-mcp` in your Codex MCP config or invoke `v1-context --prompt "<topic>"` / `v1-recall --show <id>` via shell tools.

---

## 4. Cursor, Windsurf, Cline, Zed & Any MCP Client
Add [`examples/mcp_config.json`](../examples/mcp_config.json) to your client's MCP configuration:
```json
{
  "mcpServers": {
    "v1-agent-memory": {
      "command": "v1-mcp",
      "args": []
    }
  }
}
```
This exposes five standard MCP tools over JSON-RPC stdio (zero external packages needed):
- `find_context`
- `recall_conversation`
- `backup_status`
- `sync_backup`
- `reindex_providers`

---

## 5. Custom LLM Scripts / Markdown Notes
Point `v1` at any folder containing `.jsonl` (`role`/`content`) logs or `.md` session notes:
```bash
v1-sync --extra-dir my_agent=~/my-llm-logs
```
And pipe any generic hook/prompt directly:
```bash
echo '{"prompt":"debug FDCAN prescaler","conversationId":"demo-1"}' | v1-context --format plain
```
