"""Offline unit and end-to-end tests for v1_memory (multi-LLM context finder + backup sync + MCP)."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v1_memory.cloud_sync import get_sync_status, run_sync
from v1_memory.context_finder import run_hook_payload
from v1_memory.hybrid_rag import compute_semantic_vector, cosine_similarity, expand_with_concepts, reciprocal_rank_fusion
from v1_memory.index_store import (
    find_context_matches,
    get_last_rebuild_stats,
    get_last_unindexed_cache_stats,
    rebuild_portable_index,
)
from v1_memory.mcp_server import handle_rpc
from v1_memory.recall import load_conversation_turns, search_conversations
from v1_memory.text_utils import tokenize_domain


class V1MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.state_home = self.root / "state"
        self.drive = self.root / "gdrive" / "AI-Agent-Memory"
        self.home.mkdir(parents=True, exist_ok=True)
        self.state_home.mkdir(parents=True, exist_ok=True)

        # Fixture 1: Claude Code JSONL session
        claude_proj = self.home / ".claude" / "projects" / "demo-radio"
        claude_proj.mkdir(parents=True, exist_ok=True)
        self.claude_file = claude_proj / "radio-link-01.jsonl"
        self.claude_file.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "timestamp": "2026-10-10T10:00:00Z",
                            "cwd": "/workspace/radio",
                            "message": {
                                "role": "user",
                                "content": "Tune sub-ghz 915MHz telemetry packet framing for SX1262 transceiver",
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "timestamp": "2026-10-10T10:01:00Z",
                            "message": {
                                "role": "assistant",
                                "content": "Configured explicit header CRC and 915MHz preamble length on SX1262.",
                            },
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        # Fixture 2: Codex CLI JSONL session
        codex_sess = self.home / ".codex" / "sessions" / "2026"
        codex_sess.mkdir(parents=True, exist_ok=True)
        self.codex_file = codex_sess / "fdcan-boot-02.jsonl"
        self.codex_file.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "event_msg",
                            "timestamp": "2026-10-15T12:00:00Z",
                            "payload": {
                                "type": "user_message",
                                "message": "Debug STM32H7 FDCAN arbitration prescaler and CAN bootloader timeout",
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "response_item",
                            "timestamp": "2026-10-15T12:01:00Z",
                            "payload": {
                                "role": "assistant",
                                "content": "Aligned nominal bit timing prescaler for STM32H7 FDCAN bootloader frames.",
                            },
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        count = rebuild_portable_index(self.state_home / "index.db", providers=["claude", "codex"], home=self.home)
        self.assertEqual(count, 2)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_tokenization_preserves_alnum_codes_and_drops_noise(self) -> None:
        self.assertEqual(tokenize_domain("Debug STM32 FDCAN prescaler"), ["stm32", "fdcan", "prescaler"])
        self.assertEqual(tokenize_domain("LoRa 915MHz packets"), ["lora", "915mhz", "915", "mhz", "packets"])
        self.assertEqual(tokenize_domain("can you help me fix a bug in this python script I wrote yesterday"), [])
        self.assertEqual(tokenize_domain("auto select subagent model prompt orchestrator default docs"), [])

    def test_multi_llm_hook_formats_and_permission_rule(self) -> None:
        # Antigravity format
        fmt_agy, msg_agy, m_agy = run_hook_payload(
            {"invocationNum": 1, "initialNumSteps": 1, "conversationId": "s-agy", "lastUserInput": "working on 915MHz SX1262 transceiver telemetry"},
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(fmt_agy, "antigravity")
        self.assertTrue(m_agy)
        self.assertIn("REQUIRED PERMISSION RULE", msg_agy or "")

        # Claude UserPromptSubmit format
        fmt_claude, msg_claude, m_claude = run_hook_payload(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "s-claude",
                "prompt": "checking STM32H7 FDCAN bootloader timing",
            },
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(fmt_claude, "claude")
        self.assertEqual(m_claude[0]["provider"], "codex")
        self.assertIn("25% Detected", msg_claude or "")

    def test_tumbling_five_prompt_window_catches_mid_batch_pivot(self) -> None:
        cid = "conv-5-batch"
        # Turn 1 (unmatched topic)
        _, m1, _ = run_hook_payload({"conversationId": cid, "initialNumSteps": 1, "lastUserInput": "draft kubernetes helm deployment manifest"}, home=self.home, state_home=self.state_home)
        self.assertIsNone(m1)
        # Turn 2 (same topic)
        _, m2, _ = run_hook_payload({"conversationId": cid, "initialNumSteps": 4, "lastUserInput": "tune kubernetes liveness probe interval"}, home=self.home, state_home=self.state_home)
        self.assertIsNone(m2)
        # Turn 3 (mid-batch pivot!)
        _, m3, _ = run_hook_payload({"conversationId": cid, "initialNumSteps": 7, "lastUserInput": "switch to STM32H7 FDCAN bootloader timing"}, home=self.home, state_home=self.state_home)
        self.assertIsNone(m3)
        # Turn 4 (short follow-up)
        _, m4, _ = run_hook_payload({"conversationId": cid, "initialNumSteps": 9, "lastUserInput": "why frame retry delay"}, home=self.home, state_home=self.state_home)
        self.assertIsNone(m4)
        # Turn 5 (auto pivot check fires on batch!)
        _, msg5, matches5 = run_hook_payload({"conversationId": cid, "initialNumSteps": 11, "lastUserInput": "check prescaler divider"}, home=self.home, state_home=self.state_home)
        self.assertTrue(matches5)
        self.assertIn("Auto topic-pivot check (prompt #5)", msg5 or "")
        self.assertEqual(matches5[0]["conversation_id"], "codex:fdcan-boot-02")

    def test_bare_call_word_and_greeting_deferral(self) -> None:
        cid = "conv-greet"
        # Greeting Turn 1 keeps start_checked False
        _, g1, _ = run_hook_payload({"conversationId": cid, "initialNumSteps": 1, "lastUserInput": "hi there"}, home=self.home, state_home=self.state_home)
        self.assertIsNone(g1)
        # Turn 2 gets checked automatically as start of convo
        _, g2, m2 = run_hook_payload({"conversationId": cid, "initialNumSteps": 5, "lastUserInput": "let us tune 915MHz SX1262 transceiver"}, home=self.home, state_home=self.state_home)
        self.assertTrue(m2)
        self.assertIn("New conversation start", g2 or "")

        # Bare @context uses recent_prompts and bypasses already_seen
        _, g3, m3 = run_hook_payload({"conversationId": cid, "initialNumSteps": 9, "lastUserInput": "@context"}, home=self.home, state_home=self.state_home)
        self.assertTrue(m3)
        self.assertIn("Call word triggered", g3 or "")

    def test_recall_sync_and_mcp_server(self) -> None:
        hits = search_conversations("SX1262 OR FDCAN", home=self.home, state_home=self.state_home)
        self.assertEqual(len(hits), 2)
        loaded = load_conversation_turns("claude:radio-link-01", home=self.home, state_home=self.state_home)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded[1]), 2)

        res1 = run_sync(target_dir=str(self.drive), home=self.home, state_home=self.state_home, reindex=False)
        self.assertEqual(res1["status"], "ok")
        self.assertTrue((self.drive / "providers" / "claude" / "projects" / "demo-radio" / "radio-link-01.jsonl").exists())

        res2 = run_sync(target_dir=str(self.drive), home=self.home, state_home=self.state_home, reindex=False)
        self.assertEqual(res2["status"], "skipped")

        status = get_sync_status(target_dir=str(self.drive), home=self.home, state_home=self.state_home)
        self.assertEqual(status["unsynced_changes"], {"added": 0, "modified": 0, "removed": 0})

        mcp_list = handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, home=self.home, state_home=self.state_home)
        tool_names = [t["name"] for t in mcp_list["result"]["tools"]]
        self.assertIn("find_context", tool_names)
        self.assertIn("sync_backup", tool_names)

        mcp_find = handle_rpc(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "find_context", "arguments": {"query": "STM32H7 FDCAN bootloader"}},
            },
            home=self.home,
            state_home=self.state_home,
        )
        parsed = json.loads(mcp_find["result"]["content"][0]["text"])
        self.assertEqual(parsed["matches"][0]["conversation_id"], "codex:fdcan-boot-02")

    def test_hybrid_rag_semantic_and_synonym_matching_without_word_overlap(self) -> None:
        # Query uses "wireless radio rssi link dropouts" — zero raw token overlap with
        # "Tune sub-ghz 915MHz telemetry packet framing for SX1262 transceiver"
        query = "wireless radio rssi link dropouts"
        q_tokens = set(tokenize_domain(query))
        doc_tokens = set(
            tokenize_domain(
                "Tune sub-ghz 915MHz telemetry packet framing for SX1262 transceiver "
                "Configured explicit header CRC and 915MHz preamble length on SX1262."
            )
        )
        self.assertEqual(q_tokens & doc_tokens, set())

        # Sparse-only mode returns 0 matches when raw words do not overlap
        _, sparse_matches = find_context_matches(
            query,
            home=self.home,
            state_home=self.state_home,
            hybrid_rag=False,
        )
        self.assertEqual(sparse_matches, [])

        # Hybrid-RAG mode matches the 915MHz SX1262 conversation at >= 25% via concept + semantic vector + RRF
        _, hybrid_matches = find_context_matches(
            query,
            home=self.home,
            state_home=self.state_home,
            hybrid_rag=True,
        )
        self.assertTrue(hybrid_matches)
        top = hybrid_matches[0]
        self.assertEqual(top["conversation_id"], "claude:radio-link-01")
        self.assertGreaterEqual(top["match_pct"], 25.0)
        self.assertEqual(top["retrieval_mode"], "hybrid_rrf")
        self.assertGreater(top["semantic_score"], 0.25)
        self.assertIn("concept:rf_telemetry", top["matched_tokens"])

        # Custom user synonyms also bridge arbitrary project vocabulary
        expanded, concepts = expand_with_concepts(["nighthawk", "gimbal"], {"uas_payload": ["nighthawk", "sx1262", "gimbal"]})
        self.assertIn("sx1262", expanded)
        self.assertIn("concept:uas_payload", concepts)
        v1 = compute_semantic_vector("wireless radio rssi link dropouts")
        v2 = compute_semantic_vector("915MHz SX1262 transceiver telemetry preamble")
        self.assertEqual(len(v1), 256)
        self.assertGreater(cosine_similarity(v1, v2), 0.3)
        rrf = reciprocal_rank_fusion(["a", "b"], ["b", "a"], k=60)
        self.assertAlmostEqual(rrf["a"], rrf["b"])

    def test_incremental_portable_index_and_unindexed_antigravity_cache(self) -> None:
        # Initial rebuild in setUp indexed 2 conversations
        stats1 = get_last_rebuild_stats()
        self.assertEqual(stats1, {"total": 2, "updated": 2, "skipped": 0})

        # Second rebuild with unchanged (last_modified_time, step_count) skips all 2 conversations
        count2 = rebuild_portable_index(self.state_home / "index.db", providers=["claude", "codex"], home=self.home)
        self.assertEqual(count2, 2)
        stats2 = get_last_rebuild_stats()
        self.assertEqual(stats2, {"total": 2, "updated": 0, "skipped": 2})

        # Create a synthetic Antigravity conversation_summaries.db with 1 unindexed parent conversation
        agy_dir = self.home / ".gemini" / "antigravity-cli"
        agy_dir.mkdir(parents=True, exist_ok=True)
        agy_db = agy_dir / "conversation_summaries.db"
        unindexed_cid = "unindexed-parent-01"
        t_path = agy_dir / "brain" / unindexed_cid / ".system_generated" / "logs" / "transcript.jsonl"
        t_path.parent.mkdir(parents=True, exist_ok=True)
        t_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "USER_INPUT",
                            "source": "USER_EXPLICIT",
                            "content": "Configure OAuth2 OIDC JWT refresh token rotation and RBAC session cookies",
                        }
                    ),
                    json.dumps(
                        {
                            "type": "PLANNER_RESPONSE",
                            "content": "Implemented OAuth2 JWT credential validation and RBAC permission middleware.",
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        conn = sqlite3.connect(str(agy_db))
        try:
            conn.execute(
                """
                CREATE TABLE conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    parent_conversation_id TEXT,
                    title TEXT,
                    preview TEXT,
                    last_modified_time TEXT,
                    step_count INTEGER,
                    workspace_uris TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE conversations_fts USING fts5(
                    conversation_id, title, user_body, agent_body
                )
                """
            )
            conn.execute(
                """
                INSERT INTO conversation_summaries VALUES (?, '', ?, ?, ?, ?, ?)
                """,
                (
                    unindexed_cid,
                    "Auth Service OIDC Hardening",
                    "OAuth2 session rotation",
                    "2026-11-01T09:30:00Z",
                    8,
                    "file:///workspace/auth",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # First search populates antigravity_unindexed_cache from transcript.jsonl (miss=1, hit=0)
        _, m_first = find_context_matches(
            "sso login apikey rbac permissions",
            home=self.home,
            state_home=self.state_home,
            agy_db_path=agy_db,
        )
        self.assertTrue(m_first)
        self.assertEqual(m_first[0]["conversation_id"], unindexed_cid)
        self.assertEqual(get_last_unindexed_cache_stats(), {"hits": 0, "misses": 1})

        # Delete transcript.jsonl from disk to prove run 2 uses cached digest + vector with ZERO disk reads
        t_path.unlink()
        _, m_second = find_context_matches(
            "sso login apikey rbac permissions",
            home=self.home,
            state_home=self.state_home,
            agy_db_path=agy_db,
        )
        self.assertTrue(m_second)
        self.assertEqual(m_second[0]["conversation_id"], unindexed_cid)
        self.assertEqual(get_last_unindexed_cache_stats(), {"hits": 1, "misses": 0})

    def test_zero_personal_paths_or_identifiers_in_repo(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        forbidden = re.compile(
            r"\b(" + "|".join(["ary" + "am", "My_" + "Laptop", "Arc" + "Drone", "Edmon" + "ton", "Calg" + "ary", "Dental" + " Clinic", "gma" + "il"]) + r")\b",
            re.IGNORECASE,
        )
        for path in repo_root.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() not in {".py", ".md", ".json", ".toml", ".ps1", ".sh", ".txt", ".yml", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            hit = forbidden.search(text)
            self.assertIsNone(hit, f"Forbidden personal identifier found in {path.relative_to(repo_root)}: {hit.group(0) if hit else ''}")


if __name__ == "__main__":
    unittest.main()
