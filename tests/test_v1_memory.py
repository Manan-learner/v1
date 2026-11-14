"""Offline unit and end-to-end tests for v1_memory (multi-LLM context finder + backup sync + MCP)."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v1_memory import index_store as index_store_mod
from v1_memory.cloud_sync import get_sync_status, run_sync, stage_wal_snapshots
from v1_memory.context_finder import (
    detect_batch_pivot,
    extract_call_word,
    load_seen_state,
    run_hook_payload,
    save_seen_state,
)
from v1_memory.hybrid_rag import (
    DEFAULT_CONCEPT_CLUSTERS,
    compute_rag_signature,
    compute_semantic_vector,
    cosine_similarity,
    expand_with_concepts,
    qualifying_shared_concepts,
    reciprocal_rank_fusion,
)
from v1_memory.index_store import (
    _compute_corpus_stats,
    ensure_portable_schema,
    find_context_matches,
    get_last_rebuild_stats,
    get_last_unindexed_cache_stats,
    open_readonly_snapshot,
    rebuild_portable_index,
)
from v1_memory.mcp_server import handle_rpc
from v1_memory.recall import load_conversation_turns, search_conversations
from v1_memory.text_utils import STOPWORDS, fts5_terms_for_tokens, stem_token, tokenize_domain


def _load_live_hook_module():
    hook_path = Path.home() / ".gemini" / "config" / "skills" / "context-finder" / "scripts" / "context_match_hook.py"
    if not hook_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("live_context_match_hook", hook_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        self.assertEqual(tokenize_domain("sub_915mhz uart115200"), ["sub_915mhz", "915", "mhz", "uart115200", "uart", "115200"])
        self.assertEqual(tokenize_domain("can you help me fix a bug in this python script I wrote yesterday"), [])
        self.assertEqual(tokenize_domain("auto select subagent model prompt orchestrator default docs"), [])
        self.assertEqual(tokenize_domain("fixed issues on tuesday we're you'll hi hello"), [])

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
        self.assertIn("40% Detected", msg_claude or "")

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
        query = "wireless radio rssi link dropouts"
        q_tokens = set(tokenize_domain(query))
        doc_tokens = set(
            tokenize_domain(
                "Tune sub-ghz 915MHz telemetry packet framing for SX1262 transceiver "
                "Configured explicit header CRC and 915MHz preamble length on SX1262."
            )
        )
        self.assertEqual(q_tokens & doc_tokens, set())

        _, sparse_matches = find_context_matches(
            query,
            home=self.home,
            state_home=self.state_home,
            hybrid_rag=False,
        )
        self.assertEqual(sparse_matches, [])

        _, hybrid_matches = find_context_matches(
            query,
            home=self.home,
            state_home=self.state_home,
            hybrid_rag=True,
        )
        self.assertTrue(hybrid_matches)
        top = hybrid_matches[0]
        self.assertEqual(top["conversation_id"], "claude:radio-link-01")
        self.assertGreaterEqual(top["match_pct"], 40.0)
        self.assertEqual(top["retrieval_mode"], "hybrid_rrf")
        self.assertGreater(top["semantic_score"], 0.25)
        self.assertIn("concept:rf_telemetry", top["matched_tokens"])

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
        stats1 = get_last_rebuild_stats()
        self.assertEqual(stats1, {"total": 2, "updated": 2, "skipped": 0})

        count2 = rebuild_portable_index(self.state_home / "index.db", providers=["claude", "codex"], home=self.home)
        self.assertEqual(count2, 2)
        stats2 = get_last_rebuild_stats()
        self.assertEqual(stats2, {"total": 2, "updated": 0, "skipped": 2})

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

        _, m_first = find_context_matches(
            "sso login apikey rbac permissions",
            home=self.home,
            state_home=self.state_home,
            agy_db_path=agy_db,
        )
        self.assertTrue(m_first)
        self.assertEqual(m_first[0]["conversation_id"], unindexed_cid)
        self.assertEqual(get_last_unindexed_cache_stats(), {"hits": 0, "misses": 1})

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

    def test_boilerplate_and_conversational_false_positives_rejected(self) -> None:
        self.assertEqual(tokenize_domain("where is the rag located can i see a demonstration"), ["rag"])
        self.assertEqual(tokenize_domain("problems expected located concerns verification implementation"), [])
        self.assertEqual(tokenize_domain("don't won't didn't isn't e.g. i.e. step 10 and 12"), [])
        self.assertEqual(
            tokenize_domain("sqlite table index git branch docker cli json"),
            ["sqlite", "table", "index", "git", "branch", "docker", "cli", "json"],
        )
        self.assertEqual(tokenize_domain("STM32 915MHz LoRa"), ["stm32", "915mhz", "915", "mhz", "lora"])

        extra_dir = self.root / "extra_logs"
        extra_dir.mkdir(parents=True, exist_ok=True)
        (extra_dir / "db-notes.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "role": "user",
                            "content": "Review database migration table indexing vector status and expected problems",
                        }
                    ),
                    json.dumps(
                        {
                            "role": "assistant",
                            "content": "Completed schema migration table verification with zero problems located.",
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )
        rebuild_portable_index(
            self.state_home / "index.db",
            providers=["claude", "codex", "generic"],
            home=self.home,
            custom_dirs=[extra_dir],
        )

        for noisy_prompt in (
            "where is the rag located can i see a demonstration",
            "problems expected located concerns",
            "problems expected located concerns verification implementation",
            "fixed issues on tuesday",
        ):
            _, matches = find_context_matches(noisy_prompt, home=self.home, state_home=self.state_home)
            self.assertEqual(matches, [], f"Unexpected false positive for prompt: {noisy_prompt!r}")

        _, single_rag_matches = find_context_matches(
            "rag completely_unrelated_anchor",
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(single_rag_matches, [])

        cid = "conv-noisy-batch"
        noisy_turns = [
            "don't know if verification works",
            "problems expected located concerns",
            "can you verify implementation status and summary",
            "note exact details e.g. missing results",
            "where is the rag located can i see a demonstration",
        ]
        for step_idx, prompt in enumerate(noisy_turns, start=1):
            _, msg, hook_matches = run_hook_payload(
                {"conversationId": cid, "initialNumSteps": step_idx * 2, "lastUserInput": prompt},
                home=self.home,
                state_home=self.state_home,
            )
            self.assertIsNone(msg)
            self.assertEqual(hook_matches, [])

    def test_corpus_high_df_suppression_and_idf_weighting_with_real_queries(self) -> None:
        corpus_dir = self.root / "corpus_hdf"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(1, 5):
            mid_terms = "mid_df_alpha mid_df_beta" if idx <= 3 else ""
            (corpus_dir / f"filler-{idx:02d}.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "role": "user",
                                "content": (
                                    f"workspace_common_header daily_sync_marker {mid_terms} "
                                    f"session_{idx} topic_a_{idx} topic_b_{idx} topic_c_{idx}"
                                ),
                            }
                        ),
                        json.dumps(
                            {
                                "role": "assistant",
                                "content": "Processed standard workspace_common_header daily_sync_marker check.",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
        (corpus_dir / "rag-notes-05.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "role": "user",
                            "content": "workspace_common_header daily_sync_marker tune semantic embeddings vector bm25",
                        }
                    ),
                    json.dumps(
                        {
                            "role": "assistant",
                            "content": "Evaluated hybrid_rag cosine similarity and rrf fusion.",
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )
        (corpus_dir / "rare-anchor-06.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "role": "user",
                            "content": "Investigate rare_anchor_one and rare_anchor_two calibration drift",
                        }
                    ),
                    json.dumps(
                        {
                            "role": "assistant",
                            "content": "Verified rare_anchor_one rare_anchor_two registers.",
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )
        total_indexed = rebuild_portable_index(
            self.state_home / "index.db",
            providers=["claude", "codex", "generic"],
            home=self.home,
            custom_dirs=[corpus_dir],
        )
        self.assertEqual(total_indexed, 8)

        _, hdf_only = find_context_matches(
            "workspace_common_header daily_sync_marker",
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(hdf_only, [])

        _, hdf_plus_one = find_context_matches(
            "workspace_common_header daily_sync_marker sx1262",
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(hdf_plus_one, [])

        _, idf_matches = find_context_matches(
            "mid_df_alpha mid_df_beta rare_anchor_one rare_anchor_two",
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(len(idf_matches), 1)
        self.assertEqual(idf_matches[0]["conversation_id"], "generic:rare-anchor-06")
        self.assertGreaterEqual(idf_matches[0]["match_pct"], 60.0)

        _, fdcan_matches = find_context_matches("STM32H7 FDCAN", home=self.home, state_home=self.state_home)
        self.assertTrue(fdcan_matches)
        self.assertEqual(fdcan_matches[0]["conversation_id"], "codex:fdcan-boot-02")
        self.assertGreaterEqual(fdcan_matches[0]["match_pct"], 40.0)

        _, rf_syn_matches = find_context_matches(
            "wireless radio rssi link dropouts",
            home=self.home,
            state_home=self.state_home,
        )
        self.assertTrue(rf_syn_matches)
        self.assertEqual(rf_syn_matches[0]["conversation_id"], "claude:radio-link-01")
        self.assertGreaterEqual(rf_syn_matches[0]["match_pct"], 40.0)

        _, rag_matches = find_context_matches(
            "hybrid_rag rrf cosine",
            home=self.home,
            state_home=self.state_home,
        )
        self.assertTrue(rag_matches)
        self.assertEqual(rag_matches[0]["conversation_id"], "generic:rag-notes-05")
        self.assertGreaterEqual(rag_matches[0]["match_pct"], 40.0)

    def test_vector_determinism_across_hashseed_and_input_order(self) -> None:
        tokens = [f"domain_token_{i:03d}" for i in range(220)] + ["stm32h7", "fdcan", "sx1262", "lora"]
        v_fwd = compute_semantic_vector(tokens)
        v_rev = compute_semantic_vector(list(reversed(tokens)))
        v_set = compute_semantic_vector(set(tokens))
        self.assertEqual(v_fwd, v_rev)
        self.assertEqual(v_fwd, v_set)

        src_dir = str(Path(__file__).resolve().parents[1] / "src")
        code = (
            "import json, sys; sys.path.insert(0, sys.argv[1]); "
            "from v1_memory.hybrid_rag import compute_semantic_vector; "
            "toks = set([f'domain_token_{i:03d}' for i in range(220)] + ['stm32h7', 'fdcan', 'sx1262', 'lora']); "
            "print(json.dumps(compute_semantic_vector(toks)))"
        )
        vecs: list[list[float]] = []
        for seed in ("1", "42"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            out = subprocess.check_output([sys.executable, "-c", code, src_dir], env=env, text=True)
            vecs.append(json.loads(out))
        self.assertEqual(vecs[0], v_fwd)
        self.assertEqual(vecs[1], v_fwd)

    def test_live_hook_vector_and_concept_parity(self) -> None:
        # Unconditional v1_memory cluster & signature invariants
        self.assertEqual(len(DEFAULT_CONCEPT_CLUSTERS), 8)
        sig1 = compute_rag_signature()
        sig2 = compute_rag_signature({"custom_topic": ["alpha", "beta"]})
        self.assertTrue(sig1.startswith("2.2:"))
        self.assertNotEqual(sig1, sig2)

        hook = _load_live_hook_module()
        if hook is None:
            self.skipTest("Standalone context-finder hook script not installed in current HOME")

        self.assertEqual(hook.STOPWORDS, STOPWORDS)
        self.assertEqual(hook.DEFAULT_CONCEPT_CLUSTERS, DEFAULT_CONCEPT_CLUSTERS)
        self.assertEqual(hook.compute_rag_signature(), sig1)
        self.assertEqual(hook.compute_rag_signature({"custom_topic": ["alpha", "beta"]}), sig2)

        samples = [
            "sub_915mhz uart115200 v1-context we're fixed issues on tuesday",
            "Tune sub-ghz 915MHz telemetry packet framing for SX1262 transceiver",
            "Debug STM32H7 FDCAN arbitration prescaler and bootloader timeout",
            "Configure OAuth2 OIDC JWT RBAC iam credentials",
            "pytorch transformers optimizers dataloaders gradients",
            "mavlink uav autopilot gimbal attitude imu gps ekf",
        ]
        for s in samples:
            self.assertEqual(hook.tokenize_domain(s), tokenize_domain(s))
            v_hook = hook.compute_semantic_vector(s)
            v_v1 = compute_semantic_vector(s)
            self.assertAlmostEqual(cosine_similarity(v_hook, v_v1), 1.0, places=9)
            self.assertEqual(v_hook, v_v1)
            toks = tokenize_domain(s)
            self.assertEqual(hook.expand_with_concepts(toks), expand_with_concepts(toks))

        self.assertEqual(
            hook.fts5_terms_for_tokens(["v1-context", "sub-ghz", "node.js"]),
            fts5_terms_for_tokens(["v1-context", "sub-ghz", "node.js"]),
        )

        # Verify standalone hook LRU eviction keeps newest 200 items (not oldest)
        cache_path = self.root / "hook_unindexed_cache.json"
        cache_data = {
            f"cid-{i:03d}": {"last_modified_time": f"2026-01-{(i % 28) + 1:02d} {i:03d}"}
            for i in range(250)
        }
        hook.save_unindexed_cache(cache_path, cache_data, max_entries=200)
        loaded_cache = hook.load_unindexed_cache(cache_path)
        self.assertEqual(len(loaded_cache), 200)
        # Oldest month/day entries must be evicted while newest timestamps survive
        newest_sorted = sorted(
            ((f"cid-{i:03d}", f"2026-01-{(i % 28) + 1:02d} {i:03d}") for i in range(250)),
            key=lambda kv: kv[1],
        )
        expected_kept = {k for k, _ in newest_sorted[-200:]}
        self.assertEqual(set(loaded_cache.keys()), expected_kept)

    def test_singular_plural_concept_gate_and_false_positive_rejection(self) -> None:
        extra_dir = self.root / "css_design"
        extra_dir.mkdir(parents=True, exist_ok=True)
        (extra_dir / "css-design-tokens.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "role": "user",
                            "content": "Refactor CSS design token tokens palette spacing typography for browser session layout",
                        }
                    ),
                    json.dumps(
                        {
                            "role": "assistant",
                            "content": "Updated CSS token tokens variables and session styling components.",
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )
        rebuild_portable_index(
            self.state_home / "index.db",
            providers=["claude", "codex", "generic"],
            home=self.home,
            custom_dirs=[extra_dir],
        )

        # 'oauth2 jwt' must NOT match CSS 'token tokens session'
        _, oauth_matches = find_context_matches("oauth2 jwt", home=self.home, state_home=self.state_home)
        self.assertEqual(oauth_matches, [])

        # 'playwright frame timeout retry' must NOT match STM32 FDCAN session
        _, pw_matches = find_context_matches(
            "playwright frame timeout retry",
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(pw_matches, [])

        # Unrelated domain with single generic overlap ('signal') has low cosine similarity
        cos_unix_rf = cosine_similarity(
            compute_semantic_vector("unix signal posix kill sigterm sigint"),
            compute_semantic_vector("lora sx1262 915mhz transceiver antenna signal preamble"),
        )
        self.assertLess(cos_unix_rf, 0.25)

        # Direct qualifying_shared_concepts check rejects singular/plural of single stem
        self.assertEqual(
            qualifying_shared_concepts(
                {"concept:auth_security": ["token", "tokens"]},
                {"concept:auth_security": ["token", "tokens"]},
                min_distinct_stems=2,
            ),
            {},
        )
        self.assertIn(
            "concept:auth_security",
            qualifying_shared_concepts(
                {"concept:auth_security": ["oauth2", "jwt"]},
                {"concept:auth_security": ["oauth", "oauth2", "jwt"]},
                min_distinct_stems=2,
            ),
        )
        self.assertEqual(stem_token("tokens"), "token")
        self.assertEqual(stem_token("kubernetes"), "kubernetes")

    def test_hyphen_and_dot_fts5_matching(self) -> None:
        self.assertEqual(fts5_terms_for_tokens(["v1-context", "sub-ghz", "node.js"]), ["v1", "sub", "ghz", "node", "js"])

        extra_dir = self.root / "hyphen_logs"
        extra_dir.mkdir(parents=True, exist_ok=True)
        (extra_dir / "hyphen-session.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "role": "user",
                            "content": "Deploy v1-context bridge over sub-ghz telemetry with node.js collector",
                        }
                    ),
                    json.dumps(
                        {
                            "role": "assistant",
                            "content": "Verified v1-context sub-ghz and node.js integration.",
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )
        rebuild_portable_index(
            self.state_home / "index.db",
            providers=["claude", "codex", "generic"],
            home=self.home,
            custom_dirs=[extra_dir],
        )

        for q in ("v1-context", "sub-ghz", "node.js"):
            hits = search_conversations(q, home=self.home, state_home=self.state_home)
            self.assertTrue(hits, f"Expected FTS5 search hit for {q!r}")
            self.assertEqual(hits[0]["conversation_id"], "generic:hyphen-session")

        _, ctx_matches = find_context_matches("v1-context sub-ghz node.js", home=self.home, state_home=self.state_home)
        self.assertTrue(ctx_matches)
        self.assertEqual(ctx_matches[0]["conversation_id"], "generic:hyphen-session")
        self.assertGreaterEqual(ctx_matches[0]["match_pct"], 40.0)

    def test_pure_semantic_dense_fallback_outside_clusters(self) -> None:
        ml_dir = self.root / "ml_logs"
        ml_dir.mkdir(parents=True, exist_ok=True)
        (ml_dir / "ml-train-01.jsonl").write_text(
            json.dumps(
                {
                    "role": "user",
                    "content": "pytorch transformer optimizer dataloader gradient",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (ml_dir / "ml-dense-02.jsonl").write_text(
            json.dumps(
                {
                    "role": "user",
                    "content": "cuda_kernels tensor_cores flash_attns grad_scalers",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        rebuild_portable_index(
            self.state_home / "index.db",
            providers=["claude", "codex", "generic"],
            home=self.home,
            custom_dirs=[ml_dir],
        )

        _, ml_matches = find_context_matches(
            "pytorch transformers optimizers dataloaders gradients",
            home=self.home,
            state_home=self.state_home,
        )
        self.assertTrue(ml_matches)
        self.assertEqual(ml_matches[0]["conversation_id"], "generic:ml-train-01")
        self.assertGreaterEqual(ml_matches[0]["semantic_score"], 0.42)
        self.assertGreaterEqual(ml_matches[0]["match_pct"], 40.0)
        self.assertFalse(any(t.startswith("concept:") for t in ml_matches[0]["matched_tokens"]))

        _, zero_exact_matches = find_context_matches(
            "cuda_kernel tensor_core flash_attn grad_scaler",
            home=self.home,
            state_home=self.state_home,
        )
        self.assertTrue(zero_exact_matches)
        self.assertEqual(zero_exact_matches[0]["conversation_id"], "generic:ml-dense-02")
        self.assertGreaterEqual(zero_exact_matches[0]["semantic_score"], 0.42)
        self.assertGreaterEqual(zero_exact_matches[0]["match_pct"], 40.0)
        self.assertFalse(any(t.startswith("concept:") for t in zero_exact_matches[0]["matched_tokens"]))

        _, unrelated = find_context_matches(
            "culinary sourdough fermentation oven hydration",
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(unrelated, [])

    def test_short_doc_denominator_prevents_stub_inversion(self) -> None:
        k8s_dir = self.root / "k8s_logs"
        k8s_dir.mkdir(parents=True, exist_ok=True)
        (k8s_dir / "stub-short.jsonl").write_text(
            json.dumps({"role": "user", "content": "docker nginx"}) + "\n",
            encoding="utf-8",
        )
        (k8s_dir / "rich-k8s.jsonl").write_text(
            json.dumps(
                {
                    "role": "user",
                    "content": (
                        "kubernetes helm ingress pod autoscaler hpa prometheus grafana "
                        "alertmanager certmanager cluster deployment"
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        rebuild_portable_index(
            self.state_home / "index.db",
            providers=["claude", "codex", "generic"],
            home=self.home,
            custom_dirs=[k8s_dir],
        )

        query = (
            "kubernetes helm ingress pod autoscaler hpa prometheus grafana "
            "alertmanager certmanager docker nginx"
        )
        _, matches = find_context_matches(query, home=self.home, state_home=self.state_home)
        self.assertTrue(matches)
        self.assertEqual(matches[0]["conversation_id"], "generic:rich-k8s")
        stub_hits = [m for m in matches if m["conversation_id"] == "generic:stub-short"]
        if stub_hits:
            self.assertLess(stub_hits[0]["match_pct"], matches[0]["match_pct"])

    def test_high_df_suppression_large_corpus_and_no_self_suppression(self) -> None:
        # At N=1000 with sample_n=250, tokens present in >=55% of sample are still suppressed
        sample_docs = {}
        for i in range(250):
            toks = {f"unique_{i}"}
            if i < 180:
                toks.add("boilerplate_common_token")
            if i < 5:
                toks.add("rare_signal_anchor")
            sample_docs[f"doc-{i:04d}"] = toks

        n_corpus, df, high_df = _compute_corpus_stats(sample_docs, total_db_parent_count=1000)
        self.assertEqual(n_corpus, 1000)
        self.assertIn("boilerplate_common_token", high_df)
        self.assertNotIn("rare_signal_anchor", high_df)
        self.assertGreater(df["boilerplate_common_token"], df["rare_signal_anchor"])

        # On a small corpus (N=8) where query term appears in 2/8 docs (25% < 55%), do NOT self-suppress
        small_docs = {f"d{i}": ({"rare_pair_a", "rare_pair_b"} if i < 2 else {f"other_{i}"}) for i in range(8)}
        _, _, small_hdf = _compute_corpus_stats(small_docs, total_db_parent_count=8)
        self.assertNotIn("rare_pair_a", small_hdf)
        self.assertNotIn("rare_pair_b", small_hdf)

    def test_tumbling_window_short_turn5_continuation_two_token_and_callword_boundaries(self) -> None:
        # 1. Word-boundary call word check
        call_words = ["@context", "#context", "check context", "recall check", "find context"]
        self.assertEqual(extract_call_word("recall checkpoint status", call_words), (False, "recall checkpoint status"))
        self.assertEqual(extract_call_word("using @context7 mcp server", call_words), (False, "using @context7 mcp server"))
        self.assertEqual(extract_call_word("find contextual docs", call_words), (False, "find contextual docs"))
        self.assertEqual(extract_call_word("hey @context check this", call_words), (True, "hey check this"))

        # 2. Valid 2-token short prompt (<12 chars) is NOT dropped
        _, msg_short, matches_short = run_hook_payload(
            {"conversationId": "conv-short-2tok", "initialNumSteps": 1, "lastUserInput": "stm32 uart"},
            home=self.home,
            state_home=self.state_home,
        )
        self.assertTrue(matches_short)
        self.assertIn("New conversation start", msg_short or "")
        self.assertEqual(matches_short[0]["conversation_id"], "codex:fdcan-boot-02")

        # 3. Short Turn 5 ("go ahead") still evaluates the mid-batch Turn 3 pivot!
        cid_short_t5 = "conv-short-turn5"
        turns_short_t5 = [
            "draft kubernetes helm deployment manifest",
            "tune kubernetes liveness probe interval",
            "switch to STM32H7 FDCAN bootloader timing",
            "why retry delay",
            "go ahead",
        ]
        for idx, text in enumerate(turns_short_t5, start=1):
            _, msg, matches = run_hook_payload(
                {"conversationId": cid_short_t5, "initialNumSteps": idx * 2, "lastUserInput": text},
                home=self.home,
                state_home=self.state_home,
            )
            if idx < 5:
                self.assertIsNone(msg)
            else:
                self.assertTrue(matches)
                self.assertIn("Auto topic-pivot check (prompt #5)", msg or "")
                self.assertEqual(matches[0]["conversation_id"], "codex:fdcan-boot-02")

        # 4. Non-pivot 5-turn continuation stays completely silent (detect_batch_pivot == False)
        cid_cont = "conv-non-pivot"
        cont_turns = [
            "kubernetes helm ingress deployment",
            "tune kubernetes liveness readiness probe",
            "configure horizontal pod autoscale hpa",
            "inspect kubectl namespace container pods",
            "update dockerfile compose ingress manifest",
        ]
        for idx, text in enumerate(cont_turns, start=1):
            _, msg, matches = run_hook_payload(
                {"conversationId": cid_cont, "initialNumSteps": idx * 2, "lastUserInput": text},
                home=self.home,
                state_home=self.state_home,
            )
            self.assertIsNone(msg)
            self.assertEqual(matches, [])
        is_piv, _, _ = detect_batch_pivot(
            cont_turns[1:],
            tokenize_domain(cont_turns[0]),
            min_matched_tokens=2,
        )
        self.assertFalse(is_piv)

    def test_deadline_expiry_cache_non_poisoning(self) -> None:
        agy_dir = self.home / ".gemini" / "antigravity-cli"
        agy_dir.mkdir(parents=True, exist_ok=True)
        agy_db = agy_dir / "conversation_summaries.db"
        unindexed_cid = "unindexed-deadline-01"
        t_path = agy_dir / "brain" / unindexed_cid / ".system_generated" / "logs" / "transcript.jsonl"
        t_path.parent.mkdir(parents=True, exist_ok=True)
        t_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "USER_INPUT",
                            "source": "USER_EXPLICIT",
                            "content": "Configure OAuth2 OIDC JWT credential rotation and RBAC authorization",
                        }
                    ),
                    json.dumps(
                        {
                            "type": "PLANNER_RESPONSE",
                            "content": "Enforced OAuth2 JWT login secrets and RBAC middleware.",
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
            conn.execute("CREATE VIRTUAL TABLE conversations_fts USING fts5(conversation_id, title, user_body, agent_body)")
            conn.execute(
                "INSERT INTO conversation_summaries VALUES (?, '', ?, ?, ?, ?, ?)",
                (unindexed_cid, "Generic Title", "Brief note", "2026-11-05T10:00:00Z", 6, "file:///workspace/app"),
            )
            conn.commit()
        finally:
            conn.close()

        # Simulate deadline already expired when transcript scan runs
        with mock.patch.object(index_store_mod.time, "perf_counter", side_effect=[100.0, 200.0, 200.0, 200.0]):
            _, expired_matches = find_context_matches(
                "oauth2 oidc jwt rbac",
                home=self.home,
                state_home=self.state_home,
                agy_db_path=agy_db,
            )
        self.assertEqual(expired_matches, [])

        # Confirm degraded entry was NOT written to antigravity_unindexed_cache
        idx_conn = sqlite3.connect(str(self.state_home / "index.db"))
        try:
            rows = idx_conn.execute(
                "SELECT conversation_id FROM antigravity_unindexed_cache WHERE conversation_id = ?",
                (unindexed_cid,),
            ).fetchall()
            self.assertEqual(rows, [])
        finally:
            idx_conn.close()

        # Subsequent normal run reads transcript.jsonl, finds match, and caches complete entry
        _, full_matches = find_context_matches(
            "oauth2 oidc jwt rbac",
            home=self.home,
            state_home=self.state_home,
            agy_db_path=agy_db,
        )
        self.assertTrue(full_matches)
        self.assertEqual(full_matches[0]["conversation_id"], unindexed_cid)
        self.assertEqual(get_last_unindexed_cache_stats(), {"hits": 0, "misses": 1})

    def test_atomic_seen_state_invalid_json_and_malformed_hook_payloads(self) -> None:
        seen_path = self.state_home / "context_finder_seen.json"
        seen_path.write_text("{corrupt invalid json", encoding="utf-8")

        loaded = load_seen_state(seen_path, "conv-x")
        self.assertFalse(loaded["start_checked"])
        self.assertEqual(loaded["prompt_count"], 0)

        save_seen_state(
            seen_path,
            "conv-x",
            ["id-1"],
            "hash1",
            True,
            1,
            ["stm32 fdcan"],
            [],
            ["stm32", "fdcan"],
        )
        parsed = json.loads(seen_path.read_text(encoding="utf-8"))
        self.assertIn("conv-x", parsed)
        self.assertEqual(parsed["conv-x"]["ids"], ["id-1"])
        self.assertEqual(list(self.state_home.glob("*.tmp")), [])

        # Malformed / null payload fields never crash run_hook_payload
        self.assertEqual(run_hook_payload([], home=self.home, state_home=self.state_home), ("generic", None, []))
        fmt, msg, matches = run_hook_payload(
            {
                "conversationId": "conv-null-fields",
                "invocationNum": None,
                "initialNumSteps": None,
                "lastUserInput": "checking STM32H7 FDCAN bootloader",
            },
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(fmt, "antigravity")
        self.assertTrue(matches)
        self.assertIn("REQUIRED PERMISSION RULE", msg or "")

    def test_corrupt_sqlite_wal_temp_file_unlink_and_mcp_errors(self) -> None:
        bad_db = self.root / "corrupt.db"
        bad_wal = self.root / "corrupt.db-wal"
        bad_db.write_bytes(b"not a valid sqlite database")
        bad_wal.write_bytes(b"not a valid wal file")

        temp_dir = Path(tempfile.gettempdir())
        before = set(temp_dir.glob("v1_mem_*.db"))
        with self.assertRaises(sqlite3.Error):
            open_readonly_snapshot(bad_db)
        after = set(temp_dir.glob("v1_mem_*.db"))
        self.assertEqual(after - before, set())

        # MCP missing-arg and unknown-tool JSON-RPC errors
        err_recall = handle_rpc(
            {
                "jsonrpc": "2.0",
                "id": 101,
                "method": "tools/call",
                "params": {"name": "recall_conversation", "arguments": {}},
            },
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(err_recall["error"]["code"], -32602)

        err_find = handle_rpc(
            {
                "jsonrpc": "2.0",
                "id": 102,
                "method": "tools/call",
                "params": {"name": "find_context", "arguments": {}},
            },
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(err_find["error"]["code"], -32602)

        err_unknown = handle_rpc(
            {
                "jsonrpc": "2.0",
                "id": 103,
                "method": "tools/call",
                "params": {"name": "nonexistent_tool", "arguments": {}},
            },
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(err_unknown["error"]["code"], -32601)

    def test_recall_path_traversal_rejection_and_cloud_sync_wal_paths(self) -> None:
        outside_tfile = self.home / ".gemini" / "antigravity-cli" / "secret_outside" / ".system_generated" / "logs" / "transcript.jsonl"
        outside_tfile.parent.mkdir(parents=True, exist_ok=True)
        outside_tfile.write_text(
            json.dumps({"type": "USER_INPUT", "source": "USER_EXPLICIT", "content": "sensitive text"}) + "\n",
            encoding="utf-8",
        )
        (self.home / ".gemini" / "antigravity-cli" / "brain").mkdir(parents=True, exist_ok=True)

        for bad_cid in ("../secret_outside", "..\\secret_outside", "C:secret_outside", "a/b", "a\\b", ""):
            self.assertIsNone(load_conversation_turns(bad_cid, home=self.home, state_home=self.state_home))

        # cloud_sync WAL snapshot deduplication and relative path tracking
        custom_dir = self.root / "extra_sync_src"
        nested_dir = custom_dir / "nested"
        nested_dir.mkdir(parents=True, exist_ok=True)
        nested_db = nested_dir / "index.db"
        conn = sqlite3.connect(str(nested_db))
        try:
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT)")
            conn.execute("INSERT INTO items VALUES (1, 'nested-non-wal')")
            conn.commit()
        finally:
            conn.close()

        # Ensure state_home/index.db has an active -wal file
        idx_conn = sqlite3.connect(str(self.state_home / "index.db"))
        try:
            idx_conn.execute("PRAGMA journal_mode=WAL")
            idx_conn.execute("UPDATE conversations SET step_count = step_count")
            # Keep WAL open with an uncommitted/active journal write before commit
            idx_conn.commit()
        finally:
            idx_conn.close()
        wal_file = Path(str(self.state_home / "index.db") + "-wal")
        if not wal_file.exists():
            wal_file.write_bytes(b"")

        stage_root = self.root / "stage_check"
        sources = {
            "providers/v1-memory": self.state_home,
            "providers/custom/extra": custom_dir,
        }
        snap_rel_paths, wal_count = stage_wal_snapshots(sources, stage_root)
        self.assertEqual(wal_count, 1)
        self.assertEqual(snap_rel_paths, {"providers/v1-memory/index.db"})

        sync_res = run_sync(
            target_dir=str(self.drive),
            extra_dirs=[f"extra={custom_dir}"],
            home=self.home,
            state_home=self.state_home,
            reindex=False,
            force=True,
        )
        self.assertEqual(sync_res["status"], "ok")
        self.assertTrue((self.drive / "providers" / "custom" / "extra" / "nested" / "index.db").exists())

    def test_thousand_conversation_query_latency_under_two_seconds(self) -> None:
        index_db = self.state_home / "index.db"
        cache_sig = compute_rag_signature()
        template_payloads = [
            ("Kubernetes Helm Deployment", "kubernetes helm ingress pod autoscaler hpa cluster"),
            ("OAuth2 OIDC Auth", "oauth2 oidc jwt rbac credentials authentication login"),
            ("PyTorch Training", "pytorch transformer optimizer dataloader gradient weights"),
            ("PostgreSQL Tuning", "postgresql schema migration transaction vacuum sql"),
            ("Cloud Storage Sync", "rclone gdrive google-drive backup restore storage bucket"),
        ]
        precomputed = []
        for title, body in template_payloads:
            toks = sorted(set(tokenize_domain(f"{title} {body}")))
            vec_json = json.dumps(compute_semantic_vector(toks))
            tok_json = json.dumps(toks)
            precomputed.append((title, body, tok_json, vec_json))

        conn = sqlite3.connect(str(index_db))
        try:
            ensure_portable_schema(conn)
            conv_rows = []
            fts_rows = []
            for i in range(1000):
                title, body, tok_json, vec_json = precomputed[i % len(precomputed)]
                cid = f"bulk:session-{i:04d}"
                conv_rows.append(
                    (
                        cid,
                        "generic",
                        f"{title} #{i}",
                        f"2026-10-{(i % 28) + 1:02d} 12:00:00",
                        10 + (i % 20),
                        "/workspace/bulk",
                        f"/tmp/bulk-{i}.jsonl",
                        body,
                        body,
                        body,
                        f"v2.2_{cache_sig}_{i}",
                        vec_json,
                        tok_json,
                        cache_sig,
                    )
                )
                fts_rows.append((cid, "generic", f"{title} #{i}", body, body))
            conn.executemany(
                "INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                conv_rows,
            )
            conn.executemany(
                "INSERT INTO conversations_fts (conversation_id, provider, title, user_text, agent_text) VALUES (?, ?, ?, ?, ?)",
                fts_rows,
            )
            conn.commit()
        finally:
            conn.close()

        t0 = time.perf_counter()
        _, matches = find_context_matches(
            "STM32H7 FDCAN bootloader timing",
            home=self.home,
            state_home=self.state_home,
        )
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 2.0)
        self.assertIsNotNone(matches)
        self.assertTrue(matches)
        self.assertEqual(matches[0]["conversation_id"], "codex:fdcan-boot-02")

    def test_visualizer_pca_corpus_export_and_live_server(self) -> None:
        from v1_memory.visualizer import (
            collect_vector_corpus,
            create_visualizer_server,
            export_static_html,
            fit_pca_3d,
            project_vector_3d,
            query_rag_space,
        )

        # Edge cases N=0, N=1, N=2
        pca0 = fit_pca_3d([])
        self.assertEqual(pca0["coords"], [])
        self.assertEqual(pca0["variance_explained"], [0.0, 0.0, 0.0])

        pca1 = fit_pca_3d([compute_semantic_vector(["mavlink", "telemetry"])])
        self.assertEqual(pca1["coords"], [[0.0, 0.0, 0.0]])

        pca2 = fit_pca_3d([
            compute_semantic_vector(["mavlink", "telemetry"]),
            compute_semantic_vector(["postgresql", "schema"]),
        ])
        self.assertEqual(len(pca2["coords"]), 2)
        for pt in pca2["coords"]:
            self.assertEqual(len(pt), 3)
            for coord in pt:
                self.assertGreaterEqual(coord, -1.0001)
                self.assertLessEqual(coord, 1.0001)

        # Collect corpus from isolated fixture workspace (2 portable fixtures from setUp)
        state = collect_vector_corpus(home=self.home, state_home=self.state_home, max_docs=50)
        nodes = state["public_graph"]["nodes"]
        self.assertEqual(len(nodes), 2)
        self.assertTrue(all(len(n["pos"]) == 3 for n in nodes))
        self.assertTrue(all(len(n["vector_preview"]) == 64 for n in nodes))

        q_proj = project_vector_3d(
            compute_semantic_vector(["sx1262", "telemetry"]),
            state["pca_mean"],
            state["pca_basis"],
        )
        self.assertEqual(len(q_proj), 3)

        # Query Hybrid-RAG vs Sparse-Only
        q_res = query_rag_space("wireless 915MHz telemetry packet", state, threshold_pct=20.0, hybrid_rag=True)
        self.assertIn("q_pos", q_res)
        self.assertEqual(len(q_res["q_pos"]), 3)
        self.assertIn("matches", q_res)
        self.assertTrue(any(m["conversation_id"] == "claude:radio-link-01" for m in q_res["matches"]))

        # Static offline HTML export + XSS escape check
        xss_state = collect_vector_corpus(home=self.home, state_home=self.state_home, max_docs=50)
        xss_state["public_graph"]["nodes"][0]["title"] = '</script><script>alert(1)</script>'
        out_html = Path(self.tmp.name) / "viz_export.html"
        exported = export_static_html(out_html, state=xss_state)
        self.assertTrue(exported.exists())
        html_text = exported.read_text(encoding="utf-8")
        self.assertNotIn("<script>alert(1)</script>", html_text)
        self.assertIn("const INITIAL_LIVE_API = false;", html_text)
        self.assertIn('rel="icon" href="data:,"', html_text)

        # Live ephemeral HTTP server check (/, /favicon.ico 204, /api/graph, /api/query, 400 bad JSON)
        httpd, port = create_visualizer_server(state, host="127.0.0.1", port=0, strict_port=True)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        try:
            base = f"http://127.0.0.1:{port}"
            with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                body = resp.read().decode("utf-8")
                self.assertIn("const INITIAL_LIVE_API = true;", body)

            with urllib.request.urlopen(f"{base}/favicon.ico", timeout=5) as resp:
                self.assertEqual(resp.status, 204)

            with urllib.request.urlopen(f"{base}/api/graph", timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                graph = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(len(graph["nodes"]), 2)

            req = urllib.request.Request(
                f"{base}/api/query",
                data=json.dumps({"query": "STM32H7 FDCAN", "threshold_pct": 25.0, "hybrid_rag": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                q_json = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(any(m["conversation_id"] == "codex:fdcan-boot-02" for m in q_json["matches"]))

            bad_req = urllib.request.Request(
                f"{base}/api/query",
                data=b"{not-valid-json",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(bad_req, timeout=5)
            self.assertEqual(ctx.exception.code, 400)
            ctx.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=2)

    def test_balanced_fts5_quota_and_high_idf_single_token_rescue(self) -> None:
        # 1. Backward-compatible idf_weights in compute_semantic_vector
        toks = ["quarterback", "championship", "playoffs"]
        v_default = compute_semantic_vector(toks)
        v_none = compute_semantic_vector(toks, idf_weights=None)
        self.assertEqual(v_default, v_none)
        v_weighted = compute_semantic_vector(toks, idf_weights={"quarterback": 6.5, "championship": 1.1, "playoffs": 2.0})
        self.assertEqual(len(v_weighted), 256)
        self.assertNotEqual(v_default, v_weighted)
        v_q_only = compute_semantic_vector(["quarterback"])
        self.assertGreater(cosine_similarity(v_weighted, v_q_only), cosine_similarity(v_default, v_q_only))

        # 2. Balanced FTS5 per-token quota + high-IDF single-token rescue under common-term crowding
        index_db = self.state_home / "index.db"
        cache_sig = compute_rag_signature()
        conn = sqlite3.connect(str(index_db))
        try:
            ensure_portable_schema(conn)
            conv_rows = []
            fts_rows = []
            # Insert 55 crowding docs containing "championship" repeated many times (<55% of 125 total docs so not high-DF suppressed)
            for i in range(120):
                cid = f"crowd:doc-{i:03d}"
                if i < 55:
                    title = f"Tournament Recap #{i}"
                    body = (
                        f"championship championship championship medal podium Bracket_{i} "
                        f"venue_{i} referee_{i} season_{i} league_{i}"
                    )
                else:
                    title = f"Archive Note #{i}"
                    body = f"unrelated_topic_{i} module_{i} widget_{i} metric_{i} sensor_{i}"
                tok_list = sorted(set(tokenize_domain(f"{title} {body}")))
                vec_json = json.dumps(compute_semantic_vector(tok_list))
                tok_json = json.dumps(tok_list)
                conv_rows.append(
                    (
                        cid,
                        "generic",
                        title,
                        f"2026-10-{(i % 28) + 1:02d} 11:00:00",
                        6,
                        "/workspace/sports",
                        f"/tmp/crowd-{i}.jsonl",
                        body,
                        body,
                        "",
                        f"crowd_{cache_sig}_{i}",
                        vec_json,
                        tok_json,
                        cache_sig,
                    )
                )
                fts_rows.append((cid, "generic", title, body, ""))

            # Insert 1 rare target doc containing ONLY "quarterback" from the 4-term query
            rare_cid = "generic:gridiron-rare-01"
            rare_title = "Varsity Quarterback Passing Mechanics"
            rare_body = "quarterback pocket footwork cadence spiralblitz audible"
            rare_toks = sorted(set(tokenize_domain(f"{rare_title} {rare_body}")))
            conv_rows.append(
                (
                    rare_cid,
                    "generic",
                    rare_title,
                    "2026-10-01 08:00:00",
                    8,
                    "/workspace/sports",
                    "/tmp/rare.jsonl",
                    rare_body,
                    rare_body,
                    "",
                    f"rare_{cache_sig}",
                    json.dumps(compute_semantic_vector(rare_toks)),
                    json.dumps(rare_toks),
                    cache_sig,
                )
            )
            fts_rows.append((rare_cid, "generic", rare_title, rare_body, ""))

            conn.executemany(
                "INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                conv_rows,
            )
            conn.executemany(
                "INSERT INTO conversations_fts (conversation_id, provider, title, user_text, agent_text) VALUES (?, ?, ?, ?, ?)",
                fts_rows,
            )
            conn.commit()
        finally:
            conn.close()

        _, matches = find_context_matches(
            "quarterback touchdown championship playoffs",
            threshold_pct=35.0,
            home=self.home,
            state_home=self.state_home,
        )
        self.assertTrue(matches, "Expected rare 'quarterback' doc to be rescued via balanced FTS5 quota + high-IDF rescue")
        self.assertEqual(matches[0]["conversation_id"], rare_cid)
        self.assertIn("quarterback", matches[0]["matched_tokens"])

        # Common single-token overlap (DF=55/123 > 15% max DF ratio) must NOT be rescued
        _, common_only = find_context_matches(
            "championship nonexistent_xyz_token_999",
            threshold_pct=35.0,
            home=self.home,
            state_home=self.state_home,
        )
        self.assertEqual(common_only, [])

    def test_whitened_pca_parity_dynamic_kmeans_and_slim_graph_node_api(self) -> None:
        from v1_memory.visualizer import (
            build_public_graph,
            collect_vector_corpus,
            create_visualizer_server,
            fit_pca_3d,
            project_vector_3d,
        )

        # Build >= 16 diverse non-DEFAULT_CONCEPT_CLUSTERS documents across 4 distinct domains
        diverse_topics = [
            ("Astrophysics Supernova #" + str(i), "supernova pulsar magnetar redshift parsec luminosity nebula quasar")
            for i in range(4)
        ] + [
            ("Molecular Ribosome #" + str(i), "ribosome polymerase transcription codon mrna peptide enzyme nucleotide")
            for i in range(4)
        ] + [
            ("Macroeconomic Inflation #" + str(i), "inflation bond yield tariff centralbank fiscal monetary liquidity")
            for i in range(4)
        ] + [
            ("Championship Gridiron #" + str(i), "quarterback touchdown playoffs scrimmage linebackerblitz fieldgoal")
            for i in range(4)
        ]

        topic_dir = self.root / "diverse_topics"
        topic_dir.mkdir(parents=True, exist_ok=True)
        for idx, (title, text) in enumerate(diverse_topics):
            (topic_dir / f"topic-{idx:02d}.jsonl").write_text(
                json.dumps({"role": "user", "content": f"{title} {text} sample_{idx}"}) + "\n",
                encoding="utf-8",
            )
        rebuild_portable_index(
            self.state_home / "index.db",
            providers=["claude", "codex", "generic"],
            home=self.home,
            custom_dirs=[topic_dir],
        )

        # 1. Whitened PCA projection parity with project_vector_3d
        state = collect_vector_corpus(home=self.home, state_home=self.state_home, max_docs=100)
        nodes = state["public_graph"]["nodes"]
        self.assertEqual(len(nodes), 18)  # 2 setUp fixtures + 16 diverse docs
        vecs = [state["vectors_by_id"][n["id"]] for n in nodes]
        pca_res = fit_pca_3d(vecs)
        self.assertGreater(sum(pca_res["variance_explained"]), 10.0)
        for idx, v in enumerate(vecs):
            proj = project_vector_3d(v, pca_res["mean"], pca_res["basis"])
            self.assertEqual(pca_res["coords"][idx], proj)
            radius = sum(c * c for c in proj) ** 0.5
            self.assertLessEqual(radius, 1.0002)

        # 2. Dynamic spherical k-means clustering when >75% of corpus (>= 12 docs) is outside DEFAULT_CONCEPT_CLUSTERS
        dyn_clusters = [c for c in state["cluster_names"] if c.startswith("topic:")]
        self.assertGreaterEqual(len(dyn_clusters), 4)
        general_nodes = [n for n in nodes if n["dominant_concept"] == "general"]
        self.assertEqual(len(general_nodes), 0)
        centroid_concepts = {c["concept"] for c in state["centroids"]}
        self.assertTrue(any(c.startswith("topic:") for c in centroid_concepts))

        # 3. Slim live /api/graph (>400 nodes) vs offline static export (live_api=False)
        mock_405_nodes = [
            {
                "id": f"mock:{i}",
                "provider": "generic",
                "title": f"Mock #{i}",
                "last_modified_time": "2026-11-01 12:00:00",
                "step_count": 5,
                "summary": "Detailed summary text",
                "dominant_concept": "topic:alpha_beta",
                "concepts": {},
                "top_tokens": ["alpha", "beta"],
                "vector_preview": [0.1] * 64,
                "pos": [0.0, 0.0, 0.0],
                "knn": [],
            }
            for i in range(405)
        ]
        large_state = {**state, "public_graph": {**state["public_graph"], "nodes": mock_405_nodes}}
        slim_graph = build_public_graph(large_state, live_api=True)
        self.assertEqual(len(slim_graph["nodes"]), 405)
        for stripped_key in ("summary", "top_tokens", "vector_preview", "concepts"):
            self.assertNotIn(stripped_key, slim_graph["nodes"][0])

        full_offline_graph = build_public_graph(large_state, live_api=False)
        for kept_key in ("summary", "top_tokens", "vector_preview", "concepts"):
            self.assertIn(kept_key, full_offline_graph["nodes"][0])

        # 4. GET /api/node?id=... 200 & 404 behavior
        httpd, port = create_visualizer_server(state, host="127.0.0.1", port=0, strict_port=True)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        try:
            base = f"http://127.0.0.1:{port}"
            target_id = nodes[0]["id"]
            with urllib.request.urlopen(f"{base}/api/node?id={urllib.parse.quote(target_id)}", timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                detail = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(detail["id"], target_id)
                self.assertIn("summary", detail)
                self.assertIn("top_tokens", detail)
                self.assertEqual(len(detail["vector_preview"]), 64)
                self.assertIn("concepts", detail)

            for bad_url in (f"{base}/api/node?id=nonexistent-id-999", f"{base}/api/node"):
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(bad_url, timeout=5)
                self.assertEqual(ctx.exception.code, 404)
                ctx.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=2)

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
