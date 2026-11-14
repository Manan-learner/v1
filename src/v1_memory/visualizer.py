"""3D PCA Vector Space Explorer, k-NN Topology Builder, and Live Hybrid-RAG Query Server."""

from __future__ import annotations

import argparse
import heapq
import inspect
import json
import math
import os
import re
import sqlite3
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from operator import mul
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlsplit

from v1_memory.context_finder import load_config
from v1_memory.hybrid_rag import (
    DEFAULT_CONCEPT_CLUSTERS,
    VECTOR_CACHE_VERSION,
    VECTOR_DIM,
    build_concept_lookup,
    compute_rag_signature,
    compute_semantic_vector,
    cosine_similarity,
    group_tokens_by_concept,
)
from v1_memory.index_store import (
    _bm25_idf,
    _bounded_head_tail_text,
    _compute_corpus_stats,
    _extract_antigravity_transcript_preview,
    _load_antigravity_caches_from_conn,
    _load_configured_synonyms,
    _parse_json_list,
    _save_antigravity_caches,
    default_state_home,
    find_context_matches,
    open_readonly_snapshot,
)
from v1_memory.text_utils import (
    build_summary_line,
    clean_snippet,
    ensure_utf8_stdio,
    stem_token,
    tokenize_domain,
)

MAX_POST_BODY_BYTES = 32768
KNN_TOP_K = 3
KNN_MIN_COSINE = 0.20
TOP_TOKENS_PER_DOC = 18
VECTOR_PREVIEW_BUCKETS = 64


class PCAResult(dict):
    """PCA 3D result supporting both dict key access and (coords, mean, basis, variance_explained) unpacking."""

    def __iter__(self):
        yield self["coords"]
        yield self["mean"]
        yield self["basis"]
        yield self["variance_explained"]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return (self["coords"], self["mean"], self["basis"], self["variance_explained"])[key]
        return super().__getitem__(key)


def compute_vector_preview(vec: Sequence[float], buckets: int = VECTOR_PREVIEW_BUCKETS) -> list[float]:
    """Compress a 256-dim L2 semantic vector into a 64-bucket signed RMS fingerprint rounded to 4 decimals."""
    if not vec or buckets <= 0:
        return [0.0] * max(0, buckets)
    n = len(vec)
    if n == buckets * 4:
        preview: list[float] = []
        for idx in range(0, n, 4):
            c0 = float(vec[idx])
            c1 = float(vec[idx + 1])
            c2 = float(vec[idx + 2])
            c3 = float(vec[idx + 3])
            rms = math.sqrt((c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3) * 0.25)
            if rms <= 1e-12:
                preview.append(0.0)
            else:
                sign = 1.0 if (c0 + c1 + c2 + c3) >= 0.0 else -1.0
                preview.append(round(sign * rms, 4))
        return preview
    step = max(1, n // buckets)
    preview = []
    for b in range(buckets):
        start = b * step
        end = n if b == buckets - 1 else min(n, (b + 1) * step)
        chunk = [float(vec[i]) for i in range(start, end)] if start < end else [0.0]
        mean_val = sum(chunk) / len(chunk)
        rms = math.sqrt(sum(x * x for x in chunk) / len(chunk))
        if rms <= 1e-12:
            preview.append(0.0)
        else:
            sign = 1.0 if mean_val >= 0.0 else -1.0
            preview.append(round(sign * rms, 4))
    return preview


_SUMPROD = getattr(math, "sumprod", None)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if _SUMPROD is not None:
        return float(_SUMPROD(a, b))
    return float(sum(map(mul, a, b)))


def _vec_norm(v: Sequence[float]) -> float:
    return math.sqrt(_dot(v, v))


def _to_unit_tuple(v: Sequence[float]) -> tuple[float, ...]:
    vals = [float(x) for x in v]
    norm = math.sqrt(_dot(vals, vals))
    if norm <= 1e-12:
        return tuple(vals)
    inv = 1.0 / norm
    return tuple(x * inv for x in vals)


def _orthogonalize_unit(vec: list[float], basis: Sequence[Sequence[float]]) -> list[float]:
    out = list(vec)
    for b in basis:
        b_norm_sq = _dot(b, b)
        if b_norm_sq > 1e-18:
            proj = _dot(out, b) / b_norm_sq
            out = [out[j] - proj * float(b[j]) for j in range(len(out))]
    norm = _vec_norm(out)
    if norm <= 1e-12:
        return [0.0] * len(out)
    out = [x / norm for x in out]
    for val in out:
        if abs(val) > 1e-9:
            if val < 0.0:
                out = [-x for x in out]
            break
    return out


def _fallback_orthogonal_unit(dim: int, basis: Sequence[Sequence[float]]) -> list[float]:
    for axis in range(dim):
        cand = [1.0 if j == axis else 0.0 for j in range(dim)]
        unit = _orthogonalize_unit(cand, basis)
        if _vec_norm(unit) > 0.5:
            return unit
    return [1.0 if j == 0 else 0.0 for j in range(dim)]


def project_vector_3d(
    vec: Sequence[float],
    mean: Sequence[float],
    basis: Sequence[Sequence[float]],
) -> list[float]:
    """Project a D-dimensional vector into the 3D PCA coordinate frame defined by `(mean, basis)`."""
    if not vec or not mean or not basis:
        return [0.0, 0.0, 0.0]
    dim = min(len(vec), len(mean))
    centered = [float(vec[j]) - float(mean[j]) for j in range(dim)]
    coords: list[float] = []
    for k in range(3):
        if k < len(basis) and len(basis[k]) >= dim:
            b_k = basis[k][:dim] if len(basis[k]) > dim else basis[k]
            coords.append(round(_dot(centered, b_k), 4))
        else:
            coords.append(0.0)
    return coords


def fit_pca_3d(vectors: Sequence[Sequence[float]], max_iter: int = 45) -> PCAResult:
    """Fit a deterministic SNR-conditioned 3D PCA model using deflated Power Iteration in pure Python stdlib.

    Applies anisotropic signal-to-noise variance conditioning (`w_vec`) before covariance/eigenvector extraction,
    and folds `w_vec[j] / scale` into `basis` so `project_vector_3d(vec, mean, basis)` maps queries directly
    into the exact conditioned `[-1.0, 1.0]` bounding sphere with zero signature changes.
    """
    n_docs = len(vectors)
    dim = len(vectors[0]) if n_docs > 0 and vectors[0] else VECTOR_DIM

    default_basis = [[1.0 if j == k else 0.0 for j in range(dim)] for k in range(3)]
    if n_docs == 0:
        return PCAResult(
            coords=[],
            positions=[],
            mean=[0.0] * dim,
            mean_vec=[0.0] * dim,
            basis=default_basis,
            unit_basis=default_basis,
            scale=1.0,
            variance_explained=[0.0, 0.0, 0.0],
        )

    if n_docs == 1:
        mean_vec = [round(float(x), 6) for x in vectors[0][:dim]]
        if len(mean_vec) < dim:
            mean_vec.extend([0.0] * (dim - len(mean_vec)))
        return PCAResult(
            coords=[[0.0, 0.0, 0.0]],
            positions=[[0.0, 0.0, 0.0]],
            mean=mean_vec,
            mean_vec=mean_vec,
            basis=default_basis,
            unit_basis=default_basis,
            scale=1.0,
            variance_explained=[0.0, 0.0, 0.0],
        )

    mean_vec = [0.0] * dim
    sum_sq = [0.0] * dim
    for v in vectors:
        for j in range(min(dim, len(v))):
            val = float(v[j])
            mean_vec[j] += val
            sum_sq[j] += val * val
    inv_n = 1.0 / n_docs
    mean_vec = [x * inv_n for x in mean_vec]
    var_vec = [max(0.0, (sum_sq[j] * inv_n) - (mean_vec[j] * mean_vec[j])) for j in range(dim)]
    total_raw_var = sum(var_vec) * n_docs

    if total_raw_var <= 1e-12:
        zero_coords = [[0.0, 0.0, 0.0] for _ in range(n_docs)]
        rounded_mean = [round(x, 6) for x in mean_vec]
        return PCAResult(
            coords=zero_coords,
            positions=zero_coords,
            mean=rounded_mean,
            mean_vec=rounded_mean,
            basis=default_basis,
            unit_basis=default_basis,
            scale=1.0,
            variance_explained=[0.0, 0.0, 0.0],
        )

    sorted_vars = sorted(var_vec)
    noise_floor = sorted_vars[int(0.60 * dim)] * 0.95 if n_docs >= 12 else 0.0
    w_vec = [((v - noise_floor) ** 0.75) if v > noise_floor else 0.0 for v in var_vec]
    if sum(w_vec) <= 1e-12:
        w_vec = [1.0 if v > 1e-12 else 0.0 for v in var_vec]

    sp = _SUMPROD if _SUMPROD is not None else _dot
    total_var = n_docs * sum(var_vec[j] * w_vec[j] * w_vec[j] for j in range(dim))

    if total_var <= 1e-12:
        zero_coords = [[0.0, 0.0, 0.0] for _ in range(n_docs)]
        rounded_mean = [round(x, 6) for x in mean_vec]
        return PCAResult(
            coords=zero_coords,
            positions=zero_coords,
            mean=rounded_mean,
            mean_vec=rounded_mean,
            basis=default_basis,
            unit_basis=default_basis,
            scale=1.0,
            variance_explained=[0.0, 0.0, 0.0],
        )

    def _whiten_row(v: Sequence[float]) -> tuple[float, ...]:
        if len(v) >= dim:
            return tuple((float(v[j]) - mean_vec[j]) * w_vec[j] for j in range(dim))
        return tuple(((float(v[j]) if j < len(v) else 0.0) - mean_vec[j]) * w_vec[j] for j in range(dim))

    unit_basis: list[list[float]] = []

    if n_docs > dim:
        # Fast exact/strided D x D symmetric covariance path when N > D
        if n_docs > 700:
            step_cov = n_docs / 700.0
            sample_rows = [_whiten_row(vectors[int(k * step_cov)]) for k in range(700)]
        else:
            sample_rows = [_whiten_row(v) for v in vectors]
        n_cov = len(sample_rows)
        cols = [tuple(sample_rows[i][j] for i in range(n_cov)) for j in range(dim)]
        cov = [[0.0] * dim for _ in range(dim)]
        for a in range(dim):
            ca = cols[a]
            cov[a][a] = float(sp(ca, ca))
            row_a = cov[a]
            for b in range(a + 1, dim):
                cab = float(sp(ca, cols[b]))
                row_a[b] = cab
                cov[b][a] = cab

        for comp_idx in range(3):
            seed = [1.0 if j == comp_idx else 0.0 for j in range(dim)]
            w = _orthogonalize_unit(seed, unit_basis)
            if not any(w) or sum(cov[j][j] * abs(w[j]) for j in range(dim)) <= 1e-12:
                best_axis = max(range(dim), key=lambda j: cov[j][j])
                w = _orthogonalize_unit([1.0 if idx == best_axis else 0.0 for idx in range(dim)], unit_basis)
                if not any(w):
                    w = _fallback_orthogonal_unit(dim, unit_basis)
                    unit_basis.append(w)
                    continue

            for _ in range(max_iter):
                w_next = [float(sp(cov[r], w)) for r in range(dim)]
                w_next = _orthogonalize_unit(w_next, unit_basis)
                if not any(w_next):
                    break
                diff_sq = sum((w_next[j] - w[j]) ** 2 for j in range(dim))
                w = w_next
                if diff_sq <= 1e-12:
                    break

            w = _orthogonalize_unit(w, unit_basis)
            if not any(w):
                w = _fallback_orthogonal_unit(dim, unit_basis)
            else:
                cw = [float(sp(cov[r], w)) for r in range(dim)]
                eig_val = max(0.0, float(sp(w, cw)))
                if eig_val > 1e-12:
                    for a in range(dim):
                        ev_wa = eig_val * w[a]
                        row_a = cov[a]
                        for b in range(a, dim):
                            val = row_a[b] - ev_wa * w[b]
                            row_a[b] = val
                            cov[b][a] = val
            unit_basis.append(w)
    else:
        work = [list(_whiten_row(v)) for v in vectors]
        for comp_idx in range(3):
            # Deterministic seed selection: start with unit vector e_{comp_idx}, falling back to the
            # coordinate axis or deflated row with maximal projection energy if e_{comp_idx} is orthogonal.
            seed = [1.0 if j == comp_idx else 0.0 for j in range(dim)]
            w = _orthogonalize_unit(seed, unit_basis)
            z_probe = [_dot(row, w) for row in work] if any(w) else []
            if not z_probe or sum(val * val for val in z_probe) <= 1e-12:
                best_axis = -1
                best_energy = 0.0
                for j in range(dim):
                    col_energy = sum(work[i][j] * work[i][j] for i in range(n_docs))
                    if col_energy > best_energy + 1e-12:
                        cand = [1.0 if idx == j else 0.0 for idx in range(dim)]
                        cand_u = _orthogonalize_unit(cand, unit_basis)
                        if any(cand_u):
                            e_val = sum(_dot(row, cand_u) ** 2 for row in work)
                            if e_val > best_energy:
                                best_energy = e_val
                                best_axis = j
                if best_axis >= 0:
                    w = _orthogonalize_unit([1.0 if idx == best_axis else 0.0 for idx in range(dim)], unit_basis)
                else:
                    w = _fallback_orthogonal_unit(dim, unit_basis)
                    unit_basis.append(w)
                    continue

            for _ in range(max_iter):
                z = [_dot(row, w) for row in work]
                z_energy = sum(val * val for val in z)
                if z_energy <= 1e-14:
                    break
                w_next = [0.0] * dim
                for i, z_i in enumerate(z):
                    if abs(z_i) > 1e-15:
                        row_i = work[i]
                        for j in range(dim):
                            w_next[j] += row_i[j] * z_i
                w_next = _orthogonalize_unit(w_next, unit_basis)
                if not any(w_next):
                    break
                diff_sq = sum((w_next[j] - w[j]) ** 2 for j in range(dim))
                w = w_next
                if diff_sq <= 1e-13:
                    break

            w = _orthogonalize_unit(w, unit_basis)
            if not any(w):
                w = _fallback_orthogonal_unit(dim, unit_basis)
            else:
                z_final = [_dot(row, w) for row in work]
                eig_val = sum(val * val for val in z_final)
                if eig_val > 1e-12:
                    for i, p_i in enumerate(z_final):
                        if abs(p_i) > 1e-15:
                            work[i] = [work[i][j] - p_i * w[j] for j in range(dim)]

            unit_basis.append(w)

    rb0 = tuple(unit_basis[0][j] * w_vec[j] for j in range(dim))
    rb1 = tuple(unit_basis[1][j] * w_vec[j] for j in range(dim))
    rb2 = tuple(unit_basis[2][j] * w_vec[j] for j in range(dim))
    off_r0 = float(sp(mean_vec, rb0))
    off_r1 = float(sp(mean_vec, rb1))
    off_r2 = float(sp(mean_vec, rb2))
    ev0 = 0.0
    ev1 = 0.0
    ev2 = 0.0
    max_r2 = 0.0
    for v in vectors:
        vv = v if len(v) == dim else [float(v[j]) if j < len(v) else 0.0 for j in range(dim)]
        px = float(sp(vv, rb0)) - off_r0
        py = float(sp(vv, rb1)) - off_r1
        pz = float(sp(vv, rb2)) - off_r2
        ev0 += px * px
        ev1 += py * py
        ev2 += pz * pz
        r2 = px * px + py * py + pz * pz
        if r2 > max_r2:
            max_r2 = r2

    eigenvalues = [ev0, ev1, ev2]
    max_radius = math.sqrt(max_r2)
    scale = max_radius if max_radius > 1e-9 else 1.0
    scaled_basis = [[round((b[j] * w_vec[j]) / scale, 8) for j in range(dim)] for b in unit_basis]
    rounded_mean = [round(x, 6) for x in mean_vec]
    b0, b1, b2 = tuple(scaled_basis[0]), tuple(scaled_basis[1]), tuple(scaled_basis[2])
    off0 = float(sp(rounded_mean, b0))
    off1 = float(sp(rounded_mean, b1))
    off2 = float(sp(rounded_mean, b2))
    coords_3d: list[list[float]] = []
    for v in vectors:
        if len(v) == dim:
            coords_3d.append([round(float(sp(v, b0)) - off0, 4), round(float(sp(v, b1)) - off1, 4), round(float(sp(v, b2)) - off2, 4)])
        else:
            coords_3d.append(project_vector_3d(v, rounded_mean, scaled_basis))

    var_explained = [round(min(100.0, max(0.0, (ev / total_var) * 100.0)), 2) for ev in eigenvalues]

    return PCAResult(
        coords=coords_3d,
        positions=coords_3d,
        mean=rounded_mean,
        mean_vec=rounded_mean,
        basis=scaled_basis,
        unit_basis=[[round(x, 6) for x in b] for b in unit_basis],
        scale=round(scale, 6),
        variance_explained=var_explained,
    )


def _select_dominant_concept(
    doc_tokens: Sequence[str],
    custom_synonyms: dict[str, list[str]] | None = None,
    term_to_concepts: dict[str, set[str]] | None = None,
    stem_cache: dict[str, str] | None = None,
) -> tuple[str, dict[str, list[str]]]:
    """Return `(dominant_concept, concepts_dict)` where `dominant_concept` requires >= 2 distinct stems."""
    if term_to_concepts is None:
        _, term_to_concepts = build_concept_lookup(custom_synonyms)

    raw_grouped: dict[str, list[str]] = {}
    stems_per_cid: dict[str, set[str]] = {}
    seen_toks: set[str] = set()
    for raw in doc_tokens:
        tok = str(raw).lower().strip()
        if not tok or tok in seen_toks:
            continue
        seen_toks.add(tok)
        if stem_cache is not None:
            st = stem_cache.get(tok)
            if st is None:
                st = stem_token(tok)
                stem_cache[tok] = st
        else:
            st = stem_token(tok)
        cids = term_to_concepts.get(tok)
        stem_cids = term_to_concepts.get(st)
        if cids and stem_cids:
            merged_cids = cids | stem_cids
        else:
            merged_cids = cids or stem_cids
        if not merged_cids:
            continue
        for cid in merged_cids:
            raw_grouped.setdefault(cid, []).append(tok)
            stems_per_cid.setdefault(cid, set()).add(st)

    concepts_clean: dict[str, list[str]] = {}
    best_cid = "general"
    best_score: tuple[int, int, str] = (0, 0, "")

    for cid in sorted(raw_grouped):
        sorted_hits = sorted(raw_grouped[cid])
        concepts_clean[cid] = sorted_hits[:12]
        n_stems = len(stems_per_cid[cid])
        if n_stems >= 2:
            cand_score = (n_stems, len(sorted_hits), cid)
            if best_cid == "general" or cand_score[:2] > best_score[:2]:
                best_cid = cid
                best_score = cand_score

    return best_cid, concepts_clean


def _load_all_antigravity_parents(
    agy_db: Path,
    port_db: Path | None,
    max_docs: int,
    custom_synonyms: dict[str, list[str]] | None,
    port_conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """Load parent conversations from Antigravity `conversation_summaries.db` with cache + FTS + transcript fallback."""
    records: list[dict] = []
    if not agy_db.exists():
        return records

    cache_sig = compute_rag_signature(custom_synonyms)
    cached_unindexed, cached_fts = _load_antigravity_caches_from_conn(port_conn)

    try:
        conn, tmp_path = open_readonly_snapshot(agy_db)
    except sqlite3.Error:
        return records

    summary_rows: list[sqlite3.Row] = []
    fts_chunks_by_cid: dict[str, list[sqlite3.Row]] = {}
    fts_indexed_cids: set[str] = set()

    try:
        conn.row_factory = sqlite3.Row
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        if "conversation_summaries" not in tables:
            return records

        summary_rows = conn.execute(
            """
            SELECT conversation_id, title, preview, last_modified_time, step_count, workspace_uris
            FROM conversation_summaries
            WHERE COALESCE(parent_conversation_id, '') = ''
            ORDER BY last_modified_time DESC
            LIMIT ?
            """,
            (max(1, int(max_docs)),),
        ).fetchall()

        uncached_cids: list[str] = []
        for r in summary_rows:
            cid = r["conversation_id"]
            mtime = (r["last_modified_time"] or "")[:19]
            sc = int(r["step_count"] or 0)
            c_fts = cached_fts.get(cid)
            if (
                c_fts is not None
                and c_fts.get("cache_sig") == cache_sig
                and c_fts.get("last_modified_time") == mtime
                and int(c_fts.get("step_count", -1)) == sc
                and _parse_json_list(c_fts.get("tokens_json"))
                and _parse_json_list(c_fts.get("vector_json"))
            ):
                continue
            c_un = cached_unindexed.get(cid)
            if (
                c_un is not None
                and c_un.get("cache_version") == VECTOR_CACHE_VERSION
                and c_un.get("cache_sig") == cache_sig
                and c_un.get("last_modified_time") == mtime
                and int(c_un.get("step_count", -1)) == sc
                and _parse_json_list(c_un.get("tokens_json"))
                and _parse_json_list(c_un.get("vector_json"))
            ):
                continue
            uncached_cids.append(cid)

        if uncached_cids and "conversations_fts" in tables:
            fts_cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations_fts)").fetchall()}
            if fts_cols:
                u_col = "user_body" if "user_body" in fts_cols else ("user_text" if "user_text" in fts_cols else None)
                a_col = "agent_body" if "agent_body" in fts_cols else ("agent_text" if "agent_text" in fts_cols else None)
                t_col = "title" if "title" in fts_cols else None
                u_expr = f"COALESCE(f.{u_col}, '')" if u_col else "''"
                a_expr = f"COALESCE(f.{a_col}, '')" if a_col else "''"
                t_expr = f"COALESCE(f.{t_col}, '')" if t_col else "''"

                batch_size = 150
                for i in range(0, len(uncached_cids), batch_size):
                    batch = uncached_cids[i : i + batch_size]
                    placeholders = ",".join("?" for _ in batch)
                    chunk_rows = conn.execute(
                        f"""
                        SELECT
                            f.conversation_id,
                            {u_expr} AS user_snip,
                            {a_expr} AS agent_snip,
                            {t_expr} || ' ' || {u_expr} || ' ' || {a_expr} AS chunk_text
                        FROM conversations_fts f
                        WHERE f.conversation_id IN ({placeholders})
                        """,
                        batch,
                    ).fetchall()
                    for cr in chunk_rows:
                        fcid = cr["conversation_id"]
                        fts_indexed_cids.add(fcid)
                        bucket = fts_chunks_by_cid.setdefault(fcid, [])
                        if len(bucket) < 6:
                            bucket.append(cr)
    except sqlite3.Error:
        return records
    finally:
        conn.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    new_fts_cache_rows: list[tuple] = []
    new_unindexed_cache_rows: list[tuple] = []
    brain_dir = agy_db.parent / "brain"
    deadline = time.perf_counter() + 1.25

    for r in summary_rows:
        cid = r["conversation_id"]
        mtime = (r["last_modified_time"] or "")[:19]
        sc = int(r["step_count"] or 0)
        raw_title = (r["title"] or "").strip()
        preview = (r["preview"] or "").strip()
        display_title = raw_title or clean_snippet(preview, 80) or "(Untitled Conversation)"
        workspace_str = r["workspace_uris"] or ""

        c_fts = cached_fts.get(cid)
        if (
            c_fts is not None
            and c_fts.get("cache_sig") == cache_sig
            and c_fts.get("last_modified_time") == mtime
            and int(c_fts.get("step_count", -1)) == sc
        ):
            tok_list = [str(x) for x in _parse_json_list(c_fts.get("tokens_json"))]
            vec_list = [float(x) for x in _parse_json_list(c_fts.get("vector_json"))]
            if tok_list and len(vec_list) == VECTOR_DIM:
                records.append(
                    {
                        "id": cid,
                        "provider": "antigravity",
                        "title": c_fts.get("title") or display_title,
                        "last_modified_time": mtime,
                        "step_count": sc,
                        "workspace": c_fts.get("workspace") or workspace_str,
                        "summary": c_fts.get("summary") or clean_snippet(preview or display_title, 180),
                        "tokens": tok_list,
                        "vector": vec_list,
                    }
                )
                continue

        c_un = cached_unindexed.get(cid)
        if (
            c_un is not None
            and c_un.get("cache_version") == VECTOR_CACHE_VERSION
            and c_un.get("cache_sig") == cache_sig
            and c_un.get("last_modified_time") == mtime
            and int(c_un.get("step_count", -1)) == sc
        ):
            tok_list = [str(x) for x in _parse_json_list(c_un.get("tokens_json"))]
            vec_list = [float(x) for x in _parse_json_list(c_un.get("vector_json"))]
            if tok_list and len(vec_list) == VECTOR_DIM:
                records.append(
                    {
                        "id": cid,
                        "provider": "antigravity",
                        "title": c_un.get("title") or display_title,
                        "last_modified_time": mtime,
                        "step_count": sc,
                        "workspace": c_un.get("workspace") or workspace_str,
                        "summary": c_un.get("summary") or clean_snippet(preview or display_title, 180),
                        "tokens": tok_list,
                        "vector": vec_list,
                    }
                )
                continue

        if cid in fts_indexed_cids:
            chunks = fts_chunks_by_cid.get(cid, [])
            first_u = next((cr["user_snip"] for cr in chunks if cr["user_snip"]), "")
            first_a = next((cr["agent_snip"] for cr in chunks if cr["agent_snip"]), "")
            summary = build_summary_line(first_u, first_a, preview or display_title)
            parts = [display_title, preview] + [cr["chunk_text"] for cr in chunks if cr["chunk_text"]]
            combined = " ".join(p for p in parts if p)[:120000]
            tok_list = sorted(set(tokenize_domain(combined)) | set(tokenize_domain(display_title)))
            if not tok_list:
                tok_list = [display_title.lower()[:24]] if display_title else ["session"]
            vec = compute_semantic_vector(tok_list, custom_synonyms=custom_synonyms)
            tok_json = json.dumps(tok_list)
            vec_json = json.dumps([round(v, 6) for v in vec])
            new_fts_cache_rows.append(
                (cid, mtime, sc, display_title, workspace_str, summary, tok_json, vec_json, cache_sig)
            )
            records.append(
                {
                    "id": cid,
                    "provider": "antigravity",
                    "title": display_title,
                    "last_modified_time": mtime,
                    "step_count": sc,
                    "workspace": workspace_str,
                    "summary": summary,
                    "tokens": tok_list,
                    "vector": vec,
                }
            )
        else:
            parts = [raw_title, preview]
            users: list[str] = []
            agents: list[str] = []
            tp = brain_dir / cid / ".system_generated" / "logs" / "transcript.jsonl"
            transcript_complete = False
            if time.perf_counter() < deadline:
                if tp.exists():
                    users, agents = _extract_antigravity_transcript_preview(tp)
                    parts.extend(users + agents)
                transcript_complete = True

            first_u = users[0] if users else preview
            last_a = agents[-1] if agents else ""
            summary = build_summary_line(first_u, last_a, preview or display_title)
            clean_parts = [p for p in parts if p]
            combined = _bounded_head_tail_text(" ".join(clean_parts), 120000)
            tok_list = sorted(set(tokenize_domain(combined)) | set(tokenize_domain(display_title)))
            if not tok_list:
                tok_list = [display_title.lower()[:24]] if display_title else ["session"]
            vec = compute_semantic_vector(tok_list, custom_synonyms=custom_synonyms)
            tok_json = json.dumps(tok_list)
            vec_json = json.dumps([round(v, 6) for v in vec])
            if transcript_complete:
                new_unindexed_cache_rows.append(
                    (
                        cid,
                        mtime,
                        sc,
                        display_title,
                        workspace_str,
                        summary,
                        "",
                        vec_json,
                        VECTOR_CACHE_VERSION,
                        tok_json,
                        cache_sig,
                    )
                )
            records.append(
                {
                    "id": cid,
                    "provider": "antigravity",
                    "title": display_title,
                    "last_modified_time": mtime,
                    "step_count": sc,
                    "workspace": workspace_str,
                    "summary": summary,
                    "tokens": tok_list,
                    "vector": vec,
                }
            )

    if new_unindexed_cache_rows or new_fts_cache_rows:
        _save_antigravity_caches(
            port_db,
            new_unindexed_cache_rows,
            new_fts_cache_rows,
            indexed_cids=fts_indexed_cids if fts_indexed_cids else None,
        )

    return records


def _load_all_portable_conversations(
    port_conn: sqlite3.Connection | None,
    max_docs: int,
    custom_synonyms: dict[str, list[str]] | None,
) -> list[dict]:
    """Load conversations (`claude`, `codex`, `generic`) from portable `~/.v1-memory/index.db`."""
    records: list[dict] = []
    if port_conn is None:
        return records

    cache_sig = compute_rag_signature(custom_synonyms)
    try:
        port_conn.row_factory = sqlite3.Row
        cols = {row[1] for row in port_conn.execute("PRAGMA table_info(conversations)").fetchall()}
        if not cols:
            return records
        vec_expr = "vector_json" if "vector_json" in cols else "'[]' AS vector_json"
        tok_expr = "tokens_json" if "tokens_json" in cols else "'[]' AS tokens_json"
        sig_expr = "cache_sig" if "cache_sig" in cols else "'' AS cache_sig"

        rows = port_conn.execute(
            f"""
            SELECT conversation_id, provider, title, last_modified_time, step_count,
                   workspace, summary, user_text, agent_text,
                   {vec_expr}, {tok_expr}, {sig_expr}
            FROM conversations
            ORDER BY last_modified_time DESC
            LIMIT ?
            """,
            (max(1, int(max_docs)),),
        ).fetchall()

        for r in rows:
            cid = r["conversation_id"]
            title = (r["title"] or "").strip() or "(Untitled Session)"
            mtime = (r["last_modified_time"] or "")[:19]
            sc = int(r["step_count"] or 0)
            tok_list = [str(x) for x in _parse_json_list(r["tokens_json"])]
            if not tok_list:
                combined = f"{title}\n{r['user_text'] or ''}\n{r['agent_text'] or ''}"
                tok_list = sorted(set(tokenize_domain(combined)) | set(tokenize_domain(title)))
            if not tok_list:
                tok_list = [title.lower()[:24]]

            vec_list: list[float] = []
            if r["cache_sig"] == cache_sig or not r["cache_sig"]:
                parsed_v = _parse_json_list(r["vector_json"])
                if len(parsed_v) == VECTOR_DIM:
                    vec_list = [float(x) for x in parsed_v]
            if len(vec_list) != VECTOR_DIM:
                vec_list = compute_semantic_vector(tok_list, custom_synonyms=custom_synonyms)

            records.append(
                {
                    "id": cid,
                    "provider": r["provider"] or "generic",
                    "title": title,
                    "last_modified_time": mtime,
                    "step_count": sc,
                    "workspace": r["workspace"] or "",
                    "summary": r["summary"] or clean_snippet(title, 180),
                    "tokens": tok_list,
                    "vector": vec_list,
                }
            )
    except sqlite3.Error:
        return records

    return records


def _compute_knn_topology(
    unit_vecs: Sequence[tuple[float, ...]],
    all_records: Sequence[dict],
    high_df_tokens: set[str],
    top_k: int = KNN_TOP_K,
    min_cos: float = KNN_MIN_COSINE,
) -> list[list[dict]]:
    """Compute bounded k-NN cosine topology cleanly from N=0 up to N=10,000+ documents."""
    n_nodes = len(all_records)
    if n_nodes <= 1:
        return [[] for _ in range(n_nodes)]

    heaps: list[list[tuple[float, str]]] = [[] for _ in range(n_nodes)]
    sp = _SUMPROD

    if n_nodes <= 1200:
        for i in range(n_nodes):
            ui = unit_vecs[i]
            id_i = all_records[i]["id"]
            hi = heaps[i]
            for j in range(i + 1, n_nodes):
                cos_ij = float(sp(ui, unit_vecs[j])) if sp is not None else float(sum(map(mul, ui, unit_vecs[j])))
                if cos_ij < min_cos:
                    continue
                r_cos = round(cos_ij, 4)
                cand_i = (r_cos, all_records[j]["id"])
                if len(hi) < top_k:
                    heapq.heappush(hi, cand_i)
                elif cand_i > hi[0]:
                    heapq.heapreplace(hi, cand_i)
                hj = heaps[j]
                cand_j = (r_cos, id_i)
                if len(hj) < top_k:
                    heapq.heappush(hj, cand_j)
                elif cand_j > hj[0]:
                    heapq.heapreplace(hj, cand_j)
    else:
        # Large corpus path: single-pass top-magnitude dim index + IDF-sorted token postings + 64-bit packed pair dedup
        dim = len(unit_vecs[0]) if unit_vecs[0] else VECTOR_DIM
        dim_postings: list[list[int]] = [[] for _ in range(dim)]
        top_dims_by_doc: list[list[int]] = []
        for idx, ui in enumerate(unit_vecs):
            active_d = [d for d, val in enumerate(ui) if val >= 0.085 or val <= -0.085][:5]
            top_dims_by_doc.append(active_d)
            for d in active_d:
                dim_postings[d].append(idx)

        tok_postings: dict[str, list[int]] = {}
        for idx, rec in enumerate(all_records):
            for tok in rec["tokens"]:
                if tok not in high_df_tokens:
                    lst = tok_postings.get(tok)
                    if lst is None:
                        tok_postings[tok] = [idx]
                    else:
                        lst.append(idx)

        ids = [rec["id"] for rec in all_records]
        seen_pairs: set[int] = set()
        for i in range(n_nodes):
            cands: list[int] = []
            cand_set: set[int] = {i}

            # Prioritize rarest (highest-IDF) token buckets first, stride-sampled across full bucket
            tok_lists = [
                tok_postings[t]
                for t in all_records[i]["tokens"]
                if t in tok_postings and len(tok_postings[t]) >= 2
            ]
            tok_lists.sort(key=len)
            for lst in tok_lists[:4]:
                ln = len(lst)
                if ln <= 10:
                    for j in lst:
                        if j not in cand_set:
                            cand_set.add(j)
                            cands.append(j)
                else:
                    stride = ln // 10
                    taken = 0
                    for pos in range(i % stride, ln, stride):
                        j = lst[pos]
                        if j not in cand_set:
                            cand_set.add(j)
                            cands.append(j)
                        taken += 1
                        if taken >= 10:
                            break
                if len(cands) >= 28:
                    break

            # Fill with rarest active top-magnitude dimension postings only when token candidates are sparse
            if len(cands) < 14:
                dim_lists = sorted(
                    (dim_postings[d] for d in top_dims_by_doc[i] if len(dim_postings[d]) >= 2),
                    key=len,
                )
                for lst in dim_lists:
                    ln = len(lst)
                    if ln <= 10:
                        for j in lst:
                            if j not in cand_set:
                                cand_set.add(j)
                                cands.append(j)
                    else:
                        stride = ln // 10
                        taken = 0
                        for pos in range(i % stride, ln, stride):
                            j = lst[pos]
                            if j not in cand_set:
                                cand_set.add(j)
                                cands.append(j)
                            taken += 1
                            if taken >= 10:
                                break
                    if len(cands) >= 28:
                        break

            ui = unit_vecs[i]
            id_i = ids[i]
            hi = heaps[i]
            for j in cands:
                a, b = (i, j) if i < j else (j, i)
                pair_key = (a << 32) | b
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                cos_ij = float(sp(ui, unit_vecs[j])) if sp is not None else float(sum(map(mul, ui, unit_vecs[j])))
                if cos_ij < min_cos:
                    continue
                r_cos = round(cos_ij, 4)
                cand_i = (r_cos, ids[j])
                if len(hi) < top_k:
                    heapq.heappush(hi, cand_i)
                elif cand_i > hi[0]:
                    heapq.heapreplace(hi, cand_i)
                hj = heaps[j]
                cand_j = (r_cos, id_i)
                if len(hj) < top_k:
                    heapq.heappush(hj, cand_j)
                elif cand_j > hj[0]:
                    heapq.heapreplace(hj, cand_j)

    knn_by_idx: list[list[dict]] = []
    for h in heaps:
        h.sort(reverse=True)
        knn_by_idx.append([{"id": cid, "cos": c} for c, cid in h[:top_k]])
    return knn_by_idx


def _supports_idf_weights() -> bool:
    try:
        return "idf_weights" in inspect.signature(compute_semantic_vector).parameters
    except (TypeError, ValueError):
        return False


_CLUSTER_LABEL_NOISE_WORDS: frozenset[str] = frozenset({
    "href", "http", "https", "www", "com", "org", "edu", "html", "quot", "amp",
    "reuters", "ap", "afp", "inc", "corp", "ltd", "anyone", "someone", "everyone",
    "says", "said", "told", "wrote", "writes", "article", "post", "posted",
    "subject", "organization", "lines", "nntp", "posting", "host", "university",
    "year", "years", "week", "month", "time", "day", "days", "night",
    "people", "person", "good", "better", "best", "new", "first", "two", "one",
    "also", "many", "much", "even", "well", "make", "made", "know", "think",
    "present", "based", "using", "used", "use", "system", "systems", "general",
    "information", "service", "services", "program", "programs", "number", "world",
    "state", "states", "national", "group", "work", "way", "case", "set", "high",
    "large", "small", "long", "right", "left", "place", "paper", "life",
    "company", "companies", "million", "billion", "percent", "last", "next",
    "back", "still", "take", "taken", "including", "according", "report",
    "reported", "officials", "ago", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
})


def _cluster_general_docs_kmeans(
    general_indices: Sequence[int],
    unit_vecs: Sequence[tuple[float, ...]],
    all_records: Sequence[dict],
    n_corpus: int,
    df: dict[str, int],
    high_df_tokens: set[str],
    max_iter: int = 5,
) -> tuple[dict[int, str], list[str]]:
    """Run deterministic pure-stdlib spherical k-means on unclustered 'general' docs and name via TF-IDF."""
    n_gen = len(general_indices)
    n_total = len(all_records)
    if n_gen < 2:
        return {}, []

    k = min(18, max(6, int(math.sqrt(n_total / 25.0))))
    k = min(k, n_gen)
    if k <= 1:
        return {}, []

    dim = len(unit_vecs[general_indices[0]]) if unit_vecs[general_indices[0]] else VECTOR_DIM
    gen_vecs = [unit_vecs[idx] for idx in general_indices]
    sp = _SUMPROD

    # When N_gen is large, fit spherical k-means centroids on a deterministic strided subsample (<= 500)
    # then assign all N_gen vectors in a single pass for sub-second O(N * K) scaling.
    fit_cap = 500
    if n_gen > fit_cap:
        step_km = n_gen / float(fit_cap)
        fit_vecs = [gen_vecs[int(i * step_km)] for i in range(fit_cap)]
    else:
        fit_vecs = gen_vecs
    n_fit = len(fit_vecs)

    # Deterministic farthest-first centroid initialization
    seed0 = next((m for m, gv in enumerate(fit_vecs) if any(abs(x) > 1e-12 for x in gv)), 0)
    centroids: list[tuple[float, ...]] = [fit_vecs[seed0]]
    max_sim = [
        float(sp(gv, centroids[0])) if sp is not None else float(sum(map(mul, gv, centroids[0])))
        for gv in fit_vecs
    ]
    for _ in range(1, k):
        farthest_m = min(range(n_fit), key=lambda m: max_sim[m])
        c_new = fit_vecs[farthest_m]
        centroids.append(c_new)
        for m, gv in enumerate(fit_vecs):
            s = float(sp(gv, c_new)) if sp is not None else float(sum(map(mul, gv, c_new)))
            if s > max_sim[m]:
                max_sim[m] = s

    fit_assignments = [0] * n_fit
    move_tol = max(1, n_fit // 200)
    for _ in range(max_iter):
        changed = 0
        best_sims = [0.0] * n_fit
        sums = [[0.0] * dim for _ in range(k)]
        counts = [0] * k

        for m, gv in enumerate(fit_vecs):
            best_c = 0
            best_s = -2.0
            for c_idx, cent in enumerate(centroids):
                s = float(sp(gv, cent)) if sp is not None else float(sum(map(mul, gv, cent)))
                if s > best_s:
                    best_s = s
                    best_c = c_idx
            if fit_assignments[m] != best_c:
                changed += 1
                fit_assignments[m] = best_c
            best_sims[m] = best_s
            counts[best_c] += 1
            acc = sums[best_c]
            for j in range(dim):
                acc[j] += gv[j]

        new_centroids: list[tuple[float, ...]] = []
        for c_idx in range(k):
            if counts[c_idx] > 0:
                new_centroids.append(_to_unit_tuple(sums[c_idx]))
            else:
                reseed_m = min(range(n_fit), key=lambda m: best_sims[m])
                fit_assignments[reseed_m] = c_idx
                best_sims[reseed_m] = 1.0
                new_centroids.append(fit_vecs[reseed_m])
        centroids = new_centroids
        if changed <= move_tol:
            break

    if n_gen > fit_cap:
        assignments = [0] * n_gen
        for m, gv in enumerate(gen_vecs):
            best_c = 0
            best_s = -2.0
            for c_idx, cent in enumerate(centroids):
                s = float(sp(gv, cent)) if sp is not None else float(sum(map(mul, gv, cent)))
                if s > best_s:
                    best_s = s
                    best_c = c_idx
            assignments[m] = best_c
    else:
        assignments = fit_assignments

    members_by_c: list[list[int]] = [[] for _ in range(k)]
    for m, c_idx in enumerate(assignments):
        members_by_c[c_idx].append(general_indices[m])

    # Name each non-empty cluster as "topic:<top1>_<top2>" via contrastive cluster specificity
    cluster_meta: list[tuple[int, str, list[int]]] = []
    used_names: set[str] = set()
    used_primary_stems: set[str] = set()
    ordered_c = sorted(range(k), key=lambda c: (-len(members_by_c[c]), c))

    for c_idx in ordered_c:
        members = members_by_c[c_idx]
        if not members:
            continue
        cluster_df_clean: dict[str, int] = {}
        cluster_df_all: dict[str, int] = {}
        for doc_idx in members:
            for tok in set(all_records[doc_idx]["tokens"]):
                if tok in high_df_tokens or len(tok) < 3 or tok.isdigit():
                    continue
                cluster_df_all[tok] = cluster_df_all.get(tok, 0) + 1
                if tok not in _CLUSTER_LABEL_NOISE_WORDS:
                    cluster_df_clean[tok] = cluster_df_clean.get(tok, 0) + 1

        cluster_df = cluster_df_clean if cluster_df_clean else cluster_df_all
        min_df_in_cluster = max(2, len(members) // 25)
        thresholds = [min_df_in_cluster]
        if min_df_in_cluster > 2:
            thresholds.append(2)
        if min_df_in_cluster > 1:
            thresholds.append(1)

        scored: list[tuple[float, int, str]] = []
        for thresh in thresholds:
            for tok, c_df in cluster_df.items():
                if c_df < thresh:
                    continue
                idf = _bm25_idf(tok, n_corpus, df, high_df_tokens)
                global_df = max(1, df.get(tok, c_df))
                specificity = c_df * idf * ((c_df / float(global_df)) ** 0.75)
                scored.append((specificity, c_df, tok))
            if scored:
                break

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        chosen_candidates: list[tuple[str, str]] = []
        chosen_stems: set[str] = set()
        for _, _, tok in scored:
            st = stem_token(tok)
            if st in chosen_stems:
                continue
            slug = re.sub(r"[^a-z0-9]+", "", tok.lower())
            if len(slug) < 2:
                continue
            chosen_stems.add(st)
            chosen_candidates.append((slug[:20], st))
            if len(chosen_candidates) >= 8:
                break

        if len(chosen_candidates) >= 2:
            primary_idx = next(
                (i for i, (_, st) in enumerate(chosen_candidates) if st not in used_primary_stems),
                0,
            )
            primary_slug, primary_stem = chosen_candidates[primary_idx]
            cand_name = ""
            for sec_idx, (sec_slug, _) in enumerate(chosen_candidates):
                if sec_idx == primary_idx:
                    continue
                probe = f"topic:{primary_slug}_{sec_slug}"
                if probe not in used_names:
                    cand_name = probe
                    break
            if not cand_name:
                sec_idx = 1 if primary_idx == 0 else 0
                cand_name = f"topic:{primary_slug}_{chosen_candidates[sec_idx][0]}"
            used_primary_stems.add(primary_stem)
        elif len(chosen_candidates) == 1:
            primary_slug, primary_stem = chosen_candidates[0]
            used_primary_stems.add(primary_stem)
            cand_name = f"topic:{primary_slug}"
        else:
            cand_name = f"topic:cluster_{c_idx + 1}"

        final_name = cand_name
        suffix = 2
        while final_name in used_names:
            final_name = f"{cand_name}_{suffix}"
            suffix += 1
        used_names.add(final_name)
        cluster_meta.append((len(members), final_name, members))

    assignments_out: dict[int, str] = {}
    dynamic_names: list[str] = []
    for _, cname, members in cluster_meta:
        dynamic_names.append(cname)
        for doc_idx in members:
            assignments_out[doc_idx] = cname

    return assignments_out, dynamic_names


def collect_vector_corpus(
    home: Path | None = None,
    state_home: Path | None = None,
    max_docs: int = 5000,
    agy_db_path: Path | None = None,
    index_db_path: Path | None = None,
    custom_synonyms: dict[str, list[str]] | None = None,
    default_threshold: float | None = None,
) -> dict:
    """Collect parent conversations across Antigravity and `~/.v1-memory/index.db`, project to 3D PCA, and build k-NN graph."""
    user_home = Path(home) if home is not None else Path.home()
    resolved_state = default_state_home(Path(state_home) if state_home is not None else None)
    cfg = load_config(resolved_state)

    syns = custom_synonyms if custom_synonyms is not None else (
        cfg.get("synonyms") if isinstance(cfg.get("synonyms"), dict) else _load_configured_synonyms(resolved_state)
    )
    eff_threshold = (
        float(default_threshold)
        if default_threshold is not None
        else float(cfg.get("threshold_pct", 40.0))
    )
    semantic_weight = float(cfg.get("semantic_weight", 0.45))

    agy_db = Path(agy_db_path) if agy_db_path is not None else (
        user_home / ".gemini" / "antigravity-cli" / "conversation_summaries.db"
    )
    port_db = Path(index_db_path) if index_db_path is not None else (resolved_state / "index.db")

    port_conn: sqlite3.Connection | None = None
    port_tmp: str | None = None
    if port_db.exists():
        try:
            port_conn, port_tmp = open_readonly_snapshot(port_db)
        except sqlite3.Error:
            port_conn = None

    try:
        agy_records = _load_all_antigravity_parents(
            agy_db,
            port_db,
            max_docs=max_docs,
            custom_synonyms=syns,
            port_conn=port_conn,
        )
        port_records = _load_all_portable_conversations(
            port_conn,
            max_docs=max_docs,
            custom_synonyms=syns,
        )
    finally:
        if port_conn is not None:
            port_conn.close()
            if port_tmp and os.path.exists(port_tmp):
                try:
                    os.unlink(port_tmp)
                except OSError:
                    pass

    merged_by_id: dict[str, dict] = {}
    for rec in agy_records + port_records:
        cid = rec["id"]
        if cid not in merged_by_id:
            merged_by_id[cid] = rec

    all_records = sorted(
        merged_by_id.values(),
        key=lambda item: (item.get("last_modified_time", ""), item.get("id", "")),
        reverse=True,
    )[: max(1, int(max_docs))]

    doc_tokens_by_cid: dict[str, set[str]] = {
        rec["id"]: set(rec["tokens"]) for rec in all_records
    }
    n_corpus, df, high_df_tokens = _compute_corpus_stats(doc_tokens_by_cid, len(doc_tokens_by_cid))
    idf_weights: dict[str, float] = {
        tok: _bm25_idf(tok, n_corpus, df, high_df_tokens)
        for tok in df
        if tok not in high_df_tokens
    }

    if all_records:
        for rec in all_records:
            existing_vec = rec.get("vector")
            if not existing_vec or len(existing_vec) != VECTOR_DIM:
                active_toks = (
                    [t for t in rec["tokens"] if t not in high_df_tokens]
                    if high_df_tokens
                    else rec["tokens"]
                ) or rec["tokens"]
                rec["vector"] = compute_semantic_vector(active_toks, custom_synonyms=syns)
            elif high_df_tokens:
                active_toks = [t for t in rec["tokens"] if t not in high_df_tokens]
                if active_toks and len(active_toks) < len(rec["tokens"]):
                    rec["vector"] = compute_semantic_vector(active_toks, custom_synonyms=syns)

    vectors = [rec["vector"] for rec in all_records]
    pca = fit_pca_3d(vectors)
    coords_3d = pca["coords"]
    pca_mean = pca["mean"]
    pca_basis = pca["basis"]
    variance_explained = pca["variance_explained"]

    # Precompute unit vectors once + pairwise 256-dim cosine k-NN edges
    unit_vecs = [_to_unit_tuple(v) for v in vectors]
    knn_by_idx = _compute_knn_topology(unit_vecs, all_records, high_df_tokens)

    # First pass: concept cluster matching via DEFAULT_CONCEPT_CLUSTERS + custom synonyms
    merged_clusters, term_to_concepts = build_concept_lookup(syns)
    stem_cache: dict[str, str] = {}
    dom_concepts: list[str] = []
    concepts_maps: list[dict[str, list[str]]] = []
    general_indices: list[int] = []
    for idx, rec in enumerate(all_records):
        dom, cmap = _select_dominant_concept(
            rec["tokens"],
            syns,
            term_to_concepts=term_to_concepts,
            stem_cache=stem_cache,
        )
        dom_concepts.append(dom)
        concepts_maps.append(cmap)
        if dom == "general":
            general_indices.append(idx)

    # Dynamic spherical k-means fallback when >75% of corpus (>= 12 docs) collapses into "general"
    dynamic_cluster_names: list[str] = []
    n_nodes = len(all_records)
    if n_nodes >= 12 and (len(general_indices) / n_nodes) > 0.75:
        dyn_assignments, dynamic_cluster_names = _cluster_general_docs_kmeans(
            general_indices,
            unit_vecs,
            all_records,
            n_corpus,
            df,
            high_df_tokens,
        )
        for idx, dyn_name in dyn_assignments.items():
            dom_concepts[idx] = dyn_name

    nodes: list[dict] = []
    node_details_by_id: dict[str, dict] = {}
    vectors_by_id: dict[str, list[float]] = {}
    unit_vectors_by_id: dict[str, tuple[float, ...]] = {}
    cluster_positions: dict[str, list[list[float]]] = {}
    idf_rounded = {tok: round(val, 4) for tok, val in idf_weights.items()}

    for idx, rec in enumerate(all_records):
        cid = rec["id"]
        doc_tokens = rec["tokens"]
        title_tokens = set(tokenize_domain(rec["title"]))
        dominant_concept = dom_concepts[idx]
        concepts_map = concepts_maps[idx]

        scored_tokens: list[tuple[tuple[float, int, int, str], str]] = []
        for tok in doc_tokens:
            if tok in high_df_tokens:
                continue
            r_idf = idf_rounded.get(tok)
            if r_idf is None:
                r_idf = round(_bm25_idf(tok, n_corpus, df, high_df_tokens), 4)
            in_title = 1 if tok in title_tokens else 0
            scored_tokens.append(((r_idf, in_title, min(len(tok), 16), tok), tok))
        top_tokens = [tok for _, tok in heapq.nlargest(TOP_TOKENS_PER_DOC, scored_tokens, key=lambda item: item[0])]
        if not top_tokens:
            top_tokens = list(doc_tokens[:TOP_TOKENS_PER_DOC])

        pos = coords_3d[idx] if idx < len(coords_3d) else [0.0, 0.0, 0.0]
        vec = rec["vector"]
        vec_preview = compute_vector_preview(vec, buckets=VECTOR_PREVIEW_BUCKETS)
        summary_clean = clean_snippet(rec["summary"], 220)
        vectors_by_id[cid] = vec
        unit_vectors_by_id[cid] = unit_vecs[idx]
        cluster_positions.setdefault(dominant_concept, []).append(pos)

        node_obj = {
            "id": cid,
            "provider": rec["provider"],
            "title": rec["title"],
            "last_modified_time": rec["last_modified_time"],
            "step_count": int(rec["step_count"]),
            "summary": summary_clean,
            "dominant_concept": dominant_concept,
            "concepts": concepts_map,
            "top_tokens": top_tokens,
            "vector_preview": vec_preview,
            "pos": pos,
            "knn": knn_by_idx[idx],
        }
        nodes.append(node_obj)
        node_details_by_id[cid] = {
            "id": cid,
            "provider": rec["provider"],
            "title": rec["title"],
            "last_modified_time": rec["last_modified_time"],
            "step_count": int(rec["step_count"]),
            "dominant_concept": dominant_concept,
            "summary": summary_clean,
            "top_tokens": top_tokens,
            "vector_preview": vec_preview,
            "concepts": concepts_map,
        }

    cluster_names = list(
        dict.fromkeys(
            list(DEFAULT_CONCEPT_CLUSTERS.keys())
            + list(merged_clusters.keys())
            + dynamic_cluster_names
            + ["general"]
        )
    )

    centroids: list[dict] = []
    for concept_name in cluster_names:
        pts = cluster_positions.get(concept_name)
        if not pts:
            continue
        cnt = len(pts)
        cx = round(sum(p[0] for p in pts) / cnt, 4)
        cy = round(sum(p[1] for p in pts) / cnt, 4)
        cz = round(sum(p[2] for p in pts) / cnt, 4)
        centroids.append(
            {
                "concept": concept_name,
                "count": cnt,
                "pos": [cx, cy, cz],
            }
        )

    public_graph = {
        "version": VECTOR_CACHE_VERSION,
        "live_api": True,
        "default_threshold": eff_threshold,
        "variance_explained": variance_explained,
        "cluster_names": cluster_names,
        "nodes": nodes,
        "centroids": centroids,
    }

    return {
        **public_graph,
        "public_graph": public_graph,
        "node_details_by_id": node_details_by_id,
        "vectors_by_id": vectors_by_id,
        "unit_vectors_by_id": unit_vectors_by_id,
        "idf_weights": idf_weights,
        "pca_mean": pca_mean,
        "pca_basis": pca_basis,
        "pca_scale": pca["scale"],
        "home": user_home,
        "state_home": resolved_state,
        "agy_db_path": agy_db,
        "index_db_path": port_db,
        "custom_synonyms": syns,
        "semantic_weight": semantic_weight,
    }


def build_public_graph(state: dict, live_api: bool = True) -> dict:
    """Extract the clean JSON-serializable public graph dictionary from `state`.

    When `live_api=True` and `len(nodes) > 400`, omits heavy per-node inspector fields
    (`summary`, `top_tokens`, `vector_preview`, `concepts`) so `/api/graph` and initial HTML
    remain compact while `/api/node?id=<cid>` hydrates inspector details lazily on click.
    """
    base = state.get("public_graph") if isinstance(state.get("public_graph"), dict) else state
    raw_nodes = list(base.get("nodes", []))
    if live_api and len(raw_nodes) > 400:
        nodes_out = [
            {
                k: v
                for k, v in node.items()
                if k not in ("summary", "top_tokens", "vector_preview", "concepts")
            }
            for node in raw_nodes
        ]
    else:
        nodes_out = raw_nodes

    return {
        "version": str(base.get("version", VECTOR_CACHE_VERSION)),
        "live_api": bool(live_api),
        "default_threshold": float(base.get("default_threshold", 40.0)),
        "variance_explained": list(base.get("variance_explained", [0.0, 0.0, 0.0])),
        "cluster_names": list(
            base.get("cluster_names", list(DEFAULT_CONCEPT_CLUSTERS.keys()) + ["general"])
        ),
        "nodes": nodes_out,
        "centroids": list(base.get("centroids", [])),
    }


def query_rag_space(
    query_text: str,
    state: dict,
    threshold_pct: float = 40.0,
    hybrid_rag: bool = True,
) -> dict:
    """Run a live Hybrid-RAG or Sparse-Only probe and project the query vector into the 3D PCA space."""
    t0 = time.perf_counter()
    clean_query = str(query_text or "").strip()
    eff_threshold = (
        float(threshold_pct)
        if threshold_pct is not None
        else float(state.get("default_threshold", 40.0))
    )
    eff_threshold = max(0.0, min(100.0, eff_threshold))
    use_hybrid = bool(hybrid_rag)

    syns = state.get("custom_synonyms") if isinstance(state.get("custom_synonyms"), dict) else {}
    semantic_weight = float(state.get("semantic_weight", 0.45))
    home = state.get("home")
    state_home = state.get("state_home")
    agy_db_path = state.get("agy_db_path")
    index_db_path = state.get("index_db_path")

    q_tokens = tokenize_domain(clean_query)
    raw_q_concepts = group_tokens_by_concept(q_tokens, syns)
    q_concepts = {
        (k.split(":", 1)[1] if ":" in k else k): list(v)
        for k, v in raw_q_concepts.items()
    }

    idf_weights = state.get("idf_weights") if isinstance(state.get("idf_weights"), dict) else None
    if q_tokens:
        if idf_weights and _supports_idf_weights():
            q_vec = compute_semantic_vector(q_tokens, custom_synonyms=syns, idf_weights=idf_weights)
        else:
            q_vec = compute_semantic_vector(q_tokens, custom_synonyms=syns)
    else:
        q_vec = [0.0] * VECTOR_DIM
    q_vector_preview = compute_vector_preview(q_vec, buckets=VECTOR_PREVIEW_BUCKETS)

    pca_mean = state.get("pca_mean") or ([0.0] * VECTOR_DIM)
    pca_basis = state.get("pca_basis") or [[1.0 if j == k else 0.0 for j in range(VECTOR_DIM)] for k in range(3)]
    if q_tokens and any(abs(x) > 1e-12 for x in q_vec):
        q_pos = project_vector_3d(q_vec, pca_mean, pca_basis)
        r = math.sqrt(q_pos[0] * q_pos[0] + q_pos[1] * q_pos[1] + q_pos[2] * q_pos[2])
        if r > 1.15:
            scale_clamp = 1.15 / r
            q_pos = [round(c * scale_clamp, 4) for c in q_pos]
    else:
        q_pos = [0.0, 0.0, 0.0]

    _, matches = find_context_matches(
        clean_query,
        current_conv_id="",
        threshold_pct=eff_threshold,
        min_matched_tokens=2,
        max_matches=8,
        home=home,
        state_home=state_home,
        agy_db_path=agy_db_path,
        index_db_path=index_db_path,
        hybrid_rag=use_hybrid,
        semantic_weight=semantic_weight,
        custom_synonyms=syns,
    )

    unit_vectors_by_id: dict[str, tuple[float, ...]] = state.get("unit_vectors_by_id") or {}
    vectors_by_id: dict[str, list[float]] = state.get("vectors_by_id") or {}
    nodes: list[dict] = state.get("nodes") or []
    nearest_cos: list[dict] = []
    if q_tokens:
        q_unit = _to_unit_tuple(q_vec)
        cos_scored: list[tuple[float, str, str]] = []
        for node in nodes:
            nid = str(node.get("id", ""))
            u_vec = unit_vectors_by_id.get(nid)
            if u_vec is not None:
                c = round(max(0.0, float(sum(map(mul, q_unit, u_vec)))), 4)
            else:
                d_vec = vectors_by_id.get(nid)
                if not d_vec:
                    continue
                c = round(max(0.0, cosine_similarity(q_vec, d_vec)), 4)
            if c > 0.0:
                cos_scored.append((c, nid, str(node.get("title", ""))))
        cos_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        nearest_cos = [
            {"id": nid, "title": title, "cos": c}
            for c, nid, title in cos_scored[:5]
        ]

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "query": clean_query,
        "query_tokens": q_tokens,
        "q_concepts": q_concepts,
        "q_pos": q_pos,
        "q_vector_preview": q_vector_preview,
        "hybrid_rag": use_hybrid,
        "threshold_pct": eff_threshold,
        "elapsed_ms": elapsed_ms,
        "matches": matches,
        "nearest_cos": nearest_cos,
    }


def _render_html_with_ui(initial_graph_json: str, live_api: bool = True) -> str:
    """Render the self-contained 3D HTML visualizer via `v1_memory.viz_ui`."""
    from v1_memory.viz_ui import render_visualizer_html

    return render_visualizer_html(initial_graph_json, live_api=live_api)


def export_static_html(
    output_path: Path | str,
    state: dict | None = None,
    home: Path | None = None,
    state_home: Path | None = None,
    max_docs: int = 500,
    default_threshold: float | None = None,
) -> Path:
    """Write a self-contained offline 3D RAG Vector Explorer HTML file (`live_api=False`)."""
    corpus_state = (
        state
        if state is not None
        else collect_vector_corpus(
            home=home,
            state_home=state_home,
            max_docs=max_docs,
            default_threshold=default_threshold,
        )
    )
    public_graph = build_public_graph(corpus_state, live_api=False)
    html_text = _render_html_with_ui(json.dumps(public_graph, separators=(",", ":")), live_api=False)
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return out


def make_visualizer_handler(state: dict):
    """Create a `BaseHTTPRequestHandler` class bound to the loaded corpus `state`."""

    class VisualizerHTTPRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Silence per-request console spam unless debugging
            pass

        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", raw)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path
            if path in ("/", "/index.html"):
                graph_payload = build_public_graph(state, live_api=True)
                html = _render_html_with_ui(json.dumps(graph_payload, separators=(",", ":")), live_api=True)
                self._send_bytes(200, "text/html; charset=utf-8", html.encode("utf-8"))
                return
            if path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                return
            if path == "/api/graph":
                graph_payload = build_public_graph(state, live_api=True)
                self._send_json(200, graph_payload)
                return
            if path == "/api/node":
                qs = parse_qs(parsed.query, keep_blank_values=False)
                cid_vals = qs.get("id")
                cid = cid_vals[0] if cid_vals else ""
                if not cid:
                    self._send_json(404, {"error": "Node not found", "id": ""})
                    return
                details_map = state.get("node_details_by_id")
                detail: dict | None = None
                if isinstance(details_map, dict):
                    detail = details_map.get(cid)
                if detail is None:
                    raw_nodes = state.get("nodes") or (
                        state.get("public_graph", {}).get("nodes")
                        if isinstance(state.get("public_graph"), dict)
                        else []
                    ) or []
                    for cand in raw_nodes:
                        if isinstance(cand, dict) and cand.get("id") == cid:
                            detail = {
                                "id": cid,
                                "provider": cand.get("provider", "generic"),
                                "title": cand.get("title", ""),
                                "last_modified_time": cand.get("last_modified_time", ""),
                                "step_count": int(cand.get("step_count", 0)),
                                "dominant_concept": cand.get("dominant_concept", "general"),
                                "summary": cand.get("summary", ""),
                                "top_tokens": list(cand.get("top_tokens", [])),
                                "vector_preview": list(cand.get("vector_preview", [])),
                                "concepts": dict(cand.get("concepts", {})),
                            }
                            break
                if detail is None:
                    self._send_json(404, {"error": "Node not found", "id": cid})
                    return
                self._send_json(200, detail)
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/api/query":
                self._send_json(404, {"error": "Not found"})
                return

            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                self._send_json(400, {"error": "Missing Content-Length header"})
                return
            try:
                content_len = int(raw_len)
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length header"})
                return
            if content_len < 0 or content_len > MAX_POST_BODY_BYTES:
                self._send_json(
                    400,
                    {"error": f"Payload size must be between 0 and {MAX_POST_BODY_BYTES} bytes"},
                )
                return

            try:
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                self._send_json(400, {"error": "Invalid JSON body"})
                return

            if not isinstance(payload, dict):
                self._send_json(400, {"error": "Request body must be a JSON object"})
                return

            query_val = payload.get("query", "")
            if not isinstance(query_val, str):
                self._send_json(400, {"error": "'query' must be a string"})
                return

            raw_threshold = payload.get("threshold_pct", state.get("default_threshold", 40.0))
            try:
                threshold_pct = float(raw_threshold) if raw_threshold is not None else float(state.get("default_threshold", 40.0))
            except (TypeError, ValueError):
                self._send_json(400, {"error": "'threshold_pct' must be a number"})
                return

            hybrid_val = payload.get("hybrid_rag", True)
            if not isinstance(hybrid_val, bool):
                hybrid_val = bool(hybrid_val)

            result = query_rag_space(
                query_val,
                state,
                threshold_pct=threshold_pct,
                hybrid_rag=hybrid_val,
            )
            self._send_json(200, result)

    return VisualizerHTTPRequestHandler


class VisualizerHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server with `allow_reuse_address = False` so port collisions are detected on Windows."""

    allow_reuse_address = False


def create_visualizer_server(
    state: dict,
    host: str = "127.0.0.1",
    port: int = 8765,
    strict_port: bool = False,
    max_port_tries: int = 15,
) -> tuple[ThreadingHTTPServer, int]:
    """Bind a `ThreadingHTTPServer` on `127.0.0.1`, auto-incrementing port on collision unless `strict_port=True`."""
    handler_cls = make_visualizer_handler(state)
    tries = 1 if strict_port else max(1, max_port_tries)
    last_err: OSError | None = None
    for offset in range(tries):
        cand_port = int(port) + offset
        try:
            httpd = VisualizerHTTPServer((host, cand_port), handler_cls)
            actual_port = int(httpd.server_address[1])
            return httpd, actual_port
        except OSError as exc:
            last_err = exc
            if strict_port or cand_port == 0:
                break
    raise OSError(f"Could not bind visualizer server starting at {host}:{port}: {last_err}")


def serve_visualizer(
    state: dict,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    strict_port: bool = False,
) -> None:
    """Start the local 3D Vector Space Explorer HTTP server and optionally launch the user's browser."""
    httpd, actual_port = create_visualizer_server(
        state,
        host=host,
        port=port,
        strict_port=strict_port,
    )
    url = f"http://{host}:{actual_port}/"
    n_nodes = len(state.get("nodes", []))
    var_exp = state.get("variance_explained", [0.0, 0.0, 0.0])
    print(
        f"v1-viz 3D Hybrid-RAG Explorer running at {url} "
        f"({n_nodes} conversations, PCA variance: {var_exp[0]:.1f}% / {var_exp[1]:.1f}% / {var_exp[2]:.1f}%)",
        file=sys.stderr,
    )
    print("Press Ctrl+C to stop.", file=sys.stderr)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Interactive 3D PCA Vector Space & Hybrid-RAG Topology Explorer for v1-agent-memory."
    )
    parser.add_argument("--port", type=int, default=8765, help="Local HTTP server port (default: 8765)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Default match threshold percentage (default: inherits context_finder_config.json threshold_pct)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=5000,
        help="Maximum parent conversations to project into 3D space (default: 5000)",
    )
    parser.add_argument(
        "--export",
        metavar="PATH",
        default=None,
        help="Export a self-contained offline 3D HTML file and exit 0",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the browser when starting the local server",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the 3D PCA graph JSON summary to stdout and exit 0",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    state = collect_vector_corpus(
        max_docs=max(1, int(args.max_docs)),
        default_threshold=args.threshold,
    )

    if args.json:
        public_graph = build_public_graph(state, live_api=False)
        print(json.dumps(public_graph, indent=2))
        return 0

    if args.export:
        out_path = export_static_html(args.export, state=state)
        print(f"Exported self-contained 3D RAG Explorer -> {out_path}", file=sys.stderr)
        return 0

    serve_visualizer(
        state,
        port=args.port,
        open_browser=not args.no_open,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
