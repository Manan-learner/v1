"""Hybrid-RAG engine: technical concept/synonym clusters, 2-tier 256-dim L2 semantic vectors, and Reciprocal Rank Fusion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zlib
from typing import Iterable, Sequence

from v1_memory.text_utils import STOPWORDS, split_alnum_parts, stem_token, tokenize_domain

VECTOR_DIM = 256
VECTOR_CACHE_VERSION = "2.2"
CACHE_VERSION = VECTOR_CACHE_VERSION

DEFAULT_CONCEPT_CLUSTERS: dict[str, list[str]] = {
    "rf_telemetry": [
        "rf",
        "radio",
        "lora",
        "lorawan",
        "subghz",
        "sub-ghz",
        "915mhz",
        "915",
        "868mhz",
        "433mhz",
        "mhz",
        "sx1262",
        "sx1276",
        "sx1278",
        "sx1280",
        "telemetry",
        "rssi",
        "snr",
        "dbm",
        "dropout",
        "dropouts",
        "preamble",
        "transceiver",
        "wireless",
        "antenna",
        "modulation",
        "spreading",
        "bandwidth",
        "airtime",
        "crc",
        "fsk",
        "sf7",
        "sf9",
        "sf12",
        "esp-now",
        "espnow",
        "crsf",
        "elrs",
        "downlink",
        "uplink",
    ],
    "mcu_embedded": [
        "mcu",
        "embedded",
        "stm32",
        "stm32h7",
        "stm32f4",
        "stm32g4",
        "esp32",
        "esp32s3",
        "fdcan",
        "bxcan",
        "canbus",
        "prescaler",
        "arbitration",
        "bootloader",
        "firmware",
        "uart",
        "usart",
        "spi",
        "i2c",
        "dma",
        "isr",
        "interrupt",
        "hal",
        "gpio",
        "pwm",
        "watchdog",
        "rtos",
        "freertos",
        "zephyr",
        "microcontroller",
        "peripheral",
        "baud",
        "baudrate",
        "bitrate",
    ],
    "uav_mavlink": [
        "mavlink",
        "uav",
        "drone",
        "px4",
        "ardupilot",
        "autopilot",
        "sitl",
        "gazebo",
        "waypoint",
        "waypoints",
        "gimbal",
        "offboard",
        "attitude",
        "imu",
        "gps",
        "ekf",
        "quadcopter",
        "flight",
        "airspeed",
        "altimeter",
    ],
    "rag_search": [
        "rag",
        "hybrid_rag",
        "rrf",
        "bm25",
        "fts",
        "fts5",
        "vector",
        "embedding",
        "embeddings",
        "semantic",
        "cosine",
        "retrieval",
        "rerank",
        "reranker",
    ],
    "k8s_containers": [
        "kubernetes",
        "k8s",
        "helm",
        "pod",
        "pods",
        "deployment",
        "deployments",
        "liveness",
        "readiness",
        "probe",
        "manifest",
        "manifests",
        "container",
        "containers",
        "docker",
        "dockerfile",
        "compose",
        "ingress",
        "namespace",
        "kubectl",
        "autoscale",
        "hpa",
    ],
    "auth_security": [
        "jwt",
        "oauth",
        "oauth2",
        "oidc",
        "sso",
        "rbac",
        "iam",
        "apikey",
        "api_key",
        "authentication",
        "authorization",
        "auth",
        "credential",
        "credentials",
        "login",
        "secret",
        "secrets",
        "tls",
        "ssl",
        "certificate",
    ],
    "database_storage": [
        "database",
        "db",
        "sql",
        "sqlite",
        "sqlite3",
        "wal",
        "schema",
        "migration",
        "migrations",
        "postgres",
        "postgresql",
        "bigquery",
        "spanner",
        "transaction",
        "transactions",
        "vacuum",
        "duckdb",
        "mysql",
        "nosql",
        "orm",
    ],
    "cloud_backup_sync": [
        "cloud",
        "drive",
        "gdrive",
        "google-drive",
        "rclone",
        "backup",
        "backups",
        "restore",
        "sync",
        "syncing",
        "synchronization",
        "robocopy",
        "storage",
        "bucket",
        "gcs",
        "s3",
    ],
}

# Aliases for convenience
CONCEPT_CLUSTERS = DEFAULT_CONCEPT_CLUSTERS
DEFAULT_SYNONYMS = DEFAULT_CONCEPT_CLUSTERS

_FASTEMBED_MODEL = None
_FASTEMBED_FAILED = False
_DEFAULT_LOOKUP_CACHE: tuple[dict[str, list[str]], dict[str, set[str]]] | None = None
_stem_token = stem_token


def _build_term_index(clusters: dict[str, list[str]]) -> dict[str, set[str]]:
    term_to_concepts: dict[str, set[str]] = {}
    for cid, terms in clusters.items():
        for term in terms:
            t_low = term.lower().strip()
            if not t_low:
                continue
            term_to_concepts.setdefault(t_low, set()).add(cid)
            term_to_concepts.setdefault(stem_token(t_low), set()).add(cid)
    return term_to_concepts


def build_concept_lookup(
    custom_synonyms: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Merge built-in concept clusters with user-configured custom synonyms."""
    global _DEFAULT_LOOKUP_CACHE
    if _DEFAULT_LOOKUP_CACHE is None:
        base_clusters = {k: list(v) for k, v in DEFAULT_CONCEPT_CLUSTERS.items()}
        _DEFAULT_LOOKUP_CACHE = (base_clusters, _build_term_index(base_clusters))
    if not custom_synonyms:
        return _DEFAULT_LOOKUP_CACHE

    clusters: dict[str, list[str]] = {k: list(v) for k, v in _DEFAULT_LOOKUP_CACHE[0].items()}
    for raw_key, raw_vals in custom_synonyms.items():
        if not raw_key or not isinstance(raw_vals, (list, tuple, set)):
            continue
        slug = re.sub(r"[^a-z0-9_]+", "_", str(raw_key).lower()).strip("_")
        if not slug:
            continue
        terms = [str(raw_key).lower().strip()] + [str(v).lower().strip() for v in raw_vals if str(v).strip()]
        expanded_terms: list[str] = []
        for term in terms:
            expanded_terms.append(term)
            if " " in term:
                expanded_terms.extend(tokenize_domain(term))
            else:
                expanded_terms.extend(split_alnum_parts(term))
        deduped = list(dict.fromkeys(t for t in expanded_terms if t))
        if slug in clusters:
            clusters[slug] = list(dict.fromkeys(clusters[slug] + deduped))
        else:
            clusters[slug] = deduped

    return clusters, _build_term_index(clusters)


def group_tokens_by_concept(
    tokens: Iterable[str],
    custom_synonyms: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Group input tokens by canonical `concept:<id>` using built-in + custom concept clusters."""
    _, term_to_concepts = build_concept_lookup(custom_synonyms)
    grouped: dict[str, list[str]] = {}
    seen_per_concept: dict[str, set[str]] = {}
    for raw in tokens:
        tok = str(raw).lower().strip()
        if not tok:
            continue
        cids = term_to_concepts.get(tok, set()) | term_to_concepts.get(stem_token(tok), set())
        for cid in sorted(cids):
            key = f"concept:{cid}"
            seen = seen_per_concept.setdefault(key, set())
            if tok not in seen:
                seen.add(tok)
                grouped.setdefault(key, []).append(tok)
    return grouped


def qualifying_shared_concepts(
    q_concept_hits: dict[str, list[str]],
    doc_concept_hits: dict[str, list[str]],
    min_distinct_stems: int = 2,
) -> dict[str, tuple[list[str], list[str]]]:
    """Return shared concept IDs with >= min_distinct_stems in both query and doc after lemmatization."""
    shared: dict[str, tuple[list[str], list[str]]] = {}
    for concept_id, q_hits in q_concept_hits.items():
        d_hits = doc_concept_hits.get(concept_id)
        if not d_hits:
            continue
        q_stems = {stem_token(t) for t in q_hits}
        d_stems = {stem_token(t) for t in d_hits}
        if len(q_stems) >= min_distinct_stems and len(d_stems) >= min_distinct_stems:
            shared[concept_id] = (list(q_hits), list(d_hits))
    return shared


def compute_rag_signature(custom_synonyms: dict[str, list[str]] | None = None) -> str:
    """Compute a deterministic hash signature across version, stopwords, clusters, and custom synonyms."""
    clusters, _ = build_concept_lookup(custom_synonyms)
    norm_custom: list[tuple[str, list[str]]] = []
    if isinstance(custom_synonyms, dict):
        for k, vals in sorted(custom_synonyms.items(), key=lambda kv: str(kv[0]).lower()):
            if isinstance(vals, (list, tuple, set)):
                norm_custom.append(
                    (
                        str(k).lower().strip(),
                        sorted(str(v).lower().strip() for v in vals if str(v).strip()),
                    )
                )
    payload = {
        "version": VECTOR_CACHE_VERSION,
        "stopwords": sorted(STOPWORDS),
        "clusters": {cid: sorted(set(terms)) for cid, terms in sorted(clusters.items())},
        "custom_synonyms": norm_custom,
    }
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{VECTOR_CACHE_VERSION}:{digest}"


def expand_with_concepts(
    tokens: str | Iterable[str],
    custom_synonyms: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str]]:
    """Expand domain tokens using built-in + custom concept clusters.

    Returns:
        (expanded_tokens, concept_ids) where `expanded_tokens` contains original + cluster
        synonym tokens and `concept_ids` contains canonical IDs like `concept:rf_telemetry`.
    """
    if isinstance(tokens, str):
        base_tokens = tokenize_domain(tokens)
    else:
        base_tokens = [str(t).lower().strip() for t in tokens if str(t).strip()]

    clusters, _ = build_concept_lookup(custom_synonyms)
    concept_hits = group_tokens_by_concept(base_tokens, custom_synonyms)
    concept_ids = list(concept_hits.keys())

    expanded: dict[str, None] = {t: None for t in base_tokens if t}
    for concept_id in concept_ids:
        raw_cid = concept_id.split(":", 1)[1]
        for synonym in clusters.get(raw_cid, []):
            expanded[synonym] = None

    return list(expanded.keys()), concept_ids


def token_concept_map(
    tokens: Iterable[str],
    custom_synonyms: dict[str, list[str]] | None = None,
) -> dict[str, set[str]]:
    """Map each token to its matched canonical `concept:<id>` set."""
    grouped = group_tokens_by_concept(tokens, custom_synonyms)
    out: dict[str, set[str]] = {}
    for concept_id, matched in grouped.items():
        for tok in matched:
            out.setdefault(tok, set()).add(concept_id)
    return out


def _hash_add(vec: list[float], feature: str, weight: float, dim: int = VECTOR_DIM) -> None:
    h = zlib.crc32(feature.encode("utf-8", errors="ignore")) & 0xFFFFFFFF
    idx = h % dim
    sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
    vec[idx] += sign * weight


def _char_fourgrams(tokens: Sequence[str], max_tokens: int = 256) -> list[str]:
    grams: dict[str, None] = {}
    for tok in tokens[:max_tokens]:
        padded = f"^{tok}$"
        if len(padded) < 4:
            grams[padded] = None
            continue
        for i in range(len(padded) - 3):
            grams[padded[i : i + 4]] = None
    return sorted(grams.keys())


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return [0.0] * len(vec)
    return [v / norm for v in vec]


def is_fastembed_active() -> bool:
    """Return True only when Tier-1 external fastembed dense embeddings are enabled and loaded."""
    return bool(os.environ.get("V1_USE_FASTEMBED") == "1" and _FASTEMBED_MODEL is not None and not _FASTEMBED_FAILED)


def _compute_stdlib_vector(
    tokens: Sequence[str],
    custom_synonyms: dict[str, list[str]] | None = None,
    dim: int = VECTOR_DIM,
    idf_weights: dict[str, float] | None = None,
) -> list[float]:
    if not tokens:
        return [0.0] * dim

    # Sort deduplicated tokens so PYTHONHASHSEED and set iteration order never affect vector output.
    unique_tokens = sorted(dict.fromkeys(tokens))
    vec_tokens = unique_tokens[:512]
    concept_hits = group_tokens_by_concept(vec_tokens, custom_synonyms)
    qualified_concepts = sorted(
        cid for cid, hits in concept_hits.items() if len({stem_token(t) for t in hits}) >= 2
    )
    ngrams = _char_fourgrams(vec_tokens, max_tokens=256)

    vec = [0.0] * dim

    # 1. Domain tokens + stems (weight 1.0 / 0.5, group-normalized for length invariance)
    if idf_weights is None:
        tok_scale = 1.0 / math.sqrt(len(vec_tokens))
        for tok in vec_tokens:
            _hash_add(vec, f"tok:{tok}", 1.0 * tok_scale, dim)
            stem = stem_token(tok)
            if stem != tok:
                _hash_add(vec, f"tok:{stem}", 0.5 * tok_scale, dim)
    else:
        tok_weights: list[float] = []
        for tok in vec_tokens:
            raw_w = idf_weights.get(tok)
            if raw_w is None:
                raw_w = idf_weights.get(stem_token(tok), 1.0)
            tok_weights.append(1.0 + math.log(max(1.0, float(raw_w))))
        norm_sq = sum(w * w for w in tok_weights)
        tok_scale = 1.0 / math.sqrt(norm_sq if norm_sq > 1e-12 else len(vec_tokens))
        for tok, w in zip(vec_tokens, tok_weights):
            _hash_add(vec, f"tok:{tok}", w * tok_scale, dim)
            stem = stem_token(tok)
            if stem != tok:
                _hash_add(vec, f"tok:{stem}", 0.5 * w * tok_scale, dim)

    # 2. Qualified canonical concept IDs (weight 1.6x, 4-projection hash)
    if qualified_concepts:
        projections = 4
        concept_scale = 1.6 / math.sqrt(len(qualified_concepts) * projections)
        for cid in qualified_concepts:
            for p in range(projections):
                _hash_add(vec, f"{cid}#{p}", concept_scale, dim)

    # 3. Character 4-grams (weight 0.5x, group-normalized)
    if ngrams:
        gram_scale = 0.5 / math.sqrt(len(ngrams))
        for g in ngrams:
            _hash_add(vec, f"g4:{g}", gram_scale, dim)

    return [round(v, 6) for v in _l2_normalize(vec)]


def compute_semantic_vector(
    text: str | Iterable[str],
    custom_synonyms: dict[str, list[str]] | None = None,
    dim: int = VECTOR_DIM,
    idf_weights: dict[str, float] | None = None,
) -> list[float]:
    """Compute a 256-dim L2-normalized semantic vector.

    Tier 1: Uses `fastembed` (`BAAI/bge-small-en-v1.5`) ONLY if `V1_USE_FASTEMBED=1` and installed.
    Tier 2: Fast deterministic stdlib 256-dim crc32 feature-hashed L2 vector combining domain tokens
            (1.0x, optionally damped log-IDF weighted via `idf_weights`), stems (0.5x), >=2-stem
            qualified concept IDs (1.6x), and character 4-grams (0.5x).
    """
    global _FASTEMBED_MODEL, _FASTEMBED_FAILED

    if isinstance(text, str):
        tokens = tokenize_domain(text)
    else:
        tokens = [str(x).lower().strip() for x in text if str(x).strip()]

    stdlib_vec = _compute_stdlib_vector(
        tokens,
        custom_synonyms=custom_synonyms,
        dim=dim,
        idf_weights=idf_weights,
    )

    if os.environ.get("V1_USE_FASTEMBED") == "1" and not _FASTEMBED_FAILED:
        try:
            if _FASTEMBED_MODEL is None:
                from fastembed import TextEmbedding  # type: ignore

                _FASTEMBED_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            embed_text = text if isinstance(text, str) else " ".join(sorted(dict.fromkeys(tokens)))
            emb_list = list(_FASTEMBED_MODEL.embed([embed_text[:4000]]))
            if emb_list:
                raw = [float(x) for x in emb_list[0]]
                folded = [0.0] * dim
                for idx, val in enumerate(raw):
                    folded[idx % dim] += val
                dense_norm = _l2_normalize(folded)
                blended = [0.65 * dense_norm[i] + 0.35 * stdlib_vec[i] for i in range(dim)]
                return [round(v, 6) for v in _l2_normalize(blended)]
        except Exception:
            _FASTEMBED_FAILED = True

    return stdlib_vec


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if not vec_a or not vec_b:
        return 0.0
    n = min(len(vec_a), len(vec_b))
    dot = sum(float(vec_a[i]) * float(vec_b[i]) for i in range(n))
    norm_a = math.sqrt(sum(float(vec_a[i]) * float(vec_a[i]) for i in range(n)))
    norm_b = math.sqrt(sum(float(vec_b[i]) * float(vec_b[i]) for i in range(n)))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def reciprocal_rank_fusion(
    sparse_ranked_ids: Sequence[str] | dict[str, float],
    dense_ranked_ids: Sequence[str] | dict[str, float],
    k: int = 60,
) -> dict[str, float]:
    """Fuse sparse and dense ranked candidate ID lists via Reciprocal Rank Fusion (RRF)."""
    if isinstance(sparse_ranked_ids, dict):
        sparse_list = [cid for cid, _ in sorted(sparse_ranked_ids.items(), key=lambda item: item[1], reverse=True)]
    else:
        sparse_list = list(sparse_ranked_ids)

    if isinstance(dense_ranked_ids, dict):
        dense_list = [cid for cid, _ in sorted(dense_ranked_ids.items(), key=lambda item: item[1], reverse=True)]
    else:
        dense_list = list(dense_ranked_ids)

    scores: dict[str, float] = {}
    for rank, cid in enumerate(sparse_list, start=1):
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
    for rank, cid in enumerate(dense_list, start=1):
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))

    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
