from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient


def slug_signal(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_]", "", (s or "").strip().lower())


def get_chroma_client(persist_dir: str) -> PersistentClient:
    persist_dir = os.path.abspath(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    return PersistentClient(path=persist_dir)


def _collection_name(index_name: str, namespace: str | None) -> str:
    return f"{index_name}__{namespace or 'default'}"


def _list_namespaces_for_index(client: PersistentClient, index_name: str) -> List[str]:
    """
    Discover namespaces by scanning collection names that start with '{index_name}__'.
    Returns list of suffixes (namespace names). 'default' is used for the '' namespace.
    """
    namespaces: List[str] = []
    for coll in client.list_collections():
        name = coll.name or ""
        prefix = f"{index_name}__"
        if name.startswith(prefix):
            ns = name[len(prefix):]
            # Map 'default' back to '' for convenience
            namespaces.append("" if ns == "default" else ns)
    return namespaces


def _query_collection(
    coll,
    *,
    query_vec: List[float],
    n_results: int,
    where: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """
    Query a single Chroma collection and return a flat list of match dicts with:
    {id, distance, document, metadata}
    """
    res = coll.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma groups results per query (we only send 1), so index at [0]
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    mds = (res.get("metadatas") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for i, _id in enumerate(ids):
        out.append({
            "id": _id,
            "distance": float(dists[i]) if i < len(dists) else None,
            "document": docs[i] if i < len(docs) else "",
            "metadata": mds[i] if i < len(mds) else {},
        })
    return out


def _fuse_rrf_with_overlap_chroma(
    *,
    namespace: str,
    sem_matches: List[Dict[str, Any]],
    per_signal_lists: List[List[Dict[str, Any]]],
    query_signals: List[str],
    rrf_k: int,
    lambda_signal: float,
    overlap_bonus: float,
) -> List[Tuple[str, float, Dict[str, Any], float, int, str]]:
    """
    Fuse semantic + signal-filtered results (per namespace).
    Returns list of tuples: (uid_ns, fused_score, match, sem_sim, overlap, namespace)
    """
    BIG = 10_000_000

    # Prepare semantic ranks and similarity (convert distance -> cosine similarity)
    candidates: Dict[str, Dict[str, Any]] = {}
    sem_rank: Dict[str, int] = {}
    sem_sim: Dict[str, float] = {}
    for i, m in enumerate(sem_matches):
        uid = f"{namespace}::{m['id']}"
        candidates[uid] = m
        sem_rank[uid] = i
        d = float(m.get("distance", 0.0) or 0.0)
        sim = 1.0 - d  # hnsw:space=cosine => distance ~ 1 - cosine_similarity
        sem_sim[uid] = sim

    # Signal ranks and overlap counts
    sig_rank: Dict[str, int] = defaultdict(lambda: BIG)
    overlap_counts: Dict[str, int] = defaultdict(int)

    for sig_idx, sig_list in enumerate(per_signal_lists):
        sig_name = (query_signals or [None])[sig_idx]
        sig_key = f"sig__{slug_signal(sig_name)}" if sig_name else None
        for i, m in enumerate(sig_list):
            uid = f"{namespace}::{m['id']}"
            candidates[uid] = m
            if i < sig_rank[uid]:
                sig_rank[uid] = i
            # count overlap by checking the boolean flag on metadata:
            md = m.get("metadata") or {}
            if sig_key and md.get(sig_key) is True:
                overlap_counts[uid] += 1

    # Defaults for any candidate not seen in one of the lists
    for uid in list(candidates.keys()):
        sem_rank.setdefault(uid, BIG)
        sem_sim.setdefault(uid, 0.0)
        sig_rank.setdefault(uid, BIG)
        overlap_counts.setdefault(uid, 0)

    fused: List[Tuple[str, float, Dict[str, Any], float, int, str]] = []
    for uid in candidates.keys():
        rr_sem = 1.0 / (rrf_k + sem_rank[uid])
        rr_sig = 1.0 / (rrf_k + sig_rank[uid])
        frac = (overlap_counts[uid] / max(len(query_signals), 1)) if query_signals else 0.0
        final = rr_sem + (lambda_signal * rr_sig) + (overlap_bonus * frac)
        fused.append((uid, final, candidates[uid], sem_sim[uid], overlap_counts[uid], namespace))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


def retrieve_top_k_all_namespaces_with_signals_chroma(
    *,
    persist_dir: str,
    feature_text: str,
    query_signals: List[str],
    index_name: str = "legal-clauses",
    namespaces: Optional[List[str]] = None,   # if None, auto-discover by collections
    hf_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    # Candidate sizes:
    k_semantic: int = 60,
    k_per_signal: int = 25,
    # Fusion params:
    rrf_k: int = 60,
    lambda_signal: float = 1.5,
    overlap_bonus: float = 0.3,
    # Efficiency limits:
    per_namespace_cap: int = 25,   # cap how many fused candidates we keep per ns before global sort
    top_k: int = 10,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chroma-based retriever mirroring the Pinecone version:
      - Treats each 'namespace' as a separate collection named '{index_name}__{namespace or default}'.
      - For each namespace:
          (1) query without filter -> semantic candidates
          (2) one filtered query per signal -> metadata-boolean matches
      - Fuse with Reciprocal Rank Fusion + signal-overlap bonus
      - Keep a per-namespace cap, then return global top_k across all namespaces.
    """
    if verbose:
        print(f"[DEBUG] Chroma persist dir: {persist_dir}")
        print(f"[DEBUG] Index name: {index_name}")

    # Build embedding model (normalized to match your embed pipeline)
    model = SentenceTransformer(hf_model_name)
    q_vec = model.encode([feature_text], normalize_embeddings=True)[0].tolist()

    # Open Chroma and discover namespaces from collections
    client = get_chroma_client(persist_dir)
    if namespaces is None:
        namespaces = _list_namespaces_for_index(client, index_name)
        if not namespaces:
            # Fallback to common names if nothing exists
            namespaces = ["", "US_ALL_2258A", "EU_ALL_DSA", "CA_SB976", "FL_HB3_2024", "UT_SB152_HB311"]

    if verbose:
        print(f"[DEBUG] Namespaces to search: {namespaces}")

    global_pool: Dict[str, Tuple[float, Dict[str, Any], float, int, str]] = {}

    for ns in namespaces:
        coll_name = _collection_name(index_name, ns)
        try:
            coll = client.get_collection(coll_name)
        except Exception as e:
            if verbose:
                print(f"\n[DEBUG] Namespace '{ns or '(default)'}' missing collection '{coll_name}': {e}")
            continue

        if verbose:
            print(f"\n[DEBUG] Namespace: '{ns or '(default)'}'  |  Collection: '{coll_name}'")

        # 1) semantic (no filter)
        sem_matches = _query_collection(coll, query_vec=q_vec, n_results=k_semantic, where=None)
        if verbose:
            print(f"  semantic hits: {len(sem_matches)}")

        # 2) per-signal filtered lists
        per_signal_lists: List[List[Dict[str, Any]]] = []
        for sig in (query_signals or []):
            where = {f"sig__{slug_signal(sig)}": True}
            sig_matches = _query_collection(coll, query_vec=q_vec, n_results=k_per_signal, where=where)
            per_signal_lists.append(sig_matches)
            if verbose:
                print(f"  signal '{sig}' hits: {len(sig_matches)}")

        # 3) fuse per-namespace
        fused_ns = _fuse_rrf_with_overlap_chroma(
            namespace=ns or "",
            sem_matches=sem_matches,
            per_signal_lists=per_signal_lists,
            query_signals=query_signals or [],
            rrf_k=rrf_k,
            lambda_signal=lambda_signal,
            overlap_bonus=overlap_bonus,
        )
        if verbose:
            print(f"  fused candidates (pre-cap): {len(fused_ns)}")

        for uid_ns, score, match, sem_sim, overlap, ns_out in fused_ns[:per_namespace_cap]:
            prev = global_pool.get(uid_ns)
            if (not prev) or (score > prev[0]):
                global_pool[uid_ns] = (score, match, sem_sim, overlap, ns_out)

        if verbose:
            print(f"  kept in pool (so far): {len(global_pool)}")

    if verbose:
        print(f"\n[DEBUG] Global pool size before final sort: {len(global_pool)}")

    ranked = sorted(global_pool.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
    if verbose:
        print(f"[DEBUG] Global top_k={top_k}: {len(ranked)}")

    results: List[Dict[str, Any]] = []
    for uid_ns, (score, match, sem_sim, overlap, ns) in ranked:
        md = match.get("metadata") or {}
        results.append({
            "id": match.get("id"),
            "namespace": ns,
            "clause_id": md.get("clause_id"),
            "article_number": md.get("article_number"),
            "article_title": md.get("article_title"),
            "type": md.get("type"),
            # show CSV; booleans are in metadata as sig__<signal>=True
            "signals_csv": md.get("signals_csv", ""),
            "text": md.get("clause_text", match.get("document", "")),
            "semantic_relevance": float(sem_sim),  # ~cosine sim
            "signal_overlap": int(overlap),
            "final_score": float(score),
        })

    return results