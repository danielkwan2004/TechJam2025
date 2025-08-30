# app.py
# Streamlit UI -> parser -> retriever -> reasoner -> Streamlit UI
# -----------------------------------------------------------------
# Run: streamlit run app.py
# -----------------------------------------------------------------
from __future__ import annotations
import os
import time
import typing as t
from pinecone import Pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer

# Signal extraction imports
import json

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI  # Optional alternative

from abbreviation_helper import retrieve_all_abbreviations
from signal_extractor import extract_signals
from typing import Literal
import pandas as pd
from collections import defaultdict

load_dotenv()

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Retriever imports

from typing import List, Dict, Any, Tuple, Optional, Literal

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Reasoner imports
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Feature → Clause Reasoner", layout="wide")


# =========================
# Skeleton Pipeline Functions
# =========================

# =========================
# Retriever pipeline
# =========================

def _search_with_relevance_scores(vs: PineconeVectorStore, query: str, k: int, where: dict | None = None):
    pairs = vs.similarity_search_with_score(query, k=k, filter=where)
    out = []
    for doc, dist in pairs:
        try:
            d = float(dist)
        except Exception:
            d = 1.0
        rel = 1.0 / (1.0 + d)
        out.append((doc, rel))
    return out

def _uid(doc, namespace: str) -> str:
    m = doc.metadata or {}
    return f"{namespace}::{m.get('article_number','')}__{m.get('clause_id','')}"

def _fuse_rrf_with_overlap(
    *,
    sem_pairs,
    signal_lists,
    query_signals: List[str],
    rrf_k: int,
    lambda_signal: float,
    overlap_bonus: float,
    namespace: str,
):
    BIG = 10_000_000
    candidates = {}
    sem_rank, sem_rel = {}, {}
    for i, (doc, rel) in enumerate(sem_pairs):
        uid = _uid(doc, namespace)
        candidates[uid] = doc
        sem_rank[uid] = i
        sem_rel[uid] = float(rel)

    sig_rank = defaultdict(lambda: BIG)
    overlap_counts = defaultdict(int)

    for sig_idx, pairs in enumerate(signal_lists):
        sig_val = query_signals[sig_idx] if sig_idx < len(query_signals) else None
        for i, (doc, _rel) in enumerate(pairs):
            uid = _uid(doc, namespace)
            candidates[uid] = doc
            if i < sig_rank[uid]:
                sig_rank[uid] = i
            doc_sigs = (doc.metadata or {}).get("signals", []) or []
            if sig_val and isinstance(doc_sigs, list) and sig_val in doc_sigs:
                overlap_counts[uid] += 1

    for uid in list(candidates.keys()):
        sem_rank.setdefault(uid, BIG)
        sem_rel.setdefault(uid, 0.0)
        sig_rank.setdefault(uid, BIG)
        overlap_counts.setdefault(uid, 0)

    fused = []
    for uid in candidates.keys():
        rr_sem = 1.0 / (rrf_k + sem_rank[uid])
        rr_sig = 1.0 / (rrf_k + sig_rank[uid])
        frac = (overlap_counts[uid] / max(len(query_signals), 1)) if query_signals else 0.0
        final = rr_sem + (lambda_signal * rr_sig) + (overlap_bonus * frac)
        fused.append((uid, final, candidates[uid], sem_rel[uid], overlap_counts[uid], namespace))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused

def _list_all_namespaces(index_name: str) -> List[str]:
    """List namespaces that exist on the index (keys in describe_index_stats().namespaces)."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    stats = index.describe_index_stats() or {}
    ns_map = stats.get("namespaces") or {}
    namespaces = list(ns_map.keys())
    # If you might have used the default namespace, include "" for probing:
    if "" not in namespaces:
        namespaces.append("")
    return namespaces


def _query_ns(
    index,
    *,
    vector: List[float],
    namespace: Optional[str],
    top_k: int,
    flt: Optional[dict] = None,
) -> List[dict]:
    """Query Pinecone (single namespace) and return list of matches with metadata."""
    res = index.query(
        vector=vector,
        top_k=top_k,
        namespace=(namespace or None),  # None = default namespace
        filter=flt,
        include_metadata=True,
    )
    return res.get("matches", []) or []


import os
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()


def _list_all_namespaces(index_name: str) -> List[str]:
    """List namespaces that exist on the index (keys in describe_index_stats().namespaces)."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    stats = index.describe_index_stats() or {}
    ns_map = stats.get("namespaces") or {}
    namespaces = list(ns_map.keys())
    # include default namespace probe
    if "" not in namespaces:
        namespaces.append("")
    return namespaces


def _query_ns(
    index,
    *,
    vector: List[float],
    namespace: Optional[str],
    top_k: int,
    flt: Optional[dict] = None,
) -> List[dict]:
    """Query Pinecone (single namespace) and return list of matches with metadata."""
    res = index.query(
        vector=vector,
        top_k=top_k,
        namespace=(namespace or None),  # None = default namespace
        filter=flt,
        include_metadata=True,
    )
    return res.get("matches", []) or []


def retrieve_top_k_all_namespaces_with_signals_direct(
    *,
    feature_text: str,
    query_signals: List[str],
    index_name: str,
    namespaces: Optional[List[str]] = None,   # if None, auto-discover
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
    Direct-Pinecone retriever:
      - Encodes the query locally with SentenceTransformers (normalized, for cosine).
      - For each namespace, runs:
          1) semantic query (no filter)
          2) per-signal queries (metadata filter {"signals": {"$in": [SIG]}})
      - Fuses via Reciprocal Rank Fusion + overlap bonus.
      - Caps per-namespace results, then returns global top_k across all namespaces.
    """
    if verbose:
        print(f"[DEBUG] Index name: {index_name}")

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Build query vector (normalized=True to match your embedder)
    model = SentenceTransformer(hf_model_name)
    query_vec = model.encode([feature_text], normalize_embeddings=True)[0].tolist()

    # Determine namespaces
    if namespaces is None:
        try:
            namespaces = _list_all_namespaces(index_name)
        except Exception as e:
            if verbose:
                print(f"[DEBUG] Could not list namespaces automatically: {e}")
            namespaces = ["", "US_ALL_2258A", "EU_ALL_DSA", "CA_SB976", "FL_HB3_2024", "UT_SB152_HB311"]

    if verbose:
        print(f"[DEBUG] Namespaces to search: {namespaces}")

    global_pool: Dict[str, Tuple[float, dict, float, int, str]] = {}

    for ns in namespaces:
        if verbose:
            print(f"\n[DEBUG] Namespace: '{ns or '(default)'}'")

        # 1) Semantic candidates (no filter)
        sem_matches = _query_ns(index, vector=query_vec, namespace=ns, top_k=k_semantic, flt=None)
        if verbose:
            print(f"  semantic hits: {len(sem_matches)}")

        # 2) Per-signal filtered candidates
        per_signal_lists: List[List[dict]] = []
        for sig in (query_signals or []):
            # IMPORTANT: use $in to test membership in a list-valued field
            flt = {"signals": {"$in": [sig]}}
            sig_matches = _query_ns(index, vector=query_vec, namespace=ns, top_k=k_per_signal, flt=flt)
            per_signal_lists.append(sig_matches)
            if verbose:
                print(f"  signal '{sig}' hits: {len(sig_matches)}")

        # 3) RRF fusion with overlap bonus
        BIG = 10_000_000
        candidates: Dict[str, dict] = {}
        sem_rank: Dict[str, int] = {}
        sem_sim: Dict[str, float] = {}  # cosine similarity from Pinecone 'score'
        for i, m in enumerate(sem_matches):
            uid = f"{ns}::{m.get('id')}"
            candidates[uid] = m
            sem_rank[uid] = i
            sem_sim[uid] = float(m.get("score", 0.0))

        sig_rank = defaultdict(lambda: BIG)
        overlap_counts = defaultdict(int)

        for sig_idx, sig_list in enumerate(per_signal_lists):
            sig_val = (query_signals or [None])[sig_idx]
            for i, m in enumerate(sig_list):
                uid = f"{ns}::{m.get('id')}"
                candidates[uid] = m
                if i < sig_rank[uid]:
                    sig_rank[uid] = i
                md = (m.get("metadata") or {})
                doc_sigs = md.get("signals", []) or []
                if sig_val and isinstance(doc_sigs, list) and sig_val in doc_sigs:
                    overlap_counts[uid] += 1

        for uid in list(candidates.keys()):
            sem_rank.setdefault(uid, BIG)
            sem_sim.setdefault(uid, 0.0)
            sig_rank.setdefault(uid, BIG)
            overlap_counts.setdefault(uid, 0)

        fused_ns: List[Tuple[str, float]] = []
        for uid in candidates.keys():
            rr_sem = 1.0 / (rrf_k + sem_rank[uid])
            rr_sig = 1.0 / (rrf_k + sig_rank[uid])
            frac_overlap = (overlap_counts[uid] / max(len(query_signals), 1)) if query_signals else 0.0
            final = rr_sem + (lambda_signal * rr_sig) + (overlap_bonus * frac_overlap)
            fused_ns.append((uid, final))

        fused_ns.sort(key=lambda x: x[1], reverse=True)
        if verbose:
            print(f"  fused candidates (pre-cap): {len(fused_ns)}")

        for uid, score in fused_ns[:per_namespace_cap]:
            m = candidates[uid]
            prev = global_pool.get(uid)
            if (not prev) or (score > prev[0]):
                global_pool[uid] = (score, m, sem_sim[uid], overlap_counts[uid], ns)

        if verbose:
            print(f"  kept in pool (so far): {len(global_pool)}")

    if verbose:
        print(f"\n[DEBUG] Global pool size before final sort: {len(global_pool)}")

    ranked = sorted(global_pool.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
    if verbose:
        print(f"[DEBUG] Global top_k={top_k}: {len(ranked)}")

    results: List[Dict[str, Any]] = []
    for uid, (score, match, sem_similarity, overlap, ns) in ranked:
        md = (match.get("metadata") or {})
        results.append({
            "clause_id": md.get("clause_id"),
            "article_number": md.get("article_number"),
            "article_title": md.get("article_title"),
            "type": md.get("type"),
            "signals": md.get("signals", []),
            "text": md.get("clause_text", ""),
            "semantic_relevance": float(sem_similarity),
            "signal_overlap": int(overlap),
            "final_score": float(score),
            "namespace": ns,
            "id": match.get("id"),
            "score_raw": match.get("score", 0.0),
        })
    return results


# =========================
# Reasoner Pipeline
# =========================
class GeoReasoning(BaseModel):
    """Structured output for geo compliance reasoning with explicit uncertainty."""
    # Allowed values:
    # - "yes"  -> geo-specific logic is needed
    # - "no"   -> geo-specific logic is not needed
    # - "needs_human_review" -> the model is unsure and requests human review
    needs_geo_logic: Literal["yes", "no", "needs_human_review"] = Field(
        ...,
        description="One of 'yes', 'no', or 'needs_human_review'."
    )
    reasoning: str = Field(
        ...,
        description="Clear, concise explanation referencing the feature and relevant clauses. If 'needs_human_review', state exactly why you're unsure or what is missing."
    )
    regulations: List[str] = Field(
        default_factory=list,
        description="List of related regulation/act/section names inferred from the clause metadata. If unclear, return an empty list."
    )


def _format_matches(matches: List[Dict[str, Any]]) -> str:
    """Render retriever results into a compact block for the LLM."""
    if not matches:
        return "(no clauses provided)"
    lines = []
    for m in matches:
        cid = m.get("clause_id", "UNKNOWN")
        art = m.get("article_number", "")
        ttl = m.get("article_title", "")
        typ = m.get("type", "")
        sigs = ", ".join(m.get("signals", []) or [])
        txt = (m.get("text") or "").strip()[:800]  # keep prompt lean
        lines.append(
            f"[clause_id: {cid}] (article: {art} — {ttl} — type: {typ} — signals: {sigs})\n{txt}"
        )
    return "\n\n".join(lines)


def reason_feature_geo_compliance(
        *,
        feature: Dict[str, str],
        matches: List[Dict[str, Any]],
        llm: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Decide whether the feature needs geo-specific compliance logic,
    or if the model is unsure and requires human review.

    Returns dict: {
      "needs_geo_logic": "yes" | "no" | "needs_human_review",
      "reasoning": str,
      "regulations": [str, ...]
    }
    """
    # Short-circuit when there is no evidence to reason about.
    if not matches:
        return {
            "needs_geo_logic": "needs_human_review",
            "reasoning": "No clauses were provided by the retriever; unable to determine jurisdictional variability. A human should review.",
            "regulations": []
        }

    # Default LLM (swap to Gemini if preferred)
    if llm is None:
        llm = ChatOpenAI(model=model, temperature=temperature)
        # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)

    structured_llm = llm.with_structured_output(GeoReasoning)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a compliance policy reasoner. Given a product feature and retrieved legal clauses, "
                "decide if the feature needs geo-specific compliance logic. Choose exactly one:\n"
                " - 'yes'  : obligations/bans/processes vary by jurisdiction or cited clauses clearly apply only to certain geographies.\n"
                " - 'no'   : obligations are jurisdiction-agnostic or do not vary across regions in the provided clauses.\n"
                " - 'needs_human_review' : you are unsure because the clauses are insufficient, conflicting, out-of-scope, "
                "or do not provide clear jurisdictional signals. WHEN UNSURE, YOU MUST SELECT 'needs_human_review' and explicitly state why."
                "ONLY use the provided clauses as context, and do not assume anything other than what clauses you have been provided with."
                "If the clauses are not very related to the feature, you must select 'needs_human_review' and explain why."
            ),
            (
                "human",
                "Feature:\n"
                "- Name: {name}\n"
                "- Title: {title}\n"
                "- Description: {description}\n\n"
                "Retrieved Clauses:\n{clauses}\n\n"
                "Instructions:\n"
                "- Return a structured decision using the schema.\n"
                "- Keep the reasoning concise and reference clause_ids or titles when helpful.\n"
                "- 'regulations' should list concise regulation/act/section names inferred from clause metadata; "
                "leave empty if unclear.\n"
                "- If you select 'needs_human_review', clearly declare the uncertainty (e.g., missing geography, ambiguous scope, conflicting clauses)."
            ),
        ]
    )

    clauses_block = _format_matches(matches)

    result: GeoReasoning = (prompt | structured_llm).invoke(
        {
            "name": feature.get("name", "") or feature.get("title", ""),
            "title": feature.get("title", "") or feature.get("name", ""),
            "description": feature.get("description", ""),
            "clauses": clauses_block,
        }
    )

    return result.dict()


# =========================
# Helpers
# =========================

def dedupe_preserve_order(items: t.List[str]) -> t.List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def signals_from_llm_output(llm_output: dict, min_confidence: float = 0.5) -> t.List[str]:
    """
    Extract de-duped, high-confidence signals from your LLM output dict:
    {
      "error": null,
      "data": [{"law":"...","signal":"...","reason":"...","confidence":0.8}, ...]
    }
    """
    if not isinstance(llm_output, dict) or llm_output.get("error"):
        return []
    rows = llm_output.get("data") or []
    sigs = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        sig = (r.get("signal") or "").strip()
        conf = float(r.get("confidence", 0) or 0)
        if sig and conf >= min_confidence:
            sigs.append(sig)
    return dedupe_preserve_order(sigs)


def build_feature_text(name: str, description: str, signals: t.List[str], llm_output: dict) -> str:
    """Compose a single semantic query string for the retriever."""
    reasons = []
    for r in (llm_output.get("data") or [])[:5]:
        txt = (r.get("reason") or "").strip()
        if txt:
            reasons.append(txt[:220])
    reason_block = ""
    if reasons:
        reason_block = "\nReason hints:\n- " + "\n- ".join(reasons[:5])
    sig_line = f"Signals: {', '.join(signals)}" if signals else "Signals: (none)"
    return f"{name}\n\n{description.strip()}\n\n{sig_line}{reason_block}"


# -------------------------
# UI
# -------------------------
st.title("🧭 Geo-Specific Compliance Pipeline (Pinecone Cloud)")
st.caption("PRD → Signals (LLM) → Retrieval (Pinecone + local HF embeddings) → Geo Reasoner (LLM)")

with st.sidebar:
    with st.expander("Show/hide abbreviations"):
        abbreviations = retrieve_all_abbreviations()
        st.dataframe(pd.DataFrame(abbreviations).reset_index(drop=True).iloc[:, 1:], hide_index=True)

    st.divider()
    st.subheader("Pinecone Settings")
    index_name = st.text_input("Pinecone index_name", value="legal-clauses")
    namespace = st.text_input("Pinecone namespace (optional)", value="EU_DSA")
    st.caption("Make sure your index stores clause text under metadata key 'clause_text'.")
    st.markdown("---")

    st.subheader("Retriever Settings")
    hf_model_name = st.text_input("HF Embedding Model", value="sentence-transformers/all-mpnet-base-v2")
    k_semantic = st.slider("Semantic candidate pool (k_semantic)", 20, 200, 60, 5)
    k_per_signal = st.slider("Per-signal pool (k_per_signal)", 10, 100, 25, 5)
    min_confidence = st.slider("Min signal confidence", 0.0, 1.0, 0.25, 0.05)
    lambda_signal = st.slider("Signal rank weight (λ)", 0.5, 3.0, 1.5, 0.1)
    overlap_bonus = st.slider("Overlap bonus", 0.0, 1.0, 0.3, 0.05)
    top_k = st.slider("Top-K final results", 5, 20, 10, 1)
    st.markdown("---")
    st.info("Set PINECONE_API_KEY in your environment.", icon="ℹ️")

colA, colB = st.columns(2)
with colA:
    feature_name = st.text_input("Feature Name", placeholder="e.g., Minor Accounts Privacy Defaults")
with colB:
    feature_description = st.text_area(
        "Feature Description (PRD excerpt)",
        placeholder="Paste the PRD or feature description here…",
        height=140
    )

run = st.button("Run Full Pipeline", type="primary", use_container_width=True)

# -------------------------
# Pipeline Run
# -------------------------
if run:
    if not (feature_name or feature_description):
        st.warning("Please provide a name and a description.")
        st.stop()

    prd_text = feature_description or ""

    # 1) Extract signals (LLM)
    llm_output = None
    for attempt in range(1, MAX_RETRIES + 1):
        with st.spinner(f"Extracting signals from PRD… (Attempt {attempt})"):
            llm_output = extract_signals(prd_text)

        error_val = llm_output.get("error")
        if error_val not in (None, "null"):
            st.warning(f"Extractor reported an error: {error_val}")
            if attempt < MAX_RETRIES:
                st.info(f"Retrying in {RETRY_DELAY} seconds…")
                time.sleep(RETRY_DELAY)
                continue
            else:
                st.error(f"Extractor failed after {MAX_RETRIES} attempts: {error_val}")
                st.stop()
        else:
            # Success, break out of loop
            break
    st.subheader("🔎 LLM Signal Extraction")
    st.json(llm_output, expanded=False)

    error_val = llm_output.get("error")
    if error_val and error_val.lower() != "null":
        st.error(f"Extractor reported an error: {error_val}")
        st.stop()

    # 2) Prepare signals + query text
    query_signals = signals_from_llm_output(llm_output, min_confidence=min_confidence)
    st.markdown("**Signals used for retrieval**")
    st.write(query_signals if query_signals else "(none)")
    
    def build_feature_text_from_signals(query_signals: list[str]) -> str:
        if not query_signals:
            return "Laws related to general compliance."
        cleaned = [s.replace("_", " ").strip() for s in query_signals if s]
        if not cleaned:
            return "Laws related to general compliance."
        return f"Laws related to {', '.join(cleaned)}."
        
    feature_text = build_feature_text_from_signals(query_signals)

    # 3) Retrieve top clauses (Pinecone cloud)
    with st.spinner("Retrieving relevant clauses (Pinecone)…"):
        try:
            results = retrieve_top_k_all_namespaces_with_signals_direct(
                feature_text=feature_text,
                query_signals=query_signals,
                index_name=index_name,
                hf_model_name=hf_model_name,
                k_semantic=k_semantic,
                k_per_signal=k_per_signal,
                rrf_k=60,
                lambda_signal=lambda_signal,
                overlap_bonus=overlap_bonus,
                top_k=top_k
            )
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            st.stop()

    st.subheader("📚 Retrieved Clauses (Top Matches)")
    if results:
        for r in results:
            header = f"{r.get('clause_id')} — Art. {r.get('article_number', '')} — {r.get('article_title', '')}"
            with st.expander(header):
                st.write(r.get("text", ""))
                meta_cols = st.columns(3)
                with meta_cols[0]:
                    st.caption(f"Signals: {', '.join(r.get('signals', [])) or '(none)'}")
                with meta_cols[1]:
                    st.caption(f"Semantic relevance: {r.get('semantic_relevance', 0):.3f}")
                with meta_cols[2]:
                    st.caption(f"Final score: {r.get('final_score', 0):.4f}")
    else:
        st.caption("No clauses were retrieved.")

    # 4) Geo reasoner
    with st.spinner("Determining geo-specific compliance need…"):
        try:
            reasoned = reason_feature_geo_compliance(
                feature={"name": feature_name, "description": prd_text},
                matches=results
            )
        except Exception as e:
            st.error(f"Reasoner failed: {e}")
            st.stop()

    st.subheader("✅ Geo-Specific Compliance Decision")
    decision = reasoned.get("needs_geo_logic", "needs_human_review")
    if decision == "yes":
        st.success("Needs geo-specific compliance logic ✅")
    elif decision == "no":
        st.info("Does not need geo-specific compliance logic")
    else:
        st.warning("Needs human review ⚠️")

    st.markdown("**Reasoning**")
    st.write(reasoned.get("reasoning", ""))

    regs = reasoned.get("regulations") or []
    st.markdown("**Related Regulations**")
    if regs:
        st.write(regs)
    else:
        st.caption("(none found)")

    st.markdown("---")
    with st.expander("Debug: Feature Text Sent to Retriever"):
        st.code(feature_text)
