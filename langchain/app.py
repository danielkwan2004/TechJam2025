# app.py
# Streamlit UI -> parser -> retriever -> reasoner -> Streamlit UI
# -----------------------------------------------------------------
# Run: streamlit run app.py
# -----------------------------------------------------------------
from __future__ import annotations
import json
import typing as t
from pathlib import Path

import streamlit as st

# Signal extraction imports
import json
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI  # Optional alternative

from abbreviation_helper import retrieve_abbreviations

load_dotenv()

# Retriever imports


from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Reasoner imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Feature → Clause Reasoner", layout="wide")


# =========================
# Skeleton Pipeline Functions
# =========================

# =========================
# Retriever pipeline
# =========================

def _search_with_relevance_scores(
        vs: PineconeVectorStore, query: str, k: int, where: dict | None = None
) -> List[Tuple[Any, float]]:
    """
    Use similarity_search_with_score (distance) and convert to pseudo-relevance in (0,1]:
    rel = 1 / (1 + distance). This works across providers.
    """
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


def retrieve_top10_clauses(
        *,
        feature_text: str,
        query_signals: list[str],
        index_name: str,
        namespace: str | None,
        hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        # Candidate sizes (tune as needed):
        k_semantic: int = 60,
        k_per_signal: int = 25,
        # Fusion params:
        rrf_k: int = 60,  # RRF stabilization constant
        lambda_signal: float = 1.5,  # weight for signal-driven ranking
        overlap_bonus: float = 0.3,  # bonus for proportion of query signals matched
        top_k: int = 10,
) -> list[dict]:
    """
    Retrieve top-10 clauses by fusing semantic relevance and metadata signal overlap against
    a Pinecone cloud index built with clause metadata (including 'signals' as a list).

    IMPORTANT: When you built the index, your metadata stored the clause text under "clause_text".
               We pass text_key="clause_text" so LangChain puts that in Document.page_content.
    """
    # 1) Build vector store from existing Pinecone index using local HF embeddings
    embedder = HuggingFaceEmbeddings(model_name=hf_model_name)
    vs = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedder,
        namespace=namespace,
        text_key="clause_text",  # <-- crucial: matches your upsert metadata key
    )

    # 2) Global semantic candidates (no filter)
    sem_pairs = _search_with_relevance_scores(vs, feature_text, k=k_semantic)

    def _uid(doc) -> str:
        m = doc.metadata or {}
        return f"{m.get('article_number', '')}__{m.get('clause_id', '')}"

    sem_rank: dict[str, int] = {}
    sem_rel: dict[str, float] = {}
    candidates: dict[str, t.Any] = {}
    for i, (doc, rel) in enumerate(sem_pairs):
        uid = _uid(doc)
        candidates[uid] = doc
        sem_rank[uid] = i
        sem_rel[uid] = float(rel)

    # 3) Signal-filtered candidates (using Pinecone's metadata filter $contains on lists)
    sig_rank: dict[str, int] = {}
    overlap_counts: dict[str, int] = {}
    BIG = 10_000_000
    # default big rank for any candidate missing from a list
    for uid in list(candidates.keys()):
        sig_rank[uid] = BIG
        overlap_counts[uid] = 0

    if query_signals:
        for signal in query_signals:
            where = {"signals": {"$contains": signal}}
            sig_pairs = _search_with_relevance_scores(vs, feature_text, k=k_per_signal, where=where)
            for idx, (doc, _rel) in enumerate(sig_pairs):
                uid = _uid(doc)
                candidates[uid] = doc
                # best signal rank seen
                prev = sig_rank.get(uid, BIG)
                if idx < prev:
                    sig_rank[uid] = idx
                # count overlap
                doc_sigs = doc.metadata.get("signals", []) or []
                if isinstance(doc_sigs, list) and signal in doc_sigs:
                    overlap_counts[uid] = overlap_counts.get(uid, 0) + 1

    # Ensure coverage defaults
    for uid in candidates.keys():
        sem_rank.setdefault(uid, BIG)
        sem_rel.setdefault(uid, 0.0)
        sig_rank.setdefault(uid, BIG)
        overlap_counts.setdefault(uid, 0)

    # 4) Reciprocal Rank Fusion + overlap bonus
    fused = []
    for uid in candidates.keys():
        rr_sem = 1.0 / (rrf_k + sem_rank[uid])
        rr_sig = 1.0 / (rrf_k + sig_rank[uid])
        frac = (overlap_counts[uid] / max(len(query_signals), 1)) if query_signals else 0.0
        final = rr_sem + (lambda_signal * rr_sig) + (overlap_bonus * frac)
        fused.append((uid, final))

    fused.sort(key=lambda x: x[1], reverse=True)
    top = fused[:top_k]

    # 5) Build output payload
    results: list[dict] = []
    for uid, final_score in top:
        doc = candidates[uid]
        m = doc.metadata or {}
        results.append({
            "clause_id": m.get("clause_id"),
            "article_number": m.get("article_number"),
            "article_title": m.get("article_title"),
            "type": m.get("type"),
            "signals": m.get("signals", []),
            "text": doc.page_content,  # thanks to text_key="clause_text"
            "semantic_relevance": sem_rel.get(uid, 0.0),
            "signal_overlap": overlap_counts.get(uid, 0),
            "final_score": float(final_score),
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


def build_feature_text(name: str, title: str, description: str, signals: t.List[str], llm_output: dict) -> str:
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
    return f"{name or title}\n\n{description.strip()}\n\n{sig_line}{reason_block}"


# -------------------------
# UI
# -------------------------
st.title("🧭 Geo-Specific Compliance Pipeline (Pinecone Cloud)")
st.caption("PRD → Signals (LLM) → Retrieval (Pinecone + local HF embeddings) → Geo Reasoner (LLM)")

with st.sidebar:
    st.subheader("Pinecone Settings")
    index_name = st.text_input("Pinecone index_name", value="legal-clauses")
    namespace = st.text_input("Pinecone namespace (optional)", value="eu_dsa")
    st.caption("Make sure your index stores clause text under metadata key 'clause_text'.")
    st.markdown("---")

    st.subheader("Retriever Settings")
    hf_model_name = st.text_input("HF Embedding Model", value="sentence-transformers/all-MiniLM-L6-v2")
    k_semantic = st.slider("Semantic candidate pool (k_semantic)", 20, 200, 60, 5)
    k_per_signal = st.slider("Per-signal pool (k_per_signal)", 10, 100, 25, 5)
    min_confidence = st.slider("Min signal confidence", 0.0, 1.0, 0.55, 0.05)
    lambda_signal = st.slider("Signal rank weight (λ)", 0.5, 3.0, 1.5, 0.1)
    overlap_bonus = st.slider("Overlap bonus", 0.0, 1.0, 0.3, 0.05)
    top_k = st.slider("Top-K final results", 5, 20, 10, 1)
    st.markdown("---")
    st.info("Set PINECONE_API_KEY in your environment.", icon="ℹ️")

colA, colB = st.columns(2)
with colA:
    feature_name = st.text_input("Feature Name", placeholder="e.g., Minor Accounts Privacy Defaults")
    feature_title = st.text_input("Feature Title", placeholder="e.g., Default private accounts for minors")
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
    if not (feature_title or feature_name or feature_description):
        st.warning("Please provide a name or title and a description.")
        st.stop()

    prd_text = feature_description or ""

    # 1) Extract signals (LLM)
    with st.spinner("Extracting signals from PRD…"):
        llm_output = extract_signals(prd_text)

    st.subheader("🔎 LLM Signal Extraction")
    st.json(llm_output, expanded=False)

    if llm_output.get("error"):
        st.error(f"Extractor reported an error: {llm_output['error']}")
        st.stop()

    # 2) Prepare signals + query text
    query_signals = signals_from_llm_output(llm_output, min_confidence=min_confidence)
    st.markdown("**Signals used for retrieval**")
    st.write(query_signals if query_signals else "(none)")

    feature_text = build_feature_text(
        name=feature_name,
        title=feature_title,
        description=prd_text,
        signals=query_signals,
        llm_output=llm_output
    )

    # 3) Retrieve top clauses (Pinecone cloud)
    with st.spinner("Retrieving relevant clauses (Pinecone)…"):
        try:
            results = retrieve_top10_clauses_pinecone(
                feature_text=feature_text,
                query_signals=query_signals,
                index_name=index_name,
                namespace=namespace or None,
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
                feature={"name": feature_name, "title": feature_title, "description": prd_text},
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
