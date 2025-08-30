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

from langchain_community.vectorstores import Chroma
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
# Signal Extraction Pipeline
# =========================

# --------- Structured Output Schemas (Pydantic) ---------

class Detection(BaseModel):
    law: str = Field(..., description="Precise law id from hints (e.g., 'EU Digital Services Act').")
    signal: str = Field(..., description="Signal name as detected.")
    reason: str = Field(..., description="Short explanation quoting PRD text and why it maps to the law/signal.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in detection, 0.0–1.0.")


class LLMOutput(BaseModel):
    error: Optional[str] = Field(None, description="Null if no error; otherwise a short error message.")
    data: List[Detection] = Field(default_factory=list, description="Detected compliance signals.")


def extract_hints():
    out = []
    laws = json.load(open('../knowledge_base/law_cards.json'))
    for law in laws:
        cleaned_law = {
            'id': law['laws'][0]['id'],
            'triggers': law['laws'][0]['triggers'],
            'reason_templates': law['laws'][0]['reason_templates']
        }
        out.append(cleaned_law)
    return out


def extract_signals(doc: str) -> dict:
    abbreviations = retrieve_abbreviations(doc)
    formatted_abbrevs = json.dumps(
        abbreviations + [{'term': 'PRD', 'explanation': 'Project Requirement Document'}]
    )

    hints = extract_hints()

    # --- LLM: OpenAI with Structured Output (recommended for strong JSON guarantees) ---
    base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=31082025)
    llm = base_llm.with_structured_output(LLMOutput)  # << Guarantees schema

    # If you prefer Gemini, you can try:
    # gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    # llm = gemini.with_structured_output(LLMOutput)  # Works in recent LC versions; else fall back to strict JSON prompting.

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise compliance signal extraction assistant. "
            "Your job is to analyse a Product Requirement Document (PRD) and output potential compliance issues. "
            "Look for both explicit keywords and functional hints that imply compliance signals. "
            "Take geographical location into consideration; flag signals for all regions specified by the PRD. "
            "If multiple regions are relevant, include all applicable laws. "
            "If no region is specified, assume all regions are affected. "
            "When you see an abbreviation in the PRD, look it up exactly in the abbreviations list provided below. "
            "If an abbreviation in the document cannot be found in the list, return an error describing what is missing and set data to an empty list. "
            "If no signals are found, set error to null and return an empty data array."
        ),
        (
            "system",
            "Here is the list of abbreviations as a JSON with keys 'term' and 'explanation':\n{abbreviations}\n"
        ),
        (
            "system",
            "Law hints (each law contains triggers and reason templates, and each trigger contains the signal names):\n{hints}\n"
        ),
        (
            "system",
            "Output schema (STRICT):\n"
            "Return an object with:\n"
            "- error: null OR a short string\n"
            "- data: array of objects, each with fields { law, signal, reason, confidence }."
        ),
        (
            "human",
            "The PRD:\n{document}\n"
        )
    ])

    # Build the chain: prompt → structured LLM
    chain = prompt | llm

    # Invoke
    result: LLMOutput = chain.invoke({
        "document": doc,
        "abbreviations": formatted_abbrevs,
        "hints": json.dumps(hints)
    })

    # Convert to plain dict (serializable as the exact JSON you want)
    return result.dict()


# =========================
# Retriever pipeline
# =========================

def _search_with_relevance_scores(
    vs: Chroma, query: str, k: int, where: dict | None = None
) -> List[Tuple[Any, float]]:
    """
    Try to use 'similarity_search_with_relevance_scores' (higher is better).
    Fallback to 'similarity_search_with_score' (typically distance; lower is better),
    converting to a pseudo-relevance in (0,1] as 1 / (1 + distance).
    """
    try:
        # Newer LC: returns List[(Document, relevance_score in 0..1)]
        return vs.similarity_search_with_relevance_scores(query, k=k, filter=where)
    except Exception:
        # Fallback to distance-based scores. Convert to "relevance".
        pairs = vs.similarity_search_with_score(query, k=k, filter=where)  # type: ignore[arg-type]
        conv = []
        for doc, dist in pairs:
            try:
                d = float(dist)
            except Exception:
                d = 1.0
            rel = 1.0 / (1.0 + d)
            conv.append((doc, rel))
        return conv


def retrieve_top10_clauses(
    feature_text: str,
    query_signals: List[str],
    *,
    persist_directory: str,
    collection_name: str,
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    # Candidate sizes:
    k_semantic: int = 40,
    k_per_signal: int = 20,
    # Fusion params:
    rrf_k: int = 60,              # stabilization constant in RRF
    lambda_signal: float = 1.5,   # weight for signal-driven ranking
    overlap_bonus: float = 0.3,   # bonus term for proportion of signals matched
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-10 clauses by fusing semantic relevance and metadata signal overlap.

    Args:
        feature_text: Product feature text (title + description + context).
        query_signals: Signals to prioritize (e.g., ["parental_consent_required", ...]).
        persist_directory: Chroma persist dir that contains the pre-built collection.
        collection_name: Chroma collection name.
        hf_model_name: Local Hugging Face embedding model for the query.
        k_semantic: How many candidates to pull via pure semantic search.
        k_per_signal: How many candidates to pull per signal with metadata filter.
        rrf_k: RRF constant to stabilize reciprocal ranks.
        lambda_signal: Weight multiplier for the signal-driven ranking term.
        overlap_bonus: Extra bonus for overlap proportion (0..1) of signals.
        top_k: Final number of results to return.

    Returns:
        List of dicts: [{
            "clause_id": str,
            "article_number": str,
            "article_title": str,
            "type": str,
            "signals": list[str],
            "text": str,                # clause_text / page_content
            "semantic_relevance": float,
            "signal_overlap": int,
            "final_score": float,
        }, ...] (sorted by final_score desc, length <= top_k)
    """
    # 1) Build vectorstore with LOCAL HF embeddings for the query
    embedder = HuggingFaceEmbeddings(model_name=hf_model_name)
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedder,
    )

    # 2) Global semantic candidates (no filter)
    sem_pairs = _search_with_relevance_scores(vs, feature_text, k=k_semantic)
    # sem_rank: lower rank number = better; store best rank and relevance
    sem_rank: Dict[str, int] = {}
    sem_rel: Dict[str, float] = {}
    candidates: Dict[str, Any] = {}

    def _doc_uid(d) -> str:
        m = d.metadata or {}
        return f"{m.get('article_number','')}__{m.get('clause_id','')}"

    for idx, (doc, rel) in enumerate(sem_pairs):
        uid = _doc_uid(doc)
        candidates[uid] = doc
        sem_rank[uid] = idx  # 0-based rank
        sem_rel[uid] = float(rel)

    # 3) Signal-filtered candidates (metadata where signals contains X)
    # For each signal we fetch a slice and keep the best rank across signals.
    sig_rank: Dict[str, int] = defaultdict(lambda: 10_000_000)  # large default
    overlap_counts: Dict[str, int] = defaultdict(int)

    if query_signals:
        for signal in query_signals:
            where = {"signals": {"$contains": signal}}
            sig_pairs = _search_with_relevance_scores(vs, feature_text, k=k_per_signal, where=where)
            for idx, (doc, _rel) in enumerate(sig_pairs):
                uid = _doc_uid(doc)
                candidates[uid] = doc
                # best (lowest) rank across signals
                if idx < sig_rank[uid]:
                    sig_rank[uid] = idx
                # count how many query signals this doc claims
                doc_signals = doc.metadata.get("signals", []) or []
                if isinstance(doc_signals, list) and signal in doc_signals:
                    overlap_counts[uid] += 1

    # Ensure any doc in candidates at least has a default rank and overlap
    for uid in list(candidates.keys()):
        if uid not in sem_rank:
            sem_rank[uid] = 10_000_000  # not in semantic list
            sem_rel[uid] = 0.0
        if uid not in overlap_counts:
            overlap_counts[uid] = 0

    # 4) Fuse rankings with RRF + bonus for signal overlap proportion
    # RRF score: sum(1/(k + rank_i)). Here we have:
    #   - semantic: rank_sem
    #   - signal-driven: rank_sig  (weighted by lambda_signal)
    # Then add a small overlap bonus proportional to fraction of matched signals.
    fused: List[Tuple[str, float]] = []
    for uid in candidates.keys():
        r_sem = sem_rank[uid]
        r_sig = sig_rank[uid]
        rr_sem = 1.0 / (rrf_k + r_sem)
        rr_sig = 1.0 / (rrf_k + r_sig)
        overlap = overlap_counts[uid]
        frac = (overlap / max(len(query_signals), 1)) if query_signals else 0.0
        final = rr_sem + (lambda_signal * rr_sig) + (overlap_bonus * frac)
        fused.append((uid, final))

    fused.sort(key=lambda x: x[1], reverse=True)
    top = fused[:top_k]

    # 5) Build output payload with helpful fields
    results: List[Dict[str, Any]] = []
    for uid, final_score in top:
        doc = candidates[uid]
        m = doc.metadata or {}
        results.append({
            "clause_id": m.get("clause_id"),
            "article_number": m.get("article_number"),
            "article_title": m.get("article_title"),
            "type": m.get("type"),
            "signals": m.get("signals", []),
            "text": doc.page_content,
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


# =========================
# UI
# =========================

st.title("🧭 Geo-Specific Compliance Pipeline")
st.caption("PRD → Signals (LLM) → Retrieval (Chroma + HF) → Geo Reasoner (LLM)")

with st.sidebar:
    st.subheader("Retriever Settings")
    persist_directory = st.text_input("Chroma persist_directory", value="./chroma_store")
    collection_name = st.text_input("Chroma collection_name", value="clauses")
    hf_model_name = st.text_input("HF Embedding Model", value="sentence-transformers/all-MiniLM-L6-v2")
    k_semantic = st.slider("Semantic candidate pool (k_semantic)", 20, 200, 60, 5)
    k_per_signal = st.slider("Per-signal pool (k_per_signal)", 10, 100, 25, 5)
    min_confidence = st.slider("Min signal confidence", 0.0, 1.0, 0.55, 0.05)
    top_k = st.slider("Top-K final results", 5, 20, 10, 1)
    st.markdown("---")
    st.info("Make sure your Chroma store is built with clause metadata including `signals`.", icon="ℹ️")

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

    # 3) Retrieve top clauses
    with st.spinner("Retrieving relevant clauses…"):
        try:
            results = retrieve_top10_clauses(
                feature_text=feature_text,
                query_signals=query_signals,
                persist_directory=persist_directory,
                collection_name=collection_name,
                hf_model_name=hf_model_name,
                k_semantic=k_semantic,
                k_per_signal=k_per_signal,
                top_k=top_k
            )
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            st.stop()

    st.subheader("📚 Retrieved Clauses (Top Matches)")
    if results:
        for r in results:
            header = f"{r.get('clause_id')} — Art. {r.get('article_number','')} — {r.get('article_title','')}"
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
        