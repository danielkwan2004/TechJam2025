import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.api.rag_store import RAGstore
from app.api.memory import MemoryStore
from dotenv import load_dotenv
import json
from openai import OpenAI
from pathlib import Path
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app=FastAPI()
rag = RAGstore(persist_dir="rag_db")
mem = MemoryStore(path="memory.jsonl")

BASE_DIR = Path(__file__).resolve().parents[2]          
KB_DIR   = BASE_DIR / "knowledge_base"                  
DATA_DIR = BASE_DIR / "data"                            
RAG_DIR  = DATA_DIR / "rag_db"                          
MEM_PATH = DATA_DIR / "memory.jsonl"                    
ENF_PATH = KB_DIR / "enforcement_status.json" 
LAW_CARDS_PATH = (Path(__file__).resolve().parents[3] / "knowledge_base" / "law_cards.json")
try:
    cards_raw = json.loads(LAW_CARDS_PATH.read_text(encoding="utf-8"))
    LAW_CARDS = cards_raw["laws"] if isinstance(cards_raw, dict) and "laws" in cards_raw else cards_raw
except Exception:
    LAW_CARDS = []          

# make sure runtime dirs exist
RAG_DIR.mkdir(parents=True, exist_ok=True)
MEM_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI()
rag = RAGstore(persist_dir=str(RAG_DIR))
mem = MemoryStore(path=str(MEM_PATH))

# --- load enforcement status once ---
try:
    with ENF_PATH.open("r", encoding="utf-8") as f:
        enf = json.load(f)
    # support { "laws": { ... } } or [ { "law": ... }, ... ]
    enf_map = enf["laws"] if isinstance(enf, dict) and "laws" in enf else {x["law"]: x for x in enf}
except Exception:
    enf_map = {}

def _keyword_hits(text: str, keywords: list[str]) -> int:
    text_low = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_low)

def _any_negation(text: str, negations: list[str]) -> bool:
    if not negations: return False
    low = text.lower()
    return any(neg.lower() in low for neg in negations)

def pick_candidates(feature_text: str, signals: list[str], hits: list[dict]):
    """Return [(law_id, score, matched_signal)] sorted by score desc."""
    results = []
    # flatten hit text and collect any law tags from metadata
    hit_text = "\n\n".join(h.get("text","") for h in hits)
    hit_laws = { (h.get("metadata") or {}).get("law") for h in hits }
    hit_laws.discard(None)

    text_for_kw = f"{feature_text}\n\n{hit_text}"
    sigset = set(signals)

    for law in LAW_CARDS or []:
        law_id = law.get("id")
        trig = (law.get("triggers") or {})
        kws = trig.get("keywords", [])
        sigs = trig.get("signals", [])
        must_all = set(trig.get("must_all", []))
        must_any = set(trig.get("must_any", []))
        negs = trig.get("negations", [])

        # negations
        if _any_negation(text_for_kw, negs):
            continue

        # must_all / must_any
        if must_all and not must_all.issubset(sigset):
            continue
        if must_any and not (must_any & sigset):
            continue

        # overlaps
        sig_overlap = len(sigset & set(sigs))
        kw_score = _keyword_hits(text_for_kw, kws)
        law_bonus = 2 if law_id in hit_laws else 0  # RAG found chunks tagged with this law

        score = sig_overlap * 2 + kw_score + law_bonus
        if score > 0:
            # pick any matched signal for the reason template
            matched_sig = next(iter((sigset & set(sigs)) or sigset or {"context"}))
            results.append((law_id, score, matched_sig))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
class  DecideReq(BaseModel):
    feature_id: str
    text: str
    signals: list[str] = []
    region_hint: list[str] | None = None
    age_audience_hint: str | None = None
    top_k: int = 5

class IngestItem(BaseModel):
    id:str
    text:str 
    metadata:dict = {}

class ChatReq(BaseModel):
    user_id:str
    message:str 
    top_k: int = 5

@app.post("/ingest")
def ingest(items:list[IngestItem]):
    rag.add_docs([i.model_dump() for i in items])
    return {"ok":True, "added":len(items)}

@app.post("/chat")
def chat(req: ChatReq):
    # retrive memory + context
    mem_notes = mem.get_notes(req.user_id, limit = 8) 
    ctx = rag.search(req.message, k=req.top_k)
    if not ctx:
        return {"answer":"I don't know. Please provide more info."}

    # building the prompt...
    context_block = "\n\n".join(
        [f"{i+1}] id={c['id']} meta={c.get('metadata',{})}\n{c['text']}" for i, c in enumerate(ctx)]
    )
    memory_block = "\n".join([f"- {m}" for m in mem_notes]) or "- (empty)"
    system = (
        "You answer using only the facts in CONTEXT and MEMORY."
        "Cite source ids like [1], [2], when you see them. If unsure, say you need more info."
    )
    user = f"MEMORY:\n{memory_block}\n\nCONTEXT:\n{context_block}\n\nQUESTION:\n{req.message}"

    # call LLM
    resp = client.chat.completions.create(
        model="gpt-4o", # change this if needed
        messages=[{"role":"system", "content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content

    # naive learning, add a note but in future can improve this with some LLM summarizer like notebookLM?
    mem.add_note(req.user_id, note=f"Q:{req.message} --> A:{answer[:240]}")

    return {"answer":answer,"used_sources": [c["id"] for c in ctx]}

@app.post("/decide")
def decide(req: DecideReq):
    q = " ".join(([req.text] + req.signals) if req.signals else [req.text])
    hits = rag.search(q, k=req.top_k)

    cands = pick_candidates(req.text, req.signals, hits)
    
    # thresholindg (tweakable)
    TOP = cands[0] if cands else None
    needs_geo_logic = False
    reason = "No clear legal trigger found."
    law_candidates = []
    conf = 0.35

    if TOP and TOP[1] >= 3: # simple threshold, tweakable
        needs_geo_logic = True
        law_candidates = [c[0] for c in cands if c[1] >= max(3, TOP[1] - 1)]
        reason = f"Signals/Keywords matched for {law_candidates[0]} (score = {TOP[1]})."
        conf = min(0.5 + 0.1 * TOP[1], 0.95)
    elif TOP:
        needs_geo_logic = "needs_review"
        law_candidates = [TOP[0]]
        reason = f"Some signals matched for {TOP[0]} but below threshold (score = {TOP[1]})."
        conf = 0.5
    def get_enf(law_id: str):
        v = enf_map.get(law_id) if isinstance(enf_map, dict) else None
        if isinstance(v, dict):
            return {k: v[k] for k in ("law","status","as_of","note") if k in v} | {"law": law_id}
        return {"law": law_id, "status": "unknown"}

    enf_list = [get_enf(l) for l in law_candidates]

    return {
        "needs_geo_logic": needs_geo_logic,              # true | false | "needs_review"
        "reason": reason,
        "law_candidates": law_candidates,
        "enforcement_status": enf_list,
        "confidence": round(conf, 2),
        "citations": [f"kb:{h['id']}" for h in hits],
        "evidence_ids": []  # wire B's evidence ids here when available
    }
