import os
from fastapi import FastAPI
from pydantic import BaseModel
from .rag_store import RAGstore
from .memory import MemoryStore
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app=FastAPI()
rag = RAGstore(persist_dir="rag_db")
mem = MemoryStore(path="memory.jsonl")

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
