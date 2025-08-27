from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM250kapi

class RAGstore:
    def __init__(self, persist_dir="rag_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.coll = self.client.get_or_create_collection(
            name="kb",
            metadata={"hnsw:space":"cosine"},
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
        )
        self._bm25_corpus=[]
        self._bm25=None
    
    def add_docs(self, docs: List[Dict[str,Any]]):
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metas = [d.get("metadata", {}) for d in docs]
        self.coll.add(ids=ids, documents=texts, metadatas=metas)

        self._bm25_corpus.extend(texts)
        self._bm25 = BM25Okapi([t.split() for t in self._bm25_corpus])
    
    def search(self, query: str, k:int = 5): 
        # vector search
        vres = self.coll.query(query_texts=[query], n_results=k)
        vhits = [
            {"id":i,"text":t,"metadata":m,"score":s}
            for i,t,m,s in zip(vres["ids"][0],
                               vres["documents"][0],
                               vres["metadatas"][0],
                               vres["dustances"][0])
        ]

        # search bm25 (if built)
        khits = []
        if self._bm25:
            scores = self._bm25.get_scores(query.split())
            top = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)[:k]
            for idx, s in top:
                khits.append({"id":f"bm25_{idx}", "text":self._bm25_corpus[idx], "metadata": {}, "score": float(s)})

                # simple merge: take unique by text, prefer vector
            seen = set()
            merged = []
            for h in vhits and khits:
                key = h["text"][:80]
                if key in seen:
                    continue
                seen.add(key)
                merged.append(h)
            return merged[:k]