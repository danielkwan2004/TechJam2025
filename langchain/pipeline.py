from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

def direct_query(index_name: str, namespace: str, text: str,
                 model_name="sentence-transformers/all-mpnet-base-v2", top_k=3):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    model = SentenceTransformer(model_name)
    vec = model.encode([text], normalize_embeddings=True)[0].tolist()
    res = index.query(vector=vec, top_k=top_k, namespace=(namespace or None), include_metadata=True)
    print(f"ns='{namespace or '(default)'}' matches:", len(res.get('matches', [])))
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        print("  id:", m.get("id"), "| score:", m.get("score"), "| clause_id:", md.get("clause_id"))
        

if __name__ == "__main__":
    direct_query("legal-clauses", "EU_DSA", "laws related to child safety")