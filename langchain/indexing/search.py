# Sample usage: python search.py --query "what laws relate to child safety" --index legal-clauses --namespace eu_dsa --model sentence-transformers/all-mpnet-base-v2 --retrieval_k 50 --top_k 10

import os
import argparse
from dotenv import load_dotenv
from pinecone import Pinecone

from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

load_dotenv()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="Natural-language query text")
    p.add_argument("--index", default="legal-clauses-mpnet", help="Pinecone index name")
    p.add_argument("--namespace", default="eu_dsa_mpnet", help="Pinecone namespace")

    p.add_argument("--top_k", type=int, default=10, help="Final results after rerank")
    p.add_argument("--retrieval_k", type=int, default=50, help="Initial Pinecone shortlist size")

    p.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                   help="SentenceTransformer used for indexing & query")
    p.add_argument("--max_length", type=int, default=512, help="Tokenizer max_length for ST model")

    p.add_argument("--reranker_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                   help="Cross-encoder model for reranking")
    p.add_argument("--no_rerank", action="store_true", help="Disable reranking (debug)")
    p.add_argument("--rerank_batch_size", type=int, default=32, help="Batch size for reranker")
    args = p.parse_args()

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("Set PINECONE_API_KEY in your environment")

    # Pinecone client + index
    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.index)

    # SentenceTransformer for query embedding (must match ingest model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer(args.model, device=device)
    try:
        st.max_seq_length = args.max_length
    except Exception:
        pass

    # Encode query (cosine-normalized)
    qvec = st.encode([args.query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()

    # Retrieve shortlist
    res = index.query(
        vector=qvec,
        top_k=args.retrieval_k,
        include_metadata=True,
        include_values=False,
        namespace=args.namespace,
    )
    matches = res.get("matches", []) or []
    if not matches:
        print("No results.")
        return

    if not args.no_rerank:
        # Cross-encoder reranker
        reranker = CrossEncoder(args.reranker_model, max_length=512, device=device)

        # Prepare pairs (query, passage). Truncate long texts a bit just to be safe.
        texts = [
            (m.get("metadata", {}) or {}).get("clause_text", "")[:4000]  # soft trim by chars
            for m in matches
        ]
        pairs = [(args.query, t) for t in texts]

        scores = reranker.predict(pairs, batch_size=args.rerank_batch_size)  # higher = better
        for m, s in zip(matches, scores):
            m["rerank_score"] = float(s)

        matches.sort(key=lambda m: m.get("rerank_score", 0.0), reverse=True)

    # Final top_k (either reranked, or raw Pinecone similarity if --no_rerank)
    final = matches[:args.top_k]

    # Pretty print
    for i, m in enumerate(final, 1):
        md = m.get("metadata") or {}
        pinecone_score = m.get("score")
        rerank_score = m.get("rerank_score")
        print(f"\n[{i}] id={m['id']}")
        if rerank_score is not None:
            print(f"    rerank_score={rerank_score:.4f}  pinecone_score={pinecone_score:.4f}" if pinecone_score is not None else f"    rerank_score={rerank_score:.4f}")
        elif pinecone_score is not None:
            print(f"    pinecone_score={pinecone_score:.4f}")

        print(f"    article={md.get('article_number')}  clause_id={md.get('clause_id')}")
        if md.get("article_title"):
            print(f"    title={md['article_title']}")

        snippet = (md.get("clause_text") or "").strip()
        if snippet:
            snippet = " ".join(snippet.split())
            if len(snippet) > 260:
                snippet = snippet[:260].rstrip() + " …"
            print(f"    text: {snippet}")

if __name__ == "__main__":
    main()
