# embed_to_pinecone.py
import os
import re
import json
import math
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from sentence_transformers import SentenceTransformer
import torch

load_dotenv()

def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def sanitize_id(s: str) -> str:
    # Pinecone vector IDs should be URL-safe
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s)

def load_articles(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def yield_clauses(articles_json):
    """
    Works for both:
      - articles: {"type":"article","article_number":"30","title":...,"clauses":[...]}
      - recitals: {"type":"recital","article_number":"R","recital_number":"119","title":"Recital 119","clauses":[...]}
    Yields (text, metadata) pairs.
    """
    for entry in articles_json:
        entry_type = entry.get("type", "article")  # "article" or "recital"
        article_number = str(entry.get("article_number", "")).strip()
        title = clean_whitespace(entry.get("title", ""))
        rec_num = entry.get("recital_number")  # may be None for articles

        for clause in entry.get("clauses", []):
            cid = str(clause.get("clause_id", "")).strip()       # e.g., "30.1(a)" or "R119"
            ctext = clean_whitespace(clause.get("clause_text", ""))
            if not ctext:
                continue

            # NEW: pull signals array from clause (if present)
            raw_signals = clause.get("signals", [])
            if isinstance(raw_signals, list):
                signals = [str(s) for s in raw_signals if s is not None]
            else:
                # If malformed, coerce to empty list
                signals = []

            meta = {
                "type": entry_type,
                "article_number": article_number,
                "recital_number": str(rec_num) if rec_num is not None else None,
                "article_title": title,
                "clause_id": cid,
                "clause_text": ctext,
                "signals": signals,                 # <-- NEW: include signals array in metadata
                "source": "articles.json",
                "embedding_model": None,            # will be filled right before upsert
            }
            # drop Nones to keep metadata clean
            yield ctext, {k: v for k, v in meta.items() if v is not None}

def ensure_pinecone_index(pc: Pinecone, name: str, dim: int, metric: str = "cosine",
                          cloud: str = "aws", region: str = "us-east-1"):
    existing = [ix.name for ix in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="articles.json", help="Path to articles.json")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                        help="SentenceTransformer model or local path")
    parser.add_argument("--index", default="legal-clauses", help="Pinecone index name")
    parser.add_argument("--namespace", default="eu_dsa", help="Pinecone namespace")
    parser.add_argument("--batch", type=int, default=128, help="Upsert batch size")
    parser.add_argument("--encode_batch", type=int, default=64, help="SentenceTransformer encode batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max tokens (will set model.max_seq_length)")
    parser.add_argument("--pinecone_cloud", default="aws", help="Pinecone serverless cloud")
    parser.add_argument("--pinecone_region", default="us-east-1", help="Pinecone serverless region")
    args = parser.parse_args()

    # Load JSON
    articles = load_articles(args.input)

    # Collect clauses
    texts, metas = [], []
    for txt, meta in yield_clauses(articles):
        texts.append(txt)
        metas.append(meta)

    if not texts:
        print("No clauses found to embed. Exiting.")
        return

    print(f"Loaded {len(texts)} clauses from {args.input}")

    # SentenceTransformer embedder (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = SentenceTransformer(args.model, device=device)
    # Set max sequence length for truncation consistency
    try:
        model.max_seq_length = args.max_length
    except Exception:
        pass

    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {embedding_dim}  |  Model: {args.model}")

    # Pinecone
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set PINECONE_API_KEY in your environment.")

    pc = Pinecone(api_key=api_key)
    ensure_pinecone_index(pc, args.index, embedding_dim, metric="cosine",
                          cloud=args.pinecone_cloud, region=args.pinecone_region)
    index = pc.Index(args.index)
    print(f"Upserting into Pinecone index '{args.index}' namespace '{args.namespace}'")

    # Batch embed + upsert
    total = len(texts)
    upsert_batches = math.ceil(total / args.batch)

    # We’ll encode in chunks of encode_batch (can differ from upsert batch)
    def encode_chunk(chunk_texts: List[str]):
        # normalize_embeddings=True for cosine
        return model.encode(
            chunk_texts,
            batch_size=args.encode_batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    for b in range(upsert_batches):
        start = b * args.batch
        end = min((b + 1) * args.batch, total)
        batch_texts = texts[start:end]
        batch_metas = metas[start:end]

        emb = encode_chunk(batch_texts)

        vectors = []
        for vec, meta in zip(emb, batch_metas):
            meta["embedding_model"] = args.model  # record which model produced this vector
            raw_id = f"{meta['article_number']}__{meta['clause_id']}"
            vec_id = sanitize_id(raw_id)
            vectors.append({
                "id": vec_id,
                "values": vec.tolist(),
                "metadata": meta
            })

        index.upsert(vectors=vectors, namespace=args.namespace)
        print(f"Upserted {len(vectors)} vectors [{start}:{end}]")

    print("Done.")

if __name__ == "__main__":
    main()
