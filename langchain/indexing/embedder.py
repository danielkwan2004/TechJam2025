# embed_to_pinecone.py
import os
import re
import json
import math
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from llm import LegalBERTEmbedder  # <-- import your embedder

# pip install "pinecone[grpc]"
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def sanitize_id(s: str) -> str:
    # Pinecone vector IDs should be URL-safe
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s)

def load_articles(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def yield_clauses(articles_json: List[Dict[str, Any]]):
    """
    Expects: [{ article_number, title, clauses: [{clause_id, clause_text}, ...] }]
    Yields (text, metadata).
    """
    for art in articles_json:
        article_number = str(art.get("article_number", "")).strip()
        title = clean_whitespace(art.get("title", ""))

        for clause in art.get("clauses", []):
            cid = str(clause.get("clause_id", "")).strip()
            ctext = clean_whitespace(clause.get("clause_text", ""))

            if not ctext:
                continue

            meta = {
                "article_number": article_number,
                "article_title": title,
                "clause_id": cid,                   # keep human-readable form (e.g., "30.1(a)")
                "char_count": len(ctext),
                "word_count": len(ctext.split()),
                "source": "articles.json",
            }
            yield ctext, meta

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
    parser.add_argument("--model", default="nlpaueb/legal-bert-base-uncased",
                        help="HuggingFace model or local path")
    parser.add_argument("--index", default="legal-clauses", help="Pinecone index name")
    parser.add_argument("--namespace", default="eu_dsa", help="Pinecone namespace")
    parser.add_argument("--batch", type=int, default=128, help="Embed/upsert batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Tokenizer max_length")
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

    # Initialize your LegalBERT embedder
    embedder = LegalBERTEmbedder(model_name=args.model)
    hidden_size = embedder.hidden_size
    print(f"Embedding dim: {hidden_size}")

    # Pinecone
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set PINECONE_API_KEY in your environment.")

    pc = Pinecone(api_key=api_key)
    ensure_pinecone_index(pc, args.index, hidden_size, metric="cosine",
                          cloud=args.pinecone_cloud, region=args.pinecone_region)
    index = pc.Index(args.index)
    print(f"Upserting into Pinecone index '{args.index}' namespace '{args.namespace}'")

    # Batch embed + upsert
    total = len(texts)
    batches = math.ceil(total / args.batch)

    for b in range(batches):
        start = b * args.batch
        end = min((b + 1) * args.batch, total)

        batch_texts = texts[start:end]
        batch_metas = metas[start:end]

        # Embed with your llm.py class
        emb = embedder.embed(batch_texts, max_length=args.max_length).numpy()

        vectors = []
        for vec, meta in zip(emb, batch_metas):
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