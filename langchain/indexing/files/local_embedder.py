# embedder.py
# ------------------------------------------------------------
# Upserts clauses into Pinecone Local via gRPC
# Requires Pinecone Local running on http://localhost:5080
# ------------------------------------------------------------
import os
import re
import json
import math
import argparse
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# Pinecone Local gRPC client
from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import ServerlessSpec  # still required by create_index()

load_dotenv()


# -------------------------
# Helpers
# -------------------------
def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def sanitize_id(s: str) -> str:
    # Pinecone vector IDs should be URL-safe
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def yield_clauses(articles_json: List[Dict[str, Any]]):
    """
    Works for both:
      - articles: {"type":"article","article_number":"30","title":...,"clauses":[...]}
      - recitals: {"type":"recital","article_number":"R","recital_number":"119","title":"Recital 119","clauses":[...]}

    Yields (text, metadata) pairs for each clause.
    """
    for entry in articles_json:
        entry_type = entry.get("type", "article")  # "article" or "recital"
        article_number = str(entry.get("article_number", "")).strip()
        title = clean_whitespace(entry.get("title", ""))
        rec_num = entry.get("recital_number")  # may be None for articles

        for clause in entry.get("clauses", []):
            cid = str(clause.get("clause_id", "")).strip()
            ctext = clean_whitespace(clause.get("clause_text", ""))

            if not ctext:
                continue

            raw_signals = clause.get("signals", [])
            if isinstance(raw_signals, list):
                signals = [str(s) for s in raw_signals if s is not None]
            else:
                signals = []

            meta = {
                "type": entry_type,
                "article_number": article_number,
                "recital_number": str(rec_num) if rec_num is not None else None,
                "article_title": title,
                "clause_id": cid,
                "clause_text": ctext,
                "signals": signals,
                "source": "articles.json",
                "embedding_model": None,  # set at upsert time
            }
            # drop Nones
            meta = {k: v for k, v in meta.items() if v is not None}

            yield ctext, meta


def build_namespace_map(root_dir: str) -> List[Tuple[str, str]]:
    """
    Returns a list of (filepath, namespace) tuples for the six tagged files.
    Only files that exist are returned.
    """
    mapping = [
        ("EU_DSA_tagged.json", "EU_DSA"),
        ("CA_SB976_tagged.json", "CA_SB976"),
        ("FL_HB3_2024_tagged.json", "FL_HB3_2024"),
        ("US_ALL_2258A_tagged.json", "US_ALL_2258A"),
        ("UT_HB311_tagged.json", "UT_HB311"),
        ("UT_SB152_tagged.json", "UT_SB152"),
    ]
    out = []
    for fname, ns in mapping:
        path = os.path.join(root_dir, fname)
        if os.path.exists(path):
            out.append((path, ns))
        else:
            print(f"[WARN] Missing file (skipping): {path}")
    return out


def ensure_index_local(
    pc: PineconeGRPC,
    index_name: str,
    dim: int,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
):
    """Create the dense index if it doesn't exist on Pinecone Local."""
    if not pc.has_index(index_name):
        print(f"[INFO] Creating local index '{index_name}' (dim={dim}, metric={metric}) …")
        model = pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
            deletion_protection="disabled",
            tags={"environment": "development"},
        )
        print("[INFO] Index model created:\n", model)
    else:
        print(f"[INFO] Index '{index_name}' already exists.")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Embed & upsert legal clauses to Pinecone Local via gRPC.")
    parser.add_argument("--root_dir", default=".", help="Directory containing the *_tagged.json files")
    parser.add_argument("--index", default="legal-clauses", help="Pinecone index name (local)")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                        help="SentenceTransformer model or local path")
    parser.add_argument("--batch", type=int, default=128, help="Upsert batch size")
    parser.add_argument("--encode_batch", type=int, default=64, help="SentenceTransformer encode batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max tokens (truncate) for SentenceTransformer")
    parser.add_argument("--host", default="http://localhost:5080", help="Pinecone Local host (e.g., http://localhost:5080)")
    args = parser.parse_args()

    # 1) Collect the six files we want to upsert (only those that exist)
    files_and_ns = build_namespace_map(args.root_dir)
    if not files_and_ns:
        print("[ERROR] No tagged JSON files found. Expected files like EU_DSA_tagged.json in root_dir.")
        return

    # 2) Build embedder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    model = SentenceTransformer(args.model, device=device)
    try:
        model.max_seq_length = args.max_length
    except Exception:
        pass
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dim: {embedding_dim} | Model: {args.model}")

    # 3) Connect to Pinecone Local via gRPC
    # api_key value doesn't matter with Pinecone Local, but must be present
    pc = PineconeGRPC(api_key="pclocal", host=args.host)
    ensure_index_local(pc, args.index, embedding_dim, metric="cosine")

    # Target index handle (disable TLS for local)
    index_host = pc.describe_index(name=args.index).host
    index = pc.Index(host=index_host, grpc_config=GRPCClientConfig(secure=False))

    # 4) Process each file/namespace
    for filepath, namespace in files_and_ns:
        print(f"\n[INFO] Processing file: {filepath} -> namespace: {namespace}")

        data = load_json(filepath)
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for txt, meta in yield_clauses(data):
            texts.append(txt)
            metas.append(meta)

        if not texts:
            print(f"[WARN] No clauses found in {filepath}. Skipping …")
            continue

        print(f"[INFO] Loaded {len(texts)} clauses from {os.path.basename(filepath)}")

        # encode fn
        def encode_chunk(chunk_texts: List[str]):
            return model.encode(
                chunk_texts,
                batch_size=args.encode_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,   # cosine
                show_progress_bar=False
            )

        total = len(texts)
        batches = math.ceil(total / args.batch)
        upserted_total = 0

        for b in range(batches):
            start = b * args.batch
            end = min((b + 1) * args.batch, total)

            batch_texts = texts[start:end]
            batch_metas = metas[start:end]

            emb = encode_chunk(batch_texts)

            vectors = []
            for vec, meta in zip(emb, batch_metas):
                # record which model produced the vector
                meta["embedding_model"] = args.model

                # Use article_number + clause_id for deterministic id (safe chars only)
                raw_id = f"{meta.get('article_number','')}__{meta.get('clause_id','')}"
                vec_id = sanitize_id(raw_id)

                vectors.append({
                    "id": vec_id,
                    "values": vec.tolist(),
                    "metadata": meta
                })

            # Upsert to Pinecone Local
            index.upsert(vectors=vectors, namespace=namespace)
            upserted_total += len(vectors)
            print(f"[INFO] Upserted {len(vectors)} vectors [{start}:{end}] -> ns='{namespace}'")

        print(f"[INFO] Done: {upserted_total} vectors upserted to namespace '{namespace}'")

    # Optional: show basic stats
    try:
        stats = index.describe_index_stats()
        print("\n[INFO] Index stats:\n", stats)
    except Exception as e:
        print(f"[WARN] Could not fetch index stats: {e}")

    print("\n[INFO] All done.")


if __name__ == "__main__":
    main()
