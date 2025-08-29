import os
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime

KB_DIR = (Path(__file__).parent.parent.parent / "knowledge_base").resolve()
LAWS_DIR = KB_DIR / "laws"
LAW_CARDS_FILE = KB_DIR / "law_cards.json"
ENFORCE_FILE = KB_DIR / "enforcement_status.json"
INGEST_URL = "http://127.0.0.1:8000/ingest"
BATCH_SIZE = 1

def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:8]

def chunk_markdown(text, chunk_size=1000):
    # Chunk by headings, fallback to chunk_size chars
    chunks = []
    lines = text.splitlines()
    current = []
    section = None
    for line in lines:
        if line.startswith("#"):
            if current:
                chunks.append({"section": section, "text": "\n".join(current)})
                current = []
            section = line.strip("# ").strip()
        current.append(line)
    if current:
        chunks.append({"section": section, "text": "\n".join(current)})
    # Further split large chunks
    final_chunks = []
    for c in chunks:
        txt = c["text"]
        for i in range(0, len(txt), chunk_size):
            final_chunks.append({
                "section": c["section"],
                "text": txt[i:i+chunk_size]
            })
    return final_chunks

def ingest_docs(docs):
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        try: 
            resp = requests.post(INGEST_URL, json=batch, timeout=30)
            if resp.ok:
                print(f"Added {len(batch)} docs.")
            else:
                print(f"Error: {resp.status_code} {resp.text}")
        except requests.Timeout:
            print(f"Request timed out after 30 seconds")
        except requests.RequestException as e:
            print(f"An error occurred: {e}")

def main():
    docs = []
    today = datetime.now().strftime("%Y-%m-%d")
    # Laws: Markdown files
    for md_file in LAWS_DIR.glob("*.md"):
        law_id = md_file.stem
        sha = file_sha(md_file)
        with open(md_file, encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_markdown(text)
        for idx, chunk in enumerate(chunks):
            doc_id = f"law:{law_id}:chunk:{idx:03d}"
            docs.append({
                "id": doc_id,
                "text": chunk["text"],
                "metadata": {
                    "law": law_id,
                    "file": str(md_file),
                    "section": chunk["section"] or "",
                    "as_of": today,
                    "type": "law_sheet",
                    "sha": sha
                }
            })
    # Law cards: JSON
    if LAW_CARDS_FILE.exists():
        sha = file_sha(LAW_CARDS_FILE)
        cards = json.loads(LAW_CARDS_FILE.read_text(encoding="utf-8"))
        if isinstance(cards, dict):
            laws_list = cards.get("laws", [])
        else:
            laws_list = cards  # assume already a list of law objects
        for law in laws_list:
            law_id = law["laws"][0].get("id", "unknown")
            doc_id = f"lawcards:{law_id}"
            docs.append({
                "id": doc_id,
                "text": json.dumps(law, ensure_ascii=False, indent=2),
                "metadata": {
                    "law": law_id,
                    "file": str(LAW_CARDS_FILE),
                    "type": "law_cards",
                    "sha": sha,
                    "as_of": law.get("last_updated", today)
                }
            })
    else:
        print(f"[WARN] Law cards file not found: {LAW_CARDS_FILE}")

    # --- Enforcement status: JSON (support dict-with-laws or array) ---
    if ENFORCE_FILE.exists():
        sha = file_sha(ENFORCE_FILE)
        enforce = json.loads(ENFORCE_FILE.read_text(encoding="utf-8"))
        if isinstance(enforce, dict) and "laws" in enforce:
            items = [{"law": k, **v} for k, v in enforce["laws"].items()]
        elif isinstance(enforce, list):
            items = enforce
        else:
            items = []

        for item in items:
            law_id = item.get("law", "unknown")
            doc_id = f"enforce:{law_id}:main"
            docs.append({
                "id": doc_id,
                "text": json.dumps(item, ensure_ascii=False, indent=2),
                "metadata": {
                    "law": law_id,
                    "file": str(ENFORCE_FILE),
                    "type": "enforcement",
                    "sha": sha,
                    "as_of": item.get("as_of", today)
                }
            })
    else:
        print(f"[WARN] Enforcement file not found: {ENFORCE_FILE}")

    print(f"Prepared {len(docs)} docs for ingest.")
    if docs:
        ingest_docs(docs)
    else:
        print("[INFO] Nothing to ingest.")

if __name__ == "__main__":
    main()