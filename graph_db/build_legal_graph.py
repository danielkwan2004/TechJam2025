from neo4j import GraphDatabase
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import re, hashlib, requests, time
from bs4 import BeautifulSoup 
from markdownify import markdownify as md
import os


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "gayassniggers234" 

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

LAW_SOURCES = [
    {
        "id": "US_2258A",
        "title": "18 U.S.C. § 2258A — Reporting requirements of providers",
        "jurisdiction": "US-federal",
        "url": "https://www.law.cornell.edu/uscode/text/18/2258A"
    },
    {
        "id": "UT_SMRA",
        "title": "Utah Social Media Regulation Act (S.B. 152 & H.B. 311, 2023)",
        "jurisdiction": "US-UT",
        "url": "https://en.wikipedia.org/wiki/Utah_Social_Media_Regulation_Act"
    },
    {
        "id": "FL_HB3_2024",
        "title": "Florida HB 3 (2024) — Online Protections for Minors",
        "jurisdiction": "US-FL",
        "url": "https://www.flsenate.gov/Session/Bill/2024/3"
    },
    {
        "id": "CA_SB976_2024",
        "title": "California SB 976 — Protecting Our Kids from Social Media Addiction Act",
        "jurisdiction": "US-CA",
        "url": "https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202320240SB976"
    },
    {
        "id": "EU_DSA",
        "title": "EU Digital Services Act (Regulation (EU) 2022/2065)",
        "jurisdiction": "EU/EEA",
        "url": "https://en.wikipedia.org/wiki/Digital_Services_Act"
    },
]

@dataclass
class SectionChunk:
    law_id: str
    section: str
    text:str

def http_get(url: str) -> str:
    """Fetch HTML and return text content (markdown)"""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    
    # Convert the legal info pages from HTML to markdown-ish text

    text_md = md(str(soup), strp=["a", "img"])
    return text_md

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")

def chunk_markdown_by_headings(law_id: str, text_md: str, max_chars: int = 2000):
    """Chunk by markdown headings; further split long blocks"""
    chunks: List[SectionChunk] = []
    lines = text_md.spltlines()
    current_heading_stack: List[str] = []
    buf: List[str] = []

    def flush():
        if not buf:
            return
        section_path = " / ".join(current_heading_stack) if current_heading_stack else "root"
        blob = "\n".join(buf).strip()
        if not blob: 
            return 
        # Split long blobs into smaller chunks
        for i in range(0, len(blob), max_chars):
            chunks.append(SectionChunk(law_id=law_id, section=section_path, text=blob[i:i+max_chars]))
        buf.clear()

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()

            current_heading_stack = current_heading_stack[:level - 1] + [title]
        else:
            buf.append(line)
    flush()
    chunks = [c for c in chunks if len(c.text.strip()) > 40]
    return chunks
RX_USC_SECTION = re.compile(r"§\s?(\d{3,5}[A-Za-z]?)")
RX_SUBSECT = re.compile(r"\(([a-z0-9]+)\)")
# - EU DSA: "Article 34", "Art. 28", "Annex"
RX_EU_ART = re.compile(r"\b(?:Article|Art\.)\s+(\d+)\b", re.IGNORECASE)
# - State bills: "Section 3", "Sec. 4", "Subdivision (a)"
RX_SECTION = re.compile(r"\b(?:Section|Sec\.)\s+(\d+)\b", re.IGNORECASE)
# - Common “Definitions” cues
RX_DEF_LINE = re.compile(r"^[“\"']?([A-Z][A-Za-z0-9 _/-]{2,40})[”\"']?\s+means\s+", re.IGNORECASE)

def extract_refs(law_id: str, text: str) -> Dict[str, List[str]]:
    """Return dict of detected references by type."""
    refs = {
        "usc": RX_USC_SECTION.findall(text) if law_id == "US_2258A" else [],
        "eu_art": RX_EU_ART.findall(text) if law_id == "EU_DSA" else [],
        "state_sec": RX_SECTION.findall(text) if law_id in {"CA_SB976_2024", "FL_HB3_2024", "UT_SMRA"} else [],
        "sub": RX_SUBSECT.findall(text)[:10],  # keep a few
    }
    return {k: v for k, v in refs.items() if v}


RX_USC_SECTION = re.compile(r"§\s?(\d{3,5}[A-Za-z]?)")
RX_SUBSECT = re.compile(r"\(([a-z0-9]+)\)")
# - EU DSA: "Article 34", "Art. 28", "Annex"
RX_EU_ART = re.compile(r"\b(?:Article|Art\.)\s+(\d+)\b", re.IGNORECASE)
# - State bills: "Section 3", "Sec. 4", "Subdivision (a)"
RX_SECTION = re.compile(r"\b(?:Section|Sec\.)\s+(\d+)\b", re.IGNORECASE)
# - Common “Definitions” cues
RX_DEF_LINE = re.compile(r"^[“\"']?([A-Z][A-Za-z0-9 _/-]{2,40})[”\"']?\s+means\s+", re.IGNORECASE)

def extract_refs(law_id: str, text: str) -> Dict[str, List[str]]:
    """Return dict of detected references by type."""
    refs = {
        "usc": RX_USC_SECTION.findall(text) if law_id == "US_2258A" else [],
        "eu_art": RX_EU_ART.findall(text) if law_id == "EU_DSA" else [],
        "state_sec": RX_SECTION.findall(text) if law_id in {"CA_SB976_2024", "FL_HB3_2024", "UT_SMRA"} else [],
        "sub": RX_SUBSECT.findall(text)[:10],  # keep a few
    }
    return {k: v for k, v in refs.items() if v}


def extract_definitions(text: str) -> List[Tuple[str, str]]:
    """Extract definitions from text lines."""
    definitions = []
    for line in text.splitlines():
        m = RX_DEF_LINE.match(line.strip())
        if m:
            term = m.group(1).rstrip(":").strip()
            definitions.append((term, line.strip()))
    return definitions


def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

SCHEMA_CYPHER = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Law)      REQUIRE l.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section)  REQUIRE s.uid IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept)  REQUIRE c.uid IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Obligation) REQUIRE o.uid IS UNIQUE",
]


def write_schema(tx):
    for stmt in SCHEMA_CYPHER:
        tx.run(stmt)

def upsert_law(tx, law):
        tx.run("""
        MERGE (l:Law {id:$id})
        SET l.title=$title, l.jurisdiction=$jurisdiction, l.url=$url
    """, **law)

def upsert_section(tx, law_id: str, section: SectionChunk):
    uid = f"{law_id}:{hash_text(section.section+section.text[:150])}"
    tx.run("""
        MATCH (l:Law {id:$law_id})
        MERGE (s:Section {uid:$uid})
        SET s.law_id=$law_id, s.section=$section, s.text=$text
        MERGE (l)-[:HAS_SECTION]->(s)
    """, law_id=law_id, uid=uid, section=section.section, text=section.text)
    return uid

def upsert_concept(tx, law_id: str, term: str, definition_text: str, section_uid: str):
    cuid = f"{law_id}:concept:{hash_text(term)}"
    tx.run("""
        MERGE (c:Concept {uid:$cuid})
        SET c.term=$term, c.law_id=$law_id, c.definition=$definition
        WITH c
        MATCH (s:Section {uid:$section_uid})
        MERGE (s)-[:DEFINES]->(c)
    """, cuid=cuid, term=term, law_id=law_id, definition=definition_text, section_uid=section_uid)

def link_refers_to(tx, from_uid: str, to_uid: str, kind: str):
    tx.run("""
        MATCH (a:Section {uid:$from_uid})
        MATCH (b:Section {uid:$to_uid})
        MERGE (a)-[r:REFERS_TO {kind:$kind}]->(b)
        ON CREATE SET r.cycle=false
    """, from_uid=from_uid, to_uid=to_uid, kind=kind)

def find_sections_by_label(tx, law_id: str, label_fragment: str) -> List[str]:
    res = tx.run("""
        MATCH (s:Section {law_id:$law_id})
        WHERE toLower(s.section) CONTAINS toLower($frag)
        RETURN s.uid AS uid LIMIT 25
    """, law_id=law_id, frag=label_fragment)
    return [r["uid"] for r in res]

def find_any_section_by_text(tx, law_id: str, token: str) -> List[str]:
    res = tx.run("""
        MATCH (s:Section {law_id:$law_id})
        WHERE toLower(s.text) CONTAINS toLower($tok)
        RETURN s.uid AS uid LIMIT 25
    """, law_id=law_id, tok=token)
    return [r["uid"] for r in res]

def tag_cycles(tx):
    # mark edges that participate in a cycle up to length 6
    tx.run("""
        MATCH p=(n:Section)-[r:REFERS_TO*1..6]->(n)
        WITH relationships(p) AS rels
        UNWIND rels AS rel
        SET rel.cycle = true
    """)


# Build graph for all sources

def ingest_all():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as sess:
        sess.execute_write(write_schema)

        # create Law nodes
        for law in LAW_SOURCES:
            sess.execute_write(upsert_law, law)

        # pull, chunk, and write sections
        section_uids_by_law: Dict[str, List[str]] = {law["id"]: [] for law in LAW_SOURCES}
        for law in LAW_SOURCES:
            print(f"[pull] {law['id']} {law['url']}")
            try:
                text_md = http_get(law["url"])
            except Exception as e:
                print(f"  ! fetch failed: {e}")
                continue

            chunks = chunk_markdown_by_headings(law["id"], text_md)
            print(f"  [+] {len(chunks)} chunks")

            for ch in chunks:
                uid = sess.execute_write(upsert_section, law["id"], ch)
                section_uids_by_law[law["id"]].append(uid)

                # concepts / definitions
                defs = extract_definitions(ch.text)
                for term, defline in defs:
                    sess.execute_write(upsert_concept, law["id"], term, defline, uid)

            # link intra-law references heuristically
            for ch in chunks:
                refs = extract_refs(law["id"], ch.text)
                if not refs:
                    continue
                # try to resolve refs to target sections in the same law
                # heuristics: look for sections that contain the reference tokens
                for kind, toks in refs.items():
                    for tok in toks:
                        # search label first then text
                        tgt = sess.execute_read(find_sections_by_label, law["id"], tok)
                        if not tgt:
                            tgt = sess.execute_read(find_any_section_by_text, law["id"], tok)
                        if tgt:
                            from_uid = sess.execute_read(find_any_section_by_text, law["id"], ch.section)
                            from_uid = from_uid[0] if from_uid else sess.execute_read(find_any_section_by_text, law["id"], ch.text[:30])[0]
                            # to be safe, link current chunk uid to first match
                            # (we recompute uid same as when created)
                            from_uid = f"{law['id']}:{hash_text(ch.section+ch.text[:150])}"
                            to_uid = tgt[0]
                            sess.execute_write(link_refers_to, from_uid, to_uid, kind=kind)

        # tag cycles in REFERS_TO
        print("[cycle] tagging REFERS_TO cycles")
        sess.execute_write(tag_cycles)

    driver.close()
    print("[done] graph built")

EXAMPLE_QUERIES = r"""
// Given a section uid, expand reference-aware neighborhood (2 hops), avoid cycle edges
MATCH (s:Section {uid:$uid})
CALL apoc.path.expandConfig(s, {
  relationshipFilter: 'REFERS_TO>',
  minLevel: 1, maxLevel: 2,
  bfs: true, uniqueness: 'NODE_GLOBAL'
}) YIELD path
WITH path
WHERE all(r IN relationships(path) WHERE coalesce(r.cycle,false) = false)
RETURN nodes(path) AS nodes, relationships(path) AS rels
LIMIT 25;

// Find candidate sections for a text token, then expand 1 hop
MATCH (s:Section)
WHERE toLower(s.text) CONTAINS toLower($token)
WITH s LIMIT 5
MATCH p=(s)-[:REFERS_TO*0..1]->(t)
RETURN s, t, p LIMIT 50;
"""

def main():
    ingest_all()
    print("\n example query:")
    print(EXAMPLE_QUERIES)

if __name__ == "__main__":
    main()