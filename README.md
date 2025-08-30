
# TikTok TechJam 2025 – GeoFlag
Feature → Laws Compliance Reasoning Pipeline


## Problem
From Guesswork to Governance: Automating Geo-Regulation with LLM


## Members
1. Daniel Kwan  
2. Ivan Jerrick Koh  
3. Shaun Tan Shu Ren  
4. Gordon Hong Jia Jie  

## Overview
GeoRag is a compliance reasoning tool, accessed via Streamlit.  
It takes product artifacts such as PRDs as input, analyzes them for potential compliance signals, retrieves relevant legal clauses from a Pinecone vector index, and runs a geo-specific compliance reasoner to determine whether implementation needs to vary across jurisdictions.


## Installation

### 1. Clone repo
```bash
git clone https://github.com/danielkwan2004/TechJam2025
cd TechJam2025
```

### 2. Create a virtual environment
We recommend Python 3.13.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables and run
Copy `.env.example` → `.env` and set:
```bash
OPENAI_API_KEY=yourKeyHere
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=techjam2025
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=yourKeyHere
GOOGLE_API_KEY=yourKeyHere
```

Run:
```bash
cd langchain && docker compose up -d
streamlit run app.py
```


## Tech Stack

### Tools Used
- LLM Application Framework: LangChain and LangSmith  
- Embedding Model: sentence-transformers/all-mpnet-base-v2  
- UI Framework: Streamlit  
- Databases: Pinecone  


### APIs Used
- OpenAI

## Description

### Pinecone Retriever
GeoRag uses Pinecone for vector search. Laws/clauses are pre-indexed into Pinecone with metadata:

```json
{
  "type": "...",
  "article_number": "...",
  "recital_number": "...",
  "title": "..",
  "clauses": [
    {
      "clause_id": "...",
      "clause_text": "....",
      "signals": [...]
    }
  ]
}
```

- Pinecone is run locally via Docker.  
- During retrieval:
  - Semantic search (HF embeddings like sentence-transformers/all-MiniLM-L6-v2)  
  - Signal-filtered search (boosting candidates tagged with extracted signals)  
  - Fusion (reciprocal rank + overlap bonus)  
  - Top-K results returned to the reasoner  

### Signal Extractor

#### Abbreviation Handling
- Abbreviations are stored in a Supabase table:
```sql
abbreviations(term, explanation)
```
- Includes those provided in Data Set → Terminology Table.  
- At runtime, `abbreviation_helper.py` queries Supabase for all known terms.  
- If a term appears in PRD text, it is included in the LLM context.  

Example:
```python
from abbreviation_helper import retrieve_abbreviations

retrieve_abbreviations("Upload limits apply for IMT accounts.")
# → [{'term': 'IMT', 'explanation': 'In-Market Testing'}, ...]
```

Add new abbreviations via Streamlit frontend:
```python
from abbreviation_helper import add_abbreviation

add_abbreviation("BB", "Behavioral Benchmarking")
```

### Law Cards
GeoRag relies on a structured JSON knowledge base called `law_cards.json`.  
Each card represents a regulation, act, or statute, defining keywords, triggers, and signals that map product features to legal obligations.

Example Structure:
```json
{
  "version": "1.1",
  "laws": [
    {
      "id": "US_ALL_2258A",
      "name": "U.S.Code 2258A - NCMEC reporting requirements of providers",
      "region": "US_all",
      "category": [...],
      "triggers": {
        "keywords": [...],
        "signals": [...],
        "must_all": [...],
        "must_any": [
          "ncmec_reference",
          "cyphertipline_reference",
          "include_visual_depictions",
          "include_complete_communication"
        ],
        "negations": [...]
      },
      "reason_templates": [...],
      "priority": "high",
      "last_updated": "2025-08-26"
    }
  ]
}
```

## Pipeline

### 1. Signal Extraction (LLM)
- PRD text analyzed for keyword matches  
- Corresponding signals suggested  
- must_all / must_any logic ensures valid triggers  
- negations reduce false positives  
- LLM corroborates signals (with abbreviations substituted)  
- Outputs signals with fields:  
```json
{
  "law": "...",
  "signal": "...",
  "reason": "...",
  "confidence": "..."
}
```

### 2. Retrieval (Pinecone)
- Extracted signals query Pinecone index of law clauses  
- Clauses tagged with signals are boosted in ranking  

### 3. Reasoning (LLM)
- Feature + retrieved clauses fed into reasoner  
- Outputs final decision:
```json
{
  "geo_specific_required": true,
  "reasoning": "...",
  "related_regulations": [
    {"id": "13-63-103(4)", "title": null, "snippet": ""}
  ]
}
```

