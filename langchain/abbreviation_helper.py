import os
import json
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

# -------------------------
# Defaults
# -------------------------
default_doc = "Hello World."

DEFAULT_ABBREVIATIONS: List[Dict[str, str]] = [
    {"term": "NR", "explanation": "Not recommended"},
    {"term": "PF", "explanation": "Personalized feed"},
    {"term": "GH", "explanation": "Geo-handler; a module responsible for routing features based on user region"},
    {"term": "CDS", "explanation": "Compliance Detection System"},
    {"term": "DRT", "explanation": "Data retention threshold; duration for which logs can be stored"},
    {"term": "LCP", "explanation": "Local compliance policy"},
    {"term": "Redline",
     "explanation": "Flag for legal review (different from its traditional business use for 'financial loss')"},
    {"term": "Softblock", "explanation": "A user-level limitation applied silently without notifications"},
    {"term": "Spanner", "explanation": "A synthetic name for a rule engine (not to be confused with Google Spanner)"},
    {"term": "ShadowMode", "explanation": "Deploy feature in non-user-impact way to collect analytics only"},
    {"term": "T5", "explanation": "Tier 5 sensitivity data; more critical than T1–T4 in this internal taxonomy"},
    {"term": "ASL", "explanation": "Age-sensitive logic"},
    {"term": "Glow", "explanation": "A compliance-flagging status, internally used to indicate geo-based alerts"},
    {"term": "NSP", "explanation": "Non-shareable policy (content should not be shared externally)"},
    {"term": "Jellybean", "explanation": "Feature name for internal parental control system"},
    {"term": "EchoTrace", "explanation": "Log tracing mode to verify compliance routing"},
    {"term": "BB", "explanation": "Baseline Behavior; standard user behavior used for anomaly detection"},
    {"term": "Snowcap", "explanation": "A synthetic codename for the child safety policy framework"},
    {"term": "FR", "explanation": "Feature rollout status"},
    {"term": "IMT", "explanation": "Internal monitoring trigger"},
]

# -------------------------
# Local persistence
# -------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
ABBR_PATH = os.path.join(DATA_DIR, "abbreviations.json")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_persisted() -> List[Dict[str, str]]:
    """Load locally saved abbreviations; returns [] if none."""
    try:
        if os.path.exists(ABBR_PATH):
            with open(ABBR_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or []
            # filter/normalize
            out = []
            for row in data:
                term = (row.get("term") or "").strip()
                expl = (row.get("explanation") or "").strip()
                if term and expl:
                    out.append({"term": term, "explanation": expl})
            return out
    except Exception:
        pass
    return []


def _save_persisted(rows: List[Dict[str, str]]) -> None:
    _ensure_data_dir()
    with open(ABBR_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _merge_defaults_and_persisted() -> List[Dict[str, str]]:
    """
    Merge defaults with persisted items.
    If a term exists in both, the persisted explanation wins.
    """
    merged: Dict[str, str] = {row["term"]: row["explanation"] for row in DEFAULT_ABBREVIATIONS}
    for row in _load_persisted():
        merged[row["term"]] = row["explanation"]  # override default
    return [{"term": k, "explanation": v} for k, v in merged.items()]


# -------------------------
# Public helpers (used by app)
# -------------------------
def retrieve_all_abbreviations(doc: str = default_doc) -> List[Dict[str, str]]:
    """Return merged (defaults + persisted)."""
    return _merge_defaults_and_persisted()


def retrieve_abbreviations(doc: str = default_doc) -> List[Dict[str, str]]:
    """Return abbreviations that appear (substring) in the doc."""
    out: List[Dict[str, str]] = []
    for abbr in _merge_defaults_and_persisted():
        if abbr["term"] in doc:
            out.append({"term": abbr["term"], "explanation": abbr["explanation"]})
    return out


def add_abbreviation_local(term: str, explanation: str) -> Dict[str, str]:
    """
    Add/update an abbreviation locally in data/abbreviations.json.
    If term exists, we update its explanation.
    Returns the record we wrote.
    """
    term = (term or "").strip()
    explanation = (explanation or "").strip()
    if not term or not explanation:
        raise ValueError("Both 'term' and 'explanation' are required.")

    rows = _load_persisted()
    # update if exists
    found = False
    for row in rows:
        if row["term"] == term:
            row["explanation"] = explanation
            found = True
            break
    if not found:
        rows.append({"term": term, "explanation": explanation})

    _save_persisted(rows)
    return {"term": term, "explanation": explanation}

