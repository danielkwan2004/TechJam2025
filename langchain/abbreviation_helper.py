import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)
# Helper function that takes in a query, then outputs a list of abbreviations used.
# Duplicates will be added.

default_doc = "Hello World."

abbreviations = [
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


def retrieve_all_abbreviations(doc=default_doc):
    return abbreviations


def retrieve_abbreviations(doc=default_doc):
    out = []

    for abbreviation in abbreviations:
        if abbreviation['term'] in doc:
            out.append({k: v for k, v in abbreviation.items() if k != "id"})

    return out


def add_abbreviation(term: str, explanation: str):
    response = (
        supabase.table("abbreviations")
        .insert({"term": term, "explanation": explanation})
        .execute()
    )


if __name__ == '__main__':
    abbr = retrieve_abbreviations(
        'Enable users to reshare stories from others, with auto-expiry after 48 hours. This feature logs resharing attempts with EchoTrace and stores activity under BB.')
    print(abbr)
