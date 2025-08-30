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


def retrieve_all_abbreviations(doc=default_doc):
    abbreviations = (
        supabase.table("abbreviations")
        .select("*")
        .execute()
    ).data

    return abbreviations


def retrieve_abbreviations(doc=default_doc):
    abbreviations = (
        supabase.table("abbreviations")
        .select("*")
        .execute()
    ).data
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
        'Introduce limits on video uploads from new accounts. IMT will trigger thresholds based on BB patterns. These limitations are partly for platform safety but without direct legal mapping.')
    print(abbr)
