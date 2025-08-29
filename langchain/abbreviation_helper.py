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

if __name__ == '__main__':
    abbr = retrieve_abbreviations('Notifications will be tailored by age using ASL, allowing us to throttle or suppress push alerts for minors. EchoTrace will log adjustments, and CDS will verify enforcement across rollout waves.')
    print(abbr)