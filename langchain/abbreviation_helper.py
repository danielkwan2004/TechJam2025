import json

# Helper function that takes in a query, then outputs a list of abbreviations used.
# Duplicates will be added.

filepath = '../knowledge_base/abbreviations.json'
default_doc = "Hello World."

def retrieve_abbreviations(doc=default_doc):
    abbreviations = json.load(open(filepath))
    out = []

    for abbreviation in abbreviations:
        if abbreviation['term'] in doc:
            out.append(abbreviation)

    return out

if __name__ == '__main__':
    abbr = retrieve_abbreviations()
    print(abbr)