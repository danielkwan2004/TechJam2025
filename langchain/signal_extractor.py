import json

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from abbreviation_helper import retrieve_abbreviations

load_dotenv()

def extract_hints():
    out = []

    laws = json.load(open('../knowledge_base/law_cards.json'))
    for law in laws:
        cleaned_law = {
            'id': law['laws'][0]['id'],
            'triggers': law['laws'][0]['triggers'],
            'reason_templates': law['laws'][0]['reason_templates']
        }
        out.append(cleaned_law)

    return out

def extract_signals(doc):
    abbreviations = retrieve_abbreviations(doc)
    formatted_abbrevs = ""
    for abbr in abbreviations:
        formatted_abbrevs += F"{abbr['term']} : {abbr['explanation']} \n"

    hints = extract_hints()

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise compliance signal extraction assistant."
            "Your job is to analyse a Product Requirement Document and output potential compliance issues"
            "Use the law hints provided to detect signals."
            "When you see an abbreviation, resolve it using one from the abbreviations list."
        ),
        (
            "system",
            "Here is the list of abbreviations (format: TERM : EXPLANATION ):\n{abbreviations}"
        ),
        (
            "system",
            "Law hints (each law contains triggers and reason templates, and each \n{hints}"
        ),
        (
            "system",
            "Output instructions:\n"
            "- For each detected signal, include 'signal' and 'reason'. "
            #"- Consolidate detected signals into signals listed under 'triggers'. "
            #"- Consolidate the reasons using one of the law's 'reason_templates', filling in the signal and referencing context from the document. "
            #"- Apply 'negations' to avoid false positives. "
            #"- If a law requires multiple signals ('must_all') or any of several ('must_any'), only flag the law if conditions are met. "
            "- If no signals are found, output a list with one object with 'signal': null and a 'reason' explaining why.\n"
            "- Output all the signals detected as a list with each object containing 'signal' and 'reason'."
        ),
        (
            "human",
            "The PRD:{document}"
        )
    ])

    chain = prompt | llm
    response = chain.invoke({
        "document": doc,
        "abbreviations": formatted_abbrevs,
        "hints": json.dumps(hints),
    })

    return response.content

if __name__ == '__main__':
    userInput = "Introduce limits on video uploads from new accounts. IMT will trigger thresholds based on BB patterns. These limitations are partly for platform safety but without direct legal mapping. "
    print(extract_signals(userInput))
    #print(extract_hints())
