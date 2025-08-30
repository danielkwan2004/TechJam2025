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
    formatted_abbrevs = json.dumps(abbreviations + [{ 'term': 'PRD', 'explanation': 'Project Requirement Document'}])

    hints = extract_hints()

    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise compliance signal extraction assistant. "
            "Your job is to analyse a Product Requirement Document (PRD) and output potential compliance issues. "
            "Look for both explicit keywords and functional hints that imply compliance signals. "
            "Take geographical location into consideration when looking for signals. "
            "When you see an abbreviation in the PRD, look it up exactly in the abbreviations list provided below. "
            "If an abbreviation cannot be found, return an error, stating what is missing and no signals. "
            "If there are unclear terminologies, return an error, stating what is unclear. "
            "If no signals are found, return error as null.\n"
        ),
        (
            "system",
            "Here is the list of abbreviations as a JSON with keys 'term' and 'explanation':\n{abbreviations}\n"
        ),
        (
            "system",
            "Law hints (each law contains triggers and reason templates, and each trigger contains the signal names) \n{hints}\n"
        ),
        (
            "system",
            "Output instructions:\n"
            "- For each detected signal, include 'signal' and 'reason'.\n"
            #"- Consolidate detected signals into signals listed under 'triggers'. "
            #"- Consolidate the reasons using one of the law's 'reason_templates', filling in the signal and referencing context from the document. "
            #"- Apply 'negations' to avoid false positives. "
            #"- If a law requires multiple signals ('must_all') or any of several ('must_any'), only flag the law if conditions are met. "
            "- Output all the signals detected as a list with each object containing 'signal' and 'reason'.\n"
            "- Output format is a JSON with 'error' and 'data'. 'data' is a list of objects with 'signal' and 'reason'.\n"
        ),
        (
            "human",
            "The PRD:\n{document}\n"
        )
    ])

    chain = prompt | llm

    formatted_messages = prompt.format_messages(
        abbreviations=formatted_abbrevs,
        hints=hints,
        document=doc,
    )

    print("==== Rendered Prompt ====")
    for msg in formatted_messages:
        print(f"[{msg.type.upper()}]: {msg.content}")

    response = chain.invoke({
        "document": doc,
        "abbreviations": formatted_abbrevs,
        "hints": json.dumps(hints),
    })

    return response.content.lstrip('```json\n').rstrip('\n```')

if __name__ == '__main__':
    userInput = "We are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH. The feature activates during restricted night hours and logs activity using EchoTrace for auditability. This allows parental control to be enacted without user-facing alerts, operating in ShadowMode during initial rollout."
    output = extract_signals(userInput)
    print("==== Output ====")
    print(output)