import json
import csv

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from oauthlib.uri_validate import userinfo

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
    formatted_abbrevs = json.dumps(abbreviations + [{'term': 'PRD', 'explanation': 'Project Requirement Document'}])

    hints = extract_hints()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    # llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise compliance signal extraction assistant. "
            "Your job is to analyse a Product Requirement Document (PRD) and output potential compliance issues. "
            "Look for both explicit keywords and functional hints that imply compliance signals. "
            "Take geographical location into consideration when looking for signals; only flag signals relevant to the regions affected. "
            "When you see an abbreviation in the PRD, look it up exactly in the abbreviations list provided below. "
            "If an abbreviation in the document cannot be found, return an error, stating what is missing and no signals. "
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
            "- Only flag laws relevant to the PRD's geographical scope.\n"
            "- Include signals even if they are implied by functionality, not just literal keyword matches.\n"
            "- Output all detected signals as a list with each object containing 'signal' and 'reason'.\n"
            "- Output format is JSON with 'error' and 'data'. 'data' is a list of objects with 'signal' and 'reason'.\n"
            "- Optional: include a 'confidence' score (0.0-1.0) reflecting how certain the detection is."
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

    '''
    print("==== Rendered Prompt ====")
    for msg in formatted_messages:
        print(f"[{msg.type.upper()}]: {msg.content}")
    '''

    response = chain.invoke({
        "document": doc,
        "abbreviations": formatted_abbrevs,
        "hints": json.dumps(hints),
    })
    return json.loads(response.content.lstrip('```json\n').rstrip('\n```'))


if __name__ == '__main__':
    with open('../knowledge_base/demo_dataset/test_cases.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print("==== Question ====")
            print(f"{row['feature_name'] + ".\n" + row['feature_description']}")

            userInput = row['feature_description']
            output = extract_signals(userInput)
            print("==== Output ====")
            print(json.dumps(output, indent=2))
