import json
import csv

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from typing import List, Optional
from abbreviation_helper import retrieve_abbreviations

from pydantic import BaseModel, Field

load_dotenv()


# =========================
# Signal Extraction Pipeline
# =========================

# --------- Structured Output Schemas (Pydantic) ---------

class Detection(BaseModel):
    law: str = Field(..., description="Precise law id from hints (e.g., 'EU Digital Services Act').")
    signal: str = Field(..., description="Signal name as detected.")
    reason: str = Field(..., description="Short explanation quoting PRD text and why it maps to the law/signal.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in detection, 0.0–1.0.")


class LLMOutput(BaseModel):
    error: Optional[str] = Field(None, description="Null if no error; otherwise a short error message.")
    data: List[Detection] = Field(default_factory=list, description="Detected compliance signals.")


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

    # --- LLM: OpenAI with Structured Output (recommended for strong JSON guarantees) ---
    base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=31082025)
    llm = base_llm.with_structured_output(LLMOutput)  # << Guarantees schema

    # If you prefer Gemini, you can try:
    # gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    # llm = gemini.with_structured_output(LLMOutput)  # Works in recent LC versions; else fall back to strict JSON prompting.

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise compliance signal extraction assistant. "
            "Your job is to analyse a Product Requirement Document (PRD) and output potential compliance issues. "
            "Look for both explicit keywords and functional hints that imply compliance signals. "
            "Take geographical location into consideration; flag signals for all regions specified by the PRD. "
            "If multiple regions are relevant, include all applicable laws. "
            "If no region is specified, assume all regions are affected. "
            "When you see an abbreviation in the PRD, look it up exactly in the abbreviations list provided below. "
            "If an abbreviation in the document cannot be found, return an error, stating what is missing and no signals. "
            "If no signals are found, return error as null.\n"
        ),
        (
            "system",
            "Here is the list of abbreviations as a JSON array. Each item has keys 'term' and 'explanation'."
            "Parse the JSON carefully. Match abbreviations in the PRD exactly (case-insensitive, whole words). :\n{abbreviations}\n"
        ),
        (
            "system",
            "Law hints (each law contains triggers and reason templates, and each trigger contains the signal names) \n{hints}\n"
        ),
        (
            "system",
            "Output instructions:\n"
            "- For each detected signal, include 'signal', 'reason' and 'law'.\n"
            "- The 'law' should be precisely the id of the law from the hint."
            "- Only flag laws relevant to the PRD's geographical scope as defined earlier.\n"
            "- Include signals even if they are implied by functionality, not just literal keyword matches.\n"
            "- 'reason' should quote the relevant portion of the PRD, and give a short explanation.\n"
            "- Include a 'confidence' score (0.0-1.0) reflecting how certain the detection is."
        ),
        (
            "system",
            "Output schema (STRICT):\n"
            "Return an object with:\n"
            "- error: null OR a short string\n"
            "- data: array of objects, each with fields {{ law, signal, reason, confidence }}."
        ),
        (
            "human",
            "The PRD:\n{document}\n"
        )
    ])

    chain = prompt | llm

    '''
    print("==== Rendered Prompt ====")
    for msg in formatted_messages:
        print(f"[{msg.type.upper()}]: {msg.content}")
    '''

    response: LLMOutput = chain.invoke({
        "document": doc,
        "abbreviations": formatted_abbrevs,
        "hints": json.dumps(hints),
    })

    return response.model_dump()


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
