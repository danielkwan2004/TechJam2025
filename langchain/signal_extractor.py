from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from abbreviation_helper import retrieve_abbreviations

load_dotenv()

def extract_signals(doc):
    abbreviations = retrieve_abbreviations(doc)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise assistant. Your role is to generate signals from a Product Requirements Document."
            "You are also given a list of possible abbreviations, pick the one that is most likely."
        ),
        (
            "system",
            "Here is the list of abbreviations:\n{abbreviations}"
        ),
        (
            "human",
            "{document}"
        )
    ])

    chain = prompt | llm
    response = chain.invoke({
        "document": doc,
        "abbreviations": abbreviations,
    })

    return response.content

print(extract_signals("Hello, CDS"))


