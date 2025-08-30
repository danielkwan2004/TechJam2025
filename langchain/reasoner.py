from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# 1) Define the structured schema
class RegulationRef(BaseModel):
    id: str = Field(..., description="Identifier like 'Article 30.2' or '18 USC 2258A(b)'.")
    title: Optional[str] = Field(None, description="Optional title/name of the regulation or article.")
    snippet: Optional[str] = Field(None, description="Optional short quoted snippet that justifies relevance.")

class ComplianceAnswer(BaseModel):
    geo_specific_required: bool = Field(
        ...,
        description="True if the feature needs different logic by jurisdiction; False otherwise."
    )
    reasoning: str = Field(
        ...,
        description="Clear reasoning using ONLY the provided clauses. If uncertain, say so explicitly."
    )
    related_regulations: List[RegulationRef] = Field(
        default_factory=list,
        description="Optional list of the clauses/regulations you actually used."
    )

# 2) Build the model (note: Gemini uses max_output_tokens, not max_tokens)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_output_tokens=1024,
    timeout=None,
    max_retries=2,
)

# 3) Prompt with explicit constraints
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a legal compliance assistant. Use ONLY the provided clauses. "
     "If a conclusion is uncertain or requires human review, say so explicitly. "
     "Do not invent laws. Return only the fields required by the schema."
    ),
    ("human",
     "Feature description:\n{feature}\n\n"
     "Relevant clauses (verbatim text + IDs):\n{clauses}\n")
])

# 4) Wrap the LLM to enforce structured output
structured_llm = llm.with_structured_output(ComplianceAnswer)

# 5) Compose the chain and invoke it
chain = prompt | structured_llm

if __name__ == "__main__":
    result: ComplianceAnswer = chain.invoke({
        "feature": "Friend suggestions with underage safeguards",
        "clauses": [{
            "clause_id": "13-63-103(4)",
            "clause_text": "Do not collect or use any personal information from posts, content, messages, text, or usage activities other than what is necessary to comply with and verify compliance with law, including parent or guardian name, birth date, and any other required information."
        }]
    })
    print(result.model_dump())
