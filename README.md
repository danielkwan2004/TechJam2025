# TechJam2025

TikTok TechJam 2025 Hackathon

## Problem Statement

## Features

### Tools Used

- **LLM Application Framework:** [LangChain](https://www.langchain.com/) and [LangSmith](https://www.langchain.com/langsmith)
- **Embedding Model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **UI Framework:** [Streamlit](https://streamlit.io/)
- **Databases:** [Pinecone](https://www.pinecone.io/)

### APIs Used

- [OpenAI](https://openai.com/)

### How to Use

1. Open a new Terminal at the root folder.
2. Setup virtual environment: py -3 -m venv .venv
3. Activate it: .\.venv\Scripts\Activate.ps1
4. Install dependencies: pip install -r requirements.txt

5. Run "cd langchain; docker compose up -d;", this launches the local Pinecone vector DB
6. Run the Streamlit app locally with "streamlit run app.py"
