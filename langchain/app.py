# Streamlit App
import os
import io
import time
from datetime import datetime
from typing import List

import streamlit as st

# LangChain core
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

# LangSmith observability
try:
    from langsmith import Client as LangSmithClient
except Exception:
    LangSmithClient = None

# -----------------------
# UI CONFIG
# -----------------------
st.set_page_config(
    page_title="RAG + LangSmith Demo",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 LLM RAG Demo with LangChain + LangSmith")

# -----------------------
# SIDEBAR: DOC UPLOAD & SETTINGS
# -----------------------
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or .txt files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    st.divider()
    st.header("⚙️ Settings")

    model = st.selectbox(
        "LLM model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_k = st.slider("Top-k retrieved chunks", 1, 8, 3, 1)
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 800, 50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 120, 10)

    st.caption(
        "Tip: Turn on LangSmith by setting env vars:\n"
        "`LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT`."
    )

# -----------------------
# HELPER: LOAD & CHUNK DOCS
# -----------------------
def read_docs_to_texts(files) -> List[str]:
    texts = []
    for f in files:
        name = f.name.lower()
        if name.endswith(".pdf"):
            # Save to temp and load with PyPDFLoader
            with open(f"/tmp/{f.name}", "wb") as out:
                out.write(f.read())
            loader = PyPDFLoader(f"/tmp/" + f.name)
            pages = loader.load()
            for p in pages:
                texts.append(p.page_content)
        elif name.endswith(".txt"):
            content = f.read().decode("utf-8", errors="ignore")
            texts.append(content)
    return texts

def make_vectorstore(texts: List[str], chunk_size: int, chunk_overlap: int) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vs = FAISS.from_texts(chunks, embeddings)
    return vs

# -----------------------
# RAG CHAIN (retrieval + reasoning)
# -----------------------
def build_chain(model_name: str, temperature: float):
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise assistant. Use ONLY the provided context to answer. "
         "If the answer is not in context, say you don't know."),
        ("human",
         "Question:\n{question}\n\n"
         "Context:\n{context}\n\n"
         "Answer with a concise, well-structured explanation.")
    ])

    chain = rag_prompt | llm
    return chain

def run_rag_chain(chain, vectorstore: FAISS, question: str, top_k: int):
    docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    resp = chain.invoke({"question": question, "context": context})
    return resp.content, docs

# -----------------------
# MAIN LAYOUT: QUESTION + RESULTS
# -----------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("🔎 Ask a Question")
    default_q = "Does any uploaded document mention GDPR territorial scope?"
    user_q = st.text_area("Enter your question", value=default_q, height=100)
    run_button = st.button("Run RAG")

with col_right:
    st.subheader("🧠 Chain Output")
    output_placeholder = st.empty()
    ctx_placeholder = st.expander("📚 Retrieved Context (top-k)", expanded=False)

# -----------------------
# RUN PIPELINE
# -----------------------
vectorstore = None
if uploaded_files:
    try:
        texts = read_docs_to_texts(uploaded_files)
        if texts:
            with st.spinner("Building vector store..."):
                vectorstore = make_vectorstore(texts, chunk_size, chunk_overlap)
                st.sidebar.success(f"Indexed {len(texts)} document(s).")
        else:
            st.sidebar.warning("No text extracted. Check your files.")
    except Exception as e:
        st.sidebar.error(f"Error while indexing: {e}")

if run_button:
    if not vectorstore:
        output_placeholder.error("Please upload and index some documents first.")
    elif not user_q.strip():
        output_placeholder.error("Please enter a question.")
    else:
        chain = build_chain(model, temperature)
        with st.spinner("Running RAG chain…"):
            start = time.time()
            answer, docs = run_rag_chain(chain, vectorstore, user_q, top_k)
            elapsed = (time.time() - start) * 1000.0

        # Display answer
        output_placeholder.markdown(f"**Answer (in {elapsed:.0f} ms):**\n\n{answer}")

        # Show retrieved chunks
        with ctx_placeholder:
            for i, d in enumerate(docs, start=1):
                st.markdown(f"**Chunk {i}**")
                st.code(d.page_content[:2000])  # truncation for display
                st.caption(str(d.metadata) if d.metadata else "{}")

# -----------------------
# LANGSMITH OBSERVABILITY PANEL
# -----------------------
st.divider()
st.subheader("📊 LangSmith Observability")

col_a, col_b = st.columns([2, 1])

with col_a:
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        st.success(f"Tracing is ON. Project: `{project}`")

        if LangSmithClient is None:
            st.warning("`langsmith` Python client not installed. `pip install langsmith` to fetch logs here.")
        else:
            try:
                client = LangSmithClient()
                # Fetch recent runs; adjust filters as needed
                runs = client.list_runs(
                    project_name=project,
                    # You can filter by run_type="chain" or "llm" if desired
                    limit=10,
                    # order="desc" may be supported depending on client version
                )
                rows = []
                for r in runs:
                    rows.append({
                        "Name": r.name or r.run_type,
                        "ID": str(r.id),
                        "Status": r.state.value if hasattr(r, "state") else getattr(r, "status", None),
                        "Start": r.start_time.strftime("%Y-%m-%d %H:%M:%S") if r.start_time else "",
                        "Latency (ms)": int((r.end_time - r.start_time).total_seconds() * 1000) if (r.start_time and r.end_time) else "",
                        "Tags": ", ".join(r.tags or []),
                    })
                if rows:
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.info("No recent runs found yet. Execute a chain to generate traces.")
            except Exception as e:
                st.warning(f"Could not fetch LangSmith runs: {e}")
    else:
        st.info(
            "LangSmith tracing is OFF.\n\n"
            "To enable:\n"
            "- Set `LANGCHAIN_TRACING_V2=true`\n"
            "- Set `LANGCHAIN_API_KEY=<your key>`\n"
            "- (Optional) `LANGCHAIN_PROJECT=geo-reg-compliance`\n"
            "Restart the app after setting environment variables."
        )

with col_b:
    # Quick link helper to open the project in LangSmith UI if env vars provided
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://smith.langchain.com")
    project = os.getenv("LANGCHAIN_PROJECT")
    if project:
        st.markdown(
            f"[Open LangSmith Project]({endpoint}/o/~/projects/{project})"
        )
    else:
        st.caption("Set LANGCHAIN_PROJECT to enable a direct project link.")

# Footer
st.caption("Powered by LangChain + OpenAI + FAISS — with LangSmith observability.")
