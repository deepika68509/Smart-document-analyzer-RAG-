from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_documents(pdf_path: str | Path) -> List[Document]:
    """Load a PDF file into a list of LangChain Document objects."""
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)



def get_embeddings(
    provider: str,
) -> OpenAIEmbeddings | GoogleGenerativeAIEmbeddings | HuggingFaceEmbeddings:
    """Create an embeddings client based on the chosen provider."""
    normalized = provider.strip().lower()
    if normalized == "local":
        # Local embeddings run on your machine (no API key required).
        model = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)
    if normalized == "gemini":
        # Uses GOOGLE_API_KEY from the environment.
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=model)

    # Default to OpenAI embeddings and use OPENAI_API_KEY from the environment.
    openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=openai_model)


def get_llm(provider: str) -> ChatOpenAI | ChatGoogleGenerativeAI:
    """Create an LLM client based on the chosen provider."""
    normalized = provider.strip().lower()
    if normalized == "gemini":
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=0.2)

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0.2)


def build_prompt(context: str, question: str) -> str:
    """Create a prompt that forces the model to use the provided context."""
    return (
        "Answer the question based only on the context below. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer(provider: str, prompt: str) -> str:
    """Send the prompt to an LLM provider and return the answer text."""
    llm = get_llm(provider)
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, str):
        return content

    # Some providers return a list of content blocks (e.g., [{"type": "text", ...}]).
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        if text_parts:
            return "\n".join(text_parts)

    return str(content)


def build_vector_store(
    documents: List[Document],
    provider: str,
    chunk_size: int,
    chunk_overlap: int,
) -> FAISS:
    """Chunk the documents and build a FAISS vector store."""
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = get_embeddings(provider)
    return FAISS.from_documents(chunks, embeddings)


def retrieve_similar_chunks(vector_store: FAISS, query: str, top_k: int) -> List[Document]:
    """Run a similarity search and return the most relevant chunks."""
    return vector_store.similarity_search(query, k=top_k)


def ensure_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None
    if "embed_provider" not in st.session_state:
        st.session_state.embed_provider = None


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="Smart Document Analyzer", page_icon="📄", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

        :root {
            --ink: #0f172a;
            --muted: #475569;
            --accent: #d97706;
            --accent-2: #0ea5e9;
            --paper: #fffaf3;
            --glass: rgba(255, 255, 255, 0.7);
            --stroke: rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background: radial-gradient(1200px 600px at 20% 0%, #fff1e6 0%, transparent 60%),
                        radial-gradient(900px 500px at 90% 10%, #e0f2fe 0%, transparent 55%),
                        linear-gradient(180deg, #fffaf3 0%, #f8fafc 100%);
            color: var(--ink);
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif;
            letter-spacing: -0.02em;
        }

        .hero {
            background: var(--glass);
            border: 1px solid var(--stroke);
            padding: 28px 32px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(6px);
            animation: fadeInUp 0.8s ease-out;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent);
            background: rgba(217, 119, 6, 0.12);
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: 600;
        }

        .hero-title {
            font-size: 40px;
            margin: 12px 0 8px 0;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 16px;
            max-width: 680px;
        }

        .stChatMessage {
            background: var(--glass);
            border: 1px solid var(--stroke);
            border-radius: 16px;
            padding: 8px 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.75);
            border-right: 1px solid var(--stroke);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] .stCaption {
            color: var(--muted) !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: var(--ink) !important;
        }

        label, .stMarkdown, .stText, .stCaption {
            color: var(--ink) !important;
        }

        .stSlider > div[data-baseweb="slider"] span {
            color: var(--ink) !important;
        }

        [data-baseweb="select"] *,
        [data-baseweb="select"] span,
        [data-baseweb="select"] svg {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }

        [data-baseweb="select"] div {
            background: #f8fafc !important;
        }

        [data-testid="stSelectbox"] div[role="combobox"] {
            color: var(--ink) !important;
            background: #f8fafc !important;
        }

        [data-testid="stNumberInput"] input {
            background: #f8fafc !important;
        }

        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] svg {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }

        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] input {
            color: var(--ink) !important;
        }

        [data-testid="stChatInput"] textarea {
            color: var(--ink) !important;
        }

        [data-testid="stChatInput"] {
            background: #f8fafc !important;
            border-radius: 16px;
        }

        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] textarea::placeholder {
            color: #f8fafc !important;
            caret-color: #f8fafc !important;
        }

        [data-testid="stChatInput"] div[role="textbox"] {
            background: #f8fafc !important;
        }

        [data-testid="stChatInput"] button,
        [data-testid="stChatInput"] svg {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }

        [data-testid="stFileUploader"] {
            background: var(--glass);
            border: 1px dashed rgba(15, 23, 42, 0.2);
            border-radius: 16px;
            padding: 16px;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">Smart Document Analyzer</div>
            <h1 class="hero-title">RAG-powered document intelligence</h1>
            <p class="hero-subtitle">
                Upload a PDF, ask questions in plain language, and get grounded answers
                from your document. Clean, fast, and designed for focus.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ensure_session_state()

    # Force embeddings to local to avoid any paid/billing APIs.
    embed_provider = "local"

    llm_providers = ["openai", "gemini"]
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    has_google_key = bool(os.getenv("GOOGLE_API_KEY"))
    if not has_openai_key and not has_google_key:
        default_llm = "openai"
    else:
        default_llm = "gemini" if has_google_key and not has_openai_key else "openai"
    default_llm_index = llm_providers.index(default_llm)

    with st.sidebar:
        st.subheader("Settings")
        st.caption("Embeddings: local (no API key)")
        llm_provider = st.selectbox("LLM", llm_providers, index=default_llm_index)
        top_k = st.slider("Top K results", min_value=3, max_value=5, value=3)
        chunk_size = st.number_input("Chunk size", min_value=300, max_value=2000, value=1000)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=200)

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        # Rebuild the vector store if the file or provider changes.
        if (
            st.session_state.doc_name != uploaded_file.name
            or st.session_state.embed_provider != embed_provider
        ):
            with st.spinner("Processing PDF and building embeddings..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                documents = load_pdf_documents(tmp_path)
                vector_store = build_vector_store(
                    documents,
                    provider=embed_provider,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )

                st.session_state.vector_store = vector_store
                st.session_state.doc_name = uploaded_file.name
                st.session_state.embed_provider = embed_provider

        st.success("PDF processed and embeddings ready.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input and response handling
    user_query = st.chat_input("Ask a question about the document...")
    if user_query:
        if not st.session_state.vector_store:
            st.warning("Please upload and process a PDF first.")
            return

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Searching and generating answer..."):
            results = retrieve_similar_chunks(
                st.session_state.vector_store,
                query=user_query,
                top_k=top_k,
            )

            context = "\n\n".join(doc.page_content for doc in results)
            prompt = build_prompt(context=context, question=user_query)
            answer = generate_answer(llm_provider, prompt)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()
