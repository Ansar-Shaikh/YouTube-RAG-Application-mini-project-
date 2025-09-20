# streamlit_rag_youtube_transcript.py
# High-quality Streamlit UI for RAG pipeline using YouTube transcripts
# Features:
# - Input YouTube video ID or URL
# - Fetch transcript using youtube-transcript-api (handles fetched transcript object)
# - Chunk text with LangChain RecursiveCharacterTextSplitter
# - Create/Load Chroma vector store using Google Generative AI embeddings or OpenAI embeddings
# - Query retriever and generate answer with ChatGoogleGenerativeAI (or fallback to ChatOpenAI)
# - Show retrieved documents, similarity scores, and timestamps
# - Persist vector DB to disk and allow reindexing

import streamlit as st
import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict

# --- Install-time notes (commented) ---
# To use this app ensure required packages are installed in your environment:
# pip install streamlit youtube-transcript-api langchain langchain_google_genai langchain-google-genai langchain-openai chromadb faiss-cpu tiktoken
# Note: package names and imports may vary across langchain versions. Adjust as needed.

# --- Helper functions ---
def extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video id from a full URL or return the id as-is."""
    url_or_id = url_or_id.strip()
    # If input looks like ID (11+ chars alpha-numeric), return directly
    if re.fullmatch(r"[A-Za-z0-9_-]{11,}", url_or_id):
        return url_or_id
    # Try to extract from common URL patterns
    patterns = [r"v=([A-Za-z0-9_-]{11,})", r"youtu\.be/([A-Za-z0-9_-]{11,})", r"/embed/([A-Za-z0-9_-]{11,})"]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    # fallback: return as given
    return url_or_id


def fetch_transcript(video_id: str):
    """Fetch transcript using youtube-transcript-api and return list of dicts."""
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=["en"])  # may raise if disabled
        # fetched is FetchedTranscript with .snippets
        snippets = fetched.snippets
        result = [{"text": s.text, "start": s.start, "duration": s.duration} for s in snippets]
        return result
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return []
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return []


def flatten_transcript(snippets: List[Dict]) -> str:
    """Join transcript snippets into a single string for chunking."""
    return "\n".join(s["text"] for s in snippets)


def create_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])


def make_embeddings(emb_choice: str, model_name: str):
    """Return an embedding object based on user's choice."""
    if emb_choice == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception:
            # fallback to langchain_google_genai package name
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=model_name)
    else:
        # default to OpenAI embeddings
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()


def get_vector_store(embedding, persist_dir: str, collection_name: str):
    """Create or load a Chroma vector store backed by chromadb.
    This function tries a few import paths to be robust across versions.
    """
    try:
        # Preferred import
        from langchain.vectorstores import Chroma
    except Exception:
        try:
            from langchain_community.vectorstores import Chroma
        except Exception as e:
            raise ImportError("Chroma vectorstore not available. Install chromadb and langchain vectorstores.")

    vs = Chroma(embedding_function=embedding, persist_directory=persist_dir, collection_name=collection_name)
    return vs


def index_documents(vector_store, docs):
    # docs: list of langchain Document objects created by create_documents
    vector_store.add_documents(docs)
    try:
        vector_store.persist()
    except Exception:
        pass


def retrieve_docs(vector_store, query: str, k: int = 4):
    return vector_store.similarity_search(query, k=k)


def generate_answer_with_google(llm_model: str, prompt_text: str):
    # Use ChatGoogleGenerativeAI if available
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=llm_model)
        resp = llm.invoke(prompt_text)
        # resp may be a generational object or simple response
        if hasattr(resp, "content"):
            return resp.content
        # older API
        return str(resp)
    except Exception:
        # Fallback to OpenAI Chat model if available
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI()
            out = llm.invoke(prompt_text)
            if hasattr(out, "content"):
                return out.content
            return str(out)
        except Exception as e:
            return f"Could not generate answer - missing model packages. {e}"


# ---------------- Streamlit App ----------------

st.set_page_config(page_title="YouTube RAG - Streamlit UI", layout="wide")
st.title("YouTube RAG — Streamlit UI")

# Sidebar controls
st.sidebar.header("Settings & Keys")
api_key_option = st.sidebar.selectbox("Embedding/LLM provider", ["google", "openai"], index=0)

if api_key_option == "google":
    google_key = st.sidebar.text_input("GOOGLE_API_KEY", value=os.environ.get("GOOGLE_API_KEY", ""), type="password")
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
else:
    openai_key = st.sidebar.text_input("OPENAI_API_KEY", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

collection_name = st.sidebar.text_input("Chroma collection name", value="sample")
persist_dir = st.sidebar.text_input("Chroma persist directory", value="my_chroma_db1")
chunk_size = st.sidebar.slider("Chunk size", min_value=200, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap", min_value=0, max_value=500, value=200, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("Usage:\n1. Paste YouTube URL or ID.\n2. Click Fetch Transcript.\n3. Build/Update Index.\n4. Ask questions.")

# Main area
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Step 1 — Video")
    video_input = st.text_input("YouTube Video URL or ID")
    fetch_btn = st.button("Fetch Transcript")
    if fetch_btn and video_input:
        vid = extract_video_id(video_input)
        with st.spinner("Fetching transcript..."):
            snippets = fetch_transcript(vid)
            if snippets:
                st.success(f"Fetched {len(snippets)} transcript snippets")
                st.session_state["transcript_snippets"] = snippets
                st.session_state["video_id"] = vid
            else:
                st.warning("No transcript retrieved.")

    if "transcript_snippets" in st.session_state:
        if st.checkbox("Show raw transcript snippets", value=False):
            for s in st.session_state["transcript_snippets"][:200]:
                st.markdown(f"- **{s['start']:.2f}s** ({s['duration']:.2f}s): {s['text']}")

    st.subheader("Step 2 — Indexing")
    build_idx_btn = st.button("Build / Update Index")
    if build_idx_btn:
        if "transcript_snippets" not in st.session_state:
            st.error("Please fetch a transcript first.")
        else:
            with st.spinner("Creating chunks..."):
                full_text = flatten_transcript(st.session_state["transcript_snippets"])
                docs = create_chunks(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.session_state["docs"] = docs
                st.success(f"Created {len(docs)} chunks")

            with st.spinner("Initializing embeddings & vector store..."):
                emb_model = "models/gemini-embedding-001" if api_key_option == "google" else "text-embedding-3-small"
                embedding = make_embeddings(api_key_option, emb_model)
                try:
                    vector_store = get_vector_store(embedding, persist_dir, collection_name)
                    # index docs
                    index_documents(vector_store, docs)
                    st.session_state["vector_store"] = vector_store
                    st.success("Indexed documents into Chroma vector store")
                except Exception as e:
                    st.error(f"Error creating vector store: {e}")

with col2:
    st.subheader("Step 3 — Retrieval & Q&A")
    query = st.text_input("Enter your question (ask about video)")
    k = st.slider("Retriever top-k", 1, 10, 4)
    run_query = st.button("Run Query")

    if run_query:
        if "vector_store" not in st.session_state:
            st.error("Vector store not found. Build the index first.")
        else:
            with st.spinner("Retrieving relevant chunks..."):
                vs = st.session_state["vector_store"]
                docs = retrieve_docs(vs, query, k=k)
                st.session_state["retrieved_docs"] = docs
                st.success(f"Retrieved {len(docs)} documents")

            if docs:
                st.markdown("**Retrieved snippets (top results):**")
                for i, d in enumerate(docs):
                    # Each d is a Document with page_content
                    st.markdown(f"**Result {i+1}:** {d.page_content[:300]}...")

                # Compose context and prompt
                context_text = "\n\n".join(d.page_content for d in docs)
                prompt = (
                    "You are a helpful assistant. Answer ONLY from the provided transcript context. "
                    "If the context is insufficient, just say you don't know.\n\n"
                    f"Context:\n{context_text}\n\nQuestion: {query}"
                )

                with st.spinner("Generating answer via LLM..."):
                    llm_model = "gemini-1.5-flash" if api_key_option == "google" else "gpt-4"
                    answer = generate_answer_with_google(llm_model, prompt)
                    st.markdown("### Answer")
                    st.write(answer)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by: Your Streamlit RAG App")

# End of app
