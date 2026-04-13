"""
app.py - RAG-Based Intelligent Document Q&A Chatbot
Main Streamlit Application
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ----- PATH FIX: ensure project root is in sys.path -----
ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ----- Load environment variables -----
load_dotenv()

# ----- Local imports -----
from utils.loader import save_uploaded_files, load_and_split_documents
from utils.embeddings import create_vector_store, load_vector_store, vector_store_exists
from utils.qa_chain import build_qa_chain, ask_question, format_sources
from utils.export import generate_chat_pdf

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }
  .stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e3a 0%, #12122a 100%);
    border-right: 1px solid rgba(100, 100, 255, 0.15);
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #a0a0ff;
  }

  /* ── Main header ── */
  .app-header {
    background: linear-gradient(90deg, #6c63ff 0%, #48cae4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.2rem;
  }
  .app-subtitle {
    text-align: center;
    color: #8888aa;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
  }

  /* ── Chat bubbles ── */
  .chat-wrapper {
    max-height: 62vh;
    overflow-y: auto;
    padding: 0.5rem 0;
    scrollbar-width: thin;
    scrollbar-color: #6c63ff transparent;
  }
  .msg-row {
    display: flex;
    margin-bottom: 1rem;
    gap: 10px;
  }
  .msg-row.user   { flex-direction: row-reverse; }
  .msg-row.bot    { flex-direction: row; }

  .avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
  }
  .avatar.user { background: linear-gradient(135deg, #6c63ff, #48cae4); }
  .avatar.bot  { background: linear-gradient(135deg, #f093fb, #f5576c); }

  .bubble {
    max-width: 72%;
    padding: 0.85rem 1.1rem;
    border-radius: 18px;
    line-height: 1.6;
    font-size: 0.95rem;
    word-wrap: break-word;
  }
  .bubble.user {
    background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(72,202,228,0.2));
    border: 1px solid rgba(108,99,255,0.4);
    border-top-right-radius: 4px;
    color: #e0e0ff;
  }
  .bubble.bot {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-top-left-radius: 4px;
    color: #d0d0e8;
  }

  /* ── Source cards ── */
  .source-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,100,255,0.2);
    border-left: 3px solid #6c63ff;
    border-radius: 8px;
    padding: 0.55rem 0.9rem;
    margin-top: 0.4rem;
    font-size: 0.82rem;
    color: #9090bb;
  }
  .source-card .src-title {
    font-weight: 600;
    color: #a090ff;
  }
  .source-card .src-snippet {
    margin-top: 0.2rem;
    font-style: italic;
    color: #7070aa;
  }

  /* ── Input row ── */
  .stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(108,99,255,0.4) !important;
    border-radius: 12px !important;
    color: #e0e0f0 !important;
    padding: 0.75rem 1rem !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.25) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(90deg, #6c63ff, #48cae4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s ease !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  .stDownloadButton > button {
    background: linear-gradient(90deg, #f093fb, #f5576c) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
  }

  /* ── Status / info boxes ── */
  .status-box {
    background: rgba(72,202,228,0.08);
    border: 1px solid rgba(72,202,228,0.25);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.88rem;
    color: #70c0d0;
    margin-bottom: 0.8rem;
  }
  .warning-box {
    background: rgba(255,180,0,0.08);
    border: 1px solid rgba(255,180,0,0.3);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.88rem;
    color: #c0a030;
  }

  /* ── Metrics ── */
  [data-testid="metric-container"] {
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 10px;
    padding: 0.6rem;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #6c63ff; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history": [],          # [{role, content, sources}]
        "qa_chain": None,
        "vector_store": None,
        "uploaded_doc_names": [],
        "doc_stats": {},             # {chunks, docs}
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # API Key input
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get your free API key from https://console.groq.com/",
        placeholder="gsk_...",
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    st.divider()

    # Model selection
    model_name = st.selectbox(
        "🤖 LLM Model",
        options=[
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        help="Choose the Groq-hosted LLM for answer generation."
    )

    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Lower = more factual; Higher = more creative"
    )

    k_retrievals = st.slider(
        "🔍 Retrieved Chunks (k)",
        min_value=1, max_value=10, value=4,
        help="Number of document chunks used for each answer"
    )

    chunk_size = st.slider(
        "📏 Chunk Size",
        min_value=200, max_value=2000, value=1000, step=100,
        help="Characters per text chunk during indexing"
    )

    chunk_overlap = st.slider(
        "🔗 Chunk Overlap",
        min_value=0, max_value=500, value=200, step=50,
        help="Overlap between consecutive chunks"
    )

    st.divider()

    # Document upload
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF or TXT files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    process_btn = st.button("⚡ Process Documents", use_container_width=True)

    st.divider()

    # Session stats
    if st.session_state.doc_stats:
        st.markdown("### 📊 Session Stats")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("📄 Docs", st.session_state.doc_stats.get("docs", 0))
        with col_b:
            st.metric("🧩 Chunks", st.session_state.doc_stats.get("chunks", 0))

    # Clear chat
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.session_state.vector_store = None
        st.session_state.uploaded_doc_names = []
        st.session_state.doc_stats = {}
        st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:0.78rem;color:#555577;text-align:center'>"
        "RAG Chatbot · Groq + LangChain<br>Built with ❤️ using Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
#  DOCUMENT PROCESSING
# ─────────────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.sidebar.warning("⚠️ Please upload at least one document first.")
    elif not groq_api_key:
        st.sidebar.error("❌ Please enter your Groq API key.")
    else:
        with st.spinner("🔄 Processing documents... This may take a minute on first run."):
            try:
                # 1. Save files
                file_paths = save_uploaded_files(uploaded_files)

                # 2. Load & split
                chunks = load_and_split_documents(
                    file_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                if not chunks:
                    st.sidebar.error("❌ Could not extract text from the uploaded files.")
                else:
                    # 3. Create vector store
                    vector_store = create_vector_store(chunks)
                    st.session_state.vector_store = vector_store

                    # 4. Build QA chain
                    qa_chain = build_qa_chain(
                        vector_store,
                        model_name=model_name,
                        temperature=temperature,
                        k_retrievals=k_retrievals,
                    )
                    st.session_state.qa_chain = qa_chain
                    st.session_state.uploaded_doc_names = [f.name for f in uploaded_files]
                    st.session_state.doc_stats = {
                        "docs": len(uploaded_files),
                        "chunks": len(chunks),
                    }
                    st.session_state.chat_history = []  # reset chat for new docs

                    st.sidebar.success(
                        f"✅ Processed {len(uploaded_files)} doc(s) → {len(chunks)} chunks!"
                    )
                    st.rerun()

            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")


# ─────────────────────────────────────────────────────────────────
#  MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header">🤖 RAG Document Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload documents → Ask questions → Get cited, context-aware answers</div>',
    unsafe_allow_html=True,
)

# Top toolbar: Download button
top_col1, top_col2, top_col3 = st.columns([1, 2, 1])
with top_col3:
    if st.session_state.chat_history:
        pdf_bytes = generate_chat_pdf(
            st.session_state.chat_history,
            document_names=st.session_state.uploaded_doc_names,
        )
        st.download_button(
            label="📥 Download Chat (PDF)",
            data=pdf_bytes,
            file_name=f"rag_chat_{int(time.time())}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="top_download",
        )

# Status box
if not st.session_state.qa_chain:
    st.markdown(
        '<div class="status-box">📂 <strong>Getting Started:</strong> '
        'Enter your Groq API key in the sidebar → Upload PDF/TXT documents → '
        'Click <em>Process Documents</em> → Start chatting!</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.uploaded_doc_names:
    doc_list = ", ".join(f"<strong>{n}</strong>" for n in st.session_state.uploaded_doc_names)
    st.markdown(
        f'<div class="status-box">✅ Ready! Indexed: {doc_list}</div>',
        unsafe_allow_html=True,
    )

# ── Chat history display ──
chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            st.markdown(
                f'<div class="msg-row user">'
                f'  <div class="avatar user">👤</div>'
                f'  <div class="bubble user">{content}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            # Format content for HTML
            content_html = content.replace("\n", "<br>")
            source_html = ""
            if sources:
                source_html = '<div style="margin-top:0.6rem;">'
                for src in sources:
                    snippet = src.get("snippet", "")[:150]
                    source_html += (
                        f'<div class="source-card">'
                        f'  <span class="src-title">📄 {src.get("source","?")} · Page {src.get("page","N/A")}</span>'
                        f'  <div class="src-snippet">"{snippet}…"</div>'
                        f'</div>'
                    )
                source_html += "</div>"

            st.markdown(
                f'<div class="msg-row bot">'
                f'  <div class="avatar bot">🤖</div>'
                f'  <div class="bubble bot">{content_html}{source_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── Chat input ──
st.markdown("<br>", unsafe_allow_html=True)
input_col, btn_col = st.columns([6, 1])

with input_col:
    user_question = st.text_input(
        "Ask a question",
        placeholder="e.g. What are the main findings of this document?",
        label_visibility="collapsed",
        key="user_input",
    )

with btn_col:
    send_btn = st.button("Send ➤", use_container_width=True)

# ── On send ──
if (send_btn or user_question) and user_question.strip():
    if not st.session_state.qa_chain:
        st.markdown(
            '<div class="warning-box">⚠️ Please process documents first using the sidebar.</div>',
            unsafe_allow_html=True,
        )
    else:
        question = user_question.strip()

        # Append user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "sources": [],
        })

        # Generate answer
        with st.spinner("🤔 Thinking..."):
            try:
                answer, source_docs = ask_question(st.session_state.qa_chain, question)
                sources = format_sources(source_docs)
            except Exception as e:
                answer = f"⚠️ Error generating response: {str(e)}"
                sources = []

        # Append assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

        st.rerun()

# ── Bottom download button ──
if st.session_state.chat_history:
    st.divider()
    dl_col1, dl_col2, dl_col3 = st.columns([2, 2, 2])
    with dl_col2:
        pdf_bytes_bottom = generate_chat_pdf(
            st.session_state.chat_history,
            document_names=st.session_state.uploaded_doc_names,
        )
        st.download_button(
            label="📥 Download Full Conversation as PDF",
            data=pdf_bytes_bottom,
            file_name=f"rag_chat_{int(time.time())}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="bottom_download",
        )
