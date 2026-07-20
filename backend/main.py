import os
import uuid
import pandas as pd
from io import BytesIO
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports (fixed from original langchain_classic which doesn't exist)
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import tempfile

# Load environment variables from .env file (for local dev)
load_dotenv()

# --- App Setup ---
app = FastAPI(
    title="RAG Document Analyst API",
    description="Upload documents and chat with them using Groq LLM + FAISS vector search.",
    version="1.0.0",
)

# CORS — allow all origins so the Vercel frontend can call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory session store: session_id -> FAISS vectorstore ---
sessions: Dict[str, FAISS] = {}

# --- Shared embeddings model (loaded once at startup to save time) ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --- Request/Response Models ---
class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    message: str
    filename: str
    chunks: int


# --- Helper: load documents from uploaded bytes ---
def load_documents_from_bytes(file_bytes: bytes, filename: str) -> list:
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        # Write to a temp file because PyPDFLoader needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        finally:
            os.unlink(tmp_path)
        return docs

    elif ext == "docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
        finally:
            os.unlink(tmp_path)
        return docs

    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(BytesIO(file_bytes))
        return [Document(page_content=df.to_string(), metadata={"source": filename})]

    else:
        raise ValueError(f"Unsupported file type: .{ext}. Upload PDF, DOCX, or Excel.")


# --- Routes ---

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Document Analyst API is running."}


@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(sessions)}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a document file (PDF, DOCX, Excel), chunk it, embed it,
    and store the FAISS index in memory. Returns a session_id.
    """
    allowed_extensions = {"pdf", "docx", "xlsx", "xls"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Please upload PDF, DOCX, or Excel.",
        )

    try:
        file_bytes = await file.read()
        docs = load_documents_from_bytes(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

    # Chunk the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise HTTPException(status_code=400, detail="Document appears to be empty or could not be parsed.")

    # Build FAISS vector store
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")

    # Store in session
    session_id = str(uuid.uuid4())
    sessions[session_id] = vectorstore

    return UploadResponse(
        session_id=session_id,
        message=f"✅ Successfully processed '{file.filename}'! You can now ask questions.",
        filename=file.filename,
        chunks=len(chunks),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Given a session_id and a question, retrieve relevant document chunks
    and generate an answer using the Groq LLM.
    """
    vectorstore = sessions.get(req.session_id)
    if vectorstore is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a document first.",
        )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured on the server.",
        )

    try:
        llm = ChatGroq(
            model_name="openai/gpt-oss-120b",
            temperature=0,
            api_key=groq_api_key,
        )

        prompt = ChatPromptTemplate.from_template("""
You are a helpful document analyst. Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I cannot find the answer in the provided document."
Be concise and accurate.

Context:
{context}

Question: {question}

Answer:""")

        # Modern LCEL chain — compatible with langchain 1.x
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(req.question)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return ChatResponse(answer=answer, session_id=req.session_id)


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Explicitly free a session's memory."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session cleared."}
    raise HTTPException(status_code=404, detail="Session not found.")
