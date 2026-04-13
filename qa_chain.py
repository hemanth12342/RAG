"""
qa_chain.py - RAG retrieval chain with conversational memory using Groq LLaMA3
"""

import os
from typing import List, Dict, Any, Tuple

from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import FAISS


def get_llm(model_name: str = "llama3-8b-8192", temperature: float = 0.2) -> ChatGroq:
    """
    Initialize Groq LLM with the specified model.
    Reads GROQ_API_KEY from environment.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in your .env file or environment variables."
        )

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=2048,
    )
    return llm


def build_qa_chain(
    vector_store: FAISS,
    model_name: str = "llama3-8b-8192",
    temperature: float = 0.2,
    k_retrievals: int = 4,
) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain from the vector store.
    Supports multi-turn conversation with memory.
    """

    llm = get_llm(model_name=model_name, temperature=temperature)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_retrievals}
    )

    # System prompt to guide the LLM
    system_prompt = """You are an intelligent document assistant. 
Your task is to answer user questions accurately based ONLY on the context provided from the uploaded documents.

Guidelines:
- Be precise and concise
- If the answer is not found in the context, clearly say: "I couldn't find relevant information in the uploaded documents."
- Always cite which document or section the information came from when possible
- For follow-up questions, use prior conversation history to maintain coherence
- Format responses clearly with bullet points or numbered lists when appropriate

Context from documents:
{context}

Chat History:
{chat_history}
"""

    human_prompt = "{question}"

    # Memory store (session-level)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )

    return qa_chain


def ask_question(qa_chain: ConversationalRetrievalChain, question: str) -> Tuple[str, List[Any]]:
    """
    Run a question through the QA chain.
    Returns (answer_text, source_documents).
    """
    result = qa_chain.invoke({"question": question})
    answer = result.get("answer", "No answer generated.")
    source_docs = result.get("source_documents", [])
    return answer, source_docs


def format_sources(source_docs: List[Any]) -> List[Dict[str, str]]:
    """
    Format source documents into a list of dicts with file name, page, and snippet.
    """
    seen = set()
    formatted = []

    for doc in source_docs:
        meta = doc.metadata
        source_file = meta.get("source_file", meta.get("source", "Unknown"))
        page = meta.get("page", None)
        snippet = doc.page_content[:300].strip()

        key = (source_file, page, snippet[:50])
        if key in seen:
            continue
        seen.add(key)

        entry = {
            "source": source_file,
            "page": str(page + 1) if page is not None else "N/A",
            "snippet": snippet
        }
        formatted.append(entry)

    return formatted
