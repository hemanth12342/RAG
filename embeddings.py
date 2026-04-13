"""
embeddings.py - Embedding generation and FAISS vector store management
"""

import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_STORE_DIR = Path("vector_store")
VECTOR_STORE_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and return the HuggingFace embedding model.
    Downloads model on first use and caches locally.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def create_vector_store(chunks: List[Document]) -> FAISS:
    """
    Generate embeddings for document chunks and store in FAISS.
    Saves the index to disk and returns the vector store object.
    """
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save vector store to disk
    vector_store.save_local(str(VECTOR_STORE_DIR))

    return vector_store


def load_vector_store() -> FAISS:
    """
    Load an existing FAISS vector store from disk.
    Returns the vector store object.
    """
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store


def vector_store_exists() -> bool:
    """Check if a saved vector store exists on disk."""
    index_file = VECTOR_STORE_DIR / "index.faiss"
    return index_file.exists()
