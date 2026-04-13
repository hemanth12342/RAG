"""
loader.py - Document loading and text splitting utilities
"""

import os
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Directory to temporarily store uploaded files
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def save_uploaded_files(uploaded_files) -> List[str]:
    """
    Save Streamlit uploaded file objects to the data/ directory.
    Returns a list of saved file paths.
    """
    saved_paths = []

    # Clear existing data directory for fresh session
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(file_path))

    return saved_paths


def load_and_split_documents(file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Load documents from file paths (PDF or TXT) and split into chunks.
    Returns a list of LangChain Document objects.
    """
    all_docs = []

    for path in file_paths:
        ext = Path(path).suffix.lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext == ".txt":
                loader = TextLoader(path, encoding="utf-8")
            else:
                print(f"Unsupported file type: {path}")
                continue

            documents = loader.load()

            # Add source metadata
            for doc in documents:
                doc.metadata["source_file"] = Path(path).name

            all_docs.extend(documents)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    if not all_docs:
        return []

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(all_docs)
    return chunks
