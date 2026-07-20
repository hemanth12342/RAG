import gradio as gr
import os
import pandas as pd
from dotenv import load_dotenv

# LangChain Modules
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load API Key securely from a .env file or server environment variables
load_dotenv()  # Loads GROQ_API_KEY from .env file (never hardcode secrets in source)

# Global variable to store our database
global_vectorstore = None

# --- TASK 1 & 2: Process the File ---
def process_document(file_path):
    global global_vectorstore
    if file_path is None:
        return "Please upload a file first."

    try:
        file_ext = file_path.split(".")[-1].lower()

        # Extract text based on file type
        if file_ext == "pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file_ext == "docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)
            docs = [Document(page_content=df.to_string())]
        else:
            return "Unsupported file type. Please upload PDF, DOCX, or Excel."

        # Chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # Create Embeddings and Vector DB
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        global_vectorstore = FAISS.from_documents(chunks, embeddings)

        filename = os.path.basename(file_path)
        return f"✅ Successfully processed {filename}! You can now ask questions."

    except Exception as e:
        return f"❌ Error processing file: {str(e)}"

# --- TASKS 3 & 4: Answer Generation ---
def answer_question(message, history):
    global global_vectorstore

    if global_vectorstore is None:
        return "Please upload and process a document before asking questions!"

    try:
        # Initialize Groq LLM
        llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)

        # Set up the prompt instructions
        prompt_template = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        If the answer is not in the context, say "I cannot find the answer in the provided document."

        Context: {context}
        Question: {input}
        """)

        # Build the RAG chain
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = global_vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get the answer
        response = retrieval_chain.invoke({"input": message})
        return response["answer"]

    except Exception as e:
        return f"Error connecting to AI: {str(e)}. Please check your Groq API Key."

# --- MAIN GRADIO UI ---
with gr.Blocks() as demo:
    gr.Markdown(" 📄 AI Document Analyst")
    gr.Markdown("Upload your file and chat with your data instantly using Groq's high-speed AI.")

    with gr.Row():
        # Left Column for Uploads
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload & Process")
            file_input = gr.File(label="Upload PDF, DOCX, or Excel file")
            process_btn = gr.Button("Process Document", variant="primary")
            status_text = gr.Textbox(label="System Status", interactive=False)

        # Right Column for Chat
        with gr.Column(scale=2):
            gr.Markdown("### 2. Chat with Document")
            chat_interface = gr.ChatInterface(
                fn=answer_question,
                chatbot=gr.Chatbot(height=500)
            )

    # Connect the button to the processing function
    process_btn.click(fn=process_document, inputs=file_input, outputs=status_text)

# Launch the app without fixed ports to avoid OSError
if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())