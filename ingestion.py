import asyncio
import os
from typing import List, Union

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
DOCS_DIR = "docs"
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    """
    Initializes and returns the HuggingFace embeddings model.
    """
    print(f"Loading embeddings model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return embeddings

def get_vectorstore(embeddings):
    """
    Initializes and returns the ChromaDB vector store.
    """
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def _process_and_store_documents(documents: List[Document]):
    """
    Internal helper to split documents, generate embeddings, and store them.
    This logic is shared by both file and string ingestion.
    """
    if not documents:
        print("No documents provided to store.")
        return

    # Split
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(splits)} chunks.")

    # Embed & Store
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    
    print("Adding documents to vector store...")
    vectorstore.add_documents(documents=splits)
    print("Ingestion complete!")

def ingest_from_directory(directory_path: str = DOCS_DIR):
    """
    Ingest all .txt and .pdf files from the specified directory.
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    print(f"--- Ingesting from Directory: {directory_path} ---")
    all_documents = []

    # 1. Load TXT files
    try:
        txt_loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
        txt_docs = txt_loader.load()
        print(f"Loaded {len(txt_docs)} text documents.")
        all_documents.extend(txt_docs)
    except Exception as e:
        print(f"Error loading TXT files: {e}")

    # 2. Load PDF files
    try:
        pdf_loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        print(f"Loaded {len(pdf_docs)} PDF documents.")
        all_documents.extend(pdf_docs)
    except Exception as e:
        print(f"Error loading PDF files: {e}")
    
    print(f"Total documents loaded: {len(all_documents)}")
    _process_and_store_documents(all_documents)

def ingest_from_strings(texts: List[str]):
    """
    Ingest a list of raw text strings.
    """
    if not texts:
        print("No text strings provided.")
        return
        
    print(f"--- Ingesting {len(texts)} String Inputs ---")
    documents = []
    for i, text in enumerate(texts):
        doc = Document(page_content=text, metadata={"source": f"string_input_{i}"})
        documents.append(doc)
    
    _process_and_store_documents(documents)

def get_retriever():
    """
    Returns a retriever object.
    """
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    return vectorstore.as_retriever()

async def main():
    print("Select ingestion mode:")
    print("1. Ingest from 'docs' directory")
    print("2. Ingest sample strings")
    print("3. Test Retrieval")
    
    # Simulating user choice or just running both for demo purposes
    # existing checks to avoid blocking interactive input in this environment
    
    # Demo Mode:
    print("\n--- Running Demo (Separate Calls) ---")
    
    # 1. Ingest Directory
    ingest_from_directory()
    
    # 2. Ingest Strings (Separate call)
    ingest_from_strings([
        "This is a specific string meant to be ingested separately.",
        "It is treated as a distinct ingestion event."
    ])
    
    # 3. Retrieve
    print("\n--- Testing Retrieval ---")
    retriever = get_retriever()
    result = retriever.invoke("What is a specific string?")
    print(f"Query: 'What is a specific string?'\nResult: {result[0].page_content if result else 'No result'}")

if __name__ == "__main__":
    asyncio.run(main())
