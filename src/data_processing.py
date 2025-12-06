# src/data_processing.py

from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_chunk_documents(source_directory: str) -> List[Document]:
    """
    Loads .txt documents from a directory, then splits them into smaller chunks.

    This function performs the first two steps of our RAG pipeline:
    1. Data Ingestion (Loading)
    2. Document Processing (Chunking)

    Args:
        source_directory: The path to the directory containing the .txt files.

    Returns:
        A list of LangChain Document objects, where each object is a chunk
        of text from the original files.
    """
    # 1. Loading Stage
    # We use DirectoryLoader to efficiently load all .txt files from the specified path.
    # - glob="**/*.txt" ensures we only pick up text files recursively.
    # - loader_cls=TextLoader tells it how to read these files.
    # - use_multithreading=True can speed up loading for many files.
    print(f"Loading documents from '{source_directory}'...")
    loader = DirectoryLoader(
        source_directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},  # Ensure files are read as UTF-8
        show_progress=True,
        use_multithreading=True
    )

    documents = loader.load()

    # If no documents are found, return an empty list to prevent errors downstream.
    if not documents:
        print("No documents found.")
        return []

    print(f"Loaded {len(documents)} document(s).")

    # 2. Chunking Stage
    # We use a recursive character splitter which is generally recommended.
    # It tries to split on paragraphs, then sentences, etc.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The target size for each chunk in characters.
        chunk_overlap=200,  # The number of characters to overlap between chunks.
        length_function=len,
    )

    print("Chunking documents...")
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunked_documents)} chunks.")

    return chunked_documents
