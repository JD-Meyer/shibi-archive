# tests/test_data_processing.py

import os
from pathlib import Path
import shutil

# We will create this function in our src directory later.
# PyCharm will highlight it in red because it doesn't exist yet.
from src.data_processing import load_and_chunk_documents, create_vector_store


def test_load_and_chunk_single_txt_file():
    """
    Tests that a single .txt file can be loaded and chunked.
    """
    # 1. Setup: Define the path to our test data
    # We use Path for OS-agnostic path handling.
    current_dir = Path(__file__).parent
    test_data_path = current_dir.parent / "data"

    # 2. Execution: Call the function we are testing
    # This function is expected to return a list of "Document" objects (from LangChain)
    documents = load_and_chunk_documents(str(test_data_path))

    # 3. Assertion: Check if the result is what we expect
    assert documents is not None, "The function should not return None."
    assert len(documents) > 0, "The list of documents should not be empty."

    # Check the content of the first chunk to ensure it's loaded correctly
    assert "first line of the chat log" in documents[0].page_content

    # Check that the source metadata is correctly assigned
    assert "dummy_chat.txt" in documents[0].metadata["source"]

def test_create_vector_store():
    """
    Tests the creation and persistence of a vector store from document chunks.
    """
    # 1. SETUP: Create some dummy document chunks and define a test DB path.
    # In a real scenario, this would come from our load_and_chunk_documents function.
    from langchain_core.documents import Document
    test_chunks = [
        Document(page_content="This is the first test chunk."),
        Document(page_content="This is the second test chunk.")
    ]

    db_path = Path(__file__).parent.parent / "test_chroma_db"

    # Clean up any database from a previous test run
    if db_path.exists():
        shutil.rmtree(db_path)

    # 2. EXECUTION: Call the function we are testing.
    # PyCharm will highlight 'create_vector_store' in red because it doesn't exist yet.
    create_vector_store(test_chunks, str(db_path))

    # 3. ASSERTION: Check if the database directory and its files were created.
    assert db_path.exists(), "Database directory should be created."
    # ChromaDB creates a chroma.sqlite3 file for its database.
    assert (db_path / "chroma.sqlite3").is_file(), "ChromaDB database file should exist."
    # It also creates a directory for the collection data.
    assert len(list(db_path.iterdir())) > 1, "Database directory should contain more than just the sqlite file."

    # 4. TEARDOWN: Clean up the created database directory.
    shutil.rmtree(db_path)

