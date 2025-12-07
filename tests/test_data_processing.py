# tests/test_data_processing.py

import os
from pathlib import Path
import shutil
import pytest  # <-- Import pytest
import gc # <-- garbage collector

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.data_processing import load_and_chunk_documents, create_vector_store
# Add this new import for our query function
from src.query_data import create_qa_chain



@pytest.fixture(scope="function")
def test_db_path() -> Path:
    """
    Pytest fixture to create and clean up a test database directory.
    'scope="function"' means this runs once for each test function.
    """
    # SETUP: Create a path for the test database
    db_path = Path(__file__).parent.parent / "test_chroma_db"

    # Clean up any old directory before the test runs
    if db_path.exists():
        shutil.rmtree(db_path, ignore_errors=True)
    # The 'yield' keyword passes the db_path to the test function
    yield db_path

    # TEARDOWN: This code runs after the test function completes
    # It will reliably clean up the directory
    # TEARDOWN: Forcing garbage collection before rmtree
    gc.collect()
    gc.collect()
    if db_path.exists():
        shutil.rmtree(db_path, ignore_errors=True)

def test_load_and_chunk_single_txt_file():
    """
    Tests that a single .txt file can be loaded and chunked.
    """
    # 1. Setup
    current_dir = Path(__file__).parent
    test_data_path = current_dir.parent / "data"

    # 2. Execution
    documents = load_and_chunk_documents(str(test_data_path))

    # 3. Assertion
    assert documents is not None
    assert len(documents) > 0
    assert "first line of the chat log" in documents[0].page_content
    # A more robust check for the source path
    assert "dummy_chat.txt" in str(documents[0].metadata["source"])


def test_create_vector_store_persists(test_db_path: Path):  # <-- The fixture is passed in here
    """
    Tests the creation and persistence of a vector store using a fixture for cleanup.
    """
    # 1. SETUP
    test_chunks = [
        Document(page_content="This is the first test chunk."),
        Document(page_content="This is the second test chunk.")
    ]
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 2. EXECUTION
    vector_store = create_vector_store(
        chunks=test_chunks,
        embedding_model=embedding_model,
        persist_directory=str(test_db_path)
    )

    # 3. ASSERTION
    assert test_db_path.exists()
    assert (test_db_path / "chroma.sqlite3").is_file()

    # To be absolutely sure resources are released, we can clear the object
    # before the test function ends and teardown begins.
    vector_store = None

    # Verify by loading the store back from disk
    loaded_store = Chroma(
        persist_directory=str(test_db_path),
        embedding_function=embedding_model
    )
    assert loaded_store._collection.count() == 2

    # 4. EXPLICIT CLEANUP WITHIN THE TEST
    # Set objects to None to signal they can be collected
    vector_store = None
    loaded_store = None
    # Force the garbage collector to run NOW, before the fixture's teardown phase
    gc.collect()
    gc.collect()  # Calling twice can help with complex object cycles



def test_query_vector_store(test_db_path: Path):
    """
    Tests that we can create a QA chain and query the vector store.
    """
    # 1. ARRANGE: Create a vector store with specific, queryable content.
    test_documents = [
        Document(page_content="The first line is about a happy cat."),
        Document(page_content="The second line is about a sad dog.")
    ]
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = create_vector_store(
        chunks=test_documents,
        embedding_model=embedding_model,
        persist_directory=str(test_db_path)
    )

    # 2. ACT: Create the QA chain and ask a question.
    qa_chain = create_qa_chain(
        persist_directory=str(test_db_path),
        embedding_model=embedding_model
    )
    question = "What is the first line about?"
    # Use .invoke() and expect the 'input' key
    result = qa_chain.invoke({"input": question})

    # 3. ASSERT: Check if the answer is correct in the 'answer' key.
    assert "cat" in result["answer"].lower()

    # 4. TEARDOWN
    vector_store = None
    gc.collect()
    gc.collect()