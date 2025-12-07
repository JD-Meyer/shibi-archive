# src/query_data.py

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


def create_qa_chain(persist_directory: str, embedding_model: Embeddings):
    """
    Creates a Question-Answering chain using a vector store and a language model.

    Args:
        persist_directory: The directory where the vector store is persisted.
        embedding_model: The embedding model used for the vector store.

    Returns:
        A retrieval chain.
    """
    # 1. Load the vector store and create a retriever
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever()

    # 2. Initialize the LLM
    # Make sure you have Ollama running with the 'llama3' model pulled.
    # You can run it with: ollama run llama3
    llm = ChatOllama(model="llama3")

    # 3. Create a prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    # 4. Create the "stuff" documents chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 5. Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain
