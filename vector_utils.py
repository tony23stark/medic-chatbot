import os
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Shared constants and helpers for vector store and embeddings
DB_FAISS_PATH = "vectorstore/db_faiss"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Return a HuggingFaceEmbeddings instance for the given model name."""
    return HuggingFaceEmbeddings(model_name=model_name)


def load_vectorstore(path: str = DB_FAISS_PATH, model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load an existing FAISS vectorstore from `path` using the embedding model.

    Returns the FAISS object or None if the path isn't present.
    """
    if not os.path.exists(path):
        return None
    embedding_model = get_embedding_model(model_name=model_name)
    db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    return db


def build_faiss_from_documents(documents, path: str = DB_FAISS_PATH, model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Builds and saves a FAISS index from the given documents."""
    embedding_model = get_embedding_model(model_name=model_name)
    db = FAISS.from_documents(documents, embedding_model)
    os.makedirs(path, exist_ok=True)
    db.save_local(path)
    return db
