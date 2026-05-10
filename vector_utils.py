import os
from typing import Optional

from langchain_community.vectorstores import FAISS

# Shared constants and helpers for vector store and embeddings
DB_FAISS_PATH = "vectorstore/db_faiss"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Return a HuggingFaceEmbeddings instance for the given model name."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception as e:
        raise ImportError(
            "HuggingFaceEmbeddings not available. Install sentence-transformers for local use. "
            f"Original error: {e}"
        )
    return HuggingFaceEmbeddings(model_name=model_name)


def load_vectorstore(path: str = DB_FAISS_PATH, model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load an existing FAISS vectorstore from `path` using the embedding model.

    Returns the FAISS object or None if the path isn't present.
    If embeddings module not available, tries to load without it.
    """
    if not os.path.exists(path):
        return None
    
    try:
        # Try with embeddings first (local development)
        if HAS_EMBEDDINGS:
            embedding_model = get_embedding_model(model_name=model_name)
            db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        else:
            # Fallback: load without embeddings (Streamlit Cloud)
            # The FAISS index already has embeddings serialized
            db = FAISS.load_local(path, embeddings=None, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Error loading vectorstore: {str(e)}")
        return None
    return db


def build_faiss_from_documents(documents, path: str = DB_FAISS_PATH, model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Builds and saves a FAISS index from the given documents."""
    embedding_model = get_embedding_model(model_name=model_name)
    db = FAISS.from_documents(documents, embedding_model)
    os.makedirs(path, exist_ok=True)
    db.save_local(path)
    return db
