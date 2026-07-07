"""
Configuration Module

Purpose:
    Centralized configuration management for the entire application.
    Loads environment variables, provides defaults, and exposes settings
    as a single Config object.

Why this exists:
    - Keeps configuration in one place (easy to maintain)
    - Uses environment variables (industry standard for deployment)
    - Provides sensible defaults (works out-of-the-box)
    - Follows the principle of separation of concerns

Interview Note:
    Configuration management is a common interview topic.
    Discuss why .env files are used, how environment variables work,
    and why we avoid hardcoding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Application Configuration.

    Loads settings from environment variables with sensible defaults.
    Access settings via Config.SETTING_NAME anywhere in the application.

    Example:
        >>> Config.EMBEDDING_MODEL_NAME
        'all-MiniLM-L6-v2'
    """

    # --- Project Paths ---
    # Base directory of the project
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Directory to store uploaded documents
    DOCUMENTS_DIR = BASE_DIR / "documents"
    DOCUMENTS_DIR.mkdir(exist_ok=True)

    # --- Embedding Model ---
    # Sentence-transformer model for converting text to embeddings
    # all-MiniLM-L6-v2 is lightweight, fast, and produces 384-dim embeddings
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
    )

    # --- LLM Configuration ---
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    # Ollama (local LLM)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama2")

    # HuggingFace (free, local)
    USE_HUGGINGFACE = True  # Default to free local model

    # --- ChromaDB Settings ---
    CHROMA_PERSIST_DIR = os.getenv(
        "CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db")
    )
    # Collection name in ChromaDB
    CHROMA_COLLECTION_NAME = "document_chunks"

    # --- Document Processing ---
    # Size of each text chunk in characters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    # Overlap between consecutive chunks (preserves context)
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # --- Retrieval Settings ---
    # Number of most relevant chunks to retrieve
    RETRIEVAL_K = 4

    # --- Server Settings ---
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
