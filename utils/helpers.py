"""
Utils / Helper Functions

Purpose:
    Utility functions used across the application.
    Keeps common helper code in one place to avoid duplication.

Why this exists:
    - Promotes code reuse
    - Keeps other modules clean and focused
    - Makes testing easier (isolated utilities)
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any


def generate_unique_id() -> str:
    """
    Generate a unique identifier for document chunks.

    Each chunk in ChromaDB needs a unique ID. We use UUID4
    which is practically guaranteed to be unique.

    Returns:
        str: A unique UUID string

    Interview Note:
        UUIDs are commonly used in distributed systems for
        generating unique identifiers without a central coordinator.
    """
    return str(uuid.uuid4())


def get_document_files() -> List[Path]:
    """
    Get list of all PDF files in the documents directory.

    Returns:
        List[Path]: List of paths to PDF files

    Why this function:
        - Abstracts the file discovery logic
        - Makes it easy to support other file types later
        - Centralizes file validation
    """
    from app.config import Config

    documents_dir = Config.DOCUMENTS_DIR
    if not documents_dir.exists():
        return []

    # Find all .pdf files in the documents directory
    pdf_files = list(documents_dir.glob("*.pdf"))
    return pdf_files


def format_sources(retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format retrieved documents into a clean source list for the response.

    Args:
        retrieved_docs: Raw documents from ChromaDB

    Returns:
        List[Dict]: Cleaned source information with chunk, source, page

    Why this function:
        Separates presentation logic from retrieval logic.
        The API response should be clean and consistent.
    """
    sources = []
    for doc in retrieved_docs:
        metadata = doc.get("metadata", {})
        sources.append(
            {
                "chunk": doc.get("document", "")[:200] + "...",  # Truncate for readability
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "Unknown"),
                "score": round(metadata.get("score", 0), 4),
            }
        )
    return sources
