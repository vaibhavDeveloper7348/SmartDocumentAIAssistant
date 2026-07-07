"""
ChromaDB Client Module

Purpose:
    Manages the connection to ChromaDB - our vector database.
    Provides functions to create collections, store embeddings,
    and perform similarity searches.

What is ChromaDB?
    ChromaDB is an open-source vector database. It stores text chunks
    as mathematical vectors (embeddings) and allows us to find
    similar chunks by comparing vectors.

Why ChromaDB?
    - Free and open-source
    - Simple to set up (no external server needed)
    - Designed specifically for embedding storage and search
    - Integrates well with LangChain
    - Persists data to disk (survives restarts)

Interview Note:
    Vector databases are a core RAG concept. Be prepared to explain:
    - How embeddings are stored as vectors
    - How similarity search works mathematically
    - Why ChromaDB vs. Pinecone/Weaviate/Qdrant
"""

import chromadb
from chromadb.config import Settings
from app.config import Config


def get_chroma_client():
    """
    Create and return a ChromaDB client instance.

    This function creates a persistent ChromaDB client that stores
    data on disk. Persistent storage means data survives application
    restarts - we don't need to re-index documents every time.

    Returns:
        chromadb.PersistentClient: The ChromaDB client

    Why persistent client?
        ChromaDB offers two modes:
        - EphemeralClient: in-memory, data lost on restart
        - PersistentClient: saves to disk, data persists

        We use PersistentClient because re-indexing PDFs every
        time the server starts would be slow and impractical.
    """
    client = chromadb.PersistentClient(
        path=Config.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def get_or_create_collection(client, name=None):
    """
    Get an existing collection or create a new one if it doesn't exist.

    A collection in ChromaDB is like a table in SQL databases.
    It groups related embeddings together. In our case, all document
    chunks are stored in one collection.

    Args:
        client: ChromaDB client instance
        name (str): Name of the collection

    Returns:
        chromadb.Collection: The collection object

    Why name parameter?
        Allows flexibility to use different collections for
        different document sets in the future.
    """
    collection_name = name or Config.CHROMA_COLLECTION_NAME

    # get_or_create_collection is atomic - if collection exists,
    # it returns it; otherwise creates a new one
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )
    return collection


def delete_collection(client, name=None):
    """
    Delete a collection and all its data.

    Used when we want to clear all indexed documents and start fresh.

    Args:
        client: ChromaDB client instance
        name (str): Name of the collection to delete
    """
    collection_name = name or Config.CHROMA_COLLECTION_NAME
    try:
        client.delete_collection(collection_name)
    except ValueError:
        # Collection doesn't exist - that's fine
        pass
