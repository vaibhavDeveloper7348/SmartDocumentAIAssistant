"""
Embedding Service

Purpose:
    Generates vector embeddings from text chunks using sentence-transformers.
    Embeddings are the core mathematical representations that enable
    semantic search.

What are Embeddings?
    Embeddings are numerical representations of text that capture meaning.
    Similar texts have similar embedding vectors. Think of them as
    coordinates in a "meaning space" where related concepts are nearby.

Why Sentence Transformers?
    - Free and open-source (no API costs)
    - Runs locally (no data leaves your machine)
    - all-MiniLM-L6-v2 is lightweight (80MB) and fast
    - Produces high-quality 384-dimensional embeddings
    - Good enough for interview projects

Interview Note:
    Embeddings are one of the MOST important RAG concepts to understand.
    Interviewers will ask:
    - What is an embedding? (numerical representation of meaning)
    - How are embeddings generated? (transformer encoder outputs)
    - Why 384 dimensions? (trade-off between precision and compute)
    - What is cosine similarity? (measures angle between vectors)
    - What other embedding models exist? (text-embedding-ada-002, BGE, etc.)
"""

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import Config


class EmbeddingService:
    """
    Generates text embeddings for the RAG pipeline.

    Responsibilities:
        1. Convert text chunks into vector embeddings
        2. Provide embeddings for both indexing and querying

    The same embedding model must be used for:
    - Indexing: Converting document chunks to embeddings for storage
    - Querying: Converting user questions to embeddings for search

    Why must the same model be used?
        Different models produce different embedding spaces. If Model A
        generates document embeddings and Model B generates query embeddings,
        they would exist in different "meaning spaces" and similarity
        search would be meaningless - like comparing meters to feet
        without a conversion factor.
    """

    def __init__(self):
        """
        Initialize the embedding model.

        We use HuggingFaceEmbeddings from LangChain which wraps
        sentence-transformers models. The model downloads automatically
        on first use.

        Model: all-MiniLM-L6-v2
        - Creates 384-dimensional embeddings
        - Trained on a diverse corpus of text pairs
        - Optimized for semantic similarity tasks
        - Fast inference on CPU (~10ms per chunk)
        """
        self.model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},  # Use CPU (works everywhere)
            encode_kwargs={"normalize_embeddings": True},  # Normalize for cosine similarity
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of text strings into embedding vectors.

        Args:
            texts (List[str]): Text chunks to embed

        Returns:
            List[List[float]]: List of embedding vectors (each is 384 floats)

        How it works:
            1. Each text is tokenized into tokens (words/subwords)
            2. Tokenized text passes through the transformer model
            3. The model outputs a hidden state for each token
            4. These states are pooled into a single 384-dim vector
            5. The vector is normalized (length = 1)

        Example:
            >>> embedder = EmbeddingService()
            >>> vectors = embedder.embed_texts(["What is AI?", "Define machine learning"])
            >>> len(vectors[0])
            384
        """
        embeddings = self.model.embed_documents(texts)
        # Convert numpy floats to native Python floats for ChromaDB compatibility
        return [[float(v) for v in emb] for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """
        Convert a single query string (user question) into an embedding.

        Args:
            query (str): The user's question

        Returns:
            List[float]: A single embedding vector (384 dimensions)

        Why separate method from embed_texts?
            - Some models use different encoding for queries vs documents
            - Makes the code more readable (intent is clear)
            - Simplifies future model upgrades
        """
        embedding = self.model.embed_query(query)
        # Convert numpy floats to native Python floats for ChromaDB compatibility
        return [float(v) for v in embedding]

    @property
    def embedding_dimension(self) -> int:
        """
        Return the dimension of embeddings produced by this model.

        For all-MiniLM-L6-v2, this is 384.

        Returns:
            int: Embedding dimension
        """
        return 384
