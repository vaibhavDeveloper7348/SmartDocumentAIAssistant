"""
RAG Pipeline Service

Purpose:
    Orchestrates the complete RAG pipeline bringing together
    document processing, embeddings, vector search, and LLM generation.

This is the CORE service that ties everything together.
It represents the complete "Retrieve-Augment-Generate" flow.

RAG Pipeline:
    1. Receive user question
    2. Convert question to embedding vector
    3. Search ChromaDB for similar document chunks
    4. Retrieve top-k most relevant chunks
    5. Build prompt with context + question
    6. Send prompt to LLM
    7. Return answer + source chunks

Interview Note:
    This is THE MOST IMPORTANT file in the project.
    Interviewers will ask you to walk through the entire RAG flow.
    Understanding every step here demonstrates mastery.
"""

from typing import List, Dict, Any, Optional
from app.config import Config
from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from database.chroma_client import get_chroma_client, get_or_create_collection


class RAGService:
    """
    Core RAG Pipeline Orchestrator.

    Coordinates all RAG components to answer questions based on
    uploaded documents. This class represents the complete
    "Retrieve → Augment → Generate" workflow.

    Responsibilities:
        1. Index new documents (chunk → embed → store)
        2. Answer questions (retrieve → augment → generate)
        3. Track chat history
    """

    def __init__(self):
        """Initialize all RAG components and ChromaDB connection."""
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.chroma_client = get_chroma_client()
        self.collection = get_or_create_collection(self.chroma_client)
        self.chat_history = []  # Simple in-memory chat history

    def index_document(self, file_path) -> int:
        """
        Index a PDF document into the vector database.

        RAG Pipeline Step:
            PDF → Extract Text → Split into Chunks → Generate Embeddings → Store in ChromaDB

        Args:
            file_path: Path to the PDF file

        Returns:
            int: Number of chunks indexed

        How it works:
            1. Process the PDF into chunks
            2. Extract text and metadata from chunks
            3. Generate embeddings for each chunk
            4. Store everything in ChromaDB

        Why index documents?
            Before we can answer questions, we need the document content
            to be searchable. Indexing converts human-readable text into
            machine-searchable vector embeddings.
        """
        # Step 1: Extract text and split into chunks
        chunks = self.document_processor.process_pdf(file_path)

        # Step 2: Prepare data for ChromaDB
        texts, metadatas, ids = (
            self.document_processor.get_chunk_texts_and_metadatas(chunks)
        )

        # Step 3: Generate embeddings for each text chunk
        # We use our EmbeddingService (sentence-transformers) to convert
        # text chunks into 384-dimensional vector embeddings.
        # These embeddings are passed directly to ChromaDB since we did
        # NOT set an embedding function on the collection.
        embeddings = self.embedding_service.embed_texts(texts)

        # Step 4: Store everything in ChromaDB
        # We pass pre-computed embeddings, text documents, metadata, and IDs.
        # ChromaDB stores these in its HNSW index for fast similarity search.
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        return len(texts)

    def answer_question(
        self, question: str, k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using the complete RAG pipeline.

        RAG Pipeline:
            Question → Query Embedding → Similarity Search → Retrieve Chunks
            → Create Prompt → LLM → Answer + Sources

        Args:
            question (str): The user's question
            k (int): Number of chunks to retrieve (default from Config)

        Returns:
            Dict: {
                "answer": str,
                "sources": List[Dict],
                "question": str
            }

        This is the function interviewers will ask about the most.
        Walk through each step carefully.

        How RAG works:
        1. Convert question to embedding (turn words into numbers)
        2. Search for similar vectors in ChromaDB (find relevant content)
        3. Retrieve top-k matching chunks (get the best matches)
        4. Combine chunks + question into a prompt (provide context)
        5. Ask LLM to generate answer based only on that context
        6. Return answer and show which chunks were used
        """
        k = k or Config.RETRIEVAL_K

        # Step 1: Convert question to embedding vector
        # The question must be embedded using the SAME model that
        # was used to embed the documents. Otherwise the vector
        # spaces won't match.
        query_embedding = self.embedding_service.embed_query(question)

        # Step 2: Search ChromaDB for similar chunks
        # ChromaDB compares the query embedding against all stored
        # document embeddings using cosine similarity and returns
        # the k most similar chunks.
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Step 3: Extract results
        retrieved_chunks = results["documents"][0] if results["documents"] else []
        retrieved_metadatas = (
            results["metadatas"][0] if results["metadatas"] else []
        )
        distances = results["distances"][0] if results["distances"] else []

        # Attach scores to metadata
        for i, metadata in enumerate(retrieved_metadatas):
            if i < len(distances):
                metadata["score"] = 1 - distances[i]  # Convert distance to similarity

        # Step 4: If no relevant chunks found
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents to answer this question.",
                "sources": [],
                "question": question,
            }

        # Step 5: Build the prompt with context
        # The prompt gives the LLM instructions and provides
        # the retrieved chunks as context. The LLM is told to
        # answer ONLY based on this context (reduces hallucinations).
        context = self._build_context(retrieved_chunks, retrieved_metadatas)
        prompt = self._build_prompt(context, question)

        # Step 6: Generate answer using LLM
        answer = self.llm_service.generate_response(prompt)

        # Step 7: Prepare sources for display
        sources = self._prepare_sources(retrieved_chunks, retrieved_metadatas)

        # Step 8: Save to chat history
        self.chat_history.append(
            {"question": question, "answer": answer, "sources": sources}
        )

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
        }

    def _build_context(
        self,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> str:
        """
        Combine retrieved chunks into a single context string.

        Args:
            chunks (List[str]): Retrieved text chunks
            metadatas (List[Dict]): Metadata for each chunk

        Returns:
            str: Formatted context with source information

        Why format the context?
            Clean formatting helps the LLM understand which part
            of the context is most relevant. Including source info
            helps with traceability.
        """
        context_parts = []
        for i, (chunk, metadata) in enumerate(zip(chunks, metadatas), 1):
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")
            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{chunk}\n"
            )
        return "\n---\n".join(context_parts)

    def _build_prompt(self, context: str, question: str) -> str:
        """
        Build the complete prompt for the LLM.

        The prompt structure is critical for RAG quality.
        It must:
        1. Clearly instruct the LLM to use only the provided context
        2. Include the retrieved document chunks
        3. End with the user's question

        Args:
            context (str): Retrieved document chunks
            question (str): User's question

        Returns:
            str: Complete prompt for LLM

        Interview Note:
            Prompt engineering is a key skill. Discuss:
            - Why we tell the LLM to use "only the provided context"
              (prevents hallucinations)
            - What happens if the context doesn't contain the answer
              (should say "I don't know" instead of making up info)
            - How prompt structure affects answer quality
        """
        prompt = f"""You are a helpful document assistant. Answer the question based ONLY on the provided context. 
If the context does not contain enough information to answer the question, say "I cannot find sufficient information in the documents to answer this question."

Context:
{context}

Question: {question}

Answer:"""
        return prompt

    def _prepare_sources(
        self,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Format retrieved chunks into user-friendly source display.

        Args:
            chunks (List[str]): Retrieved text chunks
            metadatas (List[Dict]): Metadata for each chunk

        Returns:
            List[Dict]: Formatted source information
        """
        import os
        sources = []
        for chunk, metadata in zip(chunks, metadatas):
            raw_source = metadata.get("source", "Unknown")
            # Extract just the filename for cleaner display
            source_name = os.path.basename(raw_source) if raw_source != "Unknown" else "Unknown"
            sources.append(
                {
                    "chunk": chunk[:300] + "..." if len(chunk) > 300 else chunk,
                    "source": source_name,
                    "page": metadata.get("page", "Unknown"),
                    "score": round(metadata.get("score", 0), 4),
                }
            )
        return sources

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Return the conversation history.

        Returns:
            List[Dict]: List of Q&A pairs with sources

        Why chat history?
            - Users can review previous answers
            - Demonstrates understanding of conversation management
            - Useful for debugging and demonstration
        """
        return self.chat_history

    def clear_chat_history(self):
        """Clear all conversation history."""
        self.chat_history = []

    def get_indexed_documents(self) -> List[str]:
        """
        Get list of all documents currently indexed in ChromaDB.

        Returns:
            List[str]: Unique source filenames

        Why this matters:
            Users need to know what documents are already indexed
            to avoid re-uploading or to know what they can ask about.
        """
        # Get all unique source filenames from metadata
        all_data = self.collection.get()
        sources = set()
        if all_data and all_data.get("metadatas"):
            for metadata in all_data["metadatas"]:
                source = metadata.get("source", "")
                if source:
                    # Extract just the filename from the path
                    import os
                    sources.add(os.path.basename(source))
        return list(sources)
