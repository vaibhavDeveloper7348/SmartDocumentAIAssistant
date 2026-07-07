"""
Document Processor Service

Purpose:
    Handles PDF document processing - extracting text, splitting into chunks,
    and preparing chunks for embedding generation.

RAG Pipeline Step: 1 (Extract Text) + 2 (Split into Chunks)

Why this is important:
    PDFs contain raw text that must be extracted before we can generate
    embeddings. Once extracted, text must be split into manageable chunks
    because:
    - LLMs have context windows (limited input size)
    - Smaller chunks mean more precise retrieval
    - Each chunk becomes a searchable unit in the vector database

Interview Note:
    Document processing is a critical RAG step. Interviewers often ask:
    - Why do we chunk text? (context limits, precision)
    - What chunk size is optimal? (trade-off between context and precision)
    - Why use overlap? (preserves context across chunk boundaries)
    - What other document types could you support? (CSV, DOCX, HTML)
"""

from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from app.config import Config
from utils.helpers import generate_unique_id


class DocumentProcessor:
    """
    Processes PDF documents for the RAG pipeline.

    Responsibilities:
        1. Load PDF files and extract text page-by-page
        2. Split extracted text into smaller chunks
        3. Track metadata (source file, page number) for each chunk

    Why this is a class (not just functions):
        - Groups related document processing functionality
        - Makes it easy to extend for other document types
        - Easier to test (can mock the class)
    """

    def __init__(self):
        """Initialize the document processor with text splitter configuration."""
        # RecursiveCharacterTextSplitter splits text recursively
        # to keep paragraphs, sentences, and words together as much as possible
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def load_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load a PDF file and return its pages with metadata.

        Args:
            file_path (Path): Path to the PDF file

        Returns:
            List[Dict]: List of pages with 'text' and 'metadata' keys

        How it works:
            1. PyPDFLoader reads the PDF
            2. It returns one Document object per page
            3. Each Document has page_content (text) and metadata

        Why PyPDFLoader?
            LangChain's PyPDFLoader is reliable and extracts
            page numbers automatically, which we need for source tracking.
        """
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        return pages

    def split_documents(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split page documents into smaller chunks.

        Args:
            pages (List[Dict]): Full pages from load_pdf()

        Returns:
            List[Dict]: Smaller chunks ready for embedding

        Why splitting matters:
            A PDF page might contain 2000+ tokens. LLMs have context limits.
            By splitting into smaller chunks (e.g., 500 tokens), we:
            - Stay within LLM context limits
            - Make retrieval more precise (only relevant chunks)
            - Can return multiple relevant chunks per query
        """
        chunks = self.text_splitter.split_documents(pages)
        return chunks

    def process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Complete PDF processing pipeline: load + split.

        This is the main method that other services will call.
        It combines loading and splitting into one step.

        Args:
            file_path (Path): Path to the PDF file

        Returns:
            List[Dict]: Processed chunks with metadata

        RAG Pipeline:
            User Uploads PDF → [Extract Text] → [Split into Chunks] → Ready for Embedding
        """
        pages = self.load_pdf(file_path)
        chunks = self.split_documents(pages)
        return chunks

    def get_chunk_texts_and_metadatas(
        self, chunks: List[Dict[str, Any]]
    ) -> tuple:
        """
        Extract texts and metadata from chunks for ChromaDB insertion.

        ChromaDB needs:
        - documents (list of text strings)
        - metadatas (list of metadata dicts)
        - ids (list of unique IDs)

        Args:
            chunks (List[Dict]): Document chunks from process_pdf()

        Returns:
            tuple: (texts, metadatas, ids)
        """
        texts = []
        metadatas = []
        ids = []

        for chunk in chunks:
            texts.append(chunk.page_content)

            # Preserve source filename and page number for source tracking
            metadatas.append(
                {
                    "source": chunk.metadata.get("source", "Unknown"),
                    "page": chunk.metadata.get("page", 0),
                }
            )
            ids.append(generate_unique_id())

        return texts, metadatas, ids
