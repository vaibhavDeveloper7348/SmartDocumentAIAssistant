"""
API Routes Module

Purpose:
    Defines all REST API endpoints for the Smart Document AI Assistant.
    Uses FastAPI for clean, modern Python API development.

API Design:
    RESTful endpoints with clear request/response models.
    Each endpoint has a single responsibility.

Interview Note:
    Interviewers will examine API design decisions:
    - Why POST vs GET for different operations
    - How file uploads work in FastAPI
    - Error handling patterns
    - Response structure consistency
"""

import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends

from api.schemas import (
    UploadResponse,
    AnswerResponse,
    QuestionRequest,
    ChatHistoryResponse,
    DocumentListResponse,
    ErrorResponse,
    SourceInfo,
)
from services.rag_service import RAGService
from app.config import Config

# Create the API router
router = APIRouter(prefix="/api", tags=["Smart Document AI Assistant"])

# Global RAG service instance
# In production, we'd use dependency injection
rag_service = RAGService()


def get_rag_service() -> RAGService:
    """
    Dependency injection for the RAG service.

    FastAPI dependencies make the code testable. We can
    easily swap the service with a mock for testing.

    Returns:
        RAGService: The RAG pipeline service instance
    """
    return rag_service


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index a PDF document",
)
async def upload_document(
    file: UploadFile = File(...),
    service: RAGService = Depends(get_rag_service),
):
    """
    Upload a PDF document and index it into the vector database.

    Request:
        - POST /api/upload
        - Body: multipart/form-data with PDF file

    Flow:
        1. Validate file type (must be PDF)
        2. Save file to documents/ directory
        3. Process PDF (extract text, split into chunks)
        4. Generate embeddings and store in ChromaDB
        5. Return success with number of chunks indexed

    Response:
        - message: Success confirmation
        - filename: Original filename
        - chunks_indexed: Number of chunks stored in ChromaDB
        - status: "success"

    Errors:
        - 400: File is not a PDF
        - 500: Internal processing error

    Interview Note:
        File upload endpoints are common in real-world applications.
        Discuss validation, security considerations (file type checking),
        and how this integrates with the RAG pipeline.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:
        # Save the uploaded file to the documents directory
        file_path = Config.DOCUMENTS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Index the document into the vector database
        chunks_indexed = service.index_document(file_path)

        return UploadResponse(
            message=f"Document '{file.filename}' uploaded and indexed successfully.",
            filename=file.filename,
            chunks_indexed=chunks_indexed,
            status="success",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}",
        )


@router.post(
    "/ask",
    response_model=AnswerResponse,
    summary="Ask a question about your documents",
)
async def ask_question(
    request: QuestionRequest,
    service: RAGService = Depends(get_rag_service),
):
    """
    Ask a question and get an answer based on indexed documents.

    Request:
        - POST /api/ask
        - Body: { "question": "Your question here", "k": 4 }

    Flow:
        1. Convert question to embedding vector
        2. Search ChromaDB for similar chunks (k results)
        3. Build prompt with retrieved context
        4. LLM generates answer based on context
        5. Return answer + source references

    Response:
        - answer: LLM-generated answer
        - sources: List of retrieved chunks with metadata
        - question: Original question

    Errors:
        - 422: Validation error (e.g., empty question)
        - 500: LLM or processing error

    Interview Note:
        This is THE core endpoint of the entire application.
        Explain every step in detail:
        - How embeddings enable semantic search
        - Why we retrieve chunks before answering
        - How the LLM uses context to generate answers
        - Why we show sources (transparency, trust)
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        result = service.answer_question(
            question=request.question,
            k=request.k,
        )

        return AnswerResponse(
            answer=result["answer"],
            sources=[
                SourceInfo(**source) for source in result["sources"]
            ],
            question=result["question"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}",
        )


@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    summary="Get chat history",
)
async def get_chat_history(
    service: RAGService = Depends(get_rag_service),
):
    """
    Retrieve the conversation history for the current session.

    Request:
        - GET /api/history

    Response:
        - history: List of all Q&A exchanges
        - total: Number of exchanges

    Why chat history?
        - Users can review previous answers
        - Demonstrates state management
        - Useful for debugging
    """
    history = service.get_chat_history()
    return ChatHistoryResponse(history=history, total=len(history))


@router.delete(
    "/history",
    summary="Clear chat history",
)
async def clear_chat_history(
    service: RAGService = Depends(get_rag_service),
):
    """
    Clear the current session's chat history.

    Request:
        - DELETE /api/history

    Response:
        - message: Confirmation
        - status: "success"
    """
    service.clear_chat_history()
    return {"message": "Chat history cleared.", "status": "success"}


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List indexed documents",
)
async def list_documents(
    service: RAGService = Depends(get_rag_service),
):
    """
    Get a list of all documents currently indexed in the vector database.

    Request:
        - GET /api/documents

    Response:
        - documents: List of filenames
        - total: Count of documents

    Why list documents?
        - Users can see what's available to query
        - Avoid duplicate uploads
        - Transparency about the knowledge base
    """
    documents = service.get_indexed_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.get(
    "/health",
    summary="Health check endpoint",
)
async def health_check():
    """
    Simple health check to verify the API is running.

    Request:
        - GET /api/health

    Response:
        - status: "healthy"
        - service: "Smart Document AI Assistant"

    Why health checks?
        - Essential for monitoring in production
        - Quick way to verify the server is running
        - Used by load balancers and orchestration tools
    """
    return {
        "status": "healthy",
        "service": "Smart Document AI Assistant",
    }
