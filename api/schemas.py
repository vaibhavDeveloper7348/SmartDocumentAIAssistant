"""
API Schemas Module

Purpose:
    Defines Pydantic models for API request/response validation.
    Pydantic ensures that data sent to our API has the correct
    types and structure.

Why Pydantic?
    - Automatic request validation (no manual type checking)
    - Clear error messages for invalid requests
    - IDE autocompletion for response models
    - Works seamlessly with FastAPI's OpenAPI documentation
    - Industry standard for Python APIs

Interview Note:
    Pydantic models demonstrate understanding of:
    - Data validation
    - Type hints (modern Python)
    - API design best practices
    - OpenAPI/Swagger documentation
"""

from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """
    Response model for document upload endpoint.

    Returned after successfully uploading and indexing a PDF document.
    """
    message: str = Field(description="Success message")
    filename: str = Field(description="Name of the uploaded file")
    chunks_indexed: int = Field(description="Number of text chunks indexed")
    status: str = Field(default="success", description="Status of the operation")


class SourceInfo(BaseModel):
    """
    Information about a retrieved source chunk.

    Displayed alongside the answer so users can verify
    where the answer came from (transparency is a key RAG benefit).
    """
    chunk: str = Field(description="The text content of the retrieved chunk")
    source: str = Field(description="Source document filename")
    page: Union[int, str] = Field(description="Page number in the source document")
    score: float = Field(description="Similarity score (higher = more relevant)")


class AnswerResponse(BaseModel):
    """
    Response model for the question-answering endpoint.

    Contains the generated answer and supporting evidence (sources).
    """
    answer: str = Field(description="Generated answer from the LLM")
    sources: List[SourceInfo] = Field(
        description="Retrieved document chunks used to generate the answer"
    )
    question: str = Field(description="The original question asked by the user")


class QuestionRequest(BaseModel):
    """
    Request model for asking a question.

    Validates that the user provides a non-empty question.
    """
    question: str = Field(..., min_length=1, description="The question to answer")
    k: Optional[int] = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (1-20)",
    )


class ChatHistoryResponse(BaseModel):
    """
    Response model for chat history.

    Returns all previous Q&A pairs in the session.
    """
    history: List[Dict[str, Any]] = Field(
        description="List of previous Q&A exchanges"
    )
    total: int = Field(description="Total number of exchanges")


class DocumentListResponse(BaseModel):
    """
    Response model for listing indexed documents.
    """
    documents: List[str] = Field(
        description="List of indexed document filenames"
    )
    total: int = Field(description="Total number of indexed documents")


class ErrorResponse(BaseModel):
    """
    Standard error response for API errors.
    """
    error: str = Field(description="Error message")
    status: str = Field(default="error", description="Status indicator")
