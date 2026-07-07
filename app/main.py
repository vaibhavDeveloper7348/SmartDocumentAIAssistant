"""
FastAPI Application Entry Point

Purpose:
    Initializes and configures the FastAPI application.
    This is the main entry point that wires together:
    - API routes
    - Middleware
    - CORS settings
    - Application metadata

Why FastAPI?
    - Modern Python web framework
    - Automatic OpenAPI/Swagger documentation
    - Built-in request validation with Pydantic
    - Async support (faster for I/O operations)
    - Excellent developer experience

Interview Note:
    FastAPI is the most popular modern Python web framework.
    Be prepared to discuss:
    - Why FastAPI over Flask/Django? (performance, auto-docs)
    - CORS and why it's needed for frontend-backend communication
    - How FastAPI generates OpenAPI docs automatically
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory function:
    1. Creates the FastAPI instance with metadata
    2. Configures CORS (allows frontend to access the API)
    3. Registers all API routes

    Returns:
        FastAPI: Configured application instance

    Why a factory function?
        - Makes the app configurable and testable
        - Allows creating multiple app instances if needed
        - Clean separation of configuration and execution
    """
    app = FastAPI(
        title="Smart Document AI Assistant",
        description=(
            "A Retrieval-Augmented Generation (RAG) based AI Assistant "
            "that answers questions from uploaded PDF documents. "
            "Built with FastAPI, LangChain, ChromaDB, and Sentence Transformers."
        ),
        version="1.0.0",
        docs_url="/docs",  # Swagger UI at /docs
        redoc_url="/redoc",  # ReDoc at /redoc
    )

    # Configure CORS (Cross-Origin Resource Sharing)
    # This allows the Streamlit frontend (running on different port)
    # to make requests to this API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins (development only)
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    app.include_router(router)

    return app


# Create the application instance
app = create_application()


@app.on_event("startup")
async def startup_event():
    """
    Execute tasks when the application starts.

    Currently used to ensure the documents directory exists.
    Future: auto-index documents on startup, load models, etc.
    """
    from app.config import Config
    Config.DOCUMENTS_DIR.mkdir(exist_ok=True)
    print(f"Smart Document AI Assistant is ready!")
    print(f"API Documentation: http://localhost:{Config.PORT}/docs")
    print(f"Upload PDFs to: /api/upload")
    print(f"Ask questions at: /api/ask")


@app.get("/")
async def root():
    """
    Root endpoint - provides a welcome message and API navigation.

    Returns basic information about the API and links to documentation.
    """
    return {
        "message": "Welcome to Smart Document AI Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "upload": "POST /api/upload (upload PDF)",
            "ask": "POST /api/ask (ask question)",
            "history": "GET /api/history (chat history)",
            "documents": "GET /api/documents (indexed docs)",
            "health": "GET /api/health",
        },
    }
