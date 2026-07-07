"""
Smart Document AI Assistant - Main Entry Point

Purpose:
    This is the entry point for running the application.
    It starts both the FastAPI backend and provides instructions
    for running the Streamlit frontend.

Usage:
    # Start the FastAPI backend
    python main.py

    # In another terminal, start the Streamlit frontend
    streamlit run frontend/streamlit_app.py

Why two commands?
    - FastAPI serves the REST API (backend)
    - Streamlit serves the web UI (frontend)
    - This separation follows the microservices pattern
    - Each can be developed and deployed independently

Interview Note:
    Discuss why we separate backend and frontend:
    - Independent scaling
    - Technology flexibility
    - Clear separation of concerns
"""

import uvicorn
from app.config import Config


def main():
    """
    Start the FastAPI backend server.

    The server runs on the configured HOST and PORT.
    Default: http://0.0.0.0:8000

    API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    """
    print("=" * 60)
    print("  Smart Document AI Assistant")
    print("=" * 60)
    print(f"\n  Starting backend server...")
    print(f"  API:      http://localhost:{Config.PORT}")
    print(f"  Docs:     http://localhost:{Config.PORT}/docs")
    print(f"  Redoc:    http://localhost:{Config.PORT}/redoc")
    print(f"\n  To start the frontend, run:")
    print(f"  streamlit run frontend/streamlit_app.py")
    print("\n" + "=" * 60)

    # Start the FastAPI application with Uvicorn
    uvicorn.run(
        "app.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True,  # Auto-restart on code changes (development only)
    )


if __name__ == "__main__":
    main()
