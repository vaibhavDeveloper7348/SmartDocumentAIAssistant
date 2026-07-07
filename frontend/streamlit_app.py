"""
Streamlit Frontend

Purpose:
    User-friendly web interface for the Smart Document AI Assistant.
    Provides a chat-like interface to upload PDFs and ask questions.

Why Streamlit?
    - Fastest way to build AI application UIs in Python
    - Built-in components for file upload, chat, and data display
    - Works with pure Python (no HTML/JS/CSS required)
    - Perfect for prototyping and interview projects

How it works:
    1. Connects to the FastAPI backend (localhost:8000)
    2. Provides a clean chat interface
    3. Displays source documents for every answer

Interview Note:
    Streamlit is very popular in the AI/ML community.
    Discuss why it's used for demos and prototypes:
    - Rapid development
    - Minimal frontend code
    - Built-in state management
"""

import os
import sys
import requests
import streamlit as st

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config

# API base URL - points to the FastAPI backend
# NOTE: Use 127.0.0.1 instead of Config.HOST because HOST may be "0.0.0.0"
# which works for binding but is not connectable from a browser.
API_HOST = "127.0.0.1" if Config.HOST == "0.0.0.0" else Config.HOST
API_BASE_URL = f"http://{API_HOST}:{Config.PORT}/api"

# Page configuration
st.set_page_config(
    page_title="Smart Document AI Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """
    Main entry point for the Streamlit frontend.

    Renders the complete UI including:
    - Sidebar with upload functionality
    - Main area with chat interface
    - Source display for each answer
    """

    # --- Sidebar ---
    with st.sidebar:
        st.title("📄 Smart Document AI")
        st.markdown("---")
        st.markdown("### Upload PDF")
        st.markdown("Upload a PDF document to index it for Q&A.")

        # File uploader widget
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to add to the knowledge base.",
        )

        if uploaded_file is not None:
            upload_pdf(uploaded_file)

        st.markdown("---")
        st.markdown("### Indexed Documents")

        # Display list of indexed documents
        indexed_docs = get_indexed_documents()
        if indexed_docs:
            for doc in indexed_docs:
                st.markdown(f"- 📄 {doc}")
        else:
            st.info("No documents indexed yet. Upload a PDF to get started.")

        st.markdown("---")
        st.markdown("### Actions")

        # Button to clear chat history
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            clear_history()
            st.rerun()

        # Button to check API health
        if st.button("🔍 Check API Status", use_container_width=True):
            health_status = check_health()
            if health_status:
                st.success("API is connected ✅")
            else:
                st.error("API is not reachable ❌")

    # --- Main Chat Area ---
    st.title("💬 Smart Document AI Assistant")
    st.markdown(
        "Ask questions about your uploaded PDF documents. "
        "The assistant will retrieve relevant information and provide "
        "answers with source references."
    )

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                response = ask_question(prompt)

            if response:
                st.markdown(response["answer"])
                if response["sources"]:
                    display_sources(response["sources"])

                # Save to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                    }
                )
            else:
                st.error("Failed to get an answer. Is the API running?")


def upload_pdf(uploaded_file):
    """
    Upload a PDF file to the FastAPI backend for indexing.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Displays progress and result using Streamlit components.
    """
    try:
        with st.spinner("Uploading and indexing PDF..."):
            # Send the file to the FastAPI backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                timeout=120,
            )

        if response.status_code == 200:
            result = response.json()
            st.success(
                f"✅ {result['filename']} indexed successfully! "
                f"({result['chunks_indexed']} chunks)"
            )
        else:
            st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API. Make sure the backend is running.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def ask_question(question: str):
    """
    Send a question to the FastAPI backend and get an answer.

    Args:
        question (str): The user's question

    Returns:
        dict: {"answer": str, "sources": list} or None on failure
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "k": 4},
            timeout=60,
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            return None

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Is the backend running?")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def display_sources(sources: list):
    """
    Display retrieved source chunks in an expandable section.

    Args:
        sources (list): List of source dictionaries with chunk, source, page, score

    Why display sources?
        - Transparency: Users see where the answer came from
        - Trust: Verifiable information improves confidence
        - Education: Helps explain how RAG works visually
    """
    with st.expander("📚 View Retrieved Sources", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}:** `{source['source']}` | Page: {source['page']} | Score: {source['score']:.4f}")
            st.markdown(f"> {source['chunk']}")
            st.markdown("---")


def get_indexed_documents():
    """
    Fetch the list of indexed documents from the backend.

    Returns:
        list: List of document filenames
    """
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception:
        pass
    return []


def clear_history():
    """Clear the chat history on the backend."""
    try:
        requests.delete(f"{API_BASE_URL}/history", timeout=10)
    except Exception:
        pass


def check_health():
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    main()
