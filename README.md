# Smart Document AI Assistant

A **Retrieval-Augmented Generation (RAG)** based AI Assistant that answers questions from uploaded PDF documents.

Built with **FastAPI**, **LangChain**, **ChromaDB**, and **Sentence Transformers**.

---

## Features

- **Upload PDF Documents** - Upload any PDF for indexing
- **Automatic Text Extraction** - Extracts text content from PDF pages
- **Intelligent Chunking** - Splits text into optimal chunks for retrieval
- **Vector Embeddings** - Converts text to searchable vector representations
- **ChromaDB Storage** - Stores embeddings in a local vector database
- **Semantic Search** - Finds relevant chunks using cosine similarity
- **RAG-based Answers** - Generates answers using retrieved context + LLM
- **Source Transparency** - Displays which chunks and pages were used
- **Chat History** - Maintains conversation context
- **Interactive UI** - Streamlit-powered chat interface

---

## Project Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                        User                              │
└──────────┬────────────────────────────────┬──────────────┘
           │ Upload PDF                     │ Ask Question
           ▼                                ▼
┌──────────────────┐              ┌──────────────────┐
│   Streamlit UI    │              │   Streamlit UI    │
│  (Frontend)       │              │  (Frontend)       │
└────────┬─────────┘              └────────┬─────────┘
         │ HTTP REST                       │ HTTP REST
         ▼                                 ▼
┌──────────────────────────────────────────────────────┐
│                    FastAPI (Backend)                   │
├──────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────┐  │
│  │Document  │  │Embedding │  │ChromaDB│  │ LLM  │  │
│  │Processor │  │Service   │  │Client  │  │Svc   │  │
│  └──────────┘  └──────────┘  └────────┘  └──────┘  │
└──────────────────────────────────────────────────────┘
```

### RAG Pipeline Flow

```text
User Uploads PDF
        │
        ▼
┌─────────────────┐
│ Extract Text     │  PyPDF extracts page-by-page content
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Split into       │  RecursiveCharacterTextSplitter chunks text
│ Chunks           │  (1000 chars per chunk, 200 overlap)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate         │  Sentence Transformers creates 384-dim vectors
│ Embeddings       │  (all-MiniLM-L6-v2)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Store in         │  ChromaDB persists vectors with metadata
│ ChromaDB         │  (source file, page number)
└─────────────────┘
         ▲
         │
User asks Question
         │
         ▼
┌─────────────────┐
│ Generate Query   │  Same embedding model converts question to vector
│ Embedding        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Similarity       │  Cosine similarity search in ChromaDB
│ Search           │  Returns top-k most similar chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieve Top-k   │  Best matching chunks with metadata
│ Chunks           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create Prompt    │  Context + Question → Structured prompt
│ (Augment)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Generates    │  LLM answers based ONLY on provided context
│ Answer           │  (reduces hallucinations)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Return Answer    │  Answer + Source chunks + Page numbers
│ + Sources        │
└─────────────────┘
```

---

## Folder Structure

```
SmartDocumentAIAssistant/
│
├── app/                          # Application configuration
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration
│   └── main.py                   # FastAPI app initialization
│
├── api/                          # API layer
│   ├── __init__.py
│   ├── routes.py                 # REST API endpoints
│   └── schemas.py                # Pydantic request/response models
│
├── services/                     # Business logic layer
│   ├── __init__.py
│   ├── document_processor.py     # PDF text extraction & chunking
│   ├── embedding_service.py      # Embedding generation
│   ├── llm_service.py            # LLM integration (OpenAI/Ollama/HF)
│   └── rag_service.py            # Core RAG pipeline orchestrator
│
├── database/                     # Database layer
│   ├── __init__.py
│   └── chroma_client.py          # ChromaDB connection management
│
├── frontend/                     # User interface
│   ├── __init__.py
│   └── streamlit_app.py          # Streamlit chat interface
│
├── utils/                        # Utility functions
│   ├── __init__.py
│   └── helpers.py                # Common helper functions
│
├── config/                       # Config directory (future use)
│
├── documents/                    # Uploaded PDF storage
│
├── chroma_db/                    # ChromaDB persistent storage (auto-created)
│
├── main.py                       # Application entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── README.md                     # This file
└── INTERVIEW_GUIDE.md            # Interview preparation guide
```

### Folder Explanations

| Folder | Purpose |
|--------|---------|
| `app/` | App initialization, configuration loading |
| `api/` | REST API routes with Pydantic validation |
| `services/` | Core business logic (RAG pipeline, embeddings, LLM) |
| `database/` | Vector database connection and management |
| `frontend/` | Streamlit web interface |
| `utils/` | Cross-cutting helper functions |
| `documents/` | Uploaded PDF files storage |
| `chroma_db/` | Persistent vector storage (auto-generated) |

---

## Installation

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Step 1: Clone or Create the Project

```bash
cd D:\SmartDocumentAIAssistant
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy the example environment file
copy .env.example .env

# (Optional) Edit .env to add your OpenAI API key or configure settings
```

---

## Running the Application

### Start the Backend (FastAPI)

```bash
# Terminal 1: Start the API server
python main.py
```

The API will start at `http://localhost:8000`

### Start the Frontend (Streamlit)

```bash
# Terminal 2: Start the web interface (in a new terminal)
streamlit run frontend/streamlit_app.py
```

The UI will open at `http://localhost:8501`

### Add Sample Documents

Place any `.pdf` files in the `documents/` folder, or use the Streamlit UI to upload them.

---

## API Endpoints

### 1. Upload and Index a PDF

```http
POST /api/upload
Content-Type: multipart/form-data

Response:
{
  "message": "Document 'report.pdf' uploaded and indexed successfully.",
  "filename": "report.pdf",
  "chunks_indexed": 45,
  "status": "success"
}
```

### 2. Ask a Question

```http
POST /api/ask
Content-Type: application/json

Request:
{
  "question": "What is the main topic of the document?",
  "k": 4
}

Response:
{
  "answer": "The document discusses...",
  "sources": [
    {
      "chunk": "The main topic is...",
      "source": "report.pdf",
      "page": 3,
      "score": 0.92
    }
  ],
  "question": "What is the main topic of the document?"
}
```

### 3. Get Chat History

```http
GET /api/history

Response:
{
  "history": [
    {
      "question": "...",
      "answer": "...",
      "sources": [...]
    }
  ],
  "total": 1
}
```

### 4. List Indexed Documents

```http
GET /api/documents

Response:
{
  "documents": ["report.pdf", "guide.pdf"],
  "total": 2
}
```

### 5. Health Check

```http
GET /api/health

Response:
{
  "status": "healthy",
  "service": "Smart Document AI Assistant"
}
```

### 6. API Documentation

```text
Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
```

---

## How RAG Works

**Retrieval-Augmented Generation (RAG)** is a technique that combines information retrieval with text generation. Instead of relying solely on the LLM's training data, RAG retrieves relevant information from a knowledge base (your documents) and provides it as context to the LLM.

### Why RAG?

| Challenge | RAG Solution |
|-----------|-------------|
| LLMs have knowledge cutoffs | RAG provides up-to-date information |
| LLMs hallucinate answers | RAG grounds answers in retrieved context |
| LLMs don't know private data | RAG retrieves from your documents |
| Expensive to retrain models | RAG requires no training, only indexing |

---

## How Embeddings Work

Embeddings convert text into numerical vectors that capture semantic meaning.

1. **Text → Tokens**: Text is split into tokens (words/subwords)
2. **Tokens → Vectors**: Each token passes through a transformer model
3. **Pooling**: Token vectors are combined into a single 384-dim vector
4. **Similarity**: Similar texts produce vectors close together in vector space

**Example**: "What is AI?" and "Tell me about artificial intelligence" produce similar embeddings because they share semantic meaning, even though they use different words.

---

## How ChromaDB Works

ChromaDB is an open-source vector database designed for AI applications.

- **Collections**: Like tables in SQL, grouping related embeddings
- **Documents**: The original text chunks
- **Embeddings**: Vector representations (384 floats each)
- **Metadata**: Additional info (source file, page number)
- **Indexing**: Uses HNSW algorithm for fast approximate nearest neighbor search

---

## How Similarity Search Works

When you ask a question:

1. Your question is converted to an embedding vector
2. ChromaDB compares it against all stored document embeddings
3. **Cosine Similarity** measures the angle between vectors:
   - Cos(θ) = 1 → identical direction (very similar)
   - Cos(θ) = 0 → perpendicular (unrelated)
   - Cos(θ) = -1 → opposite direction (very different)
4. Top-k chunks with highest similarity are retrieved

---

## Future Improvements

- **Multi-format support** (DOCX, CSV, HTML, images)
- **Advanced chunking strategies** (semantic chunking, agentic chunking)
- **Hybrid search** (combine vector + keyword search)
- **Reranking** (improve retrieval quality with cross-encoders)
- **Multiple collections** (organize documents by topic/project)
- **Persistent chat history** (save to SQLite/PostgreSQL)
- **Document summarization** (generate document summaries)
- **Streaming responses** (show answers as they're generated)
- **Authentication** (multi-user support)

---

## Interview Questions

See **[INTERVIEW_GUIDE.md](./INTERVIEW_GUIDE.md)** for a comprehensive list of interview questions with detailed answers covering:

- RAG Architecture
- Embeddings & Vector Search
- LangChain & ChromaDB
- API Design & FastAPI
- Each file, class, and function
- Trade-offs and design decisions
- Troubleshooting scenarios
