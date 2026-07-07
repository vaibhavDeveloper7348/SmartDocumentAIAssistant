# Smart Document AI Assistant - Interview Guide

## Complete Preparation Guide for Associate AI Product Engineer Interviews

---

## Table of Contents

1.  [What is RAG?](#1-what-is-rag)
2.  [Why Use RAG?](#2-why-use-rag)
3.  [Project Architecture Overview](#3-project-architecture-overview)
4.  [Technology Choices Explained](#4-technology-choices-explained)
5.  [Detailed Pipeline Walkthrough](#5-detailed-pipeline-walkthrough)
6.  [Every File Explained](#6-every-file-explained)
7.  [Every Class & Function Explained](#7-every-class--function-explained)
8.  [Every API Endpoint Explained](#8-every-api-endpoint-explained)
9.  [Interview Questions & Answers](#9-interview-questions--answers)
10. [Key Concepts to Master](#10-key-concepts-to-master)

---

## 1. What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**.

It combines two components:
1.  **Retrieval**: Finding relevant information from a knowledge base
2.  **Generation**: Using an LLM to generate answers based on retrieved information

### The Core Idea

Instead of asking an LLM to answer from its training data alone, RAG first retrieves relevant documents from your own data store, then provides those documents as context for the LLM to generate an answer.

```text
Traditional LLM:      Question -> LLM -> Answer (from training data)
RAG:                  Question -> [Retrieve relevant docs] -> Question + Context -> LLM -> Answer
```

The critical difference is that RAG **grounds** the LLM's answer in actual documents, making answers more accurate and verifiable.

---

## 2. Why Use RAG?

### Problems RAG Solves

| Problem | Description | RAG Solution |
|---------|-------------|--------------|
| **Knowledge Cutoff** | LLMs only know data up to their training date | RAG retrieves current information from your documents |
| **Hallucinations** | LLMs make up plausible-sounding false information | RAG grounds answers in actual retrieved text |
| **Private Data** | LLMs don't know your internal documents | RAG retrieves from your private document store |
| **Cost of Training** | Fine-tuning models is expensive and slow | RAG requires no training, just indexing |
| **Domain Specificity** | General LLMs lack domain expertise | RAG provides domain-specific context |
| **Verifiability** | Can't verify where an LLM got its answer | RAG shows exact source documents and pages |

### When to Use RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| Data freshness | Always up-to-date | Fixed at training time |
| Implementation complexity | Low | High |
| Cost | Low (just retrieval) | High (training compute) |
| Accuracy | High with good retrieval | High for specific tasks |
| Transparency | Shows sources | Black box |
| Use case | Question answering over documents | Task-specific behavior change |

### Interview Answer Template

> "RAG is essential for building reliable AI assistants because it addresses the fundamental limitations of LLMs. LLMs are trained on static datasets and have knowledge cutoffs. They also hallucinate - generating confident but false information. RAG solves both problems by retrieving relevant documents from a trusted knowledge base and providing them as context to the LLM. This ensures answers are grounded in actual data, are verifiable through source citations, and can work with proprietary or frequently updated information without retraining."

---

## 3. Project Architecture Overview

### High-Level Architecture

```text
+------------+     +------------+     +-------------+     +------------+
|  Streamlit  |---->|   FastAPI   |---->|   Services   |---->|  ChromaDB  |
|  (Frontend) |     |  (Backend)  |     |   (Logic)   |     |  (Vectors) |
+------------+     +------------+     +------+------+     +------------+
                                              |
                                              v
                                        +------------+
                                        |    LLM     |
                                        | (OpenAI /  |
                                        |  Ollama /  |
                                        | HuggingFace)|
                                        +------------+
```

### Data Flow

```text
INDEXING FLOW:
PDF File -> Text Extraction -> Chunking -> Embedding -> ChromaDB Store

QUERY FLOW:
Question -> Query Embedding -> ChromaDB Search -> Retrieved Chunks
-> Prompt Construction -> LLM -> Answer + Sources
```

### Design Principles Applied

1.  **Separation of Concerns**: Each layer (API, Services, Database) has distinct responsibilities
2.  **Single Responsibility**: Each class does one thing
3.  **Dependency Injection**: Services are passed to routes, making testing easier
4.  **Modularity**: Components can be swapped (different LLMs, different embedding models)
5.  **Configuration Over Hardcoding**: All settings in config.py

---

## 4. Technology Choices Explained

### Why LangChain?

LangChain is a framework for building LLM-powered applications.

**Reasons for choosing LangChain:**
1.  **Abstraction Layer**: Consistent interfaces for different LLMs, embedding models, and vector stores
2.  **Built-in Components**: Text splitters, document loaders, integrations are pre-built
3.  **RAG Primitives**: Specific tools designed for RAG workflows
4.  **Community**: Extensive documentation and community support
5.  **Flexibility**: Easy to swap components without changing application code

**What we use from LangChain:**
-   `PyPDFLoader` - Load PDF documents
-   `RecursiveCharacterTextSplitter` - Smart text chunking
-   `HuggingFaceEmbeddings` - Embedding model wrapper
-   `ChatOpenAI` / `Ollama` / `HuggingFacePipeline` - LLM interfaces

**Interview Answer:**
> "We use LangChain because it provides a unified interface for the entire RAG pipeline. Without LangChain, we'd need to write separate code for each LLM provider, each embedding model, and each vector database. LangChain abstracts these differences behind consistent APIs."

### Why ChromaDB?

ChromaDB is an open-source vector database.

**Reasons:**
1.  **No External Server**: Runs in-process, no Docker needed
2.  **Persistent Storage**: Data survives restarts
3.  **Simple API**: `add()`, `query()`, `get()` - easy to learn
4.  **Metadata Support**: Store source file, page number alongside vectors
5.  **Cosine Similarity**: Built-in support for semantic search
6.  **Lightweight**: Small footprint, fast for moderate document sets

**Database comparison:**

| Database | Pros | Cons |
|----------|------|------|
| ChromaDB | Simple, no server, free | Limited scaling |
| Pinecone | Managed, scalable | Paid, requires internet |
| Qdrant | High performance | Requires setup |
| FAISS | Fastest search | No persistence |

### Why Sentence Transformers?

Sentence Transformers (all-MiniLM-L6-v2) is a free, local embedding model.

**Reasons:**
1.  **Free**: No API costs, no API key needed
2.  **Local**: Data never leaves your machine
3.  **Lightweight**: 80MB model, runs on CPU
4.  **Quality**: 384-dim embeddings with good semantic understanding

### Why FastAPI?

**Reasons:**
1.  **Performance**: One of the fastest Python web frameworks
2.  **Auto Docs**: Automatic OpenAPI/Swagger documentation generation
3.  **Validation**: Built-in request/response validation with Pydantic
4.  **Async Support**: Native async/await for I/O operations
5.  **Modern Python**: Type hints, dependency injection

---

## 5. Detailed Pipeline Walkthrough

### Indexing Pipeline

#### Step 1: PDF Upload
File validated (.pdf), saved to `documents/`, `rag_service.index_document()` called.

#### Step 2: Text Extraction
`PyPDFLoader` reads the PDF, extracts text page-by-page, returns Document objects with page_content and metadata (source, page).

#### Step 3: Text Chunking
`RecursiveCharacterTextSplitter` splits pages into chunks:
- Chunk size: 1000 chars, Overlap: 200 chars
- Separators (in order): `\n\n`, `\n`, `.`, `!`, `?`, `,`, ` `, ``
- Tries larger separators first (keeps paragraphs together)

#### Step 4: Embedding Generation
`HuggingFaceEmbeddings` converts each text chunk to a 384-dim vector.

#### Step 5: ChromaDB Storage
Stores: text content, embedding vector, metadata (source, page), unique ID.

### Query Pipeline

#### Step 1: Question Input
User submits question via POST /api/ask. Validated (non-empty).

#### Step 2: Query Embedding
Same embedding model used for indexing converts question to vector.

**Critical: Same model must be used!** Different models produce incompatible vector spaces.

#### Step 3: Similarity Search
ChromaDB.query() finds k most similar vectors using cosine similarity.

Cosine Similarity formula:
```
cosine_similarity(A, B) = (A . B) / (||A|| * ||B||)
```

#### Step 4: Prompt Construction
Retrieved chunks + question combined into structured prompt:
```
Answer based ONLY on the provided context.
If insufficient information, say so.
Context: [Source: file, Page: N] ...
Question: ...
Answer:
```

#### Step 5: LLM Generation
Low temperature (0.2) for factual, focused answers.

#### Step 6: Response Formation
Answer + sources (chunk, file, page, score) returned to user.

---

## 6. Every File Explained

### `main.py` (Project Root)
Entry point. Starts FastAPI server via Uvicorn on configured host/port.

### `requirements.txt`
Lists dependencies: fastapi, uvicorn, langchain, chromadb, sentence-transformers, pypdf, streamlit, etc.

### `.env.example`
Template for environment configuration. Documents all configurable settings.

### `app/config.py`
Centralized configuration. Loads from env vars with sensible defaults. Key settings: embedding model, LLM choice, chunk size, ChromaDB path.

### `app/main.py`
FastAPI application initialization. Factory function for testability. CORS middleware for frontend access.

### `api/schemas.py`
Pydantic models: UploadResponse, AnswerResponse, QuestionRequest, SourceInfo, ChatHistoryResponse, DocumentListResponse, ErrorResponse.

### `api/routes.py`
API endpoints: POST /api/upload, POST /api/ask, GET /api/history, DELETE /api/history, GET /api/documents, GET /api/health.

### `services/document_processor.py`
DocumentProcessor class. Methods: load_pdf(), split_documents(), process_pdf(), get_chunk_texts_and_metadatas().

### `services/embedding_service.py`
EmbeddingService class. Methods: embed_texts(), embed_query().

### `services/llm_service.py`
LLMService class. Auto-selects OpenAI -> Ollama -> HuggingFace. Methods: _initialize_llm(), generate_response().

### `services/rag_service.py`
RAGService class - **core RAG orchestrator**. Methods: index_document(), answer_question(), _build_context(), _build_prompt(), _prepare_sources(), get_chat_history(), clear_chat_history(), get_indexed_documents().

### `database/chroma_client.py`
Functions: get_chroma_client(), get_or_create_collection(), delete_collection().

### `frontend/streamlit_app.py`
Streamlit UI. Sidebar: upload, document list, actions. Main: chat interface, source display.

### `utils/helpers.py`
Functions: generate_unique_id(), get_document_files(), format_sources().

---

## 7. Every Class & Function Explained

### Config class (app/config.py)
Single source of truth for all configuration. Centralizes env var loading with defaults.

### DocumentProcessor (services/document_processor.py)
**load_pdf(file_path)**: Extracts text from PDF page-by-page using PyPDFLoader. Preserves page numbers.
**split_documents(pages)**: Chunks pages using RecursiveCharacterTextSplitter (1000 chars, 200 overlap).
**process_pdf(file_path)**: Combines load + split. Main entry point.
**get_chunk_texts_and_metadatas(chunks)**: Extracts texts/metadatas/ids for ChromaDB insertion.

### EmbeddingService (services/embedding_service.py)
**embed_texts(texts)**: Batch embedding for indexing.
**embed_query(query)**: Single query embedding. Allows different encoding for queries vs documents.

### LLMService (services/llm_service.py)
**_initialize_llm()**: Try OpenAI -> Ollama -> HuggingFace fallback.
**generate_response(prompt)**: Unified interface for all LLM providers. Temperature 0.2.

### RAGService (services/rag_service.py) - MOST IMPORTANT
**index_document(file_path)**: Process PDF -> Generate embeddings -> Store in ChromaDB.
**answer_question(question, k)**: Complete RAG pipeline:
1.  Embed question
2.  Search ChromaDB for top-k similar chunks
3.  Build context from chunks
4.  Create prompt with context + question
5.  Generate answer with LLM
6.  Return answer + sources

**_build_context(chunks, metadatas)**: Formats chunks with source citations for LLM.
**_build_prompt(context, question)**: Constructs LLM prompt with anti-hallucination instructions.
**_prepare_sources(chunks, metadatas)**: Formats sources for API response.

### ChromaDB Functions (database/chroma_client.py)
**get_chroma_client()**: Creates persistent ChromaDB client.
**get_or_create_collection()**: Gets existing or creates new collection. Returns collection object.

### Utility Functions (utils/helpers.py)
**generate_unique_id()**: UUID4 generation for unique chunk IDs.
**get_document_files()**: Lists PDF files in documents directory.
**format_sources()**: Cleans source data for API response.

---

## 8. Every API Endpoint Explained

### POST /api/upload
Upload PDF, validate, save to documents/, index into ChromaDB. Returns message, filename, chunks_indexed.

### POST /api/ask
Ask question, validate, run RAG pipeline, return answer + sources.

### GET /api/history
Return chat history (in-memory list of Q&A pairs).

### DELETE /api/history
Clear chat history.

### GET /api/documents
List indexed document filenames.

### GET /api/health
Health check. Returns status "healthy".

---

## 9. Interview Questions & Answers

### RAG Fundamentals

**Q1: What is RAG and why is it important?**
RAG combines information retrieval with text generation. Instead of relying solely on an LLM's training data, RAG first retrieves relevant documents from a knowledge base, then provides those documents as context. It's important because it grounds answers in actual data (reducing hallucinations), enables Q&A over private/latest documents, provides verifiable sources, and requires no model retraining.

**Q2: How does RAG differ from fine-tuning?**
RAG provides relevant context at inference time - best for up-to-date or changing information. Fine-tuning modifies model weights through additional training - best for adopting specific behavior/style. Many production systems use both.

**Q3: What are the components of a RAG system?**
Document Loader, Text Splitter, Embedding Model, Vector Database, LLM.

**Q4: What are the failure modes of RAG?**
Poor retrieval (irrelevant chunks), missing context across chunks, chunk boundary issues (context split), LLM ignoring context (hallucination), low-quality sources (garbage in, garbage out).

### Embeddings

**Q5: What are embeddings?**
Dense numerical vector representations of text that capture semantic meaning. Similar texts produce similar vectors.

**Q6: How are embeddings generated?**
Tokenization -> Transformer encoding (self-attention) -> Pooling (combine token vectors) -> Normalization.

**Q7: What is cosine similarity?**
Measures the cosine of the angle between vectors. Scale-invariant, normalized (-1 to 1), computationally efficient for high-dimensional vectors.

**Q8: Why 384 dimensions?**
Trade-off between expressiveness and computational cost. 384 is sufficient for document retrieval; higher dimensions increase storage/search cost with diminishing returns.

**Q9: Why use the same embedding model for indexing and querying?**
Each model maps text into its own vector space. Different models produce incompatible spaces, making comparison meaningless.

### Vector Databases

**Q10: What is a vector database?**
Stores and indexes vector embeddings for efficient similarity search. Uses algorithms like HNSW for approximate nearest neighbor search.

**Q11: Why ChromaDB over alternatives?**
Free, no external server, simple API, persistent storage for learning/demo. Others are better for production scaling.

**Q12: What is a collection in ChromaDB?**
Like a table in SQL. Groups related embeddings. Each entry has id, embedding, metadata, document.

### LangChain

**Q13: What is LangChain?**
Framework for building LLM applications. Provides unified interfaces, pre-built components, and RAG tools.

**Q14: What LangChain components does this project use?**
PyPDFLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, ChatOpenAI/Ollama/HuggingFacePipeline.

**Q15: What is RecursiveCharacterTextSplitter?**
Splits text recursively at separator levels: paragraphs -> lines -> sentences -> phrases -> characters. Preserves semantic units.

### API Design

**Q16: Why FastAPI over Flask?**
Auto OpenAPI docs, Pydantic validation, async support, high performance, modern Python features.

**Q17: Why is CORS needed?**
Browser security mechanism blocks cross-origin requests. CORS headers allow frontend (port 8501) to talk to backend (port 8000).

**Q18: Why Pydantic models?**
Automatic validation, type safety, OpenAPI schema generation, JSON serialization.

### Project Design

**Q19: Why separate frontend and backend?**
Separation of concerns. Each can be developed/tested/scaled independently. Same API could serve mobile app or different frontend.

**Q20: Why in-memory chat history?**
Simplicity for demo. No database setup needed. In production, use PostgreSQL/Redis for persistence.

**Q21: How would you scale this?**
Add authentication/per-user collections for multi-user; use Pinecone/Qdrant for millions of vectors; add task queue for heavy processing; GPU for inference.

**Q22: How would you test this?**
Unit tests for each service, integration tests for RAG pipeline, API tests with TestClient, mock external services.

### Troubleshooting

**Q23: What if question is not in documents?**
LLM instructed to say "I cannot find sufficient information" instead of hallucinating. Low similarity scores trigger this path.

**Q24: What if PDF is corrupt?**
PyPDFLoader throws exception, caught by route handler, returns HTTP 500 with error message.

**Q25: How to choose chunk size?**
Trade-off: smaller chunks = more precise but less context; larger chunks = more context but may include irrelevant info. 1000 chars is a good starting point.

---

## 10. Key Concepts to Master

### Must-Know Topics for Interview

1.  **RAG Pipeline** - Explain every step end-to-end
2.  **Embeddings** - What they are, how created, why they work
3.  **Cosine Similarity** - Formula, intuition, advantages
4.  **Vector Databases** - Purpose, indexing algorithms (HNSW), comparison
5.  **Chunking** - Why needed, strategies, trade-offs
6.  **Prompt Engineering** - Structure, temperature, anti-hallucination instructions
7.  **FastAPI** - Features, benefits, auto-documentation
8.  **LangChain** - Purpose, components used, abstraction benefits
9.  **Separation of Concerns** - Why modular architecture matters
10. **Hallucination Mitigation** - How RAG reduces made-up answers

### Common Interview Questions to Practice

1.  "Walk me through how this application works end-to-end."
2.  "What happens when a user uploads a PDF?"
3.  "What happens when a user asks a question?"
4.  "Why does this reduce hallucinations compared to a regular LLM?"
5.  "How would you improve this system?"
6.  "What are the limitations of this approach?"
7.  "How would you handle multiple users?"
8.  "Why did you choose these specific technologies?"
9.  "Explain the embedding process in detail."
10. "What is cosine similarity and why does it work for text?"

---

*Good luck with your interview! Remember: understanding WHY each decision was made is more important than memorizing code. Focus on the concepts and trade-offs.*
