# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG CLI is a Retrieval-Augmented Generation system for document question-answering. It provides multiple interfaces: CLI, REST API (FastAPI), Web UI (React/Vite), and native macOS desktop app (Tauri).

**Key technologies:** Python 3.12+, ChromaDB, sentence-transformers (BAAI/bge-small-en-v1.5), Ollama, FastAPI, React, Tauri

## Common Commands

### Python Environment
```bash
source venv/bin/activate
pip install -r requirements.txt
pip install -e .  # Install package in editable mode
```

### Testing
```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_chunker.py -v     # Run single test file
pytest tests/ --cov=src --cov-report=term  # With coverage
```

### Linting
```bash
# Python (backend)
ruff check src/ backend/ tests/           # Check for issues
ruff check src/ backend/ tests/ --fix     # Auto-fix issues
ruff format src/ backend/ tests/          # Format code

# Frontend
cd frontend && npm run lint               # ESLint
cd frontend && npm run build              # Build (includes type checking)
```

### Running the Application

**CLI:**
```bash
rag-cli init                    # Initialize (sets up Ollama)
rag-cli add ./documents/        # Index documents
rag-cli add ./docs/ -r          # Index recursively
rag-cli query "question"        # Query knowledge base
rag-cli query "q" --show-sources  # Query with source citations
rag-cli chat                    # Interactive chat mode
rag-cli chat --stream           # Chat with streaming responses
rag-cli list                    # List indexed documents
rag-cli remove document.pdf     # Remove a document
rag-cli clear --confirm         # Clear entire knowledge base
rag-cli stats                   # Show system statistics
rag-cli config list             # Show all config values
rag-cli config get llm.ollama_model  # Get specific setting
```

**Web (Backend + Frontend):**
```bash
./start-web.sh    # Start both services
./stop-web.sh     # Stop both services
```

**Backend only (port 8000):**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Frontend only (port 5173):**
```bash
cd frontend && npm run dev
```

**Desktop app (Tauri):**
```bash
cd frontend && npm run tauri:dev    # Development
cd frontend && npm run tauri:build  # Build release
```

**HTML Scraper (download documentation):**
```bash
python tools/html_scraper.py              # Use default config
python tools/html_scraper.py --dry-run    # Preview without downloading
python tools/html_scraper.py --verbose    # Detailed progress
```

**DITA Chunker (semantic chunking for DITA HTML):**
```bash
python tools/ingest_dita_docs.py              # Ingest with semantic chunking
python tools/ingest_dita_docs.py --dry-run    # Preview without storing
python tools/ingest_dita_docs.py --clear-first # Clear and re-ingest
```

**Embedding Model Tools:**
```bash
python tools/benchmark_embeddings.py              # Benchmark 4 embedding models
python tools/benchmark_embeddings.py --output results.md  # Save results
python tools/reingest_with_new_embeddings.py --backup     # Re-ingest with new model
python tools/reingest_with_new_embeddings.py --dry-run    # Preview migration
python tools/compare_embeddings.py --old MODEL1 --new MODEL2  # Compare models
```

## Architecture

### Core RAG Pipeline (`src/`)

The RAG pipeline flows: Document → Chunks → Embeddings → Vector Store → Retrieval → LLM Response

| Module | Purpose |
|--------|---------|
| `cli.py` | Click-based CLI commands |
| `pipeline.py` | Orchestrates retrieval + generation |
| `document_loader.py` | Loads PDF, Markdown, HTML, TXT files |
| `chunker.py` | Splits documents into overlapping chunks |
| `embedder.py` | Generates embeddings via sentence-transformers |
| `vector_store.py` | ChromaDB wrapper for persistence |
| `retriever.py` | Similarity search and context retrieval |
| `llm_client.py` | Ollama and Claude API integration |
| `prompts.py` | RAG prompt templates |
| `config.py` | Pydantic settings from env vars |

### Backend (`backend/`)

FastAPI REST API wrapping the core pipeline:
- `main.py` - FastAPI app with CORS config
- `api/routes.py` - API endpoint handlers
- `api/models.py` - Pydantic request/response models
- `services/rag_service.py` - Service layer connecting to core pipeline

**REST API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check with Ollama status |
| GET | `/api/config` | Get configuration settings (defaults from .env) |
| POST | `/api/query` | Query knowledge base |
| POST | `/api/upload` | Upload and index a document |
| GET | `/api/documents` | List all indexed documents |
| DELETE | `/api/documents/{id}` | Delete a document |

Interactive docs at http://localhost:8000/docs (Swagger UI)

### Frontend (`frontend/`)

React/Vite web UI with Tauri desktop wrapper:
- `src/App.jsx` - Main app component
- `src/components/` - UI components (ChatInterface, MessageList, Sidebar, etc.)
- `src/services/api.js` - Backend API client
- `src-tauri/` - Rust code for native desktop app

## Configuration

Settings are loaded from environment variables. Copy `.env.example` to `.env`:

- `OLLAMA_MODEL` - Default: llama3.1:8b
- `DEFAULT_LLM_PROVIDER` - "ollama" (free) or "claude"
- `ANTHROPIC_API_KEY` - Required only for Claude
- `CHUNK_SIZE` / `CHUNK_OVERLAP` - Text chunking (1200/200)
- `TOP_K_RESULTS` - Chunks retrieved per query (5)

## Data Storage

- `data/vector_db/` - ChromaDB persistence
- `data/documents/` - Source documents
- `logs/` - Application logs

## Documentation

- [README.md](README.md) - Main project documentation, CLI reference, installation
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) - End-user guide
- [docs/dita_chunker.md](docs/dita_chunker.md) - DITA semantic chunker guide
- [docs/adr/](docs/adr/) - Architecture Decision Records
- [desktop/README.md](desktop/README.md) - macOS desktop app user guide
- [desktop/BUILDING.md](desktop/BUILDING.md) - Desktop app build instructions
- [frontend/README.md](frontend/README.md) - Frontend development docs
