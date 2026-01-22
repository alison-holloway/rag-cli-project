# RAG System - Project Plan

## Project Overview
Build a command-line RAG (Retrieval-Augmented Generation) system that processes documents (PDF, Markdown, HTML) and enables intelligent question-answering using local embeddings and LLM integration.

## Goals
- **Primary**: Create a functional CLI-based RAG system for learning
- **Secondary**: Design with extensibility for future web UI and macOS desktop app
- **Learning Objectives**: Python development, vector databases, embeddings, RAG architecture

## Technical Stack

### Core Technologies
- **Language**: Python 3.13
- **CLI Framework**: Click or Typer (for elegant CLI interface)
- **Document Processing**:
  - PyPDF2 or pypdf for PDF extraction
  - python-markdown for Markdown parsing
  - BeautifulSoup4 for HTML parsing
- **Embeddings**: sentence-transformers (local, no API costs)
- **Vector Store**: ChromaDB (lightweight, persistent, local)
- **LLM Integration**: 
  - **Default**: Local models via Ollama (Llama 3.1 - FREE, fully local)
  - **Optional Upgrade**: Anthropic Claude API (better quality, requires API key and costs money)
  - Note: You already have Ollama installed!

### Development Environment
- **OS**: macOS (M3 chip - ARM architecture compatible)
- **RAM**: 16GB (sufficient for local embeddings)
- **Python Version**: 3.13 (using 3.13 instead of 3.14 for ChromaDB compatibility)
- **Dependency Management**: pip + requirements.txt or Poetry

## System Architecture

### Phase 1: CLI Application (POC)

```
rag-cli/
├── src/
│   ├── __init__.py
│   ├── cli.py              # Main CLI interface
│   ├── document_loader.py  # PDF/MD/HTML processing
│   ├── chunker.py          # Text chunking strategies
│   ├── embedder.py         # Generate embeddings
│   ├── vector_store.py     # ChromaDB interface
│   ├── retriever.py        # Search & retrieve relevant chunks
│   ├── llm_client.py       # LLM API integration
│   └── config.py           # Configuration management
├── tests/
│   └── test_*.py
├── data/
│   ├── documents/          # Input documents
│   └── vector_db/          # ChromaDB persistence
├── .env.example            # Environment variables template
├── requirements.txt
├── README.md
└── pyproject.toml          # Optional: if using Poetry
```

### Core Components

#### 1. Document Loader
**Responsibility**: Extract text from various file formats
- Handle PDF text extraction (including OCR for scanned PDFs - optional)
- Parse Markdown files
- Extract text from HTML (strip tags, preserve structure)
- Metadata extraction (filename, page numbers, sections)

#### 2. Text Chunker
**Responsibility**: Split documents into semantic chunks
- **Strategy Options**:
  - Fixed-size chunks with overlap (simple, start here)
  - Semantic chunking (by paragraphs/sections)
  - Recursive character splitting
- **Parameters**: chunk_size (500-1000 chars), overlap (50-200 chars)
- Preserve context and metadata with each chunk

#### 3. Embedder
**Responsibility**: Convert text chunks to vector embeddings
- **Model**: `all-MiniLM-L6-v2` (384 dimensions, fast, good quality)
- Batch processing for efficiency
- Cache embeddings to avoid recomputation

#### 4. Vector Store (ChromaDB)
**Responsibility**: Store and retrieve embeddings
- Persistent storage (survives restarts)
- Metadata filtering capabilities
- Similarity search (cosine similarity)
- Collection management (one per document set or unified)

#### 5. Retriever
**Responsibility**: Find relevant chunks for queries
- Query embedding generation
- Top-k similarity search (k=3-5 initially)
- Re-ranking strategies (optional: MMR for diversity)
- Context window management

#### 6. LLM Client
**Responsibility**: Generate answers using retrieved context
- API integration (Anthropic Claude recommended)
- Prompt engineering:
  ```
  Context: {retrieved_chunks}
  
  Question: {user_question}
  
  Answer based only on the provided context. If the answer isn't in the context, say so.
  ```
- Streaming responses (for better UX)
- Error handling and retry logic

## CLI Commands

### Basic Commands
```bash
# Initialize a new RAG project (auto-starts Ollama and pulls model if needed)
rag-cli init [project_name]

# Add documents to the knowledge base
rag-cli add /path/to/document.pdf
rag-cli add /path/to/folder/  # Batch add

# Query the knowledge base (uses Llama by default)
rag-cli query "What is the main topic discussed?"

# Query with Claude API (optional, costs money but better quality)
rag-cli query "What is the main topic discussed?" --llm claude

# Interactive mode
rag-cli chat

# Interactive mode with Claude
rag-cli chat --llm claude

# List indexed documents
rag-cli list

# Remove documents
rag-cli remove document_id

# Clear the entire database
rag-cli clear --confirm

# Show system stats
rag-cli stats
```

### Advanced Commands (Optional)
```bash
# Configure chunk size
rag-cli config set chunk_size 800

# Switch default LLM provider
rag-cli config set llm_provider ollama  # FREE (default)
rag-cli config set llm_provider claude  # Better quality (costs money)

# Switch Ollama model
rag-cli config set ollama_model llama3.1:8b     # Default
rag-cli config set ollama_model llama3.1:70b    # Slower but smarter (if you have the RAM)

# Export/import knowledge base
rag-cli export knowledge_base.zip
rag-cli import knowledge_base.zip
```

## Implementation Phases

### Phase 1.1: Basic Infrastructure (Week 1)
- [ ] Set up Python project structure
- [ ] Implement CLI framework (Click/Typer)
- [ ] Create configuration system (.env, config files)
- [ ] Set up logging
- [ ] Basic error handling

### Phase 1.2: Document Processing (Week 1-2)
- [ ] Implement PDF text extraction
- [ ] Implement Markdown parsing
- [ ] Implement HTML parsing
- [ ] Test with sample documents
- [ ] Add metadata extraction

### Phase 1.3: Embedding & Vector Store (Week 2)
- [ ] Integrate sentence-transformers
- [ ] Implement text chunking with overlap
- [ ] Set up ChromaDB
- [ ] Create vector store interface
- [ ] Test embedding and storage pipeline

### Phase 1.4: Retrieval & Generation (Week 2-3)
- [ ] Implement similarity search
- [ ] Integrate LLM API (Claude)
- [ ] Build prompt templates
- [ ] Create query pipeline
- [ ] Test end-to-end RAG flow

### Phase 1.5: Polish & Testing (Week 3)
- [ ] Add comprehensive error messages
- [ ] Implement progress indicators
- [ ] Write unit tests
- [ ] Create user documentation
- [ ] Add example documents and queries

### Post-Phase 1 Enhancements

### Feature Additions (After POC Complete)
These features were added after the initial Phase 1 POC was completed:

#### 1. Automatic Ollama Setup (Added: Jan 2026)
- **What**: `rag-cli init` now automatically starts Ollama and pulls llama3.1:8b if not available
- **Why**: Improves user experience by eliminating manual Ollama setup steps
- **Implementation**: 
  - Check if Ollama is running, start if needed
  - Check if llama3.1:8b is pulled, pull if needed
  - Show progress indicators during setup
- **Status**: Completed

#### 2. Verbose Flag for Clean Output (Added: Jan 2026)
- **What**: Add `--verbose` flag to the main `rag-cli` command (applies to all subcommands)
- **Why**: Reduce output clutter for end users while preserving debug capability
- **Implementation**:
  - Default: Show only spinner and final results (hide INFO logs)
  - With `--verbose`: Show all internal processing details and INFO logs
  - Flag applies globally: `rag-cli --verbose <subcommand>`
- **Usage**: `rag-cli --verbose query "question"`, `rag-cli --verbose add file.pdf`, `rag-cli --verbose chat`
- **Status**: Completed

## Phase 2: Web UI (Local Development)

### Overview
Build a local web interface for the RAG system using FastAPI backend and React frontend. Focus on learning full-stack development while keeping deployment simple (local-only for now).

### Technology Stack

#### Backend
- **FastAPI** - Modern Python web framework
  - Why: Matches existing Python skills, auto API docs, async support
  - Serves REST API endpoints
  - Reuses existing RAG CLI components
  
#### Frontend  
- **React 18+** with **Vite** - Modern frontend framework
  - Why: Industry standard, great learning opportunity, fast development
  - TypeScript optional (recommend starting with JavaScript)
  - Component-based architecture
  
#### Development Setup
- **Local only** - Both frontend and backend run on localhost
- **No database needed initially** - Use in-memory session storage
- **CORS enabled** - Frontend (port 5173) talks to backend (port 8000)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (localhost:5173)             │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │           React Frontend (Vite)                 │    │
│  │  - Chat Interface                               │    │
│  │  - Document Upload                              │    │
│  │  - Settings Panel                               │    │
│  └────────────────────────────────────────────────┘    │
│                         │                                │
│                         │ HTTP/REST API                  │
│                         ▼                                │
│  ┌────────────────────────────────────────────────┐    │
│  │        FastAPI Backend (localhost:8000)         │    │
│  │  - /api/query                                   │    │
│  │  - /api/upload                                  │    │
│  │  - /api/documents                               │    │
│  │  - /api/health                                  │    │
│  └────────────────────────────────────────────────┘    │
│                         │                                │
│                         │                                │
│                         ▼                                │
│  ┌────────────────────────────────────────────────┐    │
│  │         Existing RAG Components                 │    │
│  │  - document_loader.py                           │    │
│  │  - embedder.py                                  │    │
│  │  - vector_store.py                              │    │
│  │  - llm_client.py                                │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Project Structure

```
rag-cli-project/
├── src/                    # Existing CLI code
│   ├── cli.py
│   ├── document_loader.py
│   └── ...
├── backend/                # New FastAPI backend
│   ├── __init__.py
│   ├── main.py            # FastAPI app
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py      # API endpoints
│   │   └── models.py      # Pydantic models
│   ├── services/
│   │   └── rag_service.py # Wrapper around existing RAG code
│   └── config.py
├── frontend/               # New React frontend
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── DocumentUpload.jsx
│   │   │   ├── MessageList.jsx
│   │   │   └── SettingsPanel.jsx
│   │   ├── services/
│   │   │   └── api.js     # API client
│   │   └── styles/
│   │       └── App.css
│   └── public/
├── data/                   # Existing
├── venv/                   # Existing
└── requirements-web.txt    # Additional web dependencies
```

### Implementation Phases

#### Phase 2.1: Backend API Setup (Week 1) ✅ COMPLETE
**Goal:** Create FastAPI backend that exposes RAG functionality via REST API

Tasks:
- [x] Install FastAPI and dependencies (`fastapi`, `uvicorn`, `python-multipart`)
- [x] Create basic FastAPI app structure
- [x] Implement core API endpoints:
  - `POST /api/query` - Submit questions, get answers
  - `POST /api/upload` - Upload documents
  - `GET /api/documents` - List indexed documents
  - `DELETE /api/documents/{id}` - Remove documents
  - `GET /api/health` - Health check
- [x] Create service layer that wraps existing RAG components
- [x] Add CORS middleware for local development
- [x] Test endpoints with curl/Postman
- [x] Auto-generate API docs (FastAPI provides this free!)

**API Examples:**
```python
# POST /api/query
{
  "query": "What is this about?",
  "llm_provider": "ollama",  # or "claude"
  "top_k": 5
}

# Response
{
  "answer": "This document is about...",
  "sources": [
    {"file": "doc.pdf", "page": 1, "chunk": "..."}
  ],
  "metadata": {
    "llm_provider": "ollama",
    "chunks_retrieved": 5,
    "processing_time": 2.3
  }
}
```

#### Phase 2.2: Frontend Setup (Week 1-2)
**Goal:** Create basic React app with chat interface

Tasks:
- [ ] Initialize Vite + React project
- [ ] Set up basic component structure
- [ ] Create API client service (axios or fetch)
- [ ] Implement ChatInterface component:
  - Message input box
  - Send button
  - Message list display
  - Loading indicators
- [ ] Connect to backend API
- [ ] Test end-to-end query flow
- [ ] Basic styling (can use Tailwind CSS or plain CSS)

**Key Components:**
```jsx
// ChatInterface.jsx - Main chat UI
// MessageList.jsx - Display chat history
// MessageInput.jsx - Text input + send button
// SettingsPanel.jsx - LLM provider toggle
```

#### Phase 2.3: Document Upload (Week 2)
**Goal:** Add document upload functionality to UI

Tasks:
- [ ] Create DocumentUpload component
- [ ] Implement drag-and-drop upload
- [ ] Show upload progress
- [ ] Display list of uploaded documents
- [ ] Add delete document functionality
- [ ] Handle file validation (PDF, MD, HTML only)
- [ ] Show success/error messages

#### Phase 2.4: Polish & Features (Week 2-3)
**Goal:** Improve UX and add nice-to-have features

Tasks:
- [ ] Add chat history display (session-based)
- [ ] Implement streaming responses (if using Claude API)
- [ ] Add loading states and spinners
- [ ] Improve error handling and user feedback
- [ ] Add settings panel (choose LLM provider, adjust parameters)
- [ ] Make it responsive (works on different screen sizes)
- [ ] Add keyboard shortcuts (Enter to send, etc.)
- [ ] Polish styling and UX

**Optional Enhancements:**
- [ ] Dark mode toggle
- [ ] Code syntax highlighting in responses
- [ ] Export chat history
- [ ] Document preview
- [ ] Search within documents

### Dependencies

#### Backend (requirements-web.txt)
```
# Existing dependencies from requirements.txt
# Plus new web dependencies:

# Web Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6  # For file uploads

# CORS
python-cors>=1.0.0

# Optional: Response validation
pydantic>=2.0.0
```

#### Frontend (package.json)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "vite": "^5.0.0"
  }
}
```

### Running the Application

#### Terminal 1 - Backend:
```bash
cd rag-cli-project
source venv/bin/activate
pip install -r requirements-web.txt
uvicorn backend.main:app --reload --port 8000
```

#### Terminal 2 - Frontend:
```bash
cd rag-cli-project/frontend
npm install  # First time only
npm run dev
```

Then open browser to `http://localhost:5173`

### Development Workflow

1. **Start Backend** - Terminal 1, runs on port 8000
2. **Start Frontend** - Terminal 2, runs on port 5173
3. **Make Changes** - Both have hot reload (auto-refresh on save)
4. **Test** - Use browser for frontend, check backend logs for API calls
5. **Commit** - Git commit when features work

### Success Criteria

**Phase 2 MVP:**
- ✅ Can upload documents via web UI
- ✅ Can ask questions in chat interface
- ✅ Answers appear with sources cited
- ✅ Can toggle between Ollama and Claude
- ✅ Works smoothly on localhost
- ✅ Clean, intuitive UI

**Quality Metrics:**
- Response time: <5 seconds for queries (with Llama)
- UI is responsive and doesn't freeze
- Error messages are clear and helpful
- Works in Chrome/Firefox/Safari

### Learning Resources

**FastAPI:**
- Official tutorial: https://fastapi.tiangolo.com/tutorial/
- Focus on: path operations, request/response models, file uploads

**React:**
- Official tutorial: https://react.dev/learn
- Focus on: components, props, state (useState), effects (useEffect)

**Vite:**
- Quick start: https://vitejs.dev/guide/
- Just need basics - Vite makes setup easy

### Cost Considerations

**Still 100% Free!**
- ✅ FastAPI: Free and open source
- ✅ React: Free and open source
- ✅ Vite: Free and open source
- ✅ Running locally: No hosting costs
- ✅ Same free Ollama/Llama backend

Only cost is if you use Claude API via the UI (same as CLI).

### Testing Strategy

**Manual Testing Checklist:**
- [ ] Upload a PDF document
- [ ] Ask a question, verify answer appears
- [ ] Check sources are shown
- [ ] Toggle to Claude API (if you have key), verify it works
- [ ] Try uploading multiple documents
- [ ] Test error cases (invalid file, no documents, etc.)
- [ ] Test on different browsers
- [ ] Verify chat history works within session

**Future: Automated Testing**
- Frontend: React Testing Library
- Backend: pytest with FastAPI TestClient
- (Add later, manual testing fine for learning)

### Future Enhancements (Phase 3+)

These can wait until Phase 2 is working:
- **Persistent chat history** - Save conversations to disk/DB
- **User accounts** - Multi-user support
- **Deployment** - Deploy to cloud (Vercel, Railway, etc.)
- **Real-time updates** - WebSocket for streaming responses
- **Advanced features** - Document preview, search, analytics
- **Mobile app** - React Native version

### Next Steps for Phase 2

Ready to start? Here's the order:

1. **Phase 2.1** - Build FastAPI backend first (easier to test with curl)
2. **Phase 2.2** - Build React frontend, connect to backend
3. **Phase 2.3** - Add document upload
4. **Phase 2.4** - Polish and improve UX

Each phase builds on the previous, and you can test incrementally!

---

**Want to begin Phase 2.1 (FastAPI Backend)?** Let me know and I'll give you the prompt for Claude Code!

### Phase 3: macOS Desktop App (Future)
- [ ] Evaluate frameworks (Electron, Tauri, or native Swift)
- [ ] Native file system integration
- [ ] Menu bar app design
- [ ] Local-first architecture
- [ ] App bundling and distribution

## Configuration

### Environment Variables (.env)
```
# LLM Configuration (Ollama is default - FREE and local!)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Optional: Claude API (only needed if using --llm claude flag)
# ANTHROPIC_API_KEY=your_key_here

# Vector Store
CHROMA_PERSIST_DIR=./data/vector_db

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking Parameters
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Retrieval
TOP_K_RESULTS=5

# LLM Parameters
DEFAULT_LLM_PROVIDER=ollama  # Default to free local option
LLM_TEMPERATURE=0.3
MAX_TOKENS=2000
```

## Dependencies (requirements.txt)

```
# CLI
click>=8.1.0
rich>=13.0.0  # Beautiful terminal output

# Document Processing
pypdf>=3.17.0
markdown>=3.5.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Embeddings & Vector Store
sentence-transformers>=2.2.0
chromadb>=0.4.0
numpy>=1.24.0

# LLM Integration
ollama>=0.13.0  # FREE local LLM (default) - you already have v0.13.0 installed!
# anthropic>=0.18.0  # Optional: uncomment if using --llm claude flag

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0  # Configuration validation
tqdm>=4.66.0  # Progress bars

# Development
pytest>=7.4.0
black>=23.0.0
ruff>=0.1.0
```

## Testing Strategy

### Unit Tests
- Document loader for each file type
- Chunking algorithm correctness
- Embedding generation
- Vector store CRUD operations

### Integration Tests
- End-to-end document indexing
- Query and retrieval accuracy
- LLM response generation

### Manual Testing Checklist
- [ ] Add single PDF document
- [ ] Add multiple documents in batch
- [ ] Query with expected answer in docs
- [ ] Query with answer NOT in docs (should say so)
- [ ] Handle corrupted/invalid files gracefully
- [ ] Test with large documents (>100 pages)
- [ ] Test with various document formats

## Cost Considerations (Student-Friendly!)

### 100% Free Option (Recommended for Learning)
**Total Cost: $0**
- ✅ Local embeddings (sentence-transformers): FREE
- ✅ Local vector store (ChromaDB): FREE  
- ✅ Local LLM (Ollama/Llama): FREE
- ✅ All data stays on your Mac
- ✅ Works offline

This setup is completely free and perfect for learning. You won't have any API costs.

### Optional Paid Upgrade
**When you might want Claude API:**
- Need highest quality answers for important work
- Comparing your RAG system's performance
- Presentations or demonstrations

**Cost if using Claude:**
- Claude Sonnet 4: ~$3 per million input tokens, ~$15 per million output tokens
- Estimated usage for learning: $5-20 total (can set spending limits in Anthropic console)
- You can add your API key later without changing any code!

### Command Line Examples
```bash
# Your default (FREE!)
rag-cli query "What is X?"  # Uses Llama via Ollama

# When you want premium quality (PAID)
rag-cli query "What is X?" --llm claude  # Uses Claude API
```

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ Successfully parse PDF, MD, and HTML files
- ✅ Create and persist vector embeddings
- ✅ Retrieve relevant context for queries
- ✅ Generate accurate answers using LLM
- ✅ Handle errors gracefully with helpful messages
- ✅ Complete documentation and usage examples

### Quality Metrics
- **Retrieval Accuracy**: >80% of relevant chunks retrieved in top-5
- **Response Quality**: Answers are factual and cite source chunks
- **Performance**: Process 100-page PDF in <30 seconds
- **UX**: Clear CLI output with progress indicators

## Learning Resources

### RAG Concepts
- Embeddings and vector similarity
- Semantic search vs keyword search
- Context window management
- Prompt engineering for RAG

### Python Libraries
- Click/Typer documentation
- sentence-transformers examples
- ChromaDB tutorials
- Anthropic API docs

### Future Topics (Web UI)
- FastAPI framework
- React basics
- REST API design
- Frontend-backend communication

### Future Topics (Desktop App)
- macOS app development
- Native UI frameworks
- App signing and distribution

## Risk Mitigation

### Technical Risks
1. **M3 Compatibility**: Ensure all libraries have ARM builds
   - Mitigation: Test installation early, use pre-built wheels
   - Note: All major libraries (ChromaDB, sentence-transformers) have ARM support
   
2. **Large Document Processing**: Memory constraints with 16GB
   - Mitigation: Stream processing, batch embeddings
   
3. **Llama Performance**: Slower than cloud APIs (5-15 tokens/sec)
   - Mitigation: Show progress indicators, maybe add streaming output
   - Optional: Use Claude API flag for faster responses when needed
   
4. **Vector DB Performance**: Slow searches with large datasets
   - Mitigation: Start small, optimize later, consider indices

### User Experience Risks
1. **Complex Setup**: Too many dependencies
   - Mitigation: Clear installation guide, automated setup script
   
2. **Unclear Error Messages**: User doesn't know what went wrong
   - Mitigation: Comprehensive error handling with suggestions

## Prerequisites & Setup

### What You Already Have ✅
- ✅ Python 3.13 installed (downgraded from 3.14 for ChromaDB compatibility)
- ✅ Ollama 0.13.0 installed
- ✅ macOS with M3 chip

### What You Need to Install
**Nothing else!** All dependencies will be installed via pip when you set up the project:
- ChromaDB - installed via `pip install chromadb` (no separate installation needed)
- sentence-transformers - installed via pip
- All other Python libraries - installed via pip

ChromaDB is just a Python library, not a separate database server. It runs entirely within your Python application and stores data as files on disk.

### First-Time Model Downloads (Automatic)
When you first run the RAG system, these will download automatically:
1. **Embedding model** (~80MB): `all-MiniLM-L6-v2` downloads on first use
2. **Llama model** (if not already pulled): Run `ollama pull llama3.1:8b` (about 4.7GB)

You can pre-download the Llama model now to save time later:
```bash
ollama pull llama3.1:8b
```

## Next Steps

1. **Verify your setup**
   ```bash
   python --version  # Should show 3.13.x (after activating venv)
   ollama --version  # Should show 0.13.0
   ollama list       # Check if llama3.1:8b is already downloaded
   ```

2. **Create project directory**
   ```bash
   mkdir rag-cli-project
   cd rag-cli-project
   ```

3. **Set up virtual environment**
   ```bash
   python3.13 -m venv venv
   source venv/bin/activate  # On macOS
   ```

4. **Create project structure**
   - Initialize git repository
   - Copy the plan.md into your project folder
   - Create initial folder structure

5. **Install dependencies** (after project structure is created)
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. **Start with Phase 1.1**: Build basic CLI scaffolding
   - Implement `rag-cli init` command
   - Set up configuration loading
   - Add logging infrastructure

7. **Iterate incrementally**: Build one component at a time, test thoroughly

## Questions to Resolve

- [x] Which LLM provider to start with? **Answer: Ollama/Llama (free) as default, Claude as optional upgrade**
- [x] Local vs cloud embeddings? **Answer: Local (sentence-transformers)**
- [ ] Testing approach: manual vs automated priority? **Answer: Manual first, automated later**
- [x] Should we add OCR support for scanned PDFs initially? **Answer: No, add later**
- [ ] Which Llama model size? (3.1:8b is good default, 3.1:70b if you have patience for better quality)

---

**Ready to start coding?** Begin with Phase 1.1 and build incrementally. Each phase should be functional before moving to the next.
