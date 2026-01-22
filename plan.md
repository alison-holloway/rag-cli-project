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
- **Status**: Pending implementation

## Phase 2: Web UI (Future)
- [ ] Design API backend (FastAPI)
- [ ] Create REST endpoints
- [ ] Build React or Vue.js frontend
- [ ] Document upload interface
- [ ] Chat interface with history
- [ ] Deployment considerations

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
