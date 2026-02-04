# RAG CLI

A Retrieval-Augmented Generation (RAG) system for querying your documents using local LLMs. Available as a CLI, REST API, Web UI, or native macOS Desktop App.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Commands](#commands) - CLI reference
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Examples](#examples)
- [Python API](#python-api)
- [Web API](#web-api) - REST API reference
- [Web UI](#web-ui) - Browser-based interface
- [macOS Desktop App](#macos-desktop-app) - Native desktop application
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Document Processing**: Load PDF, Markdown, and HTML files
- **Smart Chunking**: Split documents into overlapping chunks for better retrieval
- **Local Embeddings**: Generate embeddings using sentence-transformers (all-MiniLM-L6-v2)
- **Vector Storage**: Persistent storage with ChromaDB
- **LLM Integration**: Works with Ollama (free, local) or Claude API
- **Multiple Interfaces**:
  - **CLI**: Command-line interface for terminal users
  - **REST API**: FastAPI backend for programmatic access
  - **Web UI**: React-based chat interface in your browser
  - **Desktop App**: Native macOS application with system integration

## Quick Start

### Prerequisites

- **Python 3.12 or 3.13** (Python 3.14 is not yet supported)
  - macOS Apple Silicon (M1/M2/M3): Python 3.12 or 3.13
  - macOS Intel: **Python 3.12 required** (onnxruntime lacks 3.13 wheels for x86_64)
  - Linux/Windows: Python 3.12 or 3.13
- [Ollama](https://ollama.ai) installed (the CLI will start it automatically)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-cli-project

# Create virtual environment
# Use python3.12 on Intel Mac, python3.12 or python3.13 elsewhere
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Install Ollama (if not already installed)
# macOS:
brew install ollama
# Linux:
curl -fsSL https://ollama.ai/install.sh | sh
```

### Basic Usage (CLI)

```bash
# Initialize the RAG system (automatically sets up Ollama and pulls the model)
rag-cli init

# Add documents
rag-cli add ./my_documents/

# Query your documents
rag-cli query "What is machine learning?"

# Interactive chat mode
rag-cli chat
```

### Basic Usage (Web UI)

Prefer a graphical interface? Start the web application with a single command:

```bash
./start-web.sh
```

Then open http://localhost:5173 in your browser. See [Web UI](#web-ui) for more details.

The `init` command automatically:
- Creates necessary directories
- Initializes the vector database
- Starts Ollama if not running
- Downloads the LLM model (~4.7GB) if not already pulled

## Commands

### Global Options

All commands support these global options:

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Show detailed output with INFO-level logs |
| `--debug` | Show debug output (even more detailed than verbose) |
| `--help` | Show help for any command |

**Default behavior:** Clean, minimal output showing only essential information (progress indicators and results).

**Verbose mode:** Shows detailed logs including model loading, database connections, chunk processing, and timing information. Useful for debugging or understanding what's happening internally.

```bash
# Default (clean output)
rag-cli query "What is Python?"

# Verbose (detailed output)
rag-cli --verbose query "What is Python?"
rag-cli -v query "What is Python?"

# Debug (maximum detail)
rag-cli --debug query "What is Python?"
```

### `rag-cli init`

Initialize the RAG system with automatic Ollama setup.

```bash
rag-cli init
```

This command:
1. Creates necessary directories (`data/vector_db`, `data/documents`, `logs`)
2. Initializes the ChromaDB vector store
3. Checks if Ollama is installed
4. Starts Ollama service if not running
5. Downloads the LLM model (`llama3.1:8b`) if not already available

**Options:**
- `--skip-ollama`: Skip automatic Ollama setup (for manual configuration)

### `rag-cli add`

Add documents to the knowledge base.

```bash
# Add a single file
rag-cli add document.pdf

# Add all files in a directory
rag-cli add ./documents/

# Add recursively (include subdirectories)
rag-cli add ./documents/ -r

# Force re-indexing of already indexed files
rag-cli add document.pdf -f

# With verbose output (shows chunking and embedding details)
rag-cli -v add document.pdf
```

**Options:**
- `-r, --recursive`: Recursively process directories
- `-f, --force`: Re-index documents even if already indexed

**Supported formats:**
- PDF (.pdf)
- Markdown (.md)
- HTML (.html, .htm)
- Plain text (.txt)

### `rag-cli query`

Query the knowledge base and get an answer based on indexed documents.

```bash
# Basic query
rag-cli query "What is Python?"

# Retrieve more context chunks
rag-cli query "Explain neural networks" --top-k 10

# Use a different LLM provider
rag-cli query "What is REST?" --llm claude

# Adjust response creativity
rag-cli query "Summarize the document" --temperature 0.2

# Show which source documents were used
rag-cli query "What is Python?" --show-sources

# With verbose output (shows retrieval details)
rag-cli -v query "What is Python?"
```

**Options:**
- `-k, --top-k`: Number of chunks to retrieve (default: 5)
- `-l, --llm`: LLM provider (`ollama` or `claude`, default: ollama)
- `-t, --temperature`: Response creativity (0.0-1.0)
- `-s, --show-sources`: Display source documents used for the answer

**Example output (default):**
```
Query: What is Python?
Using LLM: ollama, retrieving top 5 chunks

╭─────────────────────── Answer ───────────────────────╮
│ Python is a high-level, interpreted programming      │
│ language known for its clear syntax and readability. │
╰──────────────────────────────────────────────────────╯
```

**Example output (verbose):**
```
Query: What is Python?
Using LLM: ollama, retrieving top 5 chunks

╭─────────────────────── Answer ───────────────────────╮
│ Python is a high-level, interpreted programming      │
│ language known for its clear syntax and readability. │
╰──────────────────────────────────────────────────────╯
[10:30:01] INFO  Loading embedding model: all-MiniLM-L6-v2
           INFO  Connected to collection 'rag_documents' (29 documents)
           INFO  Retrieved 5 chunks for query (avg similarity: 0.72)
           INFO  Query complete: 5 chunks, 156 chars
```

### `rag-cli chat`

Start an interactive chat session for asking multiple questions.

```bash
rag-cli chat

# With streaming responses (see answers as they're generated)
rag-cli chat --stream

# Use Claude instead of Ollama
rag-cli chat --llm claude
```

Type `exit`, `quit`, or `q` to end the session.

**Options:**
- `-l, --llm`: LLM provider (`ollama` or `claude`, default: ollama)
- `-s, --stream`: Stream responses as they are generated

### `rag-cli list`

List all indexed documents in the knowledge base.

```bash
rag-cli list
```

Shows a table with document names, file types, and chunk counts. Use this to see what has been indexed or to find document names for the `remove` command.

**Example output:**
```
Indexed Documents (2):

┏━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━┓
┃ #  ┃ Document       ┃ Type ┃ Chunks ┃
┡━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━┩
│ 1  │ guide.pdf      │ pdf  │     15 │
│ 2  │ notes.md       │ md   │      8 │
└────┴────────────────┴──────┴────────┘

Total chunks: 23
```

### `rag-cli remove`

Remove a document from the knowledge base.

```bash
# Remove by filename (as shown in 'rag-cli list')
rag-cli remove document.pdf
```

The document name should match exactly as shown in `rag-cli list`. All chunks associated with the document will be permanently deleted.

### `rag-cli clear`

Clear the entire knowledge base.

```bash
# Preview what will be deleted
rag-cli clear

# Actually delete everything
rag-cli clear --confirm
```

**WARNING:** This permanently deletes ALL indexed documents and chunks. This action cannot be undone.

**Options:**
- `--confirm`: Required flag to actually perform the deletion

### `rag-cli stats`

Show system statistics and current configuration.

```bash
rag-cli stats
```

Displays information about the knowledge base including: document count, total chunks, embedding model, chunk settings, and current LLM configuration.

**Example output:**
```
╭─────────── Stats ───────────╮
│ RAG CLI Statistics          │
╰─────────────────────────────╯
Documents indexed    2
Total chunks         23
Collection           rag_documents

Embedding model      all-MiniLM-L6-v2
Chunk size           800
Chunk overlap        100
LLM provider         ollama
```

### `rag-cli config`

View and manage configuration settings.

```bash
# List all current settings
rag-cli config list

# Get a specific setting
rag-cli config get llm.ollama_model

# Set a configuration value
rag-cli config set llm.ollama_model llama3.1:8b
```

Settings use dot notation (e.g., `llm.ollama_model`, `chunking.chunk_size`). Configuration can also be set via environment variables or `.env` file.

**Subcommands:**
- `list`: Show all current configuration values
- `get <key>`: Get a specific setting value
- `set <key> <value>`: Update a setting

### Getting Help

Use the `--help` flag on any command to see detailed usage information:

```bash
# See all available commands
rag-cli --help

# Get help for a specific command
rag-cli add --help
rag-cli query --help
rag-cli config --help

# Get help for subcommands
rag-cli config list --help
```

## Configuration

Configuration is loaded from environment variables and `.env` files.

### Environment Variables

```bash
# LLM Settings
DEFAULT_LLM_PROVIDER=ollama  # or 'claude'
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.3
MAX_TOKENS=2000

# Embedding Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking Settings
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Retrieval Settings
TOP_K_RESULTS=5

# Vector Store
CHROMA_PERSIST_DIR=./data/vector_db

# Claude API (optional)
ANTHROPIC_API_KEY=your_api_key_here
```

### Example `.env` File

```bash
# Copy from .env.example
cp .env.example .env

# Edit with your settings
nano .env
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      RAG Pipeline                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Document │───▶│ Chunker  │───▶│ Embedder         │  │
│  │ Loader   │    │          │    │ (sentence-trans) │  │
│  └──────────┘    └──────────┘    └────────┬─────────┘  │
│                                           │             │
│                                           ▼             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ LLM      │◀───│ Retriever│◀───│ Vector Store     │  │
│  │ (Ollama) │    │          │    │ (ChromaDB)       │  │
│  └──────────┘    └──────────┘    └──────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| `document_loader.py` | Loads PDF, Markdown, and HTML files |
| `chunker.py` | Splits documents into overlapping chunks |
| `embedder.py` | Generates embeddings using sentence-transformers |
| `vector_store.py` | ChromaDB wrapper for vector storage |
| `retriever.py` | Similarity search and context retrieval |
| `llm_client.py` | Ollama and Claude API integration |
| `prompts.py` | RAG prompt templates |
| `pipeline.py` | End-to-end RAG orchestration |

## Examples

### Example Documents

The `examples/documents/` directory contains sample documents:

- `python_basics.md` - Python programming tutorial
- `machine_learning.md` - ML introduction
- `software_architecture.md` - Architecture patterns

### Try It Out

```bash
# Add example documents
rag-cli add examples/documents/

# Query the examples
rag-cli query "What are the types of machine learning?"
rag-cli query "How do I define a class in Python?"
rag-cli query "What is microservices architecture?"
```

See `examples/sample_queries.txt` for more query ideas.

## Python API

You can also use RAG CLI as a Python library:

```python
from src.pipeline import RAGPipeline, create_pipeline
from src.document_loader import DocumentLoader
from src.chunker import TextChunker
from src.vector_store import VectorStore

# Load and index documents
loader = DocumentLoader()
chunker = TextChunker()
store = VectorStore()

for doc in loader.load_directory("./documents/"):
    chunks = chunker.chunk_document(doc)
    store.add_chunks(chunks)

# Create pipeline and query
pipeline = create_pipeline()
result = pipeline.query("What is Python?")

print(f"Answer: {result.answer}")
print(f"Sources: {result.sources}")
```

### Streaming Responses

```python
pipeline = create_pipeline()

for chunk in pipeline.query_stream("Explain machine learning"):
    print(chunk, end="", flush=True)
```

## Web API

RAG CLI includes a REST API built with FastAPI for web integration.

### Installation

Install the web dependencies in addition to the base requirements:

```bash
pip install -r requirements-web.txt
```

### Starting the API Server

```bash
# Start the development server
uvicorn backend.main:app --reload --port 8000

# Or run directly with Python
python -m backend.main
```

The API will be available at `http://localhost:8000`.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check with Ollama status and document stats |
| GET | `/api/config` | Get configuration settings (from `.env` file) |
| POST | `/api/query` | Query the knowledge base with a question |
| POST | `/api/upload` | Upload and index a document |
| GET | `/api/documents` | List all indexed documents |
| DELETE | `/api/documents/{id}` | Delete a document from the knowledge base |

### Example: Get Configuration

```bash
curl http://localhost:8000/api/config
```

Response:
```json
{
  "llm_provider": "ollama",
  "llm_model": "llama3.1:8b",
  "top_k": 8,
  "temperature": 0.3,
  "chunk_size": 1200,
  "chunk_overlap": 200,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

### Example: Query the Knowledge Base

```bash
# Query with default settings (from .env config)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'

# Query with custom settings (override defaults)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "top_k": 5, "temperature": 0.7}'
```

### Example: Upload a Document

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

### API Documentation

FastAPI provides auto-generated interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### CORS Configuration

The API is configured to allow requests from common local development ports:
- `http://localhost:5173` (Vite)
- `http://localhost:3000` (Create React App)

## Web UI

RAG CLI includes a React-based web interface for a chat-style experience in your browser.

### Prerequisites

- Node.js 18+ (install via `brew install node` on macOS)
- Backend dependencies installed (`pip install -r requirements-web.txt`)

### Starting the Web Application

Use the provided script to start both backend and frontend with a single command:

```bash
./start-web.sh
```

This will:
- Start the FastAPI backend on port 8000
- Start the React frontend on port 5173
- Display URLs for the Web UI and API docs

**Open the Web UI:** http://localhost:5173

### Stopping the Web Application

```bash
./stop-web.sh
```

### Viewing Logs

The scripts run services in the background. To view logs:

```bash
# View backend logs
tail -f logs/backend.log

# View frontend logs
tail -f logs/frontend.log
```

### Web UI Features

**Chat Interface:**
- Ask questions in a familiar chat-style layout
- Full chat history persisted during your session
- Real-time loading indicators with animated "Thinking..." state
- Expandable sources section shows which documents were used
- View model used and processing time for each response
- Timestamps on all messages
- Clear chat button to start fresh

**Rich Response Rendering:**
- **Markdown support**: Headings, lists, bold, italic, links, blockquotes
- **Code syntax highlighting**: Automatic language detection with support for Python, JavaScript, TypeScript, Bash, JSON, SQL, YAML, HTML/CSS, and more
- Dark theme code blocks with copy button
- Inline code styling

**Copy & Export:**
- **Copy answers**: Hover over any response to reveal a copy button
- **Copy code**: Each code block has its own copy button
- **Export chat**: Download your entire conversation as:
  - **Text file** (.txt) - Human-readable format with timestamps
  - **JSON file** (.json) - Structured data for programmatic use

**Settings Panel:**
- Toggle between **Ollama** (free, local) and **Claude** (API) providers
- Adjust **top_k** parameter (1-20 context chunks)
- Adjust **temperature** (0-1 for response creativity)
- Settings persist during your session
- Visual badges show current configuration

**Document Upload:**
- Drag-and-drop file upload directly in the browser
- Click to browse and select files
- Supports PDF, Markdown (.md), and HTML files
- Upload progress indicators with success/error feedback
- Documents are immediately available for querying after upload

**Document Management:**
- View all indexed documents in the sidebar
- See file type, name, and chunk count for each document
- Delete documents with one click (with confirmation)
- Document count badge shows total indexed documents

**Notifications & Error Handling:**
- Toast notifications for upload success/failure
- Error banner with clear messaging and dismiss button
- Graceful handling of network errors and API failures
- Auto-dismiss or manual dismiss options

**Polished UX:**
- Smooth animations and transitions throughout
- Responsive design works on desktop and mobile
- Keyboard shortcuts for efficient navigation
- User and Assistant avatars for visual distinction
- Dark mode support for comfortable viewing

**Dark Mode:**
- Toggle between light and dark themes using the sun/moon switch in the header
- Your preference is automatically saved and persists across sessions
- Detects system preference on first visit (follows your OS setting)
- Smooth transitions when switching themes
- All UI components adapt: chat, settings, sidebar, notifications
- Code blocks use a dark theme in both modes for optimal readability

### Quick Start with Web UI

1. Run `./start-web.sh`
2. Open http://localhost:5173
3. **Upload documents**: Drag files onto the upload zone in the sidebar (or click to browse)
4. **Ask questions**: Type in the chat box and press Enter
5. View answers with source citations
6. When done, run `./stop-web.sh`

**Tip:** You can also pre-index documents via CLI (`rag-cli add ./documents/`) before starting the web UI.

### Settings Panel

Click the **Settings** button in the header to configure query parameters:

**LLM Provider:**
- **Ollama** (default) - Free, runs locally on your machine
- **Claude** - Requires `ANTHROPIC_API_KEY` environment variable

**Context Chunks (top_k):**
- Controls how many document chunks are retrieved for context
- Range: 1-20 (default: 5)
- Higher values provide more context but may slow responses

**Temperature:**
- Controls response creativity/randomness
- Range: 0.0-1.0 (default: 0.7)
- Lower = more focused and deterministic
- Higher = more creative and varied

Settings are displayed as badges in the panel footer and persist during your session.

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line in message |

### Tips for Efficient Use

- **Use the sidebar** to quickly see what documents are indexed
- **Adjust top_k** based on your needs: lower for quick answers, higher for comprehensive responses
- **Export conversations** before clearing chat if you want to save them
- **Check the sources** to verify answer accuracy
- **Copy code blocks** directly with the dedicated copy button
- **Toggle providers** to compare Ollama vs Claude responses
- **Enable dark mode** for comfortable viewing during extended sessions or low-light environments

### Supported File Types

The web UI accepts the same file types as the CLI:

| Format | Extensions |
|--------|------------|
| PDF | `.pdf` |
| Markdown | `.md`, `.markdown` |
| HTML | `.html`, `.htm` |

### Manual Startup (Alternative)

If you prefer to run services manually in separate terminals:

**Terminal 1 - Backend:**
```bash
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## macOS Desktop App

RAG Assistant is also available as a native macOS desktop application, built with [Tauri](https://tauri.app/) for a lightweight, high-performance experience.

### Key Features

- **Native macOS Integration**: Full menu bar, system notifications, and dock integration
- **Keyboard Shortcuts**: Cmd+N (new chat), Cmd+K (focus search), Cmd+Shift+D (dark mode)
- **File Associations**: Double-click .txt, .md, or .pdf files to open with RAG Assistant
- **Lightweight**: 5.4MB app bundle, 2.9MB DMG installer
- **Offline-First**: All processing happens locally on your machine

### Installation

1. Download `RAG Assistant_0.1.0_aarch64.dmg` from releases
2. Open the DMG and drag RAG Assistant to Applications
3. On first launch, right-click and select "Open" (required for unsigned apps)

**Prerequisites**: Python 3.12 or 3.13 (in a virtual environment), [Ollama](https://ollama.ai), project dependencies, and the backend running via `./start-web.sh`.

### Quick Start

1. Set up the project with Python 3.12:
   ```bash
   cd /path/to/rag-cli-project
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Ensure Ollama is running (`ollama serve`)
3. Start the backend:
   ```bash
   ./start-web.sh
   ```
4. Double-click RAG Assistant in Applications
5. Upload documents via the sidebar
6. Start asking questions!

**Note**: The desktop app connects to the backend on port 8000. Documents indexed via the desktop app, Web UI, or CLI are shared across all interfaces.

### Documentation

- **User Guide**: See [desktop/README.md](desktop/README.md) for full documentation
- **Build Instructions**: See [desktop/BUILDING.md](desktop/BUILDING.md) to build from source

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage (terminal report)
pytest tests/ --cov=src --cov-report=term

# Run with coverage (HTML report)
pytest tests/ --cov=src --cov-report=html

# View the HTML coverage report
open htmlcov/index.html
```

**Note:** Coverage reporting requires `pytest-cov`, which is included in `requirements.txt`.

### Linting

```bash
# Check code style
ruff check src/ backend/ tests/

# Auto-fix issues
ruff check src/ backend/ tests/ --fix
```

### Project Structure

```
rag-cli-project/
├── src/                    # Core RAG source code
│   ├── __init__.py
│   ├── cli.py             # CLI commands
│   ├── config.py          # Configuration
│   ├── document_loader.py # Document loading
│   ├── chunker.py         # Text chunking
│   ├── embedder.py        # Embedding generation
│   ├── vector_store.py    # ChromaDB integration
│   ├── retriever.py       # Similarity search
│   ├── llm_client.py      # LLM integration
│   ├── prompts.py         # Prompt templates
│   ├── pipeline.py        # RAG pipeline
│   ├── exceptions.py      # Custom exceptions
│   ├── progress.py        # Progress indicators
│   └── logger.py          # Logging
├── backend/                # FastAPI backend
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── api/
│   │   ├── models.py      # Pydantic models
│   │   └── routes.py      # API routes
│   └── services/
│       └── rag_service.py # RAG service layer
├── frontend/               # React web UI & Tauri desktop app
│   ├── package.json       # Node.js dependencies
│   ├── vite.config.js     # Vite configuration
│   ├── index.html         # HTML entry point
│   ├── src/               # React source code
│   │   ├── App.jsx        # Main React component
│   │   ├── components/    # UI components
│   │   │   ├── ChatInterface.jsx   # Main chat container
│   │   │   ├── MessageList.jsx     # Message display with markdown/code
│   │   │   ├── MessageInput.jsx    # Input field with send button
│   │   │   ├── SettingsPanel.jsx   # LLM and query settings
│   │   │   ├── Sidebar.jsx         # Document management sidebar
│   │   │   ├── DocumentList.jsx    # Indexed document list
│   │   │   ├── DocumentUpload.jsx  # Drag-and-drop upload
│   │   │   └── Notification.jsx    # Toast notifications
│   │   └── services/
│   │       └── api.js     # Backend API client
│   └── src-tauri/         # Tauri/Rust desktop app
│       ├── Cargo.toml     # Rust dependencies
│       ├── tauri.conf.json # App configuration
│       └── src/           # Rust source code
├── desktop/                # Desktop app documentation
│   ├── README.md          # User guide
│   └── BUILDING.md        # Build instructions
├── tests/                  # Test files
├── examples/               # Example documents
├── data/                   # Data storage
│   ├── documents/         # Source documents
│   └── vector_db/         # ChromaDB storage
├── logs/                   # Log files
├── requirements.txt        # Core dependencies
├── requirements-web.txt    # Web API dependencies
├── start-web.sh           # Start web application
├── stop-web.sh            # Stop web application
├── pyproject.toml         # Project config
└── README.md              # This file
```

## Troubleshooting

### General Debugging Tip

If you encounter any issues, run commands with the `--verbose` flag to see detailed logs:

```bash
rag-cli -v add document.pdf
rag-cli -v query "your question"
rag-cli --debug init  # Even more detail
```

This will show model loading, database connections, chunk processing, and other internal operations that can help identify problems.

### Ollama Not Installed

```
Ollama is not installed.
```

**Solution:** Install Ollama before running `rag-cli init`:

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

After installing, run `rag-cli init` again - it will automatically start Ollama and pull the required model.

### Ollama Service Won't Start

```
Could not start Ollama service
```

**Solution:** Try starting Ollama manually:

```bash
ollama serve
```

If you see port conflicts, another instance may be running. Check with:
```bash
lsof -i :11434
```

### Model Download Failed

```
Failed to pull model: ...
```

**Solution:** Pull the model manually:

```bash
ollama pull llama3.1:8b
```

Ensure you have sufficient disk space (~4.7GB for llama3.1:8b).

### No Documents Indexed

```
Error: No documents have been indexed yet.
```

**Solution:** Add documents with `rag-cli add <path>`

### Memory Issues with Large Documents

For large documents, try:
- Reducing chunk size in config
- Processing documents in smaller batches
- Using a machine with more RAM

### Installation Fails with onnxruntime Error

```
ERROR: ResolutionImpossible ... onnxruntime
```

**Problem:** The `onnxruntime` package (required by ChromaDB) doesn't have pre-built wheels for your Python version and platform combination. This commonly affects macOS Intel users with Python 3.13.

**Solutions:**

1. **Use Python 3.12** (recommended for Intel Mac):
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Use conda** (works on any platform):
   ```bash
   conda create -n rag-cli python=3.12
   conda activate rag-cli
   pip install -r requirements.txt
   ```

### Web UI Issues

#### Port Already in Use

```
error: address already in use
```

**Solution:** Another process is using port 8000 (backend) or 5173 (frontend).

```bash
# Find what's using the port
lsof -i :8000
lsof -i :5173

# Kill the process or use a different port
uvicorn backend.main:app --port 8001  # Use different backend port
```

#### Backend Fails with "No module named 'fastapi'"

```
ModuleNotFoundError: No module named 'fastapi'
```

**Problem:** The web dependencies are not installed. The base `requirements.txt` only includes CLI dependencies.

**Solution:** Install the web dependencies:

```bash
pip install -r requirements-web.txt
```

#### CORS Errors in Browser Console

```
Access to fetch blocked by CORS policy
```

**Solution:** Make sure the backend is running on port 8000. The CORS configuration expects the backend at `http://localhost:8000`. If using a different port, update `frontend/src/services/api.js`.

#### Frontend Can't Connect to Backend

**Solution:** Verify both servers are running:

```bash
# Check backend is responding
curl http://localhost:8000/api/health

# Check frontend is serving
curl http://localhost:5173
```

#### Node.js Not Found

```
npm: command not found
```

**Solution:** Install Node.js:

```bash
# macOS
brew install node

# Linux (Ubuntu/Debian)
sudo apt install nodejs npm

# Or use nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install node
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [ChromaDB](https://www.trychroma.com/) for vector storage
- [sentence-transformers](https://www.sbert.net/) for embeddings
- [Ollama](https://ollama.ai/) for local LLM inference
- [Rich](https://rich.readthedocs.io/) for terminal formatting
- [Click](https://click.palletsprojects.com/) for CLI framework
- [Tauri](https://tauri.app/) for native desktop app framework
- [FastAPI](https://fastapi.tiangolo.com/) for REST API backend
