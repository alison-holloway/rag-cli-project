# RAG CLI

A command-line Retrieval-Augmented Generation (RAG) system for querying your documents using local LLMs.

## Features

- **Document Processing**: Load PDF, Markdown, and HTML files
- **Smart Chunking**: Split documents into overlapping chunks for better retrieval
- **Local Embeddings**: Generate embeddings using sentence-transformers (all-MiniLM-L6-v2)
- **Vector Storage**: Persistent storage with ChromaDB
- **LLM Integration**: Works with Ollama (free, local) or Claude API
- **Interactive CLI**: Easy-to-use command-line interface
- **REST API**: FastAPI backend for web integration

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed (the CLI will start it automatically)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-cli-project

# Create virtual environment
python -m venv venv
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

### Basic Usage

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
RAG_DEFAULT_LLM_PROVIDER=ollama  # or 'claude'
RAG_OLLAMA_MODEL=llama3.1:8b
RAG_OLLAMA_BASE_URL=http://localhost:11434
RAG_LLM_TEMPERATURE=0.7
RAG_MAX_TOKENS=1000

# Embedding Settings
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking Settings
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=100

# Retrieval Settings
RAG_TOP_K_RESULTS=5

# Vector Store
RAG_VECTOR_DB_PATH=./data/vector_db

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
| POST | `/api/query` | Query the knowledge base with a question |
| POST | `/api/upload` | Upload and index a document |
| GET | `/api/documents` | List all indexed documents |
| DELETE | `/api/documents/{id}` | Delete a document from the knowledge base |

### Example: Query the Knowledge Base

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "top_k": 5}'
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

> **Note:** This is Phase 2.1 of the web interface. Phase 2.2 will add a React frontend.

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

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
├── tests/                  # Test files
├── examples/               # Example documents
├── data/                   # Data storage
│   ├── documents/         # Source documents
│   └── vector_db/         # ChromaDB storage
├── logs/                   # Log files
├── requirements.txt        # Core dependencies
├── requirements-web.txt    # Web API dependencies
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

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [ChromaDB](https://www.trychroma.com/) for vector storage
- [sentence-transformers](https://www.sbert.net/) for embeddings
- [Ollama](https://ollama.ai/) for local LLM inference
- [Rich](https://rich.readthedocs.io/) for terminal formatting
- [Click](https://click.palletsprojects.com/) for CLI framework
