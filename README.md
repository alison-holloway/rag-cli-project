# RAG CLI

A command-line Retrieval-Augmented Generation (RAG) system for querying your documents using local LLMs.

## Features

- **Document Processing**: Load PDF, Markdown, and HTML files
- **Smart Chunking**: Split documents into overlapping chunks for better retrieval
- **Local Embeddings**: Generate embeddings using sentence-transformers (all-MiniLM-L6-v2)
- **Vector Storage**: Persistent storage with ChromaDB
- **LLM Integration**: Works with Ollama (free, local) or Claude API
- **Interactive CLI**: Easy-to-use command-line interface

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

# Add recursively
rag-cli add ./documents/ --recursive

# Force re-indexing
rag-cli add document.pdf --force
```

**Supported formats:**
- PDF (.pdf)
- Markdown (.md)
- HTML (.html, .htm)
- Plain text (.txt)

### `rag-cli query`

Query the knowledge base.

```bash
# Basic query
rag-cli query "What is Python?"

# Retrieve more context
rag-cli query "Explain neural networks" --top-k 10

# Use a different LLM provider
rag-cli query "What is REST?" --provider claude

# Adjust creativity
rag-cli query "Summarize the document" --temperature 0.2
```

**Options:**
- `--top-k`: Number of chunks to retrieve (default: 5)
- `--provider`: LLM provider (`ollama` or `claude`)
- `--temperature`: Response creativity (0.0-1.0)

### `rag-cli chat`

Start an interactive chat session.

```bash
rag-cli chat

# With streaming responses
rag-cli chat --stream
```

### `rag-cli list`

List indexed documents.

```bash
rag-cli list
```

### `rag-cli remove`

Remove documents from the knowledge base.

```bash
# Remove by filename
rag-cli remove document.pdf

# Remove all documents
rag-cli clear --confirm
```

### `rag-cli stats`

Show system statistics.

```bash
rag-cli stats
```

### `rag-cli config`

View or update configuration.

```bash
# Show current config
rag-cli config show

# Update a setting
rag-cli config set llm.ollama_model llama3.1:8b
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
ruff check src/ tests/

# Auto-fix issues
ruff check src/ tests/ --fix
```

### Project Structure

```
rag-cli-project/
├── src/                    # Source code
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
├── tests/                  # Test files
├── examples/               # Example documents
├── data/                   # Data storage
│   ├── documents/         # Source documents
│   └── vector_db/         # ChromaDB storage
├── logs/                   # Log files
├── requirements.txt        # Dependencies
├── pyproject.toml         # Project config
└── README.md              # This file
```

## Troubleshooting

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
