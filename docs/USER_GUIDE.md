# RAG CLI User Guide

Welcome to RAG CLI! This guide will help you get started with using the application to ask questions about your documents.

## Table of Contents

- [What is RAG?](#what-is-rag)
- [Getting Started](#getting-started)
- [CLI Interface](#cli-interface)
- [Web UI Interface](#web-ui-interface)
- [Examples](#examples)
- [Downloading Documentation](#downloading-documentation)
  - [HTML Scraper Tool](#html-scraper-tool)
  - [DITA Documentation Chunker](#dita-documentation-chunker)
- [Troubleshooting](#troubleshooting)
- [Quick Reference Card](#quick-reference-card)

---

## What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. It's a technique that combines the power of AI language models with your own documents to provide accurate, relevant answers to your questions.

Here's how it works: When you ask a question, RAG CLI searches through your documents to find the most relevant sections, then sends those sections along with your question to an AI model. The AI uses this context to generate an answer that's grounded in your actual documents, rather than making things up.

This is particularly useful for:
- Searching through technical documentation
- Getting answers from company knowledge bases
- Querying research papers or reports
- Finding information in large collections of documents

RAG CLI runs entirely on your computer (when using Ollama), so your documents stay private and secure.

---

## Getting Started

### Prerequisites

Before you begin, make sure you have:

1. **Python 3.12 or 3.13** installed on your computer
   - Check with: `python3 --version`
   - macOS Intel users: **Use Python 3.12** (some dependencies lack 3.13 support for x86_64)
   - macOS Apple Silicon / Linux / Windows: Python 3.12 or 3.13
   - Note: Python 3.14 is not yet supported

2. **Ollama** - A free, local AI model runner
   - Download from: https://ollama.ai
   - Or install with Homebrew (macOS): `brew install ollama`

3. **Node.js 18+** (only needed for the Web UI)
   - Download from: https://nodejs.org
   - Or install with Homebrew (macOS): `brew install node`

### Installation

1. **Download the project** from GitHub:
   ```bash
   git clone https://github.com/your-username/rag-cli-project.git
   cd rag-cli-project
   ```

2. **Create a Python virtual environment**:
   ```bash
   # Use python3.12 on Intel Mac, python3.12 or python3.13 elsewhere
   python3.12 -m venv venv
   source venv/bin/activate
   ```

   On Windows, use:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the CLI tool**:
   ```bash
   pip install -e .
   ```

### First-Time Setup

Run the initialization command to set up everything:

```bash
rag-cli init
```

This will:
- Create the necessary folders for storing data
- Set up the vector database for document search
- Start Ollama (if not running)
- Download the AI model (~4.7GB) if not already available

You'll see progress messages as each step completes. Once done, you're ready to start adding documents!

---

## CLI Interface

The command-line interface is perfect for quick tasks, scripting, and users who prefer working in the terminal.

### Starting the CLI

After installation, you can use the `rag-cli` command from any terminal:

```bash
rag-cli --help
```

This shows all available commands and options.

### Loading Documents

Add documents to your knowledge base with the `add` command:

```bash
# Add a single file
rag-cli add document.pdf

# Add all documents in a folder
rag-cli add ./my_documents/

# Add documents recursively (including subfolders)
rag-cli add ./documents/ -r

# Force re-index a document that's already added
rag-cli add document.pdf -f
```

**Supported file types:**
- PDF files (`.pdf`)
- Markdown files (`.md`)
- HTML files (`.html`, `.htm`)
- Plain text files (`.txt`)

### Configuration Options

View and manage settings with the `config` command:

```bash
# See all current settings
rag-cli config list

# Get a specific setting
rag-cli config get llm.ollama_model
```

**Available Settings:**

| Setting | Description | Default |
|---------|-------------|---------|
| `llm.default_provider` | AI provider (ollama or claude) | `ollama` |
| `llm.ollama_url` | Ollama server address | `http://localhost:11434` |
| `llm.ollama_model` | Which Ollama model to use | `llama3.1:8b` |
| `llm.temperature` | Creativity level (0.0-2.0) | `0.3` |
| `llm.max_tokens` | Maximum response length | `2000` |
| `embedding.model` | Model for document search | `all-MiniLM-L6-v2` |
| `chunking.chunk_size` | Document chunk size | `1200` |
| `chunking.chunk_overlap` | Overlap between chunks | `200` |
| `retrieval.top_k` | Number of chunks to retrieve | `5` |
| `vector_store.path` | Database storage location | `./data/vector_db` |
| `logging.level` | Log verbosity | `INFO` |

To change settings, create a `.env` file in the project folder:

```bash
# Example .env file
OLLAMA_MODEL=llama3.2
LLM_TEMPERATURE=0.5
TOP_K_RESULTS=10
```

### Querying Documents

Ask questions about your documents:

```bash
# Basic query
rag-cli query "What is machine learning?"

# Get more context (retrieve 10 chunks instead of 5)
rag-cli query "Explain the installation process" --top-k 10

# Use Claude instead of Ollama (requires API key)
rag-cli query "Summarize the main points" --llm claude

# Show which documents were used for the answer
rag-cli query "How do I configure the settings?" --show-sources

# Adjust response creativity (0.0 = focused, 1.0 = creative)
rag-cli query "What are the benefits?" --temperature 0.2
```

### Interactive Chat Mode

For back-and-forth conversations:

```bash
# Start chat mode
rag-cli chat

# Chat with streaming responses (see answers as they're generated)
rag-cli chat --stream

# Chat using Claude
rag-cli chat --llm claude
```

In chat mode:
- Type your questions and press Enter
- Type `exit`, `quit`, or `q` to leave
- Press `Ctrl+C` to exit immediately

### Managing Documents

**List all indexed documents:**
```bash
rag-cli list
```

This shows a table of all documents with their file type and number of chunks.

**Remove a specific document:**
```bash
rag-cli remove document.pdf
```

Use the exact filename shown in `rag-cli list`.

**Clear all documents:**
```bash
# Preview what will be deleted
rag-cli clear

# Actually delete everything
rag-cli clear --confirm
```

**View statistics:**
```bash
rag-cli stats
```

Shows document count, chunk count, and current settings.

### Verbose and Debug Modes

For troubleshooting or seeing what's happening:

```bash
# Verbose mode - shows detailed progress
rag-cli -v query "What is Python?"

# Debug mode - shows maximum detail
rag-cli --debug add document.pdf
```

---

## Web UI Interface

The Web UI provides a friendly chat interface in your browser, perfect for users who prefer a graphical experience.

### Starting the Web UI

1. **Make sure Ollama is running:**
   ```bash
   ollama serve
   ```

2. **Start the web application:**
   ```bash
   ./start-web.sh
   ```

3. **Open your browser** to: http://localhost:5173

[Screenshot: Main chat interface]

### Uploading Documents

1. Click on the **sidebar toggle** (hamburger menu) to open the sidebar
2. Find the **Upload Documents** section
3. Either:
   - **Drag and drop** files onto the upload area, or
   - **Click** the upload area to browse for files
4. Wait for the upload progress to complete
5. You'll see a success notification when done

[Screenshot: Uploading a document]

**Supported file types:** PDF, Markdown (.md), HTML, and plain text (.txt)

### Configuration Panel

Click the **Settings** button in the header to adjust query parameters:

[Screenshot: Settings panel]

**LLM Provider:**
- **Ollama** (default) - Free, runs locally on your computer
- **Claude** - Requires an Anthropic API key (set in `.env` file)

**Context Chunks (top_k):**
- Slider from 1 to 20
- Higher values = more context for answers, but slower
- Default: 5

**Temperature:**
- Slider from 0.0 to 1.0
- Lower = more focused, factual answers
- Higher = more creative, varied answers
- Default: 0.7

Settings are saved during your session and shown as badges below the sliders.

### Asking Questions

1. Type your question in the **chat input** at the bottom
2. Press **Enter** or click the **Send** button
3. Wait for the "Thinking..." indicator
4. Read the response, which appears in the chat area

[Screenshot: Asking a question]

**Tips:**
- Be specific in your questions for better answers
- Click **Show Sources** on any response to see which documents were used
- Use the **Copy** button to copy responses to your clipboard

### Managing Documents

In the sidebar, you can:

1. **View all documents** - See a list of indexed documents with chunk counts
2. **Delete a document** - Click the trash icon next to any document
3. **See total count** - The badge shows how many documents are indexed

[Screenshot: Document list in sidebar]

### Additional Features

**Dark Mode:**
- Toggle the sun/moon icon in the header
- Your preference is saved automatically

**Export Chat:**
- Press `Cmd+Shift+E` (Mac) or `Ctrl+Shift+E` (Windows)
- Download as text or JSON format

**Keyboard Shortcuts:**

| Shortcut | Action |
|----------|--------|
| `Cmd/Ctrl + N` | New chat |
| `Cmd/Ctrl + K` | Focus search input |
| `Cmd/Ctrl + \` | Toggle sidebar |
| `Cmd/Ctrl + Shift + D` | Toggle dark mode |
| `Cmd/Ctrl + Shift + E` | Export chat |
| `Enter` | Send message |
| `Shift + Enter` | New line in message |

### Stopping the Web UI

Run the stop script:

```bash
./stop-web.sh
```

This cleanly shuts down both the backend API and frontend servers.

---

## Examples

### CLI Example: Complete Workflow

Here's a complete example of adding documents and querying them:

```bash
# Step 1: Initialize the project (first time only)
rag-cli init

# Step 2: Add some documents
rag-cli add ./technical-docs/ -r
# Output: Processed 15 document(s), 234 chunks added

# Step 3: List what was added
rag-cli list
# Shows table of all documents

# Step 4: Ask a question
rag-cli query "How do I install the software?"
# Output: Displays answer based on your documents

# Step 5: Ask a follow-up with more context
rag-cli query "What are the system requirements?" --top-k 10 --show-sources

# Step 6: Start an interactive session for multiple questions
rag-cli chat
# You: What configuration options are available?
# Assistant: [answer based on your docs]
# You: How do I change the default settings?
# Assistant: [answer based on your docs]
# You: exit
```

### Web UI Example: Step-by-Step Workflow

1. **Start the application:**
   ```bash
   ./start-web.sh
   ```

2. **Open your browser** to http://localhost:5173

3. **Upload documents:**
   - Open the sidebar (click hamburger menu)
   - Drag your PDF or Markdown files to the upload area
   - Wait for confirmation

   [Screenshot: Document upload success]

4. **Adjust settings (optional):**
   - Click the Settings gear icon
   - Increase "Context Chunks" to 10 for more thorough answers
   - Click outside the panel to close

5. **Ask your first question:**
   - Type: "What are the main topics covered in these documents?"
   - Press Enter
   - Read the AI-generated response

   [Screenshot: Chat response with sources]

6. **Continue the conversation:**
   - Ask follow-up questions
   - Click "Show Sources" to verify information
   - Use the Copy button to save important answers

7. **When finished:**
   ```bash
   ./stop-web.sh
   ```

---

## Downloading Documentation

RAG CLI includes a tool for downloading HTML documentation from websites. This is useful for building a knowledge base from online documentation.

### HTML Scraper Tool

The HTML scraper downloads documentation from Table of Contents (TOC) pages and saves the HTML files locally for indexing.

**Basic usage:**

```bash
# Download using the default configuration
python tools/html_scraper.py

# Preview what would be downloaded (without actually downloading)
python tools/html_scraper.py --dry-run

# See detailed progress
python tools/html_scraper.py --verbose
```

**Configuration:**

Edit `config/html_scraper.yaml` to specify which documentation to download:

```yaml
# Where to save downloaded HTML files
output_dir: data/documents/html/

# Delay between requests (be respectful to servers)
delay_seconds: 1.0

# TOC pages to process
sources:
  - https://docs.example.com/guide/toc.htm
  - https://docs.example.com/reference/toc.htm

# URL patterns to skip
skip_patterns:
  - index.htm
  - toc.htm
  - copyright
```

**After downloading, index the files:**

```bash
# Add downloaded HTML files to RAG
rag-cli add data/documents/html/ -r

# Verify they were indexed
rag-cli list
```

Now you can query the documentation:

```bash
rag-cli query "How do I configure X?"
```

### DITA Documentation Chunker

If your HTML documentation was generated from DITA source (common for enterprise technical documentation), use the DITA semantic chunker for much better query results.

**Why use the DITA chunker?**

Standard chunking splits documents at arbitrary character boundaries, which can break up procedural steps mid-instruction. The DITA chunker understands document structure:

- **Task documents**: Keeps all steps together as a complete procedure
- **Concept documents**: Preserves explanations as coherent units
- **Reference documents**: Intelligently splits syntax/parameters from examples
- **Topic documents**: Chunks at section boundaries

**Basic usage:**

```bash
# Ingest DITA HTML files with semantic chunking
python tools/ingest_dita_docs.py

# Preview what would be created (without storing)
python tools/ingest_dita_docs.py --dry-run

# Clear existing documents and re-ingest fresh
python tools/ingest_dita_docs.py --clear-first

# See detailed progress
python tools/ingest_dita_docs.py --verbose
```

**Configuration:**

Edit `config/dita_chunker.yaml`:

```yaml
# Where to find DITA HTML files
input_dir: data/documents/html/

# Chunk size limits (characters)
min_chunk_size: 200
max_chunk_size: 4000
target_chunk_size: 1500
```

**Results:**

The DITA chunker typically produces ~70% fewer chunks than standard chunking while maintaining complete, meaningful content units. This means:

- Queries return complete procedures, not fragments
- Better answer quality from the AI
- Less noise in search results

For technical details, see [docs/dita_chunker.md](docs/dita_chunker.md).

---

## Troubleshooting

### Common Issues

#### "Ollama is not running"

**Problem:** You see an error about Ollama not being available.

**Solution:**
1. Start Ollama in a terminal:
   ```bash
   ollama serve
   ```
2. Keep that terminal open while using RAG CLI

#### "No documents indexed yet"

**Problem:** Queries return an error saying no documents are indexed.

**Solution:**
Add some documents first:
```bash
rag-cli add /path/to/your/documents/
```

#### "Model not found"

**Problem:** Ollama can't find the required AI model.

**Solution:**
Pull the model manually:
```bash
ollama pull llama3.1:8b
```

#### Web UI shows blank page

**Problem:** The browser shows nothing when opening http://localhost:5173

**Solution:**
1. Check if services are running: `./stop-web.sh` then `./start-web.sh`
2. Check the logs:
   ```bash
   tail -f logs/backend.log
   tail -f logs/frontend.log
   ```

#### "Port already in use"

**Problem:** Can't start services because ports 8000 or 5173 are busy.

**Solution:**
1. Stop any existing services: `./stop-web.sh`
2. If that doesn't work, find and kill the processes:
   ```bash
   lsof -i :8000
   lsof -i :5173
   kill <PID>
   ```

#### Slow responses

**Problem:** Answers take a very long time to generate.

**Solutions:**
- Use a smaller model: Set `OLLAMA_MODEL=llama3.2` in your `.env` file
- Reduce context: Use `--top-k 3` instead of higher values
- Ensure Ollama has enough memory (close other applications)

#### Python version issues

**Problem:** Errors about Python compatibility or ChromaDB.

**Solution:**
Make sure you're using Python 3.12 (recommended) or 3.13:
```bash
# Check your version
python --version

# If wrong, recreate the virtual environment
rm -rf venv
python3.12 -m venv venv  # Use 3.12 for best compatibility
source venv/bin/activate
pip install -r requirements.txt
```

**Note:** macOS Intel users must use Python 3.12 (onnxruntime lacks 3.13 wheels for x86_64).

### Where to Find Logs

- **CLI logs:** Run commands with `-v` (verbose) or `--debug` flags
- **Web UI backend logs:** `logs/backend.log`
- **Web UI frontend logs:** `logs/frontend.log`

View logs in real-time:
```bash
tail -f logs/backend.log
```

### Getting Help

If you encounter issues not covered here:

1. **Check the main README** for technical details
2. **Search existing issues** on GitHub
3. **Open a new issue** at https://github.com/anthropics/claude-code/issues

When reporting issues, include:
- What you were trying to do
- The exact error message
- Your operating system
- Output of `rag-cli --version`
- Relevant log files

---

## Quick Reference Card

### CLI Commands

| Command | Description |
|---------|-------------|
| `rag-cli init` | Set up the project |
| `rag-cli add <path>` | Add documents |
| `rag-cli query "question"` | Ask a question |
| `rag-cli chat` | Interactive chat mode |
| `rag-cli list` | List all documents |
| `rag-cli remove <name>` | Remove a document |
| `rag-cli clear --confirm` | Delete all documents |
| `rag-cli stats` | Show statistics |
| `rag-cli config list` | Show all settings |

### Web UI URLs

| URL | Description |
|-----|-------------|
| http://localhost:5173 | Main Web UI |
| http://localhost:8000/docs | API Documentation |

### Scripts

| Script | Description |
|--------|-------------|
| `./start-web.sh` | Start the Web UI |
| `./stop-web.sh` | Stop the Web UI |
