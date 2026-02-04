# RAG Assistant Desktop App

A native macOS desktop application for the RAG Assistant, providing local document intelligence powered by Retrieval-Augmented Generation.

## Features

- **Local Processing**: All documents are processed locally - no data leaves your computer
- **Native macOS Integration**: Full menu bar, keyboard shortcuts, and system notifications
- **Document Support**: PDF, Markdown, HTML, and plain text files
- **Dark Mode**: Automatic theme switching with manual toggle option
- **Powered by Ollama**: Uses local LLM for AI-powered document querying

## System Requirements

- macOS 10.15 (Catalina) or later
- Apple Silicon (M1/M2/M3) or Intel processor
- Python 3.12 or 3.13 (Python 3.14 is not yet supported)
  - **Intel Mac users**: Use Python 3.12 (onnxruntime lacks 3.13 wheels for x86_64)
- Ollama installed and running

## Installation

### From DMG (Recommended)

1. Download the latest `RAG Assistant_x.x.x_aarch64.dmg` from the releases
2. Open the DMG file
3. Drag "RAG Assistant" to your Applications folder
4. On first launch, right-click and select "Open" to bypass Gatekeeper (unsigned app)

### Prerequisites

Before running the app, ensure you have:

1. **Python 3.12** installed (recommended; 3.13 works on Apple Silicon only):
   ```bash
   # Check version
   python3.12 --version

   # Install with Homebrew if needed
   brew install python@3.12
   ```

2. **Ollama** installed and running:
   ```bash
   # Install Ollama
   brew install ollama

   # Start Ollama service
   ollama serve

   # Pull a model (e.g., llama3.2)
   ollama pull llama3.2
   ```

3. **Virtual environment with dependencies** set up:
   ```bash
   cd /path/to/rag-cli-project

   # Create venv with Python 3.12
   python3.12 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Start the backend** before launching the desktop app:
   ```bash
   cd /path/to/rag-cli-project
   ./start-web.sh
   ```

   **Important**: The desktop app connects to an existing backend on port 8000. Always start the backend first using `./start-web.sh` from the project directory.

## Usage

### First Launch

On first launch, you'll see a welcome screen that introduces the app's features. Click through or skip to get started.

### Uploading Documents

1. Click "Upload Documents" in the sidebar
2. Drag and drop files or click to select
3. Supported formats: `.pdf`, `.md`, `.txt`, `.html`

### Querying Documents

1. Type your question in the chat input
2. Press Enter or click Send
3. The AI will search your documents and provide relevant answers with source citations

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Cmd+N | New Chat |
| Cmd+K | Focus Search |
| Cmd+\ | Toggle Sidebar |
| Cmd+Shift+D | Toggle Dark Mode |
| Cmd+Shift+E | Export Chat |

### Menu Bar

- **File**: New Chat, Export Chat, Close Window
- **Edit**: Standard text editing (Undo, Redo, Cut, Copy, Paste)
- **View**: Toggle Sidebar, Toggle Dark Mode, Focus Search, Fullscreen
- **Window**: Minimize, Zoom, Close
- **Help**: Documentation, About

## Troubleshooting

### Backend Connection Error

If you see a backend connection error:

1. **Start the backend manually** (required for the desktop app):
   ```bash
   cd /path/to/rag-cli-project
   ./start-web.sh
   ```
   The desktop app connects to an existing backend on port 8000. It does not bundle its own backend.

2. **Check Python version**: Must be Python 3.12 or 3.13 (not 3.14)
   ```bash
   # Check what version is in the venv
   ./venv/bin/python --version
   ```

3. **Ensure virtual environment exists** at the project root:
   ```bash
   ls -la /path/to/rag-cli-project/venv/
   ```

4. **Verify dependencies are installed**:
   ```bash
   source venv/bin/activate
   pip list | grep chromadb
   ```

5. Make sure Ollama is running (`ollama serve`)

6. Verify the backend is healthy:
   ```bash
   curl http://localhost:8000/api/health
   ```

### Python Version Issues

ChromaDB requires Python 3.12 or 3.13. If you have Python 3.14 as your system default:

```bash
# Install Python 3.12 (recommended for broadest compatibility)
brew install python@3.12

# Recreate the virtual environment
cd /path/to/rag-cli-project
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Note:** Intel Mac users must use Python 3.12 (onnxruntime lacks 3.13 wheels for x86_64).

### Gatekeeper Warning

Since the app is not signed, macOS will show a warning. To open:

1. Right-click the app
2. Select "Open"
3. Click "Open" in the dialog

### Performance Issues

- Ensure Ollama has enough memory (adjust in Ollama settings)
- For large documents, consider splitting them into smaller files
- Close other resource-intensive applications

## Data Storage

The desktop app shares data with the Web UI and CLI. All document data is stored in:
```
<project-root>/data/
```

This includes:
- `data/vector_db/` - ChromaDB vector database
- `data/documents/` - Uploaded source documents

Documents indexed via the Web UI, CLI, or desktop app are all accessible from any interface.

## Architecture

```
RAG Assistant Desktop
├── Tauri Shell (Rust)
│   ├── Native Menu Bar
│   └── System Notifications
├── React Frontend (WebView)
│   ├── Chat Interface
│   ├── Document Upload
│   └── Settings Panel
└── FastAPI Backend (External, port 8000)
    ├── Started via ./start-web.sh
    ├── Document Processing
    ├── Vector Storage (ChromaDB)
    └── LLM Integration (Ollama)
```

**Note**: The desktop app connects to the backend started via `./start-web.sh`. This allows the desktop app, Web UI, and CLI to share the same document database.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Built with:
- [Tauri](https://tauri.app/) - Native app framework
- [React](https://react.dev/) - UI framework
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Ollama](https://ollama.com/) - Local LLM
