# RAG Assistant Frontend

This directory contains the React-based web UI and Tauri desktop app for the RAG Assistant.

## Structure

```
frontend/
├── src/                    # React source code
│   ├── App.jsx             # Main application component
│   ├── components/         # UI components
│   │   ├── ChatInterface.jsx    # Main chat container
│   │   ├── MessageList.jsx      # Message display with markdown/code
│   │   ├── MessageInput.jsx     # Input field with send button
│   │   ├── SettingsPanel.jsx    # LLM and query settings
│   │   ├── Sidebar.jsx          # Document management
│   │   ├── DocumentList.jsx     # Indexed documents list
│   │   ├── DocumentUpload.jsx   # Drag-and-drop upload
│   │   └── Notification.jsx     # Toast notifications
│   ├── context/            # React context providers
│   │   └── ThemeContext.jsx     # Dark mode state
│   └── services/
│       └── api.js          # Backend API client
├── src-tauri/              # Tauri/Rust desktop app
│   ├── Cargo.toml          # Rust dependencies
│   ├── tauri.conf.json     # App configuration
│   ├── src/                # Rust source
│   │   ├── lib.rs          # Entry point
│   │   ├── backend.rs      # Python backend management
│   │   ├── commands.rs     # Tauri IPC commands
│   │   ├── menu.rs         # Native menu bar
│   │   └── notifications.rs
│   └── icons/              # App icons
├── package.json            # Node.js dependencies
├── vite.config.js          # Vite configuration
└── index.html              # HTML entry point
```

## Development

### Web UI Development

Start the development server:

```bash
npm install
npm run dev
```

The web UI will be available at http://localhost:5173.

**Note**: The FastAPI backend must also be running on port 8000. Use `../start-web.sh` from the project root to start both services.

### Desktop App Development

For Tauri desktop development with hot-reload:

```bash
npm install
npm run tauri dev
```

**Prerequisites**:
- Rust and Cargo installed
- Tauri CLI (`cargo install tauri-cli`)
- Python 3.12 virtual environment at project root (3.12 recommended for broadest compatibility)

## Building

### Production Web Build

```bash
npm run build
```

Output goes to `dist/` directory.

### Production Desktop Build

```bash
npm run tauri build
```

Output:
- App bundle: `src-tauri/target/release/bundle/macos/RAG Assistant.app`
- DMG installer: `src-tauri/target/release/bundle/dmg/RAG Assistant_x.x.x_aarch64.dmg`

## Configuration

### Vite Configuration

The `vite.config.js` is configured to:
- Use relative paths for production builds (required for Tauri)
- Use absolute paths for development (required for Vite dev server)
- Chunk vendor dependencies for efficient caching

### Tauri Configuration

The `src-tauri/tauri.conf.json` configures:
- App metadata (name, version, identifier)
- Window settings (size, resizable, title bar)
- File associations (.txt, .md, .pdf)
- Security permissions

## See Also

- [Main README](../README.md) - Full project documentation
- [Desktop README](../desktop/README.md) - Desktop app user guide
- [Building Guide](../desktop/BUILDING.md) - Detailed build instructions
