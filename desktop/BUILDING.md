# Building RAG Assistant Desktop App

This guide covers how to build the RAG Assistant desktop application from source.

## Prerequisites

### Required Software

1. **Rust** (1.77.2 or later):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   rustc --version
   ```

2. **Node.js** (18 or later):
   ```bash
   # Using Homebrew
   brew install node

   # Or using nvm
   nvm install 18
   ```

3. **Tauri CLI**:
   ```bash
   cargo install tauri-cli
   ```

4. **Python 3.13** (for the backend - Python 3.14 is not yet supported by ChromaDB):
   ```bash
   # Install with Homebrew
   brew install python@3.13

   # Set up virtual environment at project root
   cd /path/to/rag-cli-project
   python3.13 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### macOS-Specific Requirements

- Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

## Project Structure

```
frontend/
├── src/                    # React frontend source
├── src-tauri/             # Tauri/Rust source
│   ├── Cargo.toml         # Rust dependencies
│   ├── tauri.conf.json    # Tauri configuration
│   ├── src/               # Rust source files
│   │   ├── lib.rs         # Main entry point
│   │   ├── backend.rs     # Python process management
│   │   ├── commands.rs    # Tauri IPC commands
│   │   ├── menu.rs        # Native menu
│   │   └── notifications.rs
│   ├── icons/             # App icons
│   └── scripts/           # Build scripts
├── package.json
└── vite.config.js
```

## Building

### Development Build

For development with hot-reload:

```bash
cd frontend
npm install
npm run tauri:dev
```

### Production Build

For a release build:

```bash
cd frontend
npm install
npm run tauri build
```

The build outputs:
- **App bundle**: `src-tauri/target/release/bundle/macos/RAG Assistant.app`
- **DMG installer**: `src-tauri/target/release/bundle/dmg/RAG Assistant_x.x.x_aarch64.dmg`

### Build Optimizations

The production build includes several optimizations:

**Rust (Cargo.toml)**:
```toml
[profile.release]
panic = "abort"      # Strip panic info
codegen-units = 1    # Better optimization
lto = true           # Link-time optimization
opt-level = "s"      # Optimize for size
strip = true         # Strip symbols
```

**Frontend (vite.config.js)**:
- Code splitting for React, Markdown, and Highlight.js
- Tree shaking and minification
- Source maps only for debug builds

## Generating Icons

To regenerate app icons from scratch:

```bash
cd frontend/src-tauri
python3 -m venv .venv
source .venv/bin/activate
pip install Pillow
python scripts/generate_icons.py
```

This generates:
- PNG icons for all sizes (macOS, Windows, Linux)
- `icon.icns` for macOS
- `icon.ico` for Windows

## Configuration

### App Metadata

Edit `src-tauri/tauri.conf.json`:

```json
{
  "productName": "RAG Assistant",
  "version": "0.1.0",
  "identifier": "com.student.rag-assistant",
  ...
}
```

### Window Settings

```json
{
  "windows": [{
    "width": 1200,
    "height": 800,
    "minWidth": 800,
    "minHeight": 600,
    "center": true,
    "resizable": true
  }]
}
```

### File Associations

The app registers for these file types:
- `.txt` - Plain text
- `.md` - Markdown
- `.pdf` - PDF documents

## Code Signing (Optional)

For distribution outside the Mac App Store, you can sign the app:

1. Get an Apple Developer ID certificate
2. Build with signing:
   ```bash
   npm run tauri build -- --sign
   ```

### Notarization (Optional)

For smooth installation without Gatekeeper warnings:

```bash
xcrun notarytool submit "RAG Assistant.app.zip" \
  --apple-id "your@email.com" \
  --team-id "TEAMID" \
  --password "app-specific-password"
```

## Troubleshooting

### Rust Compilation Errors

If you encounter Rust compilation errors:

1. Ensure Rust is up to date:
   ```bash
   rustup update
   ```

2. Clean and rebuild:
   ```bash
   cd frontend/src-tauri
   cargo clean
   cd ..
   npm run tauri build
   ```

### Node Module Issues

If npm packages have issues:

```bash
rm -rf node_modules package-lock.json
npm install
```

### Cargo Environment

If cargo is not found:

```bash
source ~/.cargo/env
```

Or add to your shell profile:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

### Python Version Issues

The desktop app requires Python 3.13 for ChromaDB compatibility. If you have Python 3.14:

```bash
# Check what Python version the venv uses
./venv/bin/python --version

# If it's not 3.13, recreate the venv
rm -rf venv
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Note**: The Tauri app looks for the venv at the project root (`rag-cli-project/venv/`). The backend will fail to start if this venv doesn't exist or uses an incompatible Python version.

## Build Sizes

Optimized release build sizes:
- **App bundle**: ~5.4 MB
- **DMG installer**: ~2.9 MB

## Cross-Platform Notes

While this project is optimized for macOS, Tauri supports Windows and Linux:

### Windows
```bash
npm run tauri build -- --target x86_64-pc-windows-msvc
```

### Linux
```bash
npm run tauri build -- --target x86_64-unknown-linux-gnu
```

Note: Additional dependencies may be required for non-macOS platforms.

## CI/CD

For automated builds, ensure:

1. Rust and Node.js are installed in the CI environment
2. For macOS, use `macos-latest` runner
3. Cache `~/.cargo` and `node_modules` for faster builds

Example GitHub Actions workflow snippet:
```yaml
- name: Install Rust
  uses: dtolnay/rust-action@stable

- name: Install Node
  uses: actions/setup-node@v4
  with:
    node-version: '18'

- name: Build
  run: |
    cd frontend
    npm install
    npm run tauri build
```
