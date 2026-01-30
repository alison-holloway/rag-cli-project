"""FastAPI backend for RAG CLI web interface."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as api_router
from src import __version__

# Create FastAPI application
app = FastAPI(
    title="RAG CLI API",
    description="""
    REST API for the RAG CLI system.

    This API provides endpoints for:
    - Querying the knowledge base with questions
    - Uploading and indexing documents
    - Managing indexed documents
    - Health checks and system status

    The API wraps the existing RAG CLI functionality, providing
    the same document processing, embedding, and LLM integration
    capabilities through a web interface.
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for local development
# This allows the React frontend (typically on port 5173) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # Create React App default
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "tauri://localhost",      # Tauri desktop app (macOS/Linux)
        "https://tauri.localhost",  # Tauri desktop app (Windows)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG CLI API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
