"""RAG service layer that wraps existing CLI components."""

import sys
import time
from pathlib import Path
from typing import BinaryIO

# Add the src directory to the path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.chunker import TextChunker
from src.config import get_settings
from src.document_loader import DocumentLoader
from src.llm_client import LLMClient
from src.pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore


class RAGService:
    """Service class that wraps RAG functionality for the API."""

    def __init__(self):
        """Initialize RAG service components."""
        self._settings = get_settings()
        self._vector_store: VectorStore | None = None
        self._document_loader: DocumentLoader | None = None
        self._chunker: TextChunker | None = None

    @property
    def vector_store(self) -> VectorStore:
        """Lazy-load vector store."""
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def document_loader(self) -> DocumentLoader:
        """Lazy-load document loader."""
        if self._document_loader is None:
            self._document_loader = DocumentLoader()
        return self._document_loader

    @property
    def chunker(self) -> TextChunker:
        """Lazy-load text chunker."""
        if self._chunker is None:
            self._chunker = TextChunker()
        return self._chunker

    def query(
        self,
        question: str,
        llm_provider: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        """
        Query the knowledge base and get an answer.

        Args:
            question: The question to ask.
            llm_provider: LLM provider ('ollama' or 'claude'). Defaults to config.
            top_k: Number of chunks to retrieve. Defaults to config.
            temperature: LLM temperature. Defaults to config.

        Returns:
            Dictionary with answer, sources, and metadata.
        """
        # Use settings for defaults
        if llm_provider is None:
            llm_provider = self._settings.llm.default_llm_provider
        if top_k is None:
            top_k = self._settings.retrieval.top_k_results
        if temperature is None:
            temperature = self._settings.llm.llm_temperature

        start_time = time.time()

        # Refresh to see changes made by CLI or other processes
        self.vector_store.refresh()

        # Check if we have documents
        if self.vector_store.count() == 0:
            raise ValueError("No documents indexed. Please upload documents first.")

        # Create pipeline components
        retriever = Retriever(vector_store=self.vector_store, top_k=top_k)
        llm_client = LLMClient(provider=llm_provider)

        # Check LLM availability
        if not llm_client.is_available():
            if llm_provider == "ollama":
                raise RuntimeError(
                    "Ollama is not available. Please start Ollama with 'ollama serve'."
                )
            else:
                raise RuntimeError(
                    "Claude API is not available. Check your ANTHROPIC_API_KEY."
                )

        # Create pipeline and query
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
        )

        result = pipeline.query(
            question=question,
            top_k=top_k,
            temperature=temperature,
        )

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Format sources
        sources = []
        if result.retrieval and result.retrieval.chunks:
            for chunk in result.retrieval.chunks:
                sources.append(
                    {
                        "file": chunk.metadata.get("source_file", "unknown"),
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content[:200] + "..."
                        if len(chunk.content) > 200
                        else chunk.content,
                        "similarity": chunk.similarity or 0.0,
                    }
                )

        return {
            "answer": result.answer,
            "sources": sources,
            "metadata": {
                "llm_provider": llm_provider,
                "chunks_retrieved": len(sources),
                "processing_time_ms": round(processing_time, 2),
                "model": llm_client._client.model,
            },
        }

    def upload_document(
        self,
        file: BinaryIO,
        filename: str,
        force: bool = False,
    ) -> dict:
        """
        Upload and index a document.

        Args:
            file: File-like object with document content.
            filename: Original filename.
            force: Whether to re-index if already exists.

        Returns:
            Dictionary with upload results.
        """
        # Check if document already exists
        existing_docs = self.list_documents()
        doc_names = [d["filename"] for d in existing_docs["documents"]]

        if filename in doc_names and not force:
            raise ValueError(
                f"Document '{filename}' is already indexed. Use force=true to re-index."
            )

        # Save file temporarily
        temp_dir = Path("data/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / filename

        try:
            # Write uploaded file
            content = file.read()
            temp_path.write_bytes(content)

            # Load and process document
            doc = self.document_loader.load(str(temp_path))

            if not doc.content.strip():
                raise ValueError(f"Document '{filename}' appears to be empty.")

            # Chunk document
            chunks = self.chunker.chunk_document(doc)

            if not chunks:
                raise ValueError(f"No chunks could be created from '{filename}'.")

            # If force and document exists, remove old chunks first
            if filename in doc_names and force:
                self.vector_store.delete_document(filename)

            # Add to vector store
            chunk_ids = self.vector_store.add_chunks(chunks)

            msg = f"Successfully indexed '{filename}' ({len(chunk_ids)} chunks)"
            return {
                "success": True,
                "filename": filename,
                "chunks_created": len(chunk_ids),
                "message": msg,
            }

        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def list_documents(self) -> dict:
        """
        List all indexed documents.

        Returns:
            Dictionary with document list and statistics.
        """
        # Refresh collection to see changes made by CLI or other processes
        self.vector_store.refresh()

        documents = self.vector_store.list_documents()
        total_chunks = self.vector_store.count()

        doc_list = []
        for doc in documents:
            doc_list.append(
                {
                    "id": doc.get("source_file", "unknown"),
                    "filename": doc.get("source_file", "unknown"),
                    "file_type": doc.get("file_type", "unknown"),
                    "chunk_count": doc.get("chunk_count", 0),
                    "indexed_at": None,  # Could add timestamp tracking later
                }
            )

        return {
            "documents": doc_list,
            "total_documents": len(doc_list),
            "total_chunks": total_chunks,
        }

    def delete_document(self, document_id: str) -> dict:
        """
        Delete a document from the knowledge base.

        Args:
            document_id: Document identifier (filename).

        Returns:
            Dictionary with deletion results.
        """
        # Check if document exists
        existing_docs = self.list_documents()
        doc_names = [d["filename"] for d in existing_docs["documents"]]

        if document_id not in doc_names:
            raise ValueError(
                f"Document '{document_id}' not found. "
                "Use GET /api/documents to see indexed documents."
            )

        # Delete from vector store
        deleted_count = self.vector_store.delete_document(document_id)

        return {
            "success": True,
            "filename": document_id,
            "chunks_deleted": deleted_count,
            "message": f"Successfully deleted '{document_id}' ({deleted_count} chunks)",
        }

    def get_health(self) -> dict:
        """
        Get health status of the RAG service.

        Returns:
            Dictionary with health information.
        """
        from src import __version__

        # Check Ollama availability
        ollama_available = False
        try:
            llm_client = LLMClient(provider="ollama")
            ollama_available = llm_client.is_available()
        except Exception:
            pass

        # Refresh to see changes made by CLI or other processes
        self.vector_store.refresh()

        # Get document stats
        total_chunks = self.vector_store.count()
        documents = self.vector_store.list_documents()

        return {
            "status": "healthy",
            "version": __version__,
            "ollama_available": ollama_available,
            "documents_indexed": len(documents),
            "total_chunks": total_chunks,
        }

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions."""
        return self.document_loader.supported_extensions

    def get_config(self) -> dict:
        """
        Get current configuration settings.

        Returns:
            Dictionary with configuration values.
        """
        return {
            "llm_provider": self._settings.llm.default_llm_provider,
            "llm_model": self._settings.llm.ollama_model,
            "top_k": self._settings.retrieval.top_k_results,
            "temperature": self._settings.llm.llm_temperature,
            "chunk_size": self._settings.chunking.chunk_size,
            "chunk_overlap": self._settings.chunking.chunk_overlap,
            "embedding_model": self._settings.embedding.embedding_model,
        }


# Singleton instance
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get or create the RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
