"""Vector store module for RAG CLI.

Provides persistent vector storage using ChromaDB.
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunker import Chunk
from .config import get_settings
from .embedder import Embedder, get_embedder
from .logger import get_logger

logger = get_logger(__name__)

# Default collection name
DEFAULT_COLLECTION = "rag_documents"


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""

    chunk_id: str
    content: str
    metadata: dict
    distance: float
    similarity: float

    @property
    def source_file(self) -> str | None:
        """Get the source file name."""
        return self.metadata.get("source_file")

    @property
    def chunk_index(self) -> int | None:
        """Get the chunk index within the document."""
        return self.metadata.get("chunk_index")


class VectorStore:
    """ChromaDB-based vector store for document embeddings.

    Provides persistent storage and similarity search for document chunks.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        embedder: Embedder | None = None,
    ):
        """Initialize the vector store.

        Args:
            persist_directory: Directory for persistent storage.
                              Defaults to config setting.
            collection_name: Name of the ChromaDB collection.
            embedder: Embedder instance for generating embeddings.
                     Defaults to global embedder.
        """
        settings = get_settings()

        # Set up persistence directory
        if persist_directory is None:
            self.persist_directory = settings.vector_store.persist_path
        else:
            self.persist_directory = Path(persist_directory)

        self.collection_name = collection_name
        self.embedder = embedder or get_embedder()

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG CLI document embeddings"},
        )

        logger.info(
            f"Connected to collection '{self.collection_name}' "
            f"({self.count()} documents)"
        )

    @property
    def collection(self) -> chromadb.Collection:
        """Get the ChromaDB collection."""
        return self._collection

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()

    def add_chunk(
        self,
        chunk: Chunk,
        embedding: list[float] | None = None,
        chunk_id: str | None = None,
    ) -> str:
        """Add a single chunk to the vector store.

        Args:
            chunk: The chunk to add.
            embedding: Pre-computed embedding. If None, will be generated.
            chunk_id: Optional ID for the chunk. If None, will be generated.

        Returns:
            The ID of the added chunk.
        """
        if embedding is None:
            embedding = self.embedder.embed_text(chunk.content).tolist()

        if chunk_id is None:
            chunk_id = str(uuid.uuid4())

        # Prepare metadata (ChromaDB requires simple types)
        metadata = {
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "word_count": chunk.word_count,
        }
        metadata.update(chunk.metadata)

        # Ensure all metadata values are valid types
        metadata = self._sanitize_metadata(metadata)

        self._collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[metadata],
        )

        logger.debug(f"Added chunk {chunk_id} to collection")
        return chunk_id

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
        show_progress: bool = False,
    ) -> list[str]:
        """Add multiple chunks to the vector store.

        Args:
            chunks: List of chunks to add.
            embeddings: Pre-computed embeddings. If None, will be generated.
            show_progress: If True, show progress bar during embedding.

        Returns:
            List of IDs for the added chunks.
        """
        if not chunks:
            return []

        # Generate embeddings if not provided
        if embeddings is None:
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            texts = [chunk.content for chunk in chunks]
            embeddings_array = self.embedder.embed_texts(
                texts, show_progress=show_progress
            )
            embeddings = embeddings_array.tolist()

        # Generate IDs
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

        # Prepare documents and metadata
        documents = [chunk.content for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = {
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "word_count": chunk.word_count,
            }
            metadata.update(chunk.metadata)
            metadatas.append(self._sanitize_metadata(metadata))

        # Add to collection
        self._collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks to collection")
        return chunk_ids

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks using a text query.

        Args:
            query: The search query.
            top_k: Number of results to return. Defaults to config setting.
            filter_metadata: Optional metadata filters.

        Returns:
            List of SearchResult objects, sorted by similarity.
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval.top_k_results

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query).tolist()

        return self.search_by_embedding(
            query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks using a pre-computed embedding.

        Args:
            embedding: The query embedding.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filters.

        Returns:
            List of SearchResult objects, sorted by similarity.
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval.top_k_results

        # Don't request more than we have
        top_k = min(top_k, self.count())
        if top_k == 0:
            return []

        # Build query parameters
        query_params: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_metadata:
            query_params["where"] = filter_metadata

        # Execute search
        results = self._collection.query(**query_params)

        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert distance to similarity (ChromaDB uses L2 distance by default)
                # For cosine distance: similarity = 1 - distance
                # For L2 distance: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + distance)

                content = ""
                if results["documents"]:
                    content = results["documents"][0][i]
                metadata = {}
                if results["metadatas"]:
                    metadata = results["metadatas"][0][i]

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        content=content,
                        metadata=metadata,
                        distance=distance,
                        similarity=similarity,
                    )
                )

        logger.debug(f"Search returned {len(search_results)} results")
        return search_results

    def get_chunk(self, chunk_id: str) -> SearchResult | None:
        """Get a specific chunk by ID.

        Args:
            chunk_id: The chunk ID.

        Returns:
            SearchResult if found, None otherwise.
        """
        results = self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            return None

        return SearchResult(
            chunk_id=chunk_id,
            content=results["documents"][0] if results["documents"] else "",
            metadata=results["metadatas"][0] if results["metadatas"] else {},
            distance=0.0,
            similarity=1.0,
        )

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID.

        Args:
            chunk_id: The chunk ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        try:
            self._collection.delete(ids=[chunk_id])
            logger.debug(f"Deleted chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            return False

    def delete_by_metadata(self, filter_metadata: dict) -> int:
        """Delete chunks matching metadata filters.

        Args:
            filter_metadata: Metadata filters to match.

        Returns:
            Number of chunks deleted.
        """
        # Get matching IDs first
        results = self._collection.get(
            where=filter_metadata,
            include=[],
        )

        if not results["ids"]:
            return 0

        count = len(results["ids"])
        self._collection.delete(ids=results["ids"])

        logger.info(f"Deleted {count} chunks matching filter")
        return count

    def delete_document(self, source_file: str) -> int:
        """Delete all chunks from a specific document.

        Args:
            source_file: The source filename.

        Returns:
            Number of chunks deleted.
        """
        return self.delete_by_metadata({"source_file": source_file})

    def list_documents(self) -> list[dict]:
        """List all unique documents in the collection.

        Returns:
            List of document info dictionaries.
        """
        # Get all metadata
        results = self._collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        # Group by source file
        documents: dict[str, dict] = {}
        for metadata in results["metadatas"]:
            source = metadata.get("source_file", "unknown")
            if source not in documents:
                documents[source] = {
                    "source_file": source,
                    "file_type": metadata.get("file_type", "unknown"),
                    "title": metadata.get("title"),
                    "chunk_count": 0,
                }
            documents[source]["chunk_count"] += 1

        return list(documents.values())

    def clear(self) -> int:
        """Delete all chunks from the collection.

        Returns:
            Number of chunks deleted.
        """
        count = self.count()
        if count > 0:
            # Delete and recreate collection
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG CLI document embeddings"},
            )
            logger.info(f"Cleared {count} chunks from collection")
        return count

    def get_stats(self) -> dict:
        """Get statistics about the vector store.

        Returns:
            Dictionary with store statistics.
        """
        documents = self.list_documents()
        return {
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "total_chunks": self.count(),
            "total_documents": len(documents),
            "documents": documents,
        }

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Sanitize metadata for ChromaDB storage.

        ChromaDB only supports str, int, float, and bool values.

        Args:
            metadata: Raw metadata dictionary.

        Returns:
            Sanitized metadata with valid types only.
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                # Convert to string
                sanitized[key] = str(value)
        return sanitized


# Module-level singleton
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get the global vector store instance.

    Returns:
        Singleton VectorStore instance.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def reset_vector_store() -> None:
    """Reset the global vector store instance."""
    global _vector_store
    _vector_store = None
