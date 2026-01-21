"""Retriever module for RAG CLI.

Handles searching and retrieving relevant document chunks for queries.
"""

from dataclasses import dataclass

from .config import get_settings
from .logger import get_logger
from .vector_store import SearchResult, VectorStore, get_vector_store

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval with context for LLM."""

    query: str
    chunks: list[SearchResult]
    context: str
    total_chunks: int
    avg_similarity: float

    def __str__(self) -> str:
        """String representation showing retrieval summary."""
        return (
            f"RetrievalResult(query='{self.query[:50]}...', "
            f"chunks={self.total_chunks}, avg_sim={self.avg_similarity:.3f})"
        )


class Retriever:
    """Retrieves relevant document chunks for queries.

    Wraps the vector store with additional retrieval logic like
    context formatting and relevance filtering.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        top_k: int | None = None,
        min_similarity: float = 0.0,
    ):
        """Initialize the retriever.

        Args:
            vector_store: Vector store to search. Defaults to global store.
            top_k: Number of chunks to retrieve. Defaults to config setting.
            min_similarity: Minimum similarity threshold for results.
        """
        settings = get_settings()
        self.vector_store = vector_store or get_vector_store()
        self.top_k = top_k or settings.retrieval.top_k_results
        self.min_similarity = min_similarity

        logger.debug(
            f"Retriever initialized: top_k={self.top_k}, "
            f"min_similarity={self.min_similarity}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Override default top_k for this query.
            filter_metadata: Optional metadata filters.

        Returns:
            RetrievalResult with chunks and formatted context.
        """
        top_k = top_k or self.top_k

        logger.debug(f"Retrieving top {top_k} chunks for query: {query[:50]}...")

        # Search vector store
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Filter by minimum similarity
        if self.min_similarity > 0:
            results = [r for r in results if r.similarity >= self.min_similarity]

        # Format context for LLM
        context = self._format_context(results)

        # Calculate average similarity
        avg_similarity = 0.0
        if results:
            avg_similarity = sum(r.similarity for r in results) / len(results)

        logger.info(
            f"Retrieved {len(results)} chunks for query "
            f"(avg similarity: {avg_similarity:.3f})"
        )

        return RetrievalResult(
            query=query,
            chunks=results,
            context=context,
            total_chunks=len(results),
            avg_similarity=avg_similarity,
        )

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[SearchResult, float]]:
        """Retrieve chunks with their similarity scores.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of (SearchResult, similarity_score) tuples.
        """
        top_k = top_k or self.top_k

        results = self.vector_store.search(query=query, top_k=top_k)

        return [(r, r.similarity) for r in results]

    def _format_context(
        self,
        results: list[SearchResult],
        include_metadata: bool = True,
    ) -> str:
        """Format retrieved chunks into context for LLM.

        Args:
            results: Search results to format.
            include_metadata: Whether to include source metadata.

        Returns:
            Formatted context string.
        """
        if not results:
            return ""

        context_parts = []

        for i, result in enumerate(results, 1):
            # Build chunk header
            header_parts = [f"[Chunk {i}]"]
            if include_metadata and result.source_file:
                header_parts.append(f"Source: {result.source_file}")

            header = " | ".join(header_parts)

            # Add chunk content
            context_parts.append(f"{header}\n{result.content}")

        return "\n\n".join(context_parts)

    def get_similar_chunks(
        self,
        text: str,
        top_k: int | None = None,
        exclude_self: bool = True,
    ) -> list[SearchResult]:
        """Find chunks similar to a given text.

        Useful for finding related content or detecting duplicates.

        Args:
            text: Text to find similar chunks for.
            top_k: Number of results.
            exclude_self: If True, filter out exact matches.

        Returns:
            List of similar SearchResult objects.
        """
        top_k = top_k or self.top_k

        results = self.vector_store.search(query=text, top_k=top_k + 5)

        if exclude_self:
            # Filter out chunks with very high similarity (likely exact match)
            results = [r for r in results if r.similarity < 0.99]

        return results[:top_k]


# Module-level singleton
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def reset_retriever() -> None:
    """Reset the global retriever instance."""
    global _retriever
    _retriever = None
