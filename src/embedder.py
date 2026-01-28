"""Embedding module for RAG CLI.

Generates vector embeddings from text using sentence-transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk
from .config import get_settings
from .logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Generates embeddings using sentence-transformers.

    Uses the all-MiniLM-L6-v2 model by default, which provides a good
    balance of speed and quality for semantic search.
    """

    def __init__(self, model_name: str | None = None):
        """Initialize the embedder.

        Args:
            model_name: Name of the sentence-transformer model to use.
                       Defaults to config setting (all-MiniLM-L6-v2).
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding.embedding_model

        logger.info(f"Loading embedding model: {self.model_name}")
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                f"Loaded model {self.model_name} "
                f"(dimension: {self.embedding_dimension})"
            )
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Numpy array of shape (embedding_dimension,).
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process in each batch.
            show_progress: If True, show a progress bar.

        Returns:
            Numpy array of shape (num_texts, embedding_dimension).
        """
        if not texts:
            return np.array([])

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")

        if not valid_texts:
            return np.array([])

        logger.debug(f"Embedding {len(valid_texts)} texts with batch_size={batch_size}")

        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    def embed_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[tuple[Chunk, np.ndarray]]:
        """Generate embeddings for a list of chunks.

        Args:
            chunks: List of Chunk objects to embed.
            batch_size: Batch size for embedding.
            show_progress: If True, show progress bar.

        Returns:
            List of (chunk, embedding) tuples.
        """
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(
            texts, batch_size=batch_size, show_progress=show_progress
        )

        results = list(zip(chunks, embeddings))

        logger.info(f"Generated embeddings for {len(results)} chunks")
        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a search query.

        This is an alias for embed_text, but allows for future
        query-specific optimizations (e.g., different prefixes for
        asymmetric retrieval models).

        Args:
            query: The search query.

        Returns:
            Query embedding as numpy array.
        """
        return self.embed_text(query)

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding.
            embedding2: Second embedding.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def most_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find the most similar embeddings to a query.

        Args:
            query_embedding: The query embedding.
            embeddings: Array of embeddings to search.
            top_k: Number of top results to return.

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity.
        """
        if len(embeddings) == 0:
            return []

        # Compute cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        similarities = np.dot(embeddings_norm, query_norm)

        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]


# Module-level singleton for convenience
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get the global embedder instance.

    Returns:
        Singleton Embedder instance.
    """
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
