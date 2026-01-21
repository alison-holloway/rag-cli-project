"""Tests for embedder module."""

import numpy as np
import pytest

from src.chunker import Chunk
from src.embedder import Embedder, get_embedder


class TestEmbedder:
    """Tests for Embedder class."""

    @pytest.fixture(scope="class")
    def embedder(self) -> Embedder:
        """Create an embedder instance (shared across tests for efficiency)."""
        return Embedder()

    def test_init_default_model(self) -> None:
        """Test embedder initializes with default model."""
        embedder = Embedder()
        # Model name should be from config
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_init_custom_model(self) -> None:
        """Test embedder with custom model name."""
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_lazy_loading(self) -> None:
        """Test that model is lazily loaded."""
        embedder = Embedder()
        # Model should not be loaded yet
        assert embedder._model is None

        # Accessing model property triggers loading
        _ = embedder.model
        assert embedder._model is not None

    def test_embedding_dimension(self, embedder: Embedder) -> None:
        """Test embedding dimension is correct."""
        dim = embedder.embedding_dimension
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert dim == 384

    def test_embed_text(self, embedder: Embedder) -> None:
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        # Embeddings should be non-zero
        assert np.any(embedding != 0)

    def test_embed_text_empty_raises(self, embedder: Embedder) -> None:
        """Test that empty text raises error."""
        with pytest.raises(ValueError):
            embedder.embed_text("")

        with pytest.raises(ValueError):
            embedder.embed_text("   ")

    def test_embed_texts(self, embedder: Embedder) -> None:
        """Test embedding multiple texts."""
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]
        embeddings = embedder.embed_texts(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_embed_texts_empty_list(self, embedder: Embedder) -> None:
        """Test embedding empty list returns empty array."""
        embeddings = embedder.embed_texts([])
        assert len(embeddings) == 0

    def test_embed_texts_filters_empty(self, embedder: Embedder) -> None:
        """Test that empty texts are filtered."""
        texts = ["Valid text.", "", "Another valid text.", "   "]
        embeddings = embedder.embed_texts(texts)

        # Only 2 valid texts
        assert embeddings.shape[0] == 2

    def test_embed_chunks(self, embedder: Embedder) -> None:
        """Test embedding chunks."""
        chunks = [
            Chunk(content="First chunk.", chunk_index=0, start_char=0, end_char=12),
            Chunk(content="Second chunk.", chunk_index=1, start_char=12, end_char=25),
        ]

        results = embedder.embed_chunks(chunks)

        assert len(results) == 2
        for chunk, embedding in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (384,)

    def test_embed_query(self, embedder: Embedder) -> None:
        """Test embedding a query."""
        query = "What is the meaning of life?"
        embedding = embedder.embed_query(query)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_similarity(self, embedder: Embedder) -> None:
        """Test cosine similarity calculation."""
        # Same text should have similarity ~1
        text = "This is a test."
        emb1 = embedder.embed_text(text)
        emb2 = embedder.embed_text(text)

        similarity = embedder.similarity(emb1, emb2)
        assert 0.99 <= similarity <= 1.0

    def test_similarity_different_texts(self, embedder: Embedder) -> None:
        """Test similarity of different texts."""
        emb1 = embedder.embed_text("The cat sat on the mat.")
        emb2 = embedder.embed_text("Quantum physics is complex.")

        similarity = embedder.similarity(emb1, emb2)
        # Different topics should have lower similarity
        assert similarity < 0.8

    def test_similarity_related_texts(self, embedder: Embedder) -> None:
        """Test similarity of semantically related texts."""
        emb1 = embedder.embed_text("The cat sat on the mat.")
        emb2 = embedder.embed_text("A kitten was lying on the rug.")

        similarity = embedder.similarity(emb1, emb2)
        # Similar meaning should have higher similarity
        assert similarity > 0.5

    def test_most_similar(self, embedder: Embedder) -> None:
        """Test finding most similar embeddings."""
        texts = [
            "The cat sat on the mat.",
            "A dog played in the park.",
            "The kitten was lying on the rug.",
            "Quantum physics is complex.",
            "A puppy ran through the garden.",
        ]
        embeddings = embedder.embed_texts(texts)

        query = embedder.embed_query("Where is the cat?")
        results = embedder.most_similar(query, embeddings, top_k=3)

        assert len(results) == 3
        # Results should be sorted by similarity (descending)
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]

        # First result should be the cat sentence (index 0) or kitten sentence (index 2)
        assert results[0][0] in [0, 2]

    def test_most_similar_empty_embeddings(self, embedder: Embedder) -> None:
        """Test most_similar with empty embeddings."""
        query = embedder.embed_query("Test query")
        results = embedder.most_similar(query, np.array([]))

        assert results == []


class TestGetEmbedder:
    """Tests for get_embedder singleton."""

    def test_returns_embedder(self) -> None:
        """Test that get_embedder returns an Embedder."""
        embedder = get_embedder()
        assert isinstance(embedder, Embedder)

    def test_returns_same_instance(self) -> None:
        """Test that get_embedder returns singleton."""
        embedder1 = get_embedder()
        embedder2 = get_embedder()
        assert embedder1 is embedder2
