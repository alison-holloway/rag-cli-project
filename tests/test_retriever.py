"""Tests for retriever module."""

from unittest.mock import MagicMock, patch

import pytest

from src.retriever import (
    RetrievalResult,
    Retriever,
    get_retriever,
    reset_retriever,
)
from src.vector_store import SearchResult


def make_search_result(chunk_id, content, similarity, source_file=None, metadata=None):
    """Helper to create SearchResult with proper fields."""
    meta = metadata or {}
    if source_file:
        meta["source_file"] = source_file
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        metadata=meta,
        distance=1.0 - similarity,
        similarity=similarity,
    )


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_retrieval_result_creation(self):
        """Test creating a retrieval result."""
        chunks = [
            make_search_result(
                chunk_id="1",
                content="Test content",
                similarity=0.9,
                source_file="test.txt",
            )
        ]
        result = RetrievalResult(
            query="test query",
            chunks=chunks,
            context="formatted context",
            total_chunks=1,
            avg_similarity=0.9,
        )

        assert result.query == "test query"
        assert len(result.chunks) == 1
        assert result.total_chunks == 1
        assert result.avg_similarity == 0.9

    def test_retrieval_result_str(self):
        """Test string representation."""
        result = RetrievalResult(
            query="a" * 100,  # Long query
            chunks=[],
            context="",
            total_chunks=5,
            avg_similarity=0.85,
        )

        str_repr = str(result)
        assert "chunks=5" in str_repr
        assert "0.850" in str_repr
        # Query should be truncated
        assert "..." in str_repr


class TestRetriever:
    """Tests for Retriever class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.search.return_value = [
            make_search_result(
                chunk_id="1",
                content="First chunk content",
                similarity=0.95,
                source_file="doc1.txt",
                metadata={"title": "Doc 1"},
            ),
            make_search_result(
                chunk_id="2",
                content="Second chunk content",
                similarity=0.85,
                source_file="doc2.txt",
            ),
            make_search_result(
                chunk_id="3",
                content="Third chunk content",
                similarity=0.75,
                source_file="doc1.txt",
            ),
        ]
        return store

    @pytest.fixture
    def retriever(self, mock_vector_store):
        """Create a retriever with mock vector store."""
        return Retriever(vector_store=mock_vector_store, top_k=5)

    def test_retriever_initialization(self, mock_vector_store):
        """Test retriever initialization with custom settings."""
        retriever = Retriever(
            vector_store=mock_vector_store,
            top_k=10,
            min_similarity=0.5,
        )

        assert retriever.top_k == 10
        assert retriever.min_similarity == 0.5
        assert retriever.vector_store == mock_vector_store

    def test_retriever_initialization_defaults(self, mock_vector_store):
        """Test retriever uses config defaults."""
        retriever = Retriever(vector_store=mock_vector_store)

        # Should use config defaults
        assert retriever.top_k > 0
        assert retriever.min_similarity == 0.0

    def test_retrieve_basic(self, retriever, mock_vector_store):
        """Test basic retrieval."""
        result = retriever.retrieve("test query")

        assert isinstance(result, RetrievalResult)
        assert result.query == "test query"
        assert result.total_chunks == 3
        mock_vector_store.search.assert_called_once()

    def test_retrieve_with_custom_top_k(self, retriever, mock_vector_store):
        """Test retrieval with custom top_k."""
        retriever.retrieve("test query", top_k=10)

        mock_vector_store.search.assert_called_with(
            query="test query",
            top_k=10,
            filter_metadata=None,
        )

    def test_retrieve_with_filter(self, retriever, mock_vector_store):
        """Test retrieval with metadata filter."""
        filter_metadata = {"source_file": "doc1.txt"}
        retriever.retrieve("test query", filter_metadata=filter_metadata)

        mock_vector_store.search.assert_called_with(
            query="test query",
            top_k=5,
            filter_metadata=filter_metadata,
        )

    def test_retrieve_min_similarity_filter(self, mock_vector_store):
        """Test that min_similarity filters out low-scoring results."""
        retriever = Retriever(
            vector_store=mock_vector_store,
            min_similarity=0.8,
        )

        result = retriever.retrieve("test query")

        # Should only include chunks with similarity >= 0.8
        assert result.total_chunks == 2
        for chunk in result.chunks:
            assert chunk.similarity >= 0.8

    def test_retrieve_average_similarity(self, retriever, mock_vector_store):
        """Test average similarity calculation."""
        result = retriever.retrieve("test query")

        expected_avg = (0.95 + 0.85 + 0.75) / 3
        assert abs(result.avg_similarity - expected_avg) < 0.001

    def test_retrieve_empty_results(self, mock_vector_store):
        """Test retrieval with no results."""
        mock_vector_store.search.return_value = []
        retriever = Retriever(vector_store=mock_vector_store)

        result = retriever.retrieve("obscure query")

        assert result.total_chunks == 0
        assert result.chunks == []
        assert result.context == ""
        assert result.avg_similarity == 0.0

    def test_context_formatting(self, retriever):
        """Test that context is properly formatted."""
        result = retriever.retrieve("test query")

        assert "[Chunk 1]" in result.context
        assert "Source: doc1.txt" in result.context
        assert "First chunk content" in result.context
        assert "[Chunk 2]" in result.context
        assert "[Chunk 3]" in result.context

    def test_context_formatting_without_source(self, mock_vector_store):
        """Test context formatting when source_file is None."""
        mock_vector_store.search.return_value = [
            make_search_result(
                chunk_id="1",
                content="Content without source",
                similarity=0.9,
            ),
        ]
        retriever = Retriever(vector_store=mock_vector_store)

        result = retriever.retrieve("test query")

        assert "[Chunk 1]" in result.context
        assert "Source:" not in result.context

    def test_retrieve_with_scores(self, retriever, mock_vector_store):
        """Test retrieve_with_scores method."""
        results = retriever.retrieve_with_scores("test query")

        assert len(results) == 3
        for search_result, score in results:
            assert isinstance(search_result, SearchResult)
            assert isinstance(score, float)
            assert search_result.similarity == score

    def test_get_similar_chunks(self, retriever, mock_vector_store):
        """Test finding similar chunks."""
        results = retriever.get_similar_chunks("some text")

        assert len(results) <= 5
        mock_vector_store.search.assert_called()

    def test_get_similar_chunks_excludes_exact_match(self, mock_vector_store):
        """Test that exact matches are excluded by default."""
        mock_vector_store.search.return_value = [
            make_search_result(
                chunk_id="1",
                content="Exact match",
                similarity=1.0,  # Exact match
                source_file="test.txt",
            ),
            make_search_result(
                chunk_id="2",
                content="Similar",
                similarity=0.9,
                source_file="test.txt",
            ),
        ]
        retriever = Retriever(vector_store=mock_vector_store)

        results = retriever.get_similar_chunks("Exact match")

        # Should exclude the exact match (similarity >= 0.99)
        assert all(r.similarity < 0.99 for r in results)

    def test_get_similar_chunks_include_self(self, mock_vector_store):
        """Test including exact matches when exclude_self=False."""
        mock_vector_store.search.return_value = [
            make_search_result(
                chunk_id="1",
                content="Exact match",
                similarity=1.0,
                source_file="test.txt",
            ),
        ]
        retriever = Retriever(vector_store=mock_vector_store)

        results = retriever.get_similar_chunks("Exact match", exclude_self=False)

        # Should include the exact match
        assert len(results) == 1
        assert results[0].similarity == 1.0


class TestRetrieverSingleton:
    """Tests for retriever singleton pattern."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_retriever()

    @patch("src.retriever.get_vector_store")
    def test_get_retriever_creates_instance(self, mock_get_store):
        """Test that get_retriever creates a new instance."""
        mock_get_store.return_value = MagicMock()

        retriever1 = get_retriever()
        retriever2 = get_retriever()

        # Should return the same instance
        assert retriever1 is retriever2

    @patch("src.retriever.get_vector_store")
    def test_reset_retriever(self, mock_get_store):
        """Test that reset_retriever clears the singleton."""
        mock_get_store.return_value = MagicMock()

        retriever1 = get_retriever()
        reset_retriever()
        retriever2 = get_retriever()

        # Should be different instances after reset
        assert retriever1 is not retriever2
