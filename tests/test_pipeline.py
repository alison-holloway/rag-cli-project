"""Tests for RAG pipeline module."""

from unittest.mock import MagicMock, patch

import pytest

from src.llm_client import LLMResponse
from src.pipeline import (
    QueryResult,
    RAGPipeline,
    create_pipeline,
    query,
)
from src.retriever import RetrievalResult
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


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    @pytest.fixture
    def sample_retrieval(self):
        """Create a sample retrieval result."""
        chunks = [
            make_search_result(
                chunk_id="1",
                content="Chunk 1 content",
                similarity=0.9,
                source_file="doc1.txt",
            ),
            make_search_result(
                chunk_id="2",
                content="Chunk 2 content",
                similarity=0.8,
                source_file="doc2.txt",
            ),
            make_search_result(
                chunk_id="3",
                content="Chunk 3 content",
                similarity=0.7,
                source_file="doc1.txt",  # Same as chunk 1
            ),
        ]
        return RetrievalResult(
            query="test query",
            chunks=chunks,
            context="formatted context",
            total_chunks=3,
            avg_similarity=0.8,
        )

    @pytest.fixture
    def sample_llm_response(self):
        """Create a sample LLM response."""
        return LLMResponse(
            content="This is the answer.",
            model="test-model",
            provider="ollama",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )

    def test_query_result_creation(self, sample_retrieval, sample_llm_response):
        """Test creating a query result."""
        result = QueryResult(
            query="What is AI?",
            answer="AI is artificial intelligence.",
            retrieval=sample_retrieval,
            llm_response=sample_llm_response,
            template_used="rag_default",
        )

        assert result.query == "What is AI?"
        assert result.answer == "AI is artificial intelligence."
        assert result.template_used == "rag_default"

    def test_query_result_sources(self, sample_retrieval, sample_llm_response):
        """Test sources property returns unique source files."""
        result = QueryResult(
            query="test",
            answer="answer",
            retrieval=sample_retrieval,
            llm_response=sample_llm_response,
            template_used="rag_default",
        )

        sources = result.sources
        # Should be unique and sorted
        assert sources == ["doc1.txt", "doc2.txt"]

    def test_query_result_sources_with_none(self, sample_llm_response):
        """Test sources property handles None source files."""
        chunks = [
            make_search_result(chunk_id="1", content="test", similarity=0.9),
        ]
        retrieval = RetrievalResult(
            query="test",
            chunks=chunks,
            context="",
            total_chunks=1,
            avg_similarity=0.9,
        )
        result = QueryResult(
            query="test",
            answer="answer",
            retrieval=retrieval,
            llm_response=sample_llm_response,
            template_used="rag_default",
        )

        assert result.sources == []

    def test_query_result_has_context(self, sample_retrieval, sample_llm_response):
        """Test has_context property."""
        result = QueryResult(
            query="test",
            answer="answer",
            retrieval=sample_retrieval,
            llm_response=sample_llm_response,
            template_used="rag_default",
        )

        assert result.has_context is True

    def test_query_result_no_context(self, sample_llm_response):
        """Test has_context when no chunks retrieved."""
        retrieval = RetrievalResult(
            query="test",
            chunks=[],
            context="",
            total_chunks=0,
            avg_similarity=0.0,
        )
        result = QueryResult(
            query="test",
            answer="answer",
            retrieval=retrieval,
            llm_response=sample_llm_response,
            template_used="no_context",
        )

        assert result.has_context is False

    def test_query_result_str(self, sample_retrieval, sample_llm_response):
        """Test string representation."""
        result = QueryResult(
            query="a" * 100,  # Long query
            answer="answer",
            retrieval=sample_retrieval,
            llm_response=sample_llm_response,
            template_used="rag_default",
        )

        str_repr = str(result)
        assert "QueryResult" in str_repr
        assert "sources=2" in str_repr
        assert "chunks=3" in str_repr
        # Query should be truncated
        assert "..." in str_repr

    def test_query_result_metadata(self, sample_retrieval, sample_llm_response):
        """Test metadata field."""
        result = QueryResult(
            query="test",
            answer="answer",
            retrieval=sample_retrieval,
            llm_response=sample_llm_response,
            template_used="rag_default",
            metadata={"custom_key": "custom_value"},
        )

        assert result.metadata["custom_key"] == "custom_value"


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = MagicMock()
        chunks = [
            make_search_result(
                chunk_id="1",
                content="Retrieved content",
                similarity=0.9,
                source_file="test.txt",
            ),
        ]
        retriever.retrieve.return_value = RetrievalResult(
            query="test",
            chunks=chunks,
            context="Retrieved content",
            total_chunks=1,
            avg_similarity=0.9,
        )
        retriever.vector_store.count.return_value = 10
        return retriever

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.provider = "ollama"
        client.generate.return_value = LLMResponse(
            content="Generated answer",
            model="test-model",
            provider="ollama",
        )
        client.generate_stream.return_value = iter(["Generated", " ", "answer"])
        client.is_available.return_value = True
        return client

    @pytest.fixture
    def pipeline(self, mock_retriever, mock_llm_client):
        """Create a pipeline with mocked dependencies."""
        return RAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm_client,
            top_k=5,
        )

    def test_pipeline_initialization(self, mock_retriever, mock_llm_client):
        """Test pipeline initialization."""
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm_client,
            template_name="chat",
            top_k=10,
        )

        assert pipeline.retriever == mock_retriever
        assert pipeline.llm_client == mock_llm_client
        assert pipeline.template_name == "chat"
        assert pipeline.top_k == 10

    @patch("src.pipeline.get_retriever")
    @patch("src.pipeline.get_llm_client")
    def test_pipeline_initialization_defaults(self, mock_get_llm, mock_get_retriever):
        """Test pipeline uses defaults when not specified."""
        mock_get_retriever.return_value = MagicMock()
        mock_get_llm.return_value = MagicMock(provider="ollama")

        _pipeline = RAGPipeline()  # noqa: F841

        mock_get_retriever.assert_called_once()
        mock_get_llm.assert_called_once()

    def test_query_basic(self, pipeline, mock_retriever, mock_llm_client):
        """Test basic query execution."""
        result = pipeline.query("What is AI?")

        assert isinstance(result, QueryResult)
        assert result.query == "What is AI?"
        assert result.answer == "Generated answer"
        mock_retriever.retrieve.assert_called_once()
        mock_llm_client.generate.assert_called_once()

    def test_query_with_custom_top_k(self, pipeline, mock_retriever):
        """Test query with custom top_k."""
        pipeline.query("test query", top_k=20)

        mock_retriever.retrieve.assert_called_with(
            query="test query",
            top_k=20,
            filter_metadata=None,
        )

    def test_query_with_filter(self, pipeline, mock_retriever):
        """Test query with metadata filter."""
        filter_metadata = {"source_file": "specific.txt"}
        pipeline.query("test query", filter_metadata=filter_metadata)

        mock_retriever.retrieve.assert_called_with(
            query="test query",
            top_k=5,
            filter_metadata=filter_metadata,
        )

    def test_query_with_custom_template(self, pipeline, mock_llm_client):
        """Test query with custom template."""
        result = pipeline.query("test query", template_name="chat")

        assert result.template_used == "chat"

    def test_query_with_llm_params(self, pipeline, mock_llm_client):
        """Test query passes LLM parameters."""
        pipeline.query(
            "test query",
            temperature=0.5,
            max_tokens=500,
        )

        call_kwargs = mock_llm_client.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500

    def test_query_no_context_uses_different_template(self, pipeline, mock_retriever):
        """Test that no-context queries use no_context template."""
        # Mock empty retrieval
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="obscure query",
            chunks=[],
            context="",
            total_chunks=0,
            avg_similarity=0.0,
        )

        result = pipeline.query("obscure query")

        assert result.template_used == "no_context"

    def test_query_result_metadata(self, pipeline):
        """Test query result includes metadata."""
        result = pipeline.query("test query")

        assert "top_k" in result.metadata
        assert "provider" in result.metadata
        assert result.metadata["provider"] == "ollama"

    def test_query_stream(self, pipeline, mock_llm_client):
        """Test streaming query."""
        chunks = list(pipeline.query_stream("test query"))

        assert chunks == ["Generated", " ", "answer"]
        mock_llm_client.generate_stream.assert_called_once()

    def test_query_stream_no_context(self, pipeline, mock_retriever):
        """Test streaming query with no context."""
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="test",
            chunks=[],
            context="",
            total_chunks=0,
            avg_similarity=0.0,
        )

        # Should still work (with no_context template)
        list(pipeline.query_stream("test query"))

    def test_is_ready_success(self, pipeline):
        """Test is_ready when everything is configured."""
        ready, message = pipeline.is_ready()

        assert ready is True
        assert "Ready" in message
        assert "10 chunks" in message

    def test_is_ready_no_documents(self, pipeline, mock_retriever):
        """Test is_ready when no documents are indexed."""
        mock_retriever.vector_store.count.return_value = 0

        ready, message = pipeline.is_ready()

        assert ready is False
        assert "No documents" in message

    def test_is_ready_llm_unavailable(self, pipeline, mock_llm_client):
        """Test is_ready when LLM is not available."""
        mock_llm_client.is_available.return_value = False

        ready, message = pipeline.is_ready()

        assert ready is False
        assert "not available" in message


class TestCreatePipeline:
    """Tests for create_pipeline factory function."""

    @patch("src.pipeline.get_retriever")
    @patch("src.pipeline.get_llm_client")
    def test_create_pipeline_default(self, mock_get_llm, mock_get_retriever):
        """Test creating pipeline with defaults."""
        mock_get_retriever.return_value = MagicMock()
        mock_get_llm.return_value = MagicMock(provider="ollama")

        pipeline = create_pipeline()

        assert isinstance(pipeline, RAGPipeline)

    @patch("src.pipeline.get_retriever")
    @patch("src.pipeline.get_llm_client")
    def test_create_pipeline_with_provider(self, mock_get_llm, mock_get_retriever):
        """Test creating pipeline with specific provider."""
        mock_get_retriever.return_value = MagicMock()
        mock_get_llm.return_value = MagicMock(provider="claude")

        _pipeline = create_pipeline(llm_provider="claude")  # noqa: F841

        mock_get_llm.assert_called_with("claude")

    @patch("src.pipeline.get_retriever")
    @patch("src.pipeline.get_llm_client")
    def test_create_pipeline_with_top_k(self, mock_get_llm, mock_get_retriever):
        """Test creating pipeline with custom top_k."""
        mock_get_retriever.return_value = MagicMock()
        mock_get_llm.return_value = MagicMock(provider="ollama")

        pipeline = create_pipeline(top_k=20)

        assert pipeline.top_k == 20


class TestQueryFunction:
    """Tests for query convenience function."""

    @patch("src.pipeline.create_pipeline")
    def test_query_function(self, mock_create):
        """Test query convenience function."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = QueryResult(
            query="test",
            answer="answer",
            retrieval=MagicMock(total_chunks=1, chunks=[]),
            llm_response=MagicMock(),
            template_used="rag_default",
        )
        mock_create.return_value = mock_pipeline

        result = query("What is AI?")

        mock_create.assert_called_once()
        mock_pipeline.query.assert_called_with("What is AI?")
        assert result.answer == "answer"

    @patch("src.pipeline.create_pipeline")
    def test_query_function_with_kwargs(self, mock_create):
        """Test query function passes kwargs."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = MagicMock()
        mock_create.return_value = mock_pipeline

        query("test", llm_provider="claude", top_k=10, temperature=0.5)

        mock_create.assert_called_with(llm_provider="claude", top_k=10)
        mock_pipeline.query.assert_called_with("test", temperature=0.5)
