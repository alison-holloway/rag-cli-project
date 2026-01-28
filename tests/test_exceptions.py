"""Tests for the exceptions module."""

from src.exceptions import (
    ChunkNotFoundError,
    ConfigError,
    ConfigNotFoundError,
    DocumentAlreadyIndexedError,
    DocumentError,
    DocumentNotFoundError,
    DocumentParseError,
    EmbeddingError,
    EmbeddingGenerationError,
    EmbeddingModelError,
    EmptyDocumentError,
    EmptyQueryError,
    InvalidConfigError,
    LLMError,
    LLMGenerationError,
    LLMModelNotFoundError,
    LLMNotAvailableError,
    LLMRateLimitError,
    NoDocumentsIndexedError,
    NoRelevantContextError,
    QueryError,
    RAGError,
    UnsupportedDocumentError,
    VectorStoreError,
    VectorStoreNotInitializedError,
)


class TestRAGError:
    """Tests for base RAGError class."""

    def test_basic_error(self):
        """Test creating a basic error."""
        error = RAGError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.suggestion is None

    def test_error_with_suggestion(self):
        """Test error with suggestion."""
        error = RAGError("Failed", suggestion="Try again")
        assert "Failed" in str(error)
        assert "Suggestion: Try again" in str(error)
        assert error.suggestion == "Try again"


class TestDocumentErrors:
    """Tests for document-related errors."""

    def test_document_not_found(self):
        """Test DocumentNotFoundError."""
        error = DocumentNotFoundError("/path/to/file.pdf")
        assert "/path/to/file.pdf" in str(error)
        assert error.path == "/path/to/file.pdf"
        assert "exists" in error.suggestion.lower()

    def test_unsupported_document(self):
        """Test UnsupportedDocumentError."""
        error = UnsupportedDocumentError(".xyz")
        assert ".xyz" in str(error)
        assert error.extension == ".xyz"
        assert ".pdf" in error.suggestion

    def test_unsupported_document_custom_formats(self):
        """Test UnsupportedDocumentError with custom formats."""
        error = UnsupportedDocumentError(".xyz", supported=[".doc", ".docx"])
        assert ".doc" in error.suggestion
        assert ".docx" in error.suggestion

    def test_document_parse_error(self):
        """Test DocumentParseError."""
        error = DocumentParseError("/file.pdf", reason="corrupted")
        assert "/file.pdf" in str(error)
        assert "corrupted" in str(error)
        assert error.path == "/file.pdf"
        assert error.reason == "corrupted"

    def test_document_parse_error_no_reason(self):
        """Test DocumentParseError without reason."""
        error = DocumentParseError("/file.pdf")
        assert "/file.pdf" in str(error)
        assert error.reason is None

    def test_empty_document_error(self):
        """Test EmptyDocumentError."""
        error = EmptyDocumentError("/empty.pdf")
        assert "/empty.pdf" in str(error)
        assert error.path == "/empty.pdf"


class TestVectorStoreErrors:
    """Tests for vector store errors."""

    def test_not_initialized(self):
        """Test VectorStoreNotInitializedError."""
        error = VectorStoreNotInitializedError()
        assert "not initialized" in str(error).lower()
        assert "init" in error.suggestion.lower()

    def test_no_documents_indexed(self):
        """Test NoDocumentsIndexedError."""
        error = NoDocumentsIndexedError()
        assert "no documents" in str(error).lower()
        assert "add" in error.suggestion.lower()

    def test_document_already_indexed(self):
        """Test DocumentAlreadyIndexedError."""
        error = DocumentAlreadyIndexedError("/doc.pdf")
        assert "/doc.pdf" in str(error)
        assert error.path == "/doc.pdf"
        assert "--force" in error.suggestion

    def test_chunk_not_found(self):
        """Test ChunkNotFoundError."""
        error = ChunkNotFoundError("chunk-123")
        assert "chunk-123" in str(error)
        assert error.chunk_id == "chunk-123"


class TestEmbeddingErrors:
    """Tests for embedding errors."""

    def test_embedding_model_error(self):
        """Test EmbeddingModelError."""
        error = EmbeddingModelError("all-MiniLM-L6-v2")
        assert "all-MiniLM-L6-v2" in str(error)
        assert error.model_name == "all-MiniLM-L6-v2"

    def test_embedding_model_error_with_reason(self):
        """Test EmbeddingModelError with reason."""
        error = EmbeddingModelError("model", reason="network error")
        assert "network error" in str(error)
        assert error.reason == "network error"

    def test_embedding_generation_error(self):
        """Test EmbeddingGenerationError."""
        error = EmbeddingGenerationError("Some text here")
        assert "Some text" in str(error)

    def test_embedding_generation_error_no_preview(self):
        """Test EmbeddingGenerationError without text preview."""
        error = EmbeddingGenerationError()
        assert "Failed to generate embedding" in str(error)


class TestLLMErrors:
    """Tests for LLM errors."""

    def test_llm_not_available_ollama(self):
        """Test LLMNotAvailableError for Ollama."""
        error = LLMNotAvailableError("ollama")
        assert "ollama" in str(error).lower()
        assert error.provider == "ollama"
        assert "ollama serve" in error.suggestion.lower()

    def test_llm_not_available_claude(self):
        """Test LLMNotAvailableError for Claude."""
        error = LLMNotAvailableError("claude")
        assert "claude" in str(error).lower()
        assert error.provider == "claude"
        assert "ANTHROPIC_API_KEY" in error.suggestion

    def test_llm_not_available_other(self):
        """Test LLMNotAvailableError for other providers."""
        error = LLMNotAvailableError("custom_provider")
        assert error.provider == "custom_provider"

    def test_llm_model_not_found(self):
        """Test LLMModelNotFoundError."""
        error = LLMModelNotFoundError("llama3", "ollama")
        assert "llama3" in str(error)
        assert "ollama" in str(error)
        assert error.model == "llama3"
        assert error.provider == "ollama"

    def test_llm_model_not_found_with_available(self):
        """Test LLMModelNotFoundError with available models."""
        available = ["mistral", "codellama", "phi"]
        error = LLMModelNotFoundError("llama3", "ollama", available=available)
        assert "mistral" in error.suggestion
        assert error.available == available

    def test_llm_model_not_found_many_available(self):
        """Test LLMModelNotFoundError with many available models."""
        available = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
        error = LLMModelNotFoundError("llama3", "ollama", available=available)
        assert "and 2 more" in error.suggestion

    def test_llm_generation_error(self):
        """Test LLMGenerationError."""
        error = LLMGenerationError("ollama")
        assert "ollama" in str(error)
        assert error.provider == "ollama"

    def test_llm_generation_error_with_reason(self):
        """Test LLMGenerationError with reason."""
        error = LLMGenerationError("ollama", reason="timeout")
        assert "timeout" in str(error)
        assert error.reason == "timeout"

    def test_llm_rate_limit_error(self):
        """Test LLMRateLimitError."""
        error = LLMRateLimitError("claude")
        assert "rate limit" in str(error).lower()
        assert error.provider == "claude"

    def test_llm_rate_limit_error_with_retry(self):
        """Test LLMRateLimitError with retry_after."""
        error = LLMRateLimitError("claude", retry_after=30)
        assert "30 seconds" in error.suggestion
        assert error.retry_after == 30


class TestConfigErrors:
    """Tests for configuration errors."""

    def test_config_not_found(self):
        """Test ConfigNotFoundError."""
        error = ConfigNotFoundError("/path/to/config.yaml")
        assert "/path/to/config.yaml" in str(error)
        assert error.path == "/path/to/config.yaml"

    def test_invalid_config(self):
        """Test InvalidConfigError."""
        error = InvalidConfigError("chunk_size", "must be positive")
        assert "chunk_size" in str(error)
        assert "must be positive" in str(error)
        assert error.field == "chunk_size"
        assert error.reason == "must be positive"


class TestQueryErrors:
    """Tests for query errors."""

    def test_empty_query(self):
        """Test EmptyQueryError."""
        error = EmptyQueryError()
        assert "empty" in str(error).lower()

    def test_no_relevant_context(self):
        """Test NoRelevantContextError."""
        error = NoRelevantContextError("What is quantum computing?")
        assert "no relevant" in str(error).lower()
        assert error.query == "What is quantum computing?"
        assert "rephras" in error.suggestion.lower()


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_document_errors_inherit_from_rag_error(self):
        """Test that document errors inherit from RAGError."""
        assert issubclass(DocumentError, RAGError)
        assert issubclass(DocumentNotFoundError, DocumentError)
        assert issubclass(UnsupportedDocumentError, DocumentError)
        assert issubclass(DocumentParseError, DocumentError)
        assert issubclass(EmptyDocumentError, DocumentError)

    def test_vector_store_errors_inherit_from_rag_error(self):
        """Test that vector store errors inherit from RAGError."""
        assert issubclass(VectorStoreError, RAGError)
        assert issubclass(VectorStoreNotInitializedError, VectorStoreError)
        assert issubclass(NoDocumentsIndexedError, VectorStoreError)
        assert issubclass(DocumentAlreadyIndexedError, VectorStoreError)
        assert issubclass(ChunkNotFoundError, VectorStoreError)

    def test_embedding_errors_inherit_from_rag_error(self):
        """Test that embedding errors inherit from RAGError."""
        assert issubclass(EmbeddingError, RAGError)
        assert issubclass(EmbeddingModelError, EmbeddingError)
        assert issubclass(EmbeddingGenerationError, EmbeddingError)

    def test_llm_errors_inherit_from_rag_error(self):
        """Test that LLM errors inherit from RAGError."""
        assert issubclass(LLMError, RAGError)
        assert issubclass(LLMNotAvailableError, LLMError)
        assert issubclass(LLMModelNotFoundError, LLMError)
        assert issubclass(LLMGenerationError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)

    def test_config_errors_inherit_from_rag_error(self):
        """Test that config errors inherit from RAGError."""
        assert issubclass(ConfigError, RAGError)
        assert issubclass(ConfigNotFoundError, ConfigError)
        assert issubclass(InvalidConfigError, ConfigError)

    def test_query_errors_inherit_from_rag_error(self):
        """Test that query errors inherit from RAGError."""
        assert issubclass(QueryError, RAGError)
        assert issubclass(EmptyQueryError, QueryError)
        assert issubclass(NoRelevantContextError, QueryError)

    def test_exceptions_are_catchable(self):
        """Test that all exceptions can be caught as RAGError."""
        errors = [
            DocumentNotFoundError("/test"),
            UnsupportedDocumentError(".xyz"),
            NoDocumentsIndexedError(),
            EmbeddingModelError("model"),
            LLMNotAvailableError("ollama"),
            ConfigNotFoundError("/config"),
            EmptyQueryError(),
        ]

        for error in errors:
            try:
                raise error
            except RAGError as e:
                assert isinstance(e, RAGError)
