"""Custom exceptions for RAG CLI.

Provides descriptive error messages for common failure scenarios.
"""


class RAGError(Exception):
    """Base exception for RAG CLI errors."""

    def __init__(self, message: str, suggestion: str | None = None):
        """Initialize with message and optional suggestion.

        Args:
            message: Error description.
            suggestion: Helpful suggestion for resolving the error.
        """
        self.message = message
        self.suggestion = suggestion
        super().__init__(self.full_message)

    @property
    def full_message(self) -> str:
        """Get the full error message with suggestion if available."""
        if self.suggestion:
            return f"{self.message}\n\nSuggestion: {self.suggestion}"
        return self.message


# =============================================================================
# Document Processing Errors
# =============================================================================


class DocumentError(RAGError):
    """Base exception for document-related errors."""

    pass


class DocumentNotFoundError(DocumentError):
    """Raised when a document file cannot be found."""

    def __init__(self, path: str):
        super().__init__(
            message=f"Document not found: {path}",
            suggestion="Check that the file path is correct and the file exists.",
        )
        self.path = path


class UnsupportedDocumentError(DocumentError):
    """Raised when a document type is not supported."""

    def __init__(self, extension: str, supported: list[str] | None = None):
        supported = supported or [".pdf", ".md", ".html", ".htm", ".txt"]
        super().__init__(
            message=f"Unsupported document type: {extension}",
            suggestion=f"Supported formats: {', '.join(supported)}",
        )
        self.extension = extension
        self.supported = supported


class DocumentParseError(DocumentError):
    """Raised when a document cannot be parsed."""

    def __init__(self, path: str, reason: str | None = None):
        message = f"Failed to parse document: {path}"
        if reason:
            message += f" ({reason})"
        super().__init__(
            message=message,
            suggestion="The file may be corrupted or in an unexpected format.",
        )
        self.path = path
        self.reason = reason


class EmptyDocumentError(DocumentError):
    """Raised when a document contains no extractable text."""

    def __init__(self, path: str):
        super().__init__(
            message=f"No text content found in document: {path}",
            suggestion="The document may be empty, image-only, or password-protected.",
        )
        self.path = path


# =============================================================================
# Vector Store Errors
# =============================================================================


class VectorStoreError(RAGError):
    """Base exception for vector store errors."""

    pass


class VectorStoreNotInitializedError(VectorStoreError):
    """Raised when vector store is not initialized."""

    def __init__(self):
        super().__init__(
            message="Vector store is not initialized.",
            suggestion="Run 'rag-cli init' to initialize the vector store.",
        )


class NoDocumentsIndexedError(VectorStoreError):
    """Raised when no documents have been indexed."""

    def __init__(self):
        super().__init__(
            message="No documents have been indexed yet.",
            suggestion="Add documents with 'rag-cli add <file_or_directory>'.",
        )


class DocumentAlreadyIndexedError(VectorStoreError):
    """Raised when trying to add a document that's already indexed."""

    def __init__(self, path: str):
        super().__init__(
            message=f"Document already indexed: {path}",
            suggestion="Use --force to re-index the document.",
        )
        self.path = path


class ChunkNotFoundError(VectorStoreError):
    """Raised when a chunk ID is not found."""

    def __init__(self, chunk_id: str):
        super().__init__(
            message=f"Chunk not found: {chunk_id}",
            suggestion="The chunk may have been deleted or the ID is incorrect.",
        )
        self.chunk_id = chunk_id


# =============================================================================
# Embedding Errors
# =============================================================================


class EmbeddingError(RAGError):
    """Base exception for embedding errors."""

    pass


class EmbeddingModelError(EmbeddingError):
    """Raised when embedding model fails to load."""

    def __init__(self, model_name: str, reason: str | None = None):
        message = f"Failed to load embedding model: {model_name}"
        if reason:
            message += f" ({reason})"
        super().__init__(
            message=message,
            suggestion=(
                "Check your internet connection for first-time model download. "
                "The model will be cached locally after the first load."
            ),
        )
        self.model_name = model_name
        self.reason = reason


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails."""

    def __init__(self, text_preview: str | None = None):
        message = "Failed to generate embedding"
        if text_preview:
            message += f" for text: '{text_preview[:50]}...'"
        super().__init__(
            message=message,
            suggestion="The text may be too long or contain invalid characters.",
        )


# =============================================================================
# LLM Errors
# =============================================================================


class LLMError(RAGError):
    """Base exception for LLM-related errors."""

    pass


class LLMNotAvailableError(LLMError):
    """Raised when the LLM service is not available."""

    def __init__(self, provider: str):
        if provider == "ollama":
            suggestion = (
                "Make sure Ollama is running: 'ollama serve'\n"
                "Install Ollama from: https://ollama.ai"
            )
        elif provider == "claude":
            suggestion = (
                "Set your API key: export ANTHROPIC_API_KEY=your_key\n"
                "Get an API key from: https://console.anthropic.com"
            )
        else:
            suggestion = f"Check the {provider} service configuration."

        super().__init__(
            message=f"LLM provider '{provider}' is not available.",
            suggestion=suggestion,
        )
        self.provider = provider


class LLMModelNotFoundError(LLMError):
    """Raised when the specified LLM model is not found."""

    def __init__(self, model: str, provider: str, available: list[str] | None = None):
        message = f"Model '{model}' not found for provider '{provider}'."
        suggestion = f"Pull the model with: 'ollama pull {model}'"
        if available:
            suggestion += f"\n\nAvailable models: {', '.join(available[:5])}"
            if len(available) > 5:
                suggestion += f" (and {len(available) - 5} more)"

        super().__init__(message=message, suggestion=suggestion)
        self.model = model
        self.provider = provider
        self.available = available


class LLMGenerationError(LLMError):
    """Raised when LLM generation fails."""

    def __init__(self, provider: str, reason: str | None = None):
        message = f"LLM generation failed ({provider})"
        if reason:
            message += f": {reason}"
        super().__init__(
            message=message,
            suggestion="Try again or check the service status.",
        )
        self.provider = provider
        self.reason = reason


class LLMRateLimitError(LLMError):
    """Raised when hitting API rate limits."""

    def __init__(self, provider: str, retry_after: int | None = None):
        message = f"Rate limit exceeded for {provider}."
        suggestion = "Wait a moment before trying again."
        if retry_after:
            suggestion = f"Wait {retry_after} seconds before trying again."

        super().__init__(message=message, suggestion=suggestion)
        self.provider = provider
        self.retry_after = retry_after


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigError(RAGError):
    """Base exception for configuration errors."""

    pass


class ConfigNotFoundError(ConfigError):
    """Raised when configuration file is not found."""

    def __init__(self, path: str):
        super().__init__(
            message=f"Configuration file not found: {path}",
            suggestion="Run 'rag-cli init' to create a default configuration.",
        )
        self.path = path


class InvalidConfigError(ConfigError):
    """Raised when configuration is invalid."""

    def __init__(self, field: str, reason: str):
        super().__init__(
            message=f"Invalid configuration for '{field}': {reason}",
            suggestion="Check the configuration file or environment variables.",
        )
        self.field = field
        self.reason = reason


# =============================================================================
# Query Errors
# =============================================================================


class QueryError(RAGError):
    """Base exception for query-related errors."""

    pass


class EmptyQueryError(QueryError):
    """Raised when query is empty."""

    def __init__(self):
        super().__init__(
            message="Query cannot be empty.",
            suggestion="Provide a question to search for.",
        )


class NoRelevantContextError(QueryError):
    """Raised when no relevant context is found for a query."""

    def __init__(self, query: str):
        super().__init__(
            message="No relevant documents found for your query.",
            suggestion=(
                "Try rephrasing your question or add more documents "
                "that cover the topic you're asking about."
            ),
        )
        self.query = query
