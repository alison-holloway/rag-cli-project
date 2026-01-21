"""RAG Query Pipeline for RAG CLI.

Orchestrates the complete RAG flow: retrieve → generate.
"""

from dataclasses import dataclass, field
from typing import Iterator

from .config import get_settings
from .llm_client import LLMClient, LLMResponse, get_llm_client
from .logger import get_logger
from .prompts import get_template
from .retriever import RetrievalResult, Retriever, get_retriever

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Result from a RAG query."""

    query: str
    answer: str
    retrieval: RetrievalResult
    llm_response: LLMResponse
    template_used: str
    metadata: dict = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        """Get list of source files used."""
        sources = set()
        for chunk in self.retrieval.chunks:
            if chunk.source_file:
                sources.add(chunk.source_file)
        return sorted(sources)

    @property
    def has_context(self) -> bool:
        """Check if retrieval found any context."""
        return self.retrieval.total_chunks > 0

    def __str__(self) -> str:
        """String representation."""
        return (
            f"QueryResult(query='{self.query[:50]}...', "
            f"sources={len(self.sources)}, "
            f"chunks={self.retrieval.total_chunks})"
        )


class RAGPipeline:
    """Complete RAG pipeline for question answering.

    Combines retrieval and generation into a single interface.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_client: LLMClient | None = None,
        llm_provider: str | None = None,
        template_name: str = "rag_default",
        top_k: int | None = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            retriever: Retriever instance. Defaults to global retriever.
            llm_client: LLM client. Defaults to new client with provider.
            llm_provider: LLM provider if creating new client.
            template_name: Default prompt template to use.
            top_k: Default number of chunks to retrieve.
        """
        settings = get_settings()

        self.retriever = retriever or get_retriever()
        self.llm_client = llm_client or get_llm_client(llm_provider)
        self.template_name = template_name
        self.top_k = top_k or settings.retrieval.top_k_results

        logger.info(
            f"RAG Pipeline initialized: "
            f"provider={self.llm_client.provider}, "
            f"template={self.template_name}, "
            f"top_k={self.top_k}"
        )

    def query(
        self,
        question: str,
        top_k: int | None = None,
        template_name: str | None = None,
        filter_metadata: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> QueryResult:
        """Execute a RAG query.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            template_name: Prompt template to use.
            filter_metadata: Filter retrieval by metadata.
            temperature: LLM temperature.
            max_tokens: Maximum response tokens.

        Returns:
            QueryResult with answer and metadata.
        """
        top_k = top_k or self.top_k
        template_name = template_name or self.template_name

        logger.info(f"Processing query: {question[:100]}...")

        # Step 1: Retrieve relevant chunks
        retrieval = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Step 2: Determine template based on retrieval results
        if not retrieval.chunks:
            template_name = "no_context"
            logger.warning("No relevant chunks found for query")

        # Step 3: Format prompt
        template = get_template(template_name)
        system_prompt, user_prompt = template.format(
            context=retrieval.context,
            question=question,
        )

        # Step 4: Generate answer
        llm_response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        result = QueryResult(
            query=question,
            answer=llm_response.content,
            retrieval=retrieval,
            llm_response=llm_response,
            template_used=template_name,
            metadata={
                "top_k": top_k,
                "provider": self.llm_client.provider,
            },
        )

        logger.info(
            f"Query complete: {result.retrieval.total_chunks} chunks, "
            f"{len(result.answer)} chars"
        )

        return result

    def query_stream(
        self,
        question: str,
        top_k: int | None = None,
        template_name: str | None = None,
        filter_metadata: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Execute a RAG query with streaming response.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            template_name: Prompt template to use.
            filter_metadata: Filter retrieval by metadata.
            temperature: LLM temperature.
            max_tokens: Maximum response tokens.

        Yields:
            Answer text chunks as they are generated.
        """
        top_k = top_k or self.top_k
        template_name = template_name or self.template_name

        logger.info(f"Processing streaming query: {question[:100]}...")

        # Step 1: Retrieve relevant chunks
        retrieval = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Step 2: Determine template
        if not retrieval.chunks:
            template_name = "no_context"

        # Step 3: Format prompt
        template = get_template(template_name)
        system_prompt, user_prompt = template.format(
            context=retrieval.context,
            question=question,
        )

        # Step 4: Stream answer
        yield from self.llm_client.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def is_ready(self) -> tuple[bool, str]:
        """Check if the pipeline is ready to process queries.

        Returns:
            Tuple of (is_ready, status_message).
        """
        issues = []

        # Check vector store has documents
        chunk_count = self.retriever.vector_store.count()
        if chunk_count == 0:
            issues.append("No documents indexed. Use 'rag-cli add' first.")

        # Check LLM is available
        if not self.llm_client.is_available():
            issues.append(
                f"LLM provider '{self.llm_client.provider}' is not available."
            )

        if issues:
            return False, " ".join(issues)

        return True, f"Ready. {chunk_count} chunks indexed."


def create_pipeline(
    llm_provider: str | None = None,
    top_k: int | None = None,
) -> RAGPipeline:
    """Create a new RAG pipeline.

    Convenience factory function.

    Args:
        llm_provider: LLM provider ('ollama' or 'claude').
        top_k: Number of chunks to retrieve.

    Returns:
        Configured RAGPipeline instance.
    """
    return RAGPipeline(
        llm_provider=llm_provider,
        top_k=top_k,
    )


# Quick query function for simple use cases
def query(
    question: str,
    llm_provider: str | None = None,
    top_k: int | None = None,
    **kwargs,
) -> QueryResult:
    """Execute a RAG query using the default pipeline.

    Convenience function for quick queries.

    Args:
        question: The user's question.
        llm_provider: LLM provider to use.
        top_k: Number of chunks to retrieve.
        **kwargs: Additional arguments passed to pipeline.query().

    Returns:
        QueryResult with answer and metadata.
    """
    pipeline = create_pipeline(llm_provider=llm_provider, top_k=top_k)
    return pipeline.query(question, **kwargs)
