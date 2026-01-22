"""Pydantic models for the RAG API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Request Models


class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""

    query: str = Field(..., min_length=1, description="The question to ask")
    llm_provider: Literal["ollama", "claude"] = Field(
        default="ollama", description="LLM provider to use"
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of chunks to retrieve"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="LLM temperature"
    )
    show_sources: bool = Field(default=True, description="Include source information")


# Response Models


class SourceChunk(BaseModel):
    """A source chunk used in generating the answer."""

    file: str = Field(..., description="Source filename")
    chunk_id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content preview")
    similarity: float = Field(..., description="Similarity score")


class QueryMetadata(BaseModel):
    """Metadata about the query processing."""

    llm_provider: str = Field(..., description="LLM provider used")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    model: str = Field(..., description="Model used for generation")


class QueryResponse(BaseModel):
    """Response model for query results."""

    answer: str = Field(..., description="The generated answer")
    sources: list[SourceChunk] = Field(
        default_factory=list, description="Source chunks used"
    )
    metadata: QueryMetadata = Field(..., description="Query metadata")


class DocumentInfo(BaseModel):
    """Information about an indexed document."""

    id: str = Field(..., description="Document identifier (filename)")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (pdf, md, html, txt)")
    chunk_count: int = Field(..., description="Number of chunks")
    indexed_at: datetime | None = Field(None, description="When document was indexed")


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""

    documents: list[DocumentInfo] = Field(..., description="List of indexed documents")
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")


class UploadResponse(BaseModel):
    """Response model for document upload."""

    success: bool = Field(..., description="Whether upload succeeded")
    filename: str = Field(..., description="Uploaded filename")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Status message")


class DeleteResponse(BaseModel):
    """Response model for document deletion."""

    success: bool = Field(..., description="Whether deletion succeeded")
    filename: str = Field(..., description="Deleted filename")
    chunks_deleted: int = Field(..., description="Number of chunks deleted")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    ollama_available: bool = Field(..., description="Whether Ollama is available")
    documents_indexed: int = Field(..., description="Number of indexed documents")
    total_chunks: int = Field(..., description="Total chunks in knowledge base")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional details")
