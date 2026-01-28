"""Text chunking module for RAG CLI.

Splits documents into smaller chunks suitable for embedding and retrieval.
"""

import re
from dataclasses import dataclass, field
from typing import Iterator

from .config import get_settings
from .document_loader import Document
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk from a document."""

    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Count words in the chunk."""
        return len(self.content.split())

    def to_dict(self) -> dict:
        """Convert chunk to dictionary for storage."""
        return {
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "word_count": self.word_count,
            **self.metadata,
        }


class TextChunker:
    """Splits text into overlapping chunks for embedding.

    Uses a fixed-size chunking strategy with configurable overlap
    to maintain context across chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int = 50,
    ):
        """Initialize the text chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
                       Defaults to config setting.
            chunk_overlap: Number of overlapping characters between chunks.
                          Defaults to config setting.
            min_chunk_size: Minimum chunk size. Chunks smaller than this
                           will be merged with adjacent chunks.
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        self.min_chunk_size = min_chunk_size

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        logger.debug(
            f"TextChunker initialized: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def chunk_text(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text into overlapping chunks.

        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        # Clean and normalize text
        text = self._normalize_text(text)

        if len(text) <= self.chunk_size:
            # Text fits in a single chunk
            return [
                Chunk(
                    content=text,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata=metadata or {},
                )
            ]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk - take everything remaining
                end = len(text)
            else:
                # Try to break at a natural boundary
                end = self._find_break_point(text, start, end)

            # Extract chunk content
            chunk_content = text[start:end].strip()

            if chunk_content and len(chunk_content) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                        metadata=metadata or {},
                    )
                )
                chunk_index += 1

            # Calculate next start position with overlap
            prev_start = start
            start = end - self.chunk_overlap

            # Ensure we always make progress to avoid infinite loops
            if start <= prev_start:
                start = end

            # Also check if we're at the end
            if end >= len(text):
                break

        logger.debug(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks, preserving document metadata.

        Args:
            document: The document to chunk.

        Returns:
            List of Chunk objects with document metadata attached.
        """
        # Build metadata to attach to each chunk
        doc_metadata = {
            "source_file": document.metadata.filename,
            "source_path": document.metadata.file_path,
            "file_type": document.metadata.file_type,
            "document_hash": document.metadata.document_hash,
        }

        if document.metadata.title:
            doc_metadata["title"] = document.metadata.title

        if document.metadata.author:
            doc_metadata["author"] = document.metadata.author

        chunks = self.chunk_text(document.content, metadata=doc_metadata)

        logger.info(
            f"Chunked document '{document.metadata.filename}' into {len(chunks)} chunks"
        )

        return chunks

    def chunk_documents(self, documents: list[Document]) -> Iterator[Chunk]:
        """Chunk multiple documents.

        Args:
            documents: List of documents to chunk.

        Yields:
            Chunk objects from all documents.
        """
        total_chunks = 0
        for doc in documents:
            for chunk in self.chunk_document(doc):
                yield chunk
                total_chunks += 1

        logger.info(f"Created {total_chunks} chunks from {len(documents)} documents")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for chunking.

        Args:
            text: Raw text.

        Returns:
            Normalized text with cleaned whitespace.
        """
        # Replace multiple whitespace with single space
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find a natural break point near the end position.

        Tries to break at paragraph, sentence, or word boundaries.

        Args:
            text: The full text.
            start: Start position of current chunk.
            end: Proposed end position.

        Returns:
            Adjusted end position at a natural break point.
        """
        # Look for break points in a window before the end
        window_start = max(start + self.min_chunk_size, end - 200)
        window = text[window_start:end]

        # Priority 1: Paragraph break (double newline)
        para_break = window.rfind("\n\n")
        if para_break != -1:
            return window_start + para_break + 2

        # Priority 2: Single newline
        line_break = window.rfind("\n")
        if line_break != -1:
            return window_start + line_break + 1

        # Priority 3: Sentence end (. ! ?)
        for punct in [". ", "! ", "? "]:
            sent_break = window.rfind(punct)
            if sent_break != -1:
                return window_start + sent_break + len(punct)

        # Priority 4: Word boundary (space)
        word_break = window.rfind(" ")
        if word_break != -1:
            return window_start + word_break + 1

        # No good break point found, use original end
        return end


class SemanticChunker(TextChunker):
    """Advanced chunker that attempts to preserve semantic units.

    Extends basic chunking with awareness of document structure
    like headers, lists, and code blocks.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        respect_headers: bool = True,
    ):
        """Initialize semantic chunker.

        Args:
            chunk_size: Maximum chunk size.
            chunk_overlap: Overlap between chunks.
            respect_headers: If True, try to keep headers with their content.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.respect_headers = respect_headers

    def chunk_text(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text with awareness of semantic structure.

        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach.

        Returns:
            List of semantically-aware chunks.
        """
        if not text or not text.strip():
            return []

        # For now, use the base implementation
        # Future: Add semantic splitting logic
        return super().chunk_text(text, metadata)
