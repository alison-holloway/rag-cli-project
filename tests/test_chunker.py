"""Tests for text chunker module."""

from datetime import datetime

import pytest

from src.chunker import Chunk, SemanticChunker, TextChunker
from src.document_loader import Document, DocumentMetadata


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_word_count(self) -> None:
        """Test word count calculation."""
        chunk = Chunk(
            content="Hello world this is a test",
            chunk_index=0,
            start_char=0,
            end_char=26,
        )
        assert chunk.word_count == 6

    def test_to_dict(self) -> None:
        """Test chunk serialization."""
        chunk = Chunk(
            content="Test content",
            chunk_index=1,
            start_char=100,
            end_char=112,
            metadata={"source": "test.txt"},
        )
        result = chunk.to_dict()

        assert result["content"] == "Test content"
        assert result["chunk_index"] == 1
        assert result["start_char"] == 100
        assert result["end_char"] == 112
        assert result["source"] == "test.txt"
        assert "word_count" in result


class TestTextChunker:
    """Tests for TextChunker class."""

    @pytest.fixture
    def chunker(self) -> TextChunker:
        """Create a chunker with known settings."""
        return TextChunker(chunk_size=100, chunk_overlap=20)

    def test_init_default_settings(self) -> None:
        """Test chunker uses config defaults."""
        chunker = TextChunker()
        # Should use values from config (800, 100)
        assert chunker.chunk_size > 0
        assert chunker.chunk_overlap > 0
        assert chunker.chunk_overlap < chunker.chunk_size

    def test_init_custom_settings(self) -> None:
        """Test chunker with custom settings."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_init_invalid_overlap(self) -> None:
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=150)

    def test_chunk_empty_text(self, chunker: TextChunker) -> None:
        """Test chunking empty text returns empty list."""
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []
        assert chunker.chunk_text(None) == []  # type: ignore

    def test_chunk_small_text(self, chunker: TextChunker) -> None:
        """Test text smaller than chunk_size returns single chunk."""
        text = "This is a short text."
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].start_char == 0

    def test_chunk_text_creates_overlapping_chunks(self) -> None:
        """Test that chunks overlap correctly."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
        text = "A" * 100  # 100 characters

        chunks = chunker.chunk_text(text)

        # Should have multiple chunks
        assert len(chunks) >= 2

        # Verify overlap: each chunk should start before the previous one ends
        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1].end_char
            curr_start = chunks[i].start_char
            # The overlap should mean curr_start is before prev_end
            assert curr_start < prev_end, "Chunks should overlap"

    def test_chunk_text_breaks_at_sentences(self) -> None:
        """Test that chunker prefers sentence boundaries."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=20)
        text = (
            "This is the first sentence. "
            "This is the second sentence. "
            "This is the third sentence. "
            "This is the fourth sentence."
        )

        chunks = chunker.chunk_text(text)

        # Check that chunks tend to end at sentence boundaries
        for chunk in chunks[:-1]:  # Exclude last chunk
            content = chunk.content.strip()
            # Should end with punctuation or be at a natural break
            assert content[-1] in ".!?" or content[-1].isalpha()

    def test_chunk_text_preserves_metadata(self, chunker: TextChunker) -> None:
        """Test that metadata is preserved in chunks."""
        text = "A" * 200
        metadata = {"source": "test.txt", "author": "Test"}

        chunks = chunker.chunk_text(text, metadata=metadata)

        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["author"] == "Test"

    def test_chunk_document(self) -> None:
        """Test chunking a document."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)

        metadata = DocumentMetadata(
            filename="test.md",
            file_path="/path/test.md",
            file_type="markdown",
            file_size=500,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            document_hash="abc123",
            title="Test Document",
            author="Test Author",
        )

        doc = Document(content="A" * 300, metadata=metadata)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1

        # Check that document metadata is attached
        for chunk in chunks:
            assert chunk.metadata["source_file"] == "test.md"
            assert chunk.metadata["file_type"] == "markdown"
            assert chunk.metadata["document_hash"] == "abc123"
            assert chunk.metadata["title"] == "Test Document"
            assert chunk.metadata["author"] == "Test Author"

    def test_chunk_documents_generator(self) -> None:
        """Test chunking multiple documents as generator."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)

        docs = []
        for i in range(3):
            metadata = DocumentMetadata(
                filename=f"test{i}.md",
                file_path=f"/path/test{i}.md",
                file_type="markdown",
                file_size=200,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                document_hash=f"hash{i}",
            )
            docs.append(Document(content="A" * 200, metadata=metadata))

        chunks = list(chunker.chunk_documents(docs))

        # Should have chunks from all documents
        assert len(chunks) >= 3

        # Verify chunks are from different documents
        sources = {chunk.metadata["source_file"] for chunk in chunks}
        assert len(sources) == 3

    def test_chunk_text_handles_newlines(self, chunker: TextChunker) -> None:
        """Test that chunker handles newlines properly."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunker.chunk_text(text)

        # Text should be normalized
        for chunk in chunks:
            # No excessive newlines
            assert "\n\n\n" not in chunk.content

    def test_chunk_indices_are_sequential(self) -> None:
        """Test that chunk indices are sequential."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
        text = "A" * 200

        chunks = chunker.chunk_text(text)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestSemanticChunker:
    """Tests for SemanticChunker class."""

    def test_inherits_from_text_chunker(self) -> None:
        """Test that SemanticChunker inherits TextChunker."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
        assert isinstance(chunker, TextChunker)

    def test_chunk_text_works(self) -> None:
        """Test that basic chunking works."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
        text = "A" * 200

        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
