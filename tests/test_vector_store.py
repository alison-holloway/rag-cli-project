"""Tests for vector store module."""

import gc
import tempfile
from pathlib import Path

import pytest

from src.chunker import Chunk
from src.vector_store import SearchResult, VectorStore, reset_vector_store


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests.

    Yields the path and ensures cleanup happens after ChromaDB is closed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
    # Force garbage collection to ensure file handles are released
    gc.collect()


@pytest.fixture
def vector_store(temp_dir: Path) -> VectorStore:
    """Create a vector store with temporary storage.

    Properly cleans up ChromaDB connections after each test.
    """
    # Reset the global singleton to avoid interference
    reset_vector_store()
    store = VectorStore(
        persist_directory=temp_dir / "chroma",
        collection_name="test_collection",
    )
    yield store
    # Cleanup: reset (clear data and close) ChromaDB before temp_dir cleanup
    store.reset()
    reset_vector_store()
    gc.collect()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            content="The cat sat on the mat.",
            chunk_index=0,
            start_char=0,
            end_char=24,
            metadata={"source_file": "doc1.txt", "file_type": "text"},
        ),
        Chunk(
            content="A dog played in the park.",
            chunk_index=1,
            start_char=24,
            end_char=49,
            metadata={"source_file": "doc1.txt", "file_type": "text"},
        ),
        Chunk(
            content="Quantum physics explains the nature of matter.",
            chunk_index=0,
            start_char=0,
            end_char=46,
            metadata={"source_file": "doc2.txt", "file_type": "text"},
        ),
    ]


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_source_file_property(self) -> None:
        """Test source_file property."""
        result = SearchResult(
            chunk_id="123",
            content="Test content",
            metadata={"source_file": "test.txt"},
            distance=0.1,
            similarity=0.9,
        )
        assert result.source_file == "test.txt"

    def test_source_file_missing(self) -> None:
        """Test source_file when not in metadata."""
        result = SearchResult(
            chunk_id="123",
            content="Test content",
            metadata={},
            distance=0.1,
            similarity=0.9,
        )
        assert result.source_file is None

    def test_chunk_index_property(self) -> None:
        """Test chunk_index property."""
        result = SearchResult(
            chunk_id="123",
            content="Test content",
            metadata={"chunk_index": 5},
            distance=0.1,
            similarity=0.9,
        )
        assert result.chunk_index == 5


class TestVectorStore:
    """Tests for VectorStore class."""

    def test_init_creates_directory(self, temp_dir: Path) -> None:
        """Test that init creates the persist directory."""
        store_path = temp_dir / "new_store" / "chroma"
        store = VectorStore(persist_directory=store_path)
        assert store_path.exists()
        store.reset()  # Full cleanup since we don't need the data

    def test_init_creates_collection(self, vector_store: VectorStore) -> None:
        """Test that init creates a collection."""
        assert vector_store.collection is not None
        assert vector_store.collection_name == "test_collection"

    def test_count_empty(self, vector_store: VectorStore) -> None:
        """Test count on empty store."""
        assert vector_store.count() == 0

    def test_add_chunk(self, vector_store: VectorStore) -> None:
        """Test adding a single chunk."""
        chunk = Chunk(
            content="Test content for embedding.",
            chunk_index=0,
            start_char=0,
            end_char=27,
            metadata={"source_file": "test.txt"},
        )

        chunk_id = vector_store.add_chunk(chunk)

        assert chunk_id is not None
        assert vector_store.count() == 1

    def test_add_chunk_with_id(self, vector_store: VectorStore) -> None:
        """Test adding a chunk with custom ID."""
        chunk = Chunk(
            content="Test content.",
            chunk_index=0,
            start_char=0,
            end_char=13,
        )

        chunk_id = vector_store.add_chunk(chunk, chunk_id="custom_id_123")

        assert chunk_id == "custom_id_123"

    def test_add_chunks(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test adding multiple chunks."""
        chunk_ids = vector_store.add_chunks(sample_chunks)

        assert len(chunk_ids) == 3
        assert vector_store.count() == 3

    def test_search(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test searching for similar chunks."""
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search("Where is the cat?", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        # First result should be about the cat
        assert "cat" in results[0].content.lower()

    def test_search_empty_store(self, vector_store: VectorStore) -> None:
        """Test searching empty store."""
        results = vector_store.search("test query")
        assert results == []

    def test_search_with_metadata_filter(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test searching with metadata filter."""
        vector_store.add_chunks(sample_chunks)

        # Search only in doc1.txt
        results = vector_store.search(
            "animals",
            top_k=5,
            filter_metadata={"source_file": "doc1.txt"},
        )

        # Should only return chunks from doc1.txt
        for result in results:
            assert result.metadata.get("source_file") == "doc1.txt"

    def test_get_chunk(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test getting a chunk by ID."""
        chunk_ids = vector_store.add_chunks(sample_chunks)

        result = vector_store.get_chunk(chunk_ids[0])

        assert result is not None
        assert result.chunk_id == chunk_ids[0]
        assert result.content == sample_chunks[0].content

    def test_get_chunk_not_found(self, vector_store: VectorStore) -> None:
        """Test getting non-existent chunk."""
        result = vector_store.get_chunk("nonexistent_id")
        assert result is None

    def test_delete_chunk(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test deleting a chunk."""
        chunk_ids = vector_store.add_chunks(sample_chunks)
        initial_count = vector_store.count()

        success = vector_store.delete_chunk(chunk_ids[0])

        assert success
        assert vector_store.count() == initial_count - 1
        assert vector_store.get_chunk(chunk_ids[0]) is None

    def test_delete_by_metadata(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test deleting chunks by metadata."""
        vector_store.add_chunks(sample_chunks)

        # Delete all chunks from doc1.txt (2 chunks)
        deleted = vector_store.delete_by_metadata({"source_file": "doc1.txt"})

        assert deleted == 2
        assert vector_store.count() == 1  # Only doc2.txt remains

    def test_delete_document(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test deleting all chunks from a document."""
        vector_store.add_chunks(sample_chunks)

        deleted = vector_store.delete_document("doc1.txt")

        assert deleted == 2
        assert vector_store.count() == 1

    def test_list_documents(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test listing unique documents."""
        vector_store.add_chunks(sample_chunks)

        docs = vector_store.list_documents()

        assert len(docs) == 2
        source_files = {d["source_file"] for d in docs}
        assert source_files == {"doc1.txt", "doc2.txt"}

        # Check chunk counts
        for doc in docs:
            if doc["source_file"] == "doc1.txt":
                assert doc["chunk_count"] == 2
            else:
                assert doc["chunk_count"] == 1

    def test_list_documents_empty(self, vector_store: VectorStore) -> None:
        """Test listing documents on empty store."""
        docs = vector_store.list_documents()
        assert docs == []

    def test_clear(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test clearing the store."""
        vector_store.add_chunks(sample_chunks)
        assert vector_store.count() > 0

        deleted = vector_store.clear()

        assert deleted == 3
        assert vector_store.count() == 0

    def test_clear_empty(self, vector_store: VectorStore) -> None:
        """Test clearing empty store."""
        deleted = vector_store.clear()
        assert deleted == 0

    def test_get_stats(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test getting store statistics."""
        vector_store.add_chunks(sample_chunks)

        stats = vector_store.get_stats()

        assert stats["collection_name"] == "test_collection"
        assert stats["total_chunks"] == 3
        assert stats["total_documents"] == 2
        assert len(stats["documents"]) == 2

    def test_persistence(self, temp_dir: Path, sample_chunks: list[Chunk]) -> None:
        """Test that data persists across store instances."""
        persist_path = temp_dir / "persist_test"

        # Create store and add data
        store1 = VectorStore(
            persist_directory=persist_path,
            collection_name="persist_test",
        )
        store1.add_chunks(sample_chunks)
        count1 = store1.count()
        # Close first store (release resources, keep data)
        store1.close()
        gc.collect()  # Ensure file handles are released

        # Create new store instance with same path
        store2 = VectorStore(
            persist_directory=persist_path,
            collection_name="persist_test",
        )

        # Data should still be there
        assert store2.count() == count1
        store2.reset()  # Full cleanup

    def test_search_results_sorted_by_similarity(
        self, vector_store: VectorStore, sample_chunks: list[Chunk]
    ) -> None:
        """Test that search results are sorted by similarity."""
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search("cat mat", top_k=3)

        # Results should be sorted by similarity (descending)
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity
