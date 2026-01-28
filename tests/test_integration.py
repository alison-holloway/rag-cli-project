"""Integration tests for the RAG pipeline.

These tests verify the complete end-to-end flow from document loading
through retrieval and (mocked) generation.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.chunker import TextChunker
from src.document_loader import DocumentLoader
from src.llm_client import LLMResponse
from src.pipeline import QueryResult, RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore


class TestEndToEndRAGFlow:
    """Tests for the complete RAG pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_documents(self, temp_dir):
        """Create sample documents for testing."""
        # Create markdown document about Python
        python_doc = temp_dir / "python_guide.md"
        python_doc.write_text("""# Python Programming Guide

## Introduction
Python is a high-level, interpreted programming language known for its simplicity
and readability. It was created by Guido van Rossum and first released in 1991.

## Key Features
- **Easy to Learn**: Python has a clean syntax that makes it easy for beginners.
- **Versatile**: Used for web development, data science, AI, automation, and more.
- **Large Community**: Extensive libraries and active community support.

## Data Types
Python supports several built-in data types:
- Integers (int): Whole numbers like 1, 42, -17
- Floats (float): Decimal numbers like 3.14, -0.5
- Strings (str): Text data like "Hello, World!"
- Lists (list): Ordered collections like [1, 2, 3]
- Dictionaries (dict): Key-value pairs like {"name": "Alice", "age": 30}

## Functions
Functions in Python are defined using the `def` keyword:
```python
def greet(name):
    return f"Hello, {name}!"
```
""")

        # Create document about machine learning
        ml_doc = temp_dir / "machine_learning.md"
        ml_doc.write_text("""# Machine Learning Basics

## What is Machine Learning?
Machine learning is a subset of artificial intelligence (AI) that enables
systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning
1. **Supervised Learning**: Learning from labeled data
2. **Unsupervised Learning**: Finding patterns in unlabeled data
3. **Reinforcement Learning**: Learning through trial and error

## Common Algorithms
- Linear Regression: Predicting continuous values
- Decision Trees: Classification and regression
- Neural Networks: Deep learning for complex patterns
- K-Means: Clustering similar data points

## Applications
Machine learning is used in:
- Image recognition
- Natural language processing
- Recommendation systems
- Fraud detection
""")

        return [python_doc, ml_doc]

    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create a vector store for testing."""
        store = VectorStore(
            persist_directory=temp_dir / "vector_db",
            collection_name="test_collection",
        )
        yield store
        # Cleanup happens automatically with temp_dir

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.provider = "mock"
        client.is_available.return_value = True

        def generate_response(prompt, system_prompt=None, **kwargs):
            # Generate a contextual response based on the prompt
            if "Python" in prompt:
                content = "Python is a programming language by Guido van Rossum."
            elif "machine learning" in prompt.lower():
                content = "Machine learning is a subset of AI for learning from data."
            else:
                content = "I don't have enough information to answer that."

            return LLMResponse(
                content=content,
                model="mock-model",
                provider="mock",
            )

        client.generate.side_effect = generate_response
        return client

    def test_document_loading_and_chunking(self, sample_documents, temp_dir):
        """Test loading documents and chunking them."""
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)

        # Load documents (generator -> list)
        documents = list(loader.load_directory(temp_dir, recursive=False))
        assert len(documents) == 2

        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            assert len(chunks) > 0

        # Verify we have multiple chunks
        assert len(all_chunks) >= 4  # At least a few chunks from each doc

        # Verify chunks have metadata
        for chunk in all_chunks:
            assert chunk.metadata.get("source_file") is not None
            assert chunk.content

    def test_vector_store_indexing_and_search(
        self, sample_documents, temp_dir, vector_store
    ):
        """Test indexing documents and searching."""
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)

        # Load and chunk
        documents = list(loader.load_directory(temp_dir, recursive=False))
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunker.chunk_document(doc))

        # Index
        chunk_ids = vector_store.add_chunks(all_chunks)
        assert len(chunk_ids) == len(all_chunks)

        # Verify count
        assert vector_store.count() == len(all_chunks)

        # Search for Python content
        results = vector_store.search("What are Python data types?", top_k=3)
        assert len(results) > 0

        # Top result should be from Python document
        assert any("python" in r.source_file.lower() for r in results if r.source_file)

        # Search for ML content
        ml_results = vector_store.search(
            "types of machine learning algorithms", top_k=3
        )
        assert len(ml_results) > 0

    def test_retriever_integration(self, sample_documents, temp_dir, vector_store):
        """Test the retriever with real documents."""
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)

        # Load, chunk, and index
        documents = list(loader.load_directory(temp_dir, recursive=False))
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            vector_store.add_chunks(chunks)

        # Create retriever
        retriever = Retriever(vector_store=vector_store, top_k=5)

        # Test retrieval
        result = retriever.retrieve("Who created Python?")

        assert result.total_chunks > 0
        assert result.avg_similarity > 0
        assert "Guido" in result.context or "Python" in result.context

    def test_full_pipeline_with_mock_llm(
        self, sample_documents, temp_dir, vector_store, mock_llm_client
    ):
        """Test the complete RAG pipeline with mocked LLM."""
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)

        # Load, chunk, and index documents
        documents = list(loader.load_directory(temp_dir, recursive=False))
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            vector_store.add_chunks(chunks)

        # Create pipeline
        retriever = Retriever(vector_store=vector_store, top_k=5)
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=mock_llm_client,
        )

        # Check readiness
        ready, message = pipeline.is_ready()
        assert ready

        # Query about Python
        result = pipeline.query("What is Python?")

        assert isinstance(result, QueryResult)
        assert result.query == "What is Python?"
        assert result.answer  # Should have an answer
        assert result.has_context  # Should have retrieved context
        assert len(result.sources) > 0  # Should have source files

        # Query about ML
        ml_result = pipeline.query("What are the types of machine learning?")
        assert ml_result.answer
        assert ml_result.has_context

    def test_pipeline_with_no_relevant_context(
        self, sample_documents, temp_dir, vector_store, mock_llm_client
    ):
        """Test pipeline when query doesn't match any documents."""
        # Create pipeline with empty vector store
        retriever = Retriever(vector_store=vector_store, top_k=5)
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=mock_llm_client,
        )

        # Query completely unrelated topic
        result = pipeline.query("What is the capital of Mongolia?")

        # Should use no_context template since nothing matches
        assert result.template_used == "no_context"
        assert result.has_context is False

    def test_streaming_query(self, sample_documents, temp_dir, vector_store):
        """Test streaming query functionality."""
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)

        # Load, chunk, and index
        documents = list(loader.load_directory(temp_dir, recursive=False))
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            vector_store.add_chunks(chunks)

        # Create mock LLM that streams
        mock_client = MagicMock()
        mock_client.provider = "mock"
        mock_client.generate_stream.return_value = iter(
            ["Python ", "is ", "a ", "programming ", "language."]
        )

        # Create pipeline
        retriever = Retriever(vector_store=vector_store, top_k=5)
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=mock_client,
        )

        # Stream query
        chunks = list(pipeline.query_stream("What is Python?"))

        assert len(chunks) == 5
        assert "".join(chunks) == "Python is a programming language."

    def test_multiple_queries_same_pipeline(
        self, sample_documents, temp_dir, vector_store, mock_llm_client
    ):
        """Test making multiple queries with the same pipeline."""
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)

        # Index documents
        documents = list(loader.load_directory(temp_dir, recursive=False))
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            vector_store.add_chunks(chunks)

        # Create pipeline
        retriever = Retriever(vector_store=vector_store, top_k=5)
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=mock_llm_client,
        )

        # Make multiple queries
        queries = [
            "What is Python?",
            "What are Python's key features?",
            "What is machine learning?",
            "What are the types of machine learning?",
        ]

        results = []
        for q in queries:
            result = pipeline.query(q)
            results.append(result)

        # All queries should succeed
        assert len(results) == 4
        for result in results:
            assert result.answer
            assert result.has_context


class TestRAGWithRealEmbeddings:
    """Tests that verify embeddings and similarity search work correctly."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_embedding_similarity(self, temp_dir):
        """Test that similar texts get similar embeddings."""
        vector_store = VectorStore(
            persist_directory=temp_dir / "test_db",
            collection_name="similarity_test",
        )

        # Add chunks with clearly different topics
        from src.chunker import Chunk

        chunks = [
            Chunk(
                content="Python is a programming language for web and data science.",
                chunk_index=0,
                start_char=0,
                end_char=100,
                metadata={"source_file": "python.txt", "topic": "programming"},
            ),
            Chunk(
                content="JavaScript is a programming language for web and frontend.",
                chunk_index=0,
                start_char=0,
                end_char=100,
                metadata={"source_file": "javascript.txt", "topic": "programming"},
            ),
            Chunk(
                content="Chocolate cake is made with flour, sugar, cocoa, and eggs.",
                chunk_index=0,
                start_char=0,
                end_char=100,
                metadata={"source_file": "recipe.txt", "topic": "cooking"},
            ),
        ]

        vector_store.add_chunks(chunks)

        # Search for programming content
        results = vector_store.search(
            "What programming language should I learn?", top_k=3
        )

        # Programming-related chunks should rank higher than cooking
        assert results[0].metadata.get("topic") == "programming"
        assert results[1].metadata.get("topic") == "programming"
        # Cooking should be last (least similar)
        assert results[2].metadata.get("topic") == "cooking"

    def test_semantic_search_accuracy(self, temp_dir):
        """Test semantic search returns contextually relevant results."""
        vector_store = VectorStore(
            persist_directory=temp_dir / "semantic_test",
            collection_name="semantic_test",
        )

        from src.chunker import Chunk

        # Add chunks with semantic variations
        chunks = [
            Chunk(
                content="The dog ran quickly through the park chasing a ball.",
                chunk_index=0,
                start_char=0,
                end_char=100,
                metadata={"source_file": "animals.txt", "topic": "pets"},
            ),
            Chunk(
                content="The canine sprinted across the field pursuing a toy.",
                chunk_index=1,
                start_char=0,
                end_char=100,
                metadata={"source_file": "animals.txt", "topic": "pets"},
            ),
            Chunk(
                content="The stock market experienced significant volatility today.",
                chunk_index=0,
                start_char=0,
                end_char=100,
                metadata={"source_file": "finance.txt", "topic": "finance"},
            ),
        ]

        vector_store.add_chunks(chunks)

        # Search for semantically similar content (not exact match)
        results = vector_store.search("A pet running in a garden", top_k=3)

        # Both pet-related chunks should rank higher than finance
        top_two_topics = {r.metadata.get("topic") for r in results[:2]}
        assert "pets" in top_two_topics
