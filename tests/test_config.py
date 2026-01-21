"""Tests for configuration module."""


from src.config import (
    Settings,
    get_settings,
)


def test_get_settings_returns_settings_instance() -> None:
    """Test that get_settings returns a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_default_llm_provider_is_ollama() -> None:
    """Test that the default LLM provider is ollama."""
    settings = get_settings()
    assert settings.llm.default_llm_provider == "ollama"


def test_default_embedding_model() -> None:
    """Test the default embedding model."""
    settings = get_settings()
    assert settings.embedding.embedding_model == "all-MiniLM-L6-v2"


def test_default_chunk_settings() -> None:
    """Test default chunking settings."""
    settings = get_settings()
    assert settings.chunking.chunk_size == 800
    assert settings.chunking.chunk_overlap == 100


def test_default_retrieval_settings() -> None:
    """Test default retrieval settings."""
    settings = get_settings()
    assert settings.retrieval.top_k_results == 5
