"""Configuration management for RAG CLI."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    """Find the project root by looking for .env or data directory."""
    current = Path.cwd()

    # Check current directory and parents for project indicators
    for path in [current] + list(current.parents):
        if (path / ".env").exists() or (path / "data").exists():
            return path
        # Stop at home directory
        if path == Path.home():
            break

    return current


# Load environment variables from .env file
PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")


class LLMSettings(BaseSettings):
    """LLM-related configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model to use",
    )

    # Claude settings
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude",
    )

    # General LLM settings
    default_llm_provider: Literal["ollama", "claude"] = Field(
        default="ollama",
        description="Default LLM provider",
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation",
    )
    max_tokens: int = Field(
        default=2000,
        ge=1,
        description="Maximum tokens for LLM response",
    )


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    chroma_persist_dir: str = Field(
        default="./data/vector_db",
        description="Directory for ChromaDB persistence",
    )

    @property
    def persist_path(self) -> Path:
        """Get the absolute path for ChromaDB persistence."""
        path = Path(self.chroma_persist_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )


class ChunkingSettings(BaseSettings):
    """Text chunking configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    chunk_size: int = Field(
        default=1200,
        ge=100,
        le=10000,
        description="Size of text chunks in characters",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks in characters",
    )


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    top_k_results: int = Field(
        default=8,
        ge=1,
        le=100,
        description="Number of chunks to retrieve",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_file: str | None = Field(
        default=None,
        description="Log file path (None for console only)",
    )

    @property
    def log_path(self) -> Path | None:
        """Get the absolute path for log file."""
        if not self.log_file:
            return None
        path = Path(self.log_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


class Settings(BaseSettings):
    """Main settings class combining all configuration sections."""

    model_config = SettingsConfigDict(extra="ignore")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return PROJECT_ROOT / "data"

    @property
    def documents_dir(self) -> Path:
        """Get the documents directory path."""
        return self.data_dir / "documents"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    _settings = Settings()
    return _settings
