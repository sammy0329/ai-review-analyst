"""
Configuration module using pydantic-settings.
Loads environment variables from .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API Key")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for generation",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model for embeddings",
    )

    # ChromaDB Configuration
    chroma_persist_directory: Path = Field(
        default=Path("./data/chroma_db"),
        description="Directory for ChromaDB persistence",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Streamlit Configuration
    streamlit_server_port: int = Field(
        default=8501,
        description="Streamlit server port",
    )

    # Crawler Configuration
    crawler_timeout: int = Field(
        default=30000,
        description="Crawler timeout in milliseconds",
    )
    crawler_max_retries: int = Field(
        default=3,
        description="Maximum number of crawler retries",
    )

    # RAG Configuration
    retriever_top_k: int = Field(
        default=5,
        description="Number of documents to retrieve",
    )
    chunk_size: int = Field(
        default=500,
        description="Maximum chunk size for text splitting",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are only loaded once.

    Raises:
        ValidationError: If required environment variables are not set.
    """
    return Settings()


class _SettingsProxy:
    """
    Lazy proxy for settings.
    Only loads settings when accessed, allowing import without .env file.
    """

    _settings: Settings | None = None

    def __getattr__(self, name: str):
        if self._settings is None:
            self._settings = get_settings()
        return getattr(self._settings, name)


# Convenience instance (lazy loaded)
settings = _SettingsProxy()
