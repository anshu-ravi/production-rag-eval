"""Configuration management using pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM API Keys
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama base URL"
    )

    # LLM Model Configuration (per provider)
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    anthropic_model: str = Field(default="claude-haiku-4-5", description="Anthropic model name")
    ollama_model: str = Field(default="llama3.2", description="Ollama model name")

    # Vector Store
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")

    # Embeddings and Retrieval
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model for embeddings"
    )
    top_k: int = Field(default=10, description="Number of documents to retrieve")
    chunk_size: int = Field(default=512, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Chunk overlap in tokens")
    semantic_similarity_threshold: float = Field(
        default=0.85, description="Threshold for semantic chunking"
    )


# Global config instance
config = Config()
