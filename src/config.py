import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # =========================================================================
    # Application & Environment Settings
    # =========================================================================
    PROJECT_NAME: str = "Agentic IRDAI Life Insurance Advisory System"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # =========================================================================
    # OpenAI LLM Configuration
    # =========================================================================
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.1

    # =========================================================================
    # Hugging Face Embeddings Configuration
    # =========================================================================
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = None
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # =========================================================================
    # Qdrant Vector Store Configuration
    # =========================================================================
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "term_insurance_policies"

    # =========================================================================
    # Database Persistence Configuration (Neon / Supabase PostgreSQL)
    # =========================================================================
    DATABASE_URL: Optional[str] = None

    # =========================================================================
    # Pydantic V2 Configuration Settings
    # =========================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()