from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List


class Settings(BaseSettings):
    gemini_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    llm_provider: str = "gemini"  # gemini or ollama

    # Primary model — overridable via LLM_MODEL in .env
    llm_model: str = "gemini-2.5-flash"

    # Ordered fallback list used when the primary model hits quota / rate limits.
    # The service iterates through these before giving up.
    # Override via GEMINI_FALLBACK_MODELS (comma-separated) in .env.
    gemini_fallback_models: List[str] = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-3.5-flash",
    ]

    embedding_model: str = "gemini-embedding-001"
    chroma_db_dir: str = "./chroma_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
