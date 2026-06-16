from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    gemini_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    llm_provider: str = "gemini"  # gemini or ollama
    llm_model: str = "gemini-1.5-flash"
    embedding_model: str = "text-embedding-004"
    chroma_db_dir: str = "./chroma_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
