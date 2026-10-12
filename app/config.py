from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration for MMpact RAG."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # -- logging to disk ----------------------------------------------------
    log_dir: str = ""
    log_file_name: str = "rag-service.log"
    log_file_max_mb: int = 20
    log_file_keep: int = 10

    # Shared secret every caller presents as `Authorization: Bearer <token>`.
    rag_api_token: str = ""

    # -- MongoDB logging ----------------------------------------------------
    mongodb_url: str = ""
    mongodb_database: str = "universal_rag"
    mongodb_collection: str = "rag_logs"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

