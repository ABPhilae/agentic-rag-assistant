from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = 'gpt-4o-mini'
    openai_embedding_model: str = 'text-embedding-3-small'

    # Qdrant
    qdrant_host: str = 'qdrant'
    qdrant_port: int = 6333
    qdrant_collection: str = 'audit_documents'

    # Redis
    redis_url: str = 'redis://redis:6379'

    # NeMo Guardrails
    guardrails_url: str = 'http://guardrails:8080'
    use_guardrails: bool = True

    # App
    app_env: str = 'development'
    log_level: str = 'INFO'

    class Config:
        env_file = '.env'
        extra = 'ignore'


@lru_cache()
def get_settings() -> Settings:
    return Settings()
