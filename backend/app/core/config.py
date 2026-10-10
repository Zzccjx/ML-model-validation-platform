from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str = 'postgresql+asyncpg://postgres:postgres@localhost:5432/ai_validation'
    SECRET_KEY: str = 'change-me-in-production'
    JWT_SECRET: str = 'change-me-in-production'
    JWT_ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    UPLOAD_DIRECTORY: str = './uploads'
    MAX_UPLOAD_SIZE_MB: int = 100
    CORS_ORIGINS: list[str] = ['http://localhost:5173']
    LOG_LEVEL: str = 'INFO'
    APP_NAME: str = 'AI/ML Validation Platform'
    APP_VERSION: str = '0.1.0'

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False)

@lru_cache
def get_settings():
    return Settings()
