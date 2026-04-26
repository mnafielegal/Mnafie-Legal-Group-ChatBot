import os
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    API_TITLE: str = "Mnafie Legal Group Chatbot Backend"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    HUGGINGFACE_API_KEY: str | None = os.getenv("HUGGINGFACE_API_KEY")
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    API_PROVIDER: Literal["openai", "anthropic"] = os.getenv("API_PROVIDER", "anthropic")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_EMBEDDING_DIMENSIONS: int = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "256"))
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    DOCKER_IMAGE: str = "chatbot:latest"

    MAX_CONCURRENT_SESSIONS: int = 10
    SESSION_TIMEOUT_MINUTES: int = 60
    MAX_TRANSFER_REPLY_LENGTH: int = 100

    BASE_DIR: Path = Path(__file__).parent.parent

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
