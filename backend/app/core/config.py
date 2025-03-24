"""
Configuration Module

This module provides configuration settings for the application,
loading values from environment variables with sensible defaults.
"""

import os
import secrets
from typing import List, Optional
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MCP Scientific Paper Analyzer"

    # Security settings
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)

    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:8000",  # Backend
        "http://localhost:8501",  # Streamlit default
        "http://localhost:3000",  # Potential React frontend
    ]

    # Database settings
    POSTGRES_SERVER: str = Field(default="localhost")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_DB: str = Field(default="mcp_scientific")
    DATABASE_URL: Optional[str] = None

    # Computed property for SQLAlchemy DB URI
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return self.DATABASE_URL or f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"

    # MCP settings
    MCP_STORAGE_PATH: Path = Field(default=Path("./storage"))

    # LLM settings
    ANTHROPIC_API_KEY: str = Field(default="")
    ANTHROPIC_MODEL: str = Field(default="claude-3-5-sonnet-20240229")

    # Storage settings
    UPLOAD_DIR: Path = Field(default=Path("./uploads"))
    MAX_UPLOAD_SIZE: int = Field(default=10_485_760)  # 10MB

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO")

    class Config:
        """Pydantic config class."""
        env_file = ".env"
        case_sensitive = True


# Global settings object
settings = Settings()


def ensure_directories():
    """Ensure necessary directories exist."""
    settings.MCP_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Create directories when the module is imported
ensure_directories()

# Computed paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
