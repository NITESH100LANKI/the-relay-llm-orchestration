"""
Configuration settings for The Relay.
Loaded from environment variables or .env file.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Groq API Configuration
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="qwen/qwen3.8-27b", alias="GROQ_MODEL")
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Provider Mode: "real" or "stand-in"
    provider_mode: Literal["real", "stand-in"] = Field(default="stand-in", alias="RELAY_PROVIDER_MODE")

    # Server settings
    host: str = Field(default="127.0.0.1", alias="RELAY_HOST")
    port: int = Field(default=8000, alias="RELAY_PORT")

    # Concurrency and Reliability
    max_concurrency: int = Field(default=5, alias="RELAY_MAX_CONCURRENCY")
    request_deadline_sec: float = Field(default=30.0, alias="RELAY_REQUEST_DEADLINE_SEC")
    rate_limit_rpm: int = Field(default=30, alias="RELAY_RATE_LIMIT_RPM")
    rate_limit_tpm: int = Field(default=6000, alias="RELAY_RATE_LIMIT_TPM")
    rate_limit_rpd: int = Field(default=14400, alias="RELAY_RATE_LIMIT_RPD")

    # Paths
    base_dir: Path = BASE_DIR
    corpus_dir: Path = BASE_DIR / "corpus" / "papers"
    processed_dir: Path = BASE_DIR / "corpus" / "processed"
    indexes_dir: Path = BASE_DIR / "corpus" / "indexes"
    logs_dir: Path = BASE_DIR / "logs"
    configs_dir: Path = BASE_DIR / "configs"
    submission_dir: Path = BASE_DIR / "submission"

settings = Settings()

