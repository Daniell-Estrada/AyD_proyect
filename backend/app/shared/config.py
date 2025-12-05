"""Centralized configuration for the clean architecture stack."""

from typing import List, Literal, Optional

import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str = Field(
        default="", description="Anthropic API key (empty for tests)"
    )
    openai_api_key: str = Field(
        default="", description="OpenAI API key (empty for tests)"
    )
    gemini_api_key: str = Field(
        default="", description="Gemini API key (empty for tests)"
    )
    mistral_api_key: Optional[str] = Field(
        default=None, description="Mistral API key (optional)"
    )

    google_model: str = Field(
        default="gemini-2.5-flash", description="Default Google Gemini model identifier"
    )

    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI",
    )
    mongodb_db_name: str = Field(default="AyD", description="MongoDB database name")

    enable_state_persistence: bool = Field(
        default=True,
        description="Enable persistence of workflow snapshots in Redis",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for workflow state snapshots",
    )
    redis_namespace: str = Field(
        default="complexity_workflow",
        description="Redis key namespace for workflow artifacts",
    )
    redis_state_ttl: int = Field(
        default=86400,
        description="Expiration time (seconds) for workflow state snapshots",
    )

    app_env: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )
    api_debug: bool = Field(default=True, description="Enable debug mode")
    secret_key: str = Field(
        default="dev-secret-key", description="Secret key for JWT and encryption"
    )
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=5000, description="API port")

    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins",
    )

    primary_llm_provider: Literal[
        "anthropic", "openai", "github", "google", "mistral"
    ] = Field(default="anthropic", description="Primary LLM provider")
    primary_llm_model: str = Field(default="gpt4.1", description="Primary LLM model")
    fallback_llm_provider: Literal[
        "anthropic", "openai", "github", "google", "mistral"
    ] = Field(default="openai", description="Fallback LLM provider")
    fallback_llm_model: str = Field(
        default="gpt-4-turbo-preview", description="Fallback LLM model"
    )
    max_tokens: int = Field(default=4096, description="Maximum tokens per request")
    temperature: float = Field(default=0.1, description="LLM temperature")

    github_token: Optional[str] = Field(
        default=None, description="GitHub Models access token"
    )
    github_endpoint: str = Field(
        default="https://models.github.ai/inference",
        description="GitHub Models inference endpoint",
    )
    github_model: str = Field(
        default="openai/gpt-4.1", description="Default GitHub Models identifier"
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    max_session_duration: int = Field(
        default=3600, description="Maximum session duration in seconds"
    )
    enable_hitl: bool = Field(default=True, description="Enable Human-in-the-Loop mode")

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string if needed."""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins


settings = Settings()
