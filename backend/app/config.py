"""
Application settings loaded from environment variables.

Uses pydantic-settings to validate and type-check all config values at startup.
If a required env var is missing, the app fails fast with a clear error message
instead of crashing at runtime when the value is first used.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application configuration.

    Values are read from environment variables (case-insensitive).
    A `.env` file in the backend/ directory is loaded automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────
    app_name: str = "Decisio"
    debug: bool = True
    api_prefix: str = "/api/v1"

    # ── Database ─────────────────────────────────────────
    # SQLite for local dev; Postgres for production.
    # Example Postgres: postgresql+asyncpg://user:pass@localhost:5432/decisio
    database_url: str = "sqlite+aiosqlite:///./decisio.db"

    # ── GitHub Integration ───────────────────────────────
    github_client_id: str = ""
    github_client_secret: str = ""
    # Optional personal token for testing or higher API rate limits (5,000 req/hr vs 60 req/hr)
    github_token: str = ""

    # ── LLM Multi-Provider Configuration ─────────────────
    # Active default provider: "groq" | "openai" | "anthropic" | "grok" | "mock"
    llm_provider: Literal["groq", "openai", "anthropic", "grok", "mock"] = "groq"

    # 1. Groq (GroqCloud - https://console.groq.com/keys) - Ultra-fast inference
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # 2. OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # 3. Claude (Anthropic)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # 4. Grok (xAI) - Supports both GROK_API_KEY and XAI_API_KEY
    grok_api_key: str = ""
    xai_api_key: str = ""
    grok_model: str = "grok-2-latest"
    grok_base_url: str = "https://api.x.ai/v1"

    @property
    def effective_grok_api_key(self) -> str:
        return self.grok_api_key or self.xai_api_key

    # ── Security ─────────────────────────────────────────
    # Used to encrypt stored GitHub tokens and sign JWTs.
    # Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
    secret_key: str = "CHANGE-ME-IN-PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week

    # ── Extraction ───────────────────────────────────────
    # Max PRs to process per sync job (prevents runaway costs).
    max_prs_per_sync: int = 100

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def data_dir(self) -> Path:
        """Directory for local data files (SQLite DB, etc.)."""
        path = Path("./data")
        path.mkdir(exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — settings are read once at startup."""
    return Settings()
