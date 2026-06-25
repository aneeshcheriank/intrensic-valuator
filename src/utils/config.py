"""
Configuration management.

Loads settings from .env via python-dotenv and exposes a typed
Config dataclass consumed by the rest of the application.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve the project root (one level up from src/utils/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_dotenv() -> Path | None:
    """Return the path to the .env file if it exists, else None."""
    candidate = _PROJECT_ROOT / ".env"
    return candidate if candidate.exists() else None


# Load .env once at import time.  If the file is missing we still
# want the process to continue — the caller is responsible for
# handling missing keys.
_load_result = load_dotenv(dotenv_path=_find_dotenv(), override=False)

# ---------------------------------------------------------------------------
# Config holder
# ---------------------------------------------------------------------------


class Config:
    """Typed container for application configuration."""

    def __init__(self) -> None:
        # -- LLM --
        self.deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")

        # -- Data APIs --
        self.fred_api_key: str = os.getenv("FRED_API_KEY", "")
        self.fmp_api_key: str = os.getenv("FMP_API_KEY", "")
        self.alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

        # -- Cache --
        cache_dir = os.getenv("CACHE_DIR", str(_PROJECT_ROOT / "cache"))
        self.cache_dir: Path = Path(cache_dir).resolve()
        self.cache_db_path: Path = self.cache_dir / "valuator_cache.db"
        self.http_cache_path: Path = self.cache_dir / "valuator_http_cache"

    @property
    def has_deepseek(self) -> bool:
        return bool(self.deepseek_api_key and not self.deepseek_api_key.startswith("sk-your-"))

    @property
    def has_fred(self) -> bool:
        return bool(self.fred_api_key and self.fred_api_key != "your-fred-api-key-here")

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key and self.tavily_api_key != "your-tavily-api-key-here")

    def ensure_cache_dir(self) -> Path:
        """Create the cache directory if it doesn't exist. Returns its path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir


# Singleton
config = Config()


def get_config() -> Config:
    """Return the application-wide Config singleton."""
    return config
