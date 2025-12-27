"""Configuration management for OptionsRadar."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AlpacaConfig:
    """Alpaca API configuration."""

    api_key: str
    secret_key: str
    paper: bool

    @property
    def base_url(self) -> str:
        """REST API base URL."""
        if self.paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"

    @property
    def data_url(self) -> str:
        """Market data API base URL."""
        return "https://data.alpaca.markets"

    @property
    def options_stream_url(self) -> str:
        """Options WebSocket stream URL."""
        return "wss://stream.data.alpaca.markets/v1beta1/opra"

    @property
    def stocks_stream_url(self) -> str:
        """Stocks WebSocket stream URL (for underlying prices)."""
        return "wss://stream.data.alpaca.markets/v2/iex"


@dataclass(frozen=True)
class ORATSConfig:
    """ORATS API configuration."""

    api_token: str
    base_url: str = "https://api.orats.io/datav2"


@dataclass(frozen=True)
class FinnhubConfig:
    """Finnhub API configuration."""

    api_key: str
    base_url: str = "https://finnhub.io/api/v1"


@dataclass(frozen=True)
class QuiverConfig:
    """Quiver Quant API configuration."""

    api_key: str
    base_url: str = "https://api.quiverquant.com/beta"


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""

    alpaca: AlpacaConfig
    orats: ORATSConfig
    finnhub: FinnhubConfig
    quiver: QuiverConfig
    log_level: str

    # MVP watchlist
    watchlist: tuple[str, ...] = ("NVDA",)

    # Staleness thresholds (seconds)
    quote_stale_threshold: float = 5.0
    greeks_stale_threshold: float = 90.0
    underlying_stale_threshold: float = 2.0

    # Reconnection settings
    ws_reconnect_base_delay: float = 1.0
    ws_reconnect_max_delay: float = 60.0
    ws_reconnect_max_attempts: int | None = None  # None = infinite


def _get_env_or_raise(key: str) -> str:
    """Get environment variable or raise if not set."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes")


@lru_cache
def load_config() -> AppConfig:
    """Load configuration from environment.

    Returns:
        AppConfig instance with all settings loaded.

    Raises:
        ValueError: If required environment variables are missing.
    """
    # Load .env file if present
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    return AppConfig(
        alpaca=AlpacaConfig(
            api_key=_get_env_or_raise("ALPACA_API_KEY"),
            secret_key=_get_env_or_raise("ALPACA_SECRET_KEY"),
            paper=_get_env_bool("ALPACA_PAPER", default=True),
        ),
        orats=ORATSConfig(
            api_token=os.getenv("ORATS_API_TOKEN", ""),
        ),
        finnhub=FinnhubConfig(
            api_key=os.getenv("FINNHUB_API_KEY", ""),
        ),
        quiver=QuiverConfig(
            api_key=os.getenv("QUIVER_API_KEY", ""),
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
