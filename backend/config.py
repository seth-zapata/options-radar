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
class EODHDConfig:
    """EODHD API configuration for historical options data."""

    api_key: str
    base_url: str = "https://eodhd.com/api"  # Marketplace API at /mp/unicornbay/options


@dataclass(frozen=True)
class SignalQualityConfig:
    """Signal quality filter configuration.

    Based on backtest results showing which symbols/conditions have predictive power.
    """
    # Symbols where signals are disabled (poor backtest accuracy ~50% or worse)
    # These can still be monitored in scanner but won't generate recommendations
    # AMD/AAPL: ~50% accuracy, QQQ: 46.7% (worse than random for ETFs)
    # GME: 34.9% 5-day (inverse), SMCI: 38.1% 1-day (inverse)
    signals_disabled: tuple[str, ...] = ("AMD", "AAPL", "QQQ", "GME", "SMCI")

    # IV Rank thresholds for buy signals
    # Buy signals only fire when IV is relatively cheap
    iv_rank_max_for_buys: float = 45.0  # Only buy when IV Rank <= 45
    iv_rank_elevated_threshold: float = 70.0  # Flag as "elevated IV risk" above 70

    # WSB mention thresholds
    # Higher mentions = more reliable signal (backtest hypothesis)
    min_mentions_for_signal: int = 5  # Require at least 5 mentions
    high_mention_threshold: int = 20  # 20+ mentions = "high conviction"

    # Sentiment alignment requirement
    # When True, signals only fire when news AND WSB sentiment agree
    require_sentiment_alignment: bool = True

    # Confidence boosts/penalties
    alignment_confidence_boost: int = 10  # Bonus when news + WSB strongly aligned
    high_mentions_confidence_boost: int = 5  # Bonus for high mention count
    elevated_iv_confidence_penalty: int = 15  # Penalty for IV > 70

    # Recency weighting for sentiment data
    # Fresh sentiment data gets a confidence boost, stale data gets penalized
    sentiment_fresh_hours: float = 4.0  # Sentiment < 4 hours old = fresh
    sentiment_stale_hours: float = 24.0  # Sentiment > 24 hours old = stale
    recency_fresh_boost: int = 10  # Bonus for fresh sentiment
    recency_stale_penalty: int = 10  # Penalty for stale sentiment


@dataclass(frozen=True)
class RegimeStrategyConfig:
    """Regime-filtered intraday strategy configuration.

    Validated through backtesting on TSLA 2024-01 to 2025-01:
    - 71 trades, 43.7% win rate, +17.4% avg return
    - +1238% total return (with 10% position sizing)

    Regime Detection Thresholds (calibrated from WSB sentiment distribution):
    - 10th percentile bearish: -0.103
    - 90th percentile bullish: +0.071
    - Strong signals at top 5%: roughly +/-0.12 to 0.15
    """

    # Regime thresholds (WSB sentiment values)
    strong_bullish_threshold: float = 0.12
    moderate_bullish_threshold: float = 0.07
    moderate_bearish_threshold: float = -0.08
    strong_bearish_threshold: float = -0.15

    # Regime window (trading days regime stays active after trigger)
    regime_window_days: int = 7

    # Entry thresholds
    pullback_threshold: float = 1.5  # % pullback from high for bullish entry
    bounce_threshold: float = 1.5  # % bounce from low for bearish entry

    # Option selection
    target_dte: int = 7  # Target days to expiration (weeklies)
    min_dte: int = 4  # Minimum DTE to consider
    max_dte: int = 14  # Maximum DTE to consider

    # Liquidity gates
    min_open_interest: int = 500
    min_volume: int = 100

    # Exit rules
    take_profit_percent: float = 40.0  # Exit at +40%
    stop_loss_percent: float = -20.0  # Exit at -20%
    min_dte_exit: int = 1  # Exit when DTE falls to 1

    # Position sizing
    position_size_pct: float = 10.0  # % of portfolio per trade
    max_concurrent_positions: int = 3
    min_days_between_entries: int = 1  # Cooldown between entries

    # Enabled symbols (TSLA only validated, others need backtesting)
    enabled_symbols: tuple[str, ...] = ("TSLA",)


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""

    alpaca: AlpacaConfig
    orats: ORATSConfig
    finnhub: FinnhubConfig
    quiver: QuiverConfig
    eodhd: EODHDConfig
    log_level: str

    # Watchlist - expanded with WSB favorites
    # Signal-enabled: NVDA, TSLA, PLTR, COIN, MARA, RKLB, ASTS (>55% backtest accuracy)
    # Scanner-only: QQQ, AAPL, SPY, AMD, GOOGL, AMZN, META, MSFT, GME, SMCI
    watchlist: tuple[str, ...] = (
        # Signal-enabled (high backtest accuracy)
        "NVDA", "TSLA", "PLTR", "COIN", "MARA", "RKLB", "ASTS",
        # Scanner-only (disabled for signals)
        "QQQ", "AAPL", "SPY", "AMD", "GOOGL", "AMZN", "META", "MSFT", "GME", "SMCI",
    )

    # Staleness thresholds (seconds)
    quote_stale_threshold: float = 5.0
    greeks_stale_threshold: float = 90.0
    underlying_stale_threshold: float = 2.0

    # Reconnection settings
    ws_reconnect_base_delay: float = 1.0
    ws_reconnect_max_delay: float = 60.0
    ws_reconnect_max_attempts: int | None = None  # None = infinite

    # Signal quality filters
    signal_quality: SignalQualityConfig = SignalQualityConfig()

    # Regime-filtered strategy configuration
    regime_strategy: RegimeStrategyConfig = RegimeStrategyConfig()


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
        eodhd=EODHDConfig(
            api_key=os.getenv("EODHD_API_KEY", ""),
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
