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
class AutoExecutionConfig:
    """Auto-execution configuration for automated paper trading.

    When enabled, signals are automatically executed via Alpaca paper trading.
    Positions are monitored for exit conditions and auto-closed.

    Trading modes:
    - "off": No auto-execution, manual trading only
    - "simulation": Use MockAlpacaTrader for testing without Alpaca
    - "paper": Use Alpaca paper trading (requires credentials)
    """

    enabled: bool = False  # Must explicitly enable auto-execution
    mode: str = "off"  # "off" | "simulation" | "paper"
    position_size_pct: float = 10.0  # % of portfolio per trade
    max_positions: int = 3  # Maximum concurrent positions
    max_contract_price: float = 20.0  # Don't buy options over this price
    min_contract_price: float = 0.50  # Don't buy options under this price
    exit_check_interval: float = 1.0  # Seconds between exit checks (1s for responsive exits)
    use_limit_orders: bool = True  # Use limit orders at mid price
    limit_offset_pct: float = 0.5  # Offset from mid for limit orders
    simulation_speed: float = 5.0  # Speed multiplier for simulation mode
    simulation_balance: float = 100000.0  # Starting balance for simulation


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
class DualRegimeConfig:
    """Dual-regime signal generation configuration.

    When bear market is detected, switches from sentiment-based signals
    to momentum-based PUT signals ("sell the rally" strategy).
    """

    # Master enable for dual-regime mode
    enabled: bool = True

    # Bounce detection (identifies relief rally to sell into)
    bounce_threshold: float = 0.03  # 3% bounce from recent low required
    bounce_lookback_days: int = 5  # Days to look back for recent low

    # Resistance detection (50-day SMA as resistance in bear markets)
    resistance_proximity: float = 0.02  # Within 2% of 50-day SMA

    # RSI thresholds for momentum signals
    oversold_rsi: float = 25.0  # Don't short below this (capitulation)
    neutral_rsi: float = 50.0  # Bearish below this

    # VIX limits for bear market trading
    vix_max_for_puts: float = 35.0  # Too volatile above this


@dataclass(frozen=True)
class RiskManagementConfig:
    """Risk management configuration for the 5 improvements.

    Improvement 1: Earnings Blackout
    Improvement 2: VIX Regime Filter
    Improvement 3: Trading Hours Optimization
    Improvement 4: Dynamic Position Sizing
    Improvement 5: Price-Sentiment Divergence (Bear Market Protection)
    """

    # Improvement 1: Earnings Blackout
    earnings_blackout_enabled: bool = True
    earnings_blackout_days_before: int = 5  # Block X days before earnings
    earnings_blackout_days_after: int = 1  # Block X days after earnings

    # Improvement 2: VIX Regime Filter
    vix_filter_enabled: bool = True
    vix_panic_threshold: float = 35.0  # Block entries above this
    vix_elevated_threshold: float = 25.0  # Reduce position size above this

    # Improvement 3: Trading Hours Optimization
    trading_hours_enabled: bool = True
    block_market_open_minutes: int = 30  # Block first X minutes
    block_market_close_minutes: int = 15  # Block last X minutes

    # Improvement 4: Dynamic Position Sizing
    dynamic_sizing_enabled: bool = True
    base_position_size: float = 0.10  # 10% base
    max_position_size: float = 0.15  # 15% max
    min_position_size: float = 0.05  # 5% min

    # Improvement 5: Price-Sentiment Divergence (Bear Market Protection)
    # Blocks bullish signals when price action indicates bear market
    price_sentiment_divergence_enabled: bool = True
    bear_drawdown_threshold: float = 0.80  # Block if price < 80% of 52-week high (20% drawdown)
    bear_sma_threshold: float = 0.95  # Block if price < 95% of 200-day SMA
    bear_sma_days_required: int = 10  # Must be below SMA for X consecutive days
    death_cross_lookback_days: int = 60  # Consider death cross if within X days


@dataclass(frozen=True)
class ScalpingModuleConfig:
    """Scalping module configuration (high-level enable/disable).

    Detailed scalping config is in backend/scalping/config.py.
    This provides the top-level enable and data directory settings.
    """

    enabled: bool = False  # Master enable for scalping module
    databento_data_dir: str | None = None  # Path to DataBento CBBO data


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
    underlying_stale_threshold: float = 5.0  # Increased to match quote threshold

    # Reconnection settings
    ws_reconnect_base_delay: float = 1.0
    ws_reconnect_max_delay: float = 60.0
    ws_reconnect_max_attempts: int | None = None  # None = infinite

    # Signal quality filters
    signal_quality: SignalQualityConfig = SignalQualityConfig()

    # Regime-filtered strategy configuration
    regime_strategy: RegimeStrategyConfig = RegimeStrategyConfig()

    # Auto-execution configuration
    auto_execution: AutoExecutionConfig = AutoExecutionConfig()

    # Risk management configuration (4 improvements)
    risk_management: RiskManagementConfig = RiskManagementConfig()

    # Dual-regime configuration (bear market momentum signals)
    dual_regime: DualRegimeConfig = DualRegimeConfig()

    # Scalping module configuration
    scalping: ScalpingModuleConfig = ScalpingModuleConfig()


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
        auto_execution=AutoExecutionConfig(
            enabled=_get_env_bool("AUTO_EXECUTE", default=False),
            mode=os.getenv("TRADING_MODE", "off"),  # "off" | "simulation" | "paper"
            position_size_pct=float(os.getenv("AUTO_EXECUTE_POSITION_SIZE", "10.0")),
            max_positions=int(os.getenv("AUTO_EXECUTE_MAX_POSITIONS", "3")),
            simulation_speed=float(os.getenv("SIMULATION_SPEED", "5.0")),
            simulation_balance=float(os.getenv("SIMULATION_BALANCE", "100000.0")),
        ),
        risk_management=RiskManagementConfig(
            # Improvement 1: Earnings Blackout
            earnings_blackout_enabled=_get_env_bool("EARNINGS_BLACKOUT_ENABLED", default=True),
            earnings_blackout_days_before=int(os.getenv("EARNINGS_BLACKOUT_DAYS", "5")),
            earnings_blackout_days_after=int(os.getenv("EARNINGS_POST_BLACKOUT_DAYS", "1")),
            # Improvement 2: VIX Regime Filter
            vix_filter_enabled=_get_env_bool("VIX_ENABLED", default=True),
            vix_panic_threshold=float(os.getenv("VIX_PANIC_THRESHOLD", "35")),
            vix_elevated_threshold=float(os.getenv("VIX_ELEVATED_THRESHOLD", "25")),
            # Improvement 3: Trading Hours Optimization
            trading_hours_enabled=_get_env_bool("TRADING_HOURS_ENABLED", default=True),
            block_market_open_minutes=int(os.getenv("BLOCK_MARKET_OPEN_MINUTES", "30")),
            block_market_close_minutes=int(os.getenv("BLOCK_MARKET_CLOSE_MINUTES", "15")),
            # Improvement 4: Dynamic Position Sizing
            dynamic_sizing_enabled=_get_env_bool("DYNAMIC_SIZING_ENABLED", default=True),
            base_position_size=float(os.getenv("BASE_POSITION_SIZE", "0.10")),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.15")),
            min_position_size=float(os.getenv("MIN_POSITION_SIZE", "0.05")),
            # Improvement 5: Price-Sentiment Divergence (Bear Market Protection)
            price_sentiment_divergence_enabled=_get_env_bool("PRICE_SENTIMENT_DIVERGENCE_ENABLED", default=True),
            bear_drawdown_threshold=float(os.getenv("BEAR_DRAWDOWN_THRESHOLD", "0.80")),
            bear_sma_threshold=float(os.getenv("BEAR_SMA_THRESHOLD", "0.95")),
            bear_sma_days_required=int(os.getenv("BEAR_SMA_DAYS_REQUIRED", "10")),
            death_cross_lookback_days=int(os.getenv("DEATH_CROSS_LOOKBACK_DAYS", "60")),
        ),
        dual_regime=DualRegimeConfig(
            enabled=_get_env_bool("DUAL_REGIME_ENABLED", default=True),
            bounce_threshold=float(os.getenv("MOMENTUM_BOUNCE_THRESHOLD", "0.03")),
            bounce_lookback_days=int(os.getenv("MOMENTUM_BOUNCE_LOOKBACK", "5")),
            resistance_proximity=float(os.getenv("MOMENTUM_RESISTANCE_PROXIMITY", "0.02")),
            oversold_rsi=float(os.getenv("MOMENTUM_OVERSOLD_RSI", "25")),
            neutral_rsi=float(os.getenv("MOMENTUM_NEUTRAL_RSI", "50")),
            vix_max_for_puts=float(os.getenv("MOMENTUM_VIX_MAX", "35")),
        ),
        scalping=ScalpingModuleConfig(
            enabled=_get_env_bool("SCALP_ENABLED", default=False),
            databento_data_dir=os.getenv("DATABENTO_DATA_DIR"),
        ),
    )
