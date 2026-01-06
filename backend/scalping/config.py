"""Configuration for the scalping module.

These configs control momentum detection, option selection,
risk management, and position limits for 0DTE/1DTE scalping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScalpExitConfig:
    """Exit-specific configuration for scalp positions.

    These are tighter than regime strategy exits:
    - Scalp: +30% profit, -15% stop, 15 min max hold
    - Regime: +40% profit, -20% stop, DTE-based exit
    """

    take_profit_pct: float = 30.0  # Exit at +30%
    stop_loss_pct: float = 15.0  # Exit at -15%
    max_hold_minutes: int = 15  # Force exit after 15 minutes


@dataclass(frozen=True)
class ScalpConfig:
    """Configuration for scalping module.

    Attributes:
        enabled: Master enable for scalping (default False)
        eval_interval_seconds: How often to check for signals (backtest)
        eval_interval_ms: How often to check for signals (live, 200ms default)

        Momentum Detection:
        momentum_threshold_pct: Min % move to trigger signal (0.5%)
        momentum_window_seconds: Time window for momentum calc (30s)

        Volume Confirmation:
        volume_spike_ratio: Required volume vs baseline (1.5x)

        Option Selection:
        target_delta: Ideal delta for entries (0.35)
        delta_tolerance: Accept this range around target (±0.10)
        max_spread_pct: Max bid-ask spread as % of mid (8%)
        min_open_interest: Minimum OI required (100)
        max_dte: Maximum days to expiry (1 = 0DTE + 1DTE)
        prefer_0dte: Prefer same-day expiry when available

        Risk Management:
        take_profit_pct: Exit at this profit (30%)
        stop_loss_pct: Exit at this loss (15%)
        max_hold_minutes: Force exit after this time (15 min)

        Position Limits:
        max_daily_scalps: Max trades per day (10)
        max_concurrent_scalps: Max open at once (1)
        scalp_position_size_pct: % of portfolio per trade (5%)
        max_contract_price: Don't buy options over this ($5)

        Cooldowns:
        min_signal_interval_seconds: Min time between signals (60s)
        cooldown_after_loss_seconds: Extra wait after a loss (300s)
    """

    # Master enable
    enabled: bool = False

    # Evaluation frequency
    eval_interval_seconds: float = 1.0  # For backtesting
    eval_interval_ms: int = 200  # For live trading (200ms)

    # Momentum thresholds
    momentum_threshold_pct: float = 0.5  # 0.5% move triggers signal
    momentum_window_seconds: int = 30  # Over 30 second window

    # Volume thresholds
    volume_spike_ratio: float = 1.5  # 1.5x normal volume required

    # Option selection
    target_delta: float = 0.35
    delta_tolerance: float = 0.10  # Accept 0.25-0.45 delta
    max_spread_pct: float = 8.0  # Max 8% spread
    min_open_interest: int = 100  # Minimum OI
    max_dte: int = 1  # 0DTE or 1DTE only
    prefer_0dte: bool = True

    # Risk management
    take_profit_pct: float = 30.0
    stop_loss_pct: float = 15.0
    max_hold_minutes: int = 15

    # Position limits
    max_daily_scalps: int = 10
    max_concurrent_scalps: int = 1
    scalp_position_size_pct: float = 5.0  # % of portfolio
    max_contract_price: float = 5.00  # Don't buy options > $5

    # Cooldowns
    min_signal_interval_seconds: float = 60.0  # 1 min between signals
    cooldown_after_loss_seconds: float = 300.0  # 5 min after loss

    @property
    def exit_config(self) -> ScalpExitConfig:
        """Get exit configuration derived from this config."""
        return ScalpExitConfig(
            take_profit_pct=self.take_profit_pct,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_minutes=self.max_hold_minutes,
        )


def load_scalp_config_from_env() -> ScalpConfig:
    """Load scalping configuration from environment variables.

    Environment variables (all optional, defaults shown):
        SCALP_ENABLED=false
        SCALP_EVAL_INTERVAL_MS=200
        SCALP_MOMENTUM_THRESHOLD=0.5
        SCALP_MOMENTUM_WINDOW=30
        SCALP_VOLUME_SPIKE_RATIO=1.5
        SCALP_TARGET_DELTA=0.35
        SCALP_DELTA_TOLERANCE=0.10
        SCALP_MAX_SPREAD_PCT=8.0
        SCALP_MIN_OI=100
        SCALP_MAX_DTE=1
        SCALP_PREFER_0DTE=true
        SCALP_TAKE_PROFIT=30.0
        SCALP_STOP_LOSS=15.0
        SCALP_MAX_HOLD_MINUTES=15
        SCALP_MAX_DAILY=10
        SCALP_MAX_CONCURRENT=1
        SCALP_POSITION_SIZE=5.0
        SCALP_MAX_CONTRACT_PRICE=5.0
        SCALP_MIN_SIGNAL_INTERVAL=60.0
        SCALP_COOLDOWN_AFTER_LOSS=300.0

    Returns:
        ScalpConfig with values from environment or defaults
    """

    def get_bool(key: str, default: bool) -> bool:
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes")

    def get_float(key: str, default: float) -> float:
        return float(os.getenv(key, str(default)))

    def get_int(key: str, default: int) -> int:
        return int(os.getenv(key, str(default)))

    return ScalpConfig(
        enabled=get_bool("SCALP_ENABLED", False),
        eval_interval_ms=get_int("SCALP_EVAL_INTERVAL_MS", 200),
        eval_interval_seconds=get_float("SCALP_EVAL_INTERVAL_MS", 200) / 1000,
        momentum_threshold_pct=get_float("SCALP_MOMENTUM_THRESHOLD", 0.5),
        momentum_window_seconds=get_int("SCALP_MOMENTUM_WINDOW", 30),
        volume_spike_ratio=get_float("SCALP_VOLUME_SPIKE_RATIO", 1.5),
        target_delta=get_float("SCALP_TARGET_DELTA", 0.35),
        delta_tolerance=get_float("SCALP_DELTA_TOLERANCE", 0.10),
        max_spread_pct=get_float("SCALP_MAX_SPREAD_PCT", 8.0),
        min_open_interest=get_int("SCALP_MIN_OI", 100),
        max_dte=get_int("SCALP_MAX_DTE", 1),
        prefer_0dte=get_bool("SCALP_PREFER_0DTE", True),
        take_profit_pct=get_float("SCALP_TAKE_PROFIT", 30.0),
        stop_loss_pct=get_float("SCALP_STOP_LOSS", 15.0),
        max_hold_minutes=get_int("SCALP_MAX_HOLD_MINUTES", 15),
        max_daily_scalps=get_int("SCALP_MAX_DAILY", 10),
        max_concurrent_scalps=get_int("SCALP_MAX_CONCURRENT", 1),
        scalp_position_size_pct=get_float("SCALP_POSITION_SIZE", 5.0),
        max_contract_price=get_float("SCALP_MAX_CONTRACT_PRICE", 5.0),
        min_signal_interval_seconds=get_float("SCALP_MIN_SIGNAL_INTERVAL", 60.0),
        cooldown_after_loss_seconds=get_float("SCALP_COOLDOWN_AFTER_LOSS", 300.0),
    )


@dataclass
class DataBentoConfig:
    """Configuration for DataBento data loading.

    Attributes:
        data_dir: Path to directory containing .csv.zst files
        max_dte: Max DTE to include when filtering (default 1)
        min_bid: Min bid price filter (default 0.05)
        max_spread_pct: Max spread % filter (default 10.0)
        chunk_size: Rows per chunk for large files (default 100k)
    """

    data_dir: Path
    max_dte: int = 1
    min_bid: float = 0.05
    max_spread_pct: float = 10.0
    min_open_interest: int = 100
    chunk_size: int = 100_000


def load_databento_config_from_env() -> DataBentoConfig | None:
    """Load DataBento config from environment.

    Environment variables:
        DATABENTO_DATA_DIR: Path to data directory (required for backtest)
        DATABENTO_MAX_DTE: Max DTE filter (default 1)
        DATABENTO_MIN_BID: Min bid price (default 0.05)
        DATABENTO_MAX_SPREAD_PCT: Max spread % (default 10.0)

    Returns:
        DataBentoConfig if data dir is set, None otherwise
    """
    data_dir = os.getenv("DATABENTO_DATA_DIR")
    if not data_dir:
        return None

    return DataBentoConfig(
        data_dir=Path(data_dir),
        max_dte=int(os.getenv("DATABENTO_MAX_DTE", "1")),
        min_bid=float(os.getenv("DATABENTO_MIN_BID", "0.05")),
        max_spread_pct=float(os.getenv("DATABENTO_MAX_SPREAD_PCT", "10.0")),
    )
