"""Scalping module for momentum-based 0DTE/1DTE options trading.

This module runs parallel to the existing regime-based system,
targeting rapid intraday moves in TSLA options.

Key distinction from regime system:
- Regime: Multi-day sentiment signals, 7-30 DTE, ~1 trade/week
- Scalping: Intraday momentum signals, 0-1 DTE, multiple trades/day

Phase 1 (DataBento Integration):
- DataBentoLoader: Load historical CBBO-1m data
- QuoteReplaySystem: Replay quotes for backtesting
- ScalpConfig: Configuration dataclass

Phase 2 (Core Components) - TODO:
- PriceVelocityTracker: Track price velocity
- VolumeAnalyzer: Detect volume spikes
- TechnicalScalper: VWAP, S/R levels

Phase 3 (Signal Generation) - TODO:
- ScalpSignalGenerator: Generate scalp signals
- ScalpBacktester: Backtest scalping strategy
"""

from backend.scalping.config import (
    DataBentoConfig,
    ScalpConfig,
    ScalpExitConfig,
    load_databento_config_from_env,
    load_scalp_config_from_env,
)
from backend.scalping.replay import (
    BacktestClock,
    QuoteReplaySystem,
    ReplayQuote,
    ReplayTick,
)

__all__ = [
    # Config
    "ScalpConfig",
    "ScalpExitConfig",
    "DataBentoConfig",
    "load_scalp_config_from_env",
    "load_databento_config_from_env",
    # Replay
    "QuoteReplaySystem",
    "ReplayQuote",
    "ReplayTick",
    "BacktestClock",
]
