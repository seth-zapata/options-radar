"""Scalping module for momentum-based 0DTE/1DTE options trading.

This module runs parallel to the existing regime-based system,
targeting rapid intraday moves in TSLA options.

Key distinction from regime system:
- Regime: Multi-day sentiment signals, 7-30 DTE, ~1 trade/week
- Scalping: Intraday momentum signals, 0-1 DTE, multiple trades/day

Phase 1 (DataBento Integration): COMPLETE
- DataBentoLoader: Load historical CBBO-1m data
- QuoteReplaySystem: Replay quotes for backtesting
- ScalpConfig: Configuration dataclass

Phase 2 (Core Components): COMPLETE
- PriceVelocityTracker: Track price velocity over multiple windows
- VolumeAnalyzer: Detect volume spikes and put/call ratios
- TechnicalScalper: VWAP with bands, S/R level detection

Phase 3 (Signal Generation): COMPLETE
- ScalpSignal: Complete scalping trade signal
- ScalpSignalGenerator: Generate scalp signals from components
- ScalpTrade: Completed trade record with P&L
- BacktestResult: Aggregate backtest statistics
- ScalpBacktester: Backtest scalping strategy on historical data
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
from backend.scalping.velocity_tracker import (
    PricePoint,
    PriceVelocityTracker,
    SpikeSignal,
    VelocityReading,
)
from backend.scalping.volume_analyzer import (
    VolumeAnalyzer,
    VolumeBar,
    VolumeSpike,
)
from backend.scalping.technical_scalper import (
    ScalpTechnicalSignal,
    SupportResistance,
    TechnicalScalper,
    VWAPState,
)
from backend.scalping.signal_generator import (
    ScalpSignal,
    ScalpSignalGenerator,
)
from backend.scalping.scalp_backtester import (
    BacktestResult,
    ScalpBacktester,
    ScalpTrade,
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
    # Velocity Tracker
    "PriceVelocityTracker",
    "VelocityReading",
    "SpikeSignal",
    "PricePoint",
    # Volume Analyzer
    "VolumeAnalyzer",
    "VolumeBar",
    "VolumeSpike",
    # Technical Scalper
    "TechnicalScalper",
    "ScalpTechnicalSignal",
    "SupportResistance",
    "VWAPState",
    # Signal Generator
    "ScalpSignal",
    "ScalpSignalGenerator",
    # Backtester
    "ScalpTrade",
    "BacktestResult",
    "ScalpBacktester",
]
