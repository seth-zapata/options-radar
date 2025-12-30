"""Gating engine for OptionsRadar.

Provides:
- Gate definitions (hard and soft)
- Pipeline orchestration
- Abstain generation

See spec section 5 for gating pipeline details.
"""

from backend.engine.gates import (
    ALL_GATES,
    DATA_FRESHNESS_GATES,
    Gate,
    GateContext,
    GateResult,
    GateSeverity,
    LIQUIDITY_GATES,
    PORTFOLIO_CONSTRAINT_GATES,
    SENTIMENT_GATES,
    SIGNAL_QUALITY_GATES,
    STRATEGY_FIT_GATES,
    TECHNICAL_GATES,
    get_hard_gates,
    get_soft_gates,
    # Individual gates
    CashAvailableGate,
    DeltaInRangeGate,
    GreeksFreshGate,
    HighMentionsBoostGate,
    IVRankExtremesGate,
    MinimumMentionsGate,
    OpenInterestSufficientGate,
    PositionSizeLimitGate,
    QuoteFreshGate,
    RetailMomentumGate,
    RSIOverboughtGate,
    SectorConcentrationGate,
    SentimentAlignmentGate,
    SentimentDirectionGate,
    SentimentRecencyGate,
    SignalEnabledGate,
    SpreadAcceptableGate,
    StrongSentimentBoostGate,
    UnderlyingPriceFreshGate,
    VolumeSufficientGate,
)
from backend.engine.pipeline import (
    Abstain,
    AbstainReason,
    DataFreshness,
    GatingPipeline,
    PipelineResult,
    PipelineStage,
    PortfolioState,
    evaluate_option_for_signal,
)
from backend.engine.recommender import (
    Recommendation,
    Recommender,
    RecommenderConfig,
)
from backend.engine.session_tracker import (
    SessionConfig,
    SessionStats,
    SessionTracker,
)
from backend.engine.position_tracker import (
    ExitSignal,
    PositionTracker,
    PositionTrackerConfig,
    TrackedPosition,
)
from backend.engine.scanner import (
    DailyScanner,
    ScanResult,
)
from backend.engine.auto_executor import (
    AutoExecutor,
    AutoExecutorConfig,
    ExecutionResult,
)

__all__ = [
    # Gate types
    "Gate",
    "GateContext",
    "GateResult",
    "GateSeverity",
    # Gate collections
    "ALL_GATES",
    "DATA_FRESHNESS_GATES",
    "LIQUIDITY_GATES",
    "PORTFOLIO_CONSTRAINT_GATES",
    "SENTIMENT_GATES",
    "SIGNAL_QUALITY_GATES",
    "STRATEGY_FIT_GATES",
    "TECHNICAL_GATES",
    "get_hard_gates",
    "get_soft_gates",
    # Individual gates
    "CashAvailableGate",
    "DeltaInRangeGate",
    "GreeksFreshGate",
    "HighMentionsBoostGate",
    "IVRankExtremesGate",
    "MinimumMentionsGate",
    "OpenInterestSufficientGate",
    "PositionSizeLimitGate",
    "QuoteFreshGate",
    "RetailMomentumGate",
    "RSIOverboughtGate",
    "SectorConcentrationGate",
    "SentimentAlignmentGate",
    "SentimentDirectionGate",
    "SentimentRecencyGate",
    "SignalEnabledGate",
    "SpreadAcceptableGate",
    "StrongSentimentBoostGate",
    "UnderlyingPriceFreshGate",
    "VolumeSufficientGate",
    # Pipeline
    "Abstain",
    "AbstainReason",
    "DataFreshness",
    "GatingPipeline",
    "PipelineResult",
    "PipelineStage",
    "PortfolioState",
    "evaluate_option_for_signal",
    # Recommender
    "Recommendation",
    "Recommender",
    "RecommenderConfig",
    # Session Tracker
    "SessionConfig",
    "SessionStats",
    "SessionTracker",
    # Position Tracker
    "ExitSignal",
    "PositionTracker",
    "PositionTrackerConfig",
    "TrackedPosition",
    # Scanner
    "DailyScanner",
    "ScanResult",
    # Auto Executor
    "AutoExecutor",
    "AutoExecutorConfig",
    "ExecutionResult",
]
