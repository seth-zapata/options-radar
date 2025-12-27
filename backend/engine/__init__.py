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
    STRATEGY_FIT_GATES,
    get_hard_gates,
    get_soft_gates,
    # Individual gates
    CashAvailableGate,
    DeltaInRangeGate,
    GreeksFreshGate,
    IVRankAppropriateGate,
    OpenInterestSufficientGate,
    PositionSizeLimitGate,
    QuoteFreshGate,
    SectorConcentrationGate,
    SpreadAcceptableGate,
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
    "STRATEGY_FIT_GATES",
    "PORTFOLIO_CONSTRAINT_GATES",
    "get_hard_gates",
    "get_soft_gates",
    # Individual gates
    "CashAvailableGate",
    "DeltaInRangeGate",
    "GreeksFreshGate",
    "IVRankAppropriateGate",
    "OpenInterestSufficientGate",
    "PositionSizeLimitGate",
    "QuoteFreshGate",
    "SectorConcentrationGate",
    "SpreadAcceptableGate",
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
]
