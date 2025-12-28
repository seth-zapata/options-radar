"""Evaluation logging infrastructure for OptionsRadar.

Provides:
- Shadow mode logging of all recommendations/abstains
- Outcome recording for post-hoc analysis
- Metrics calculation
- Replay framework for offline testing

See spec section 6 for details.
"""

from backend.logging.models import (
    GreeksSnapshot,
    InputSnapshot,
    Outcome,
    PortfolioSnapshot,
    QuoteSnapshot,
    RecommendationLog,
    UnderlyingSnapshot,
)
from backend.logging.logger import (
    EvaluationLogger,
)
from backend.logging.metrics import (
    EvaluationMetrics,
    MetricsCalculator,
)
from backend.logging.replay import (
    ExpectedDecision,
    MarketTick,
    ReplayComparator,
    ReplayResult,
    ReplaySession,
    SessionRecorder,
    SessionReplayer,
)

__all__ = [
    # Models
    "GreeksSnapshot",
    "InputSnapshot",
    "Outcome",
    "PortfolioSnapshot",
    "QuoteSnapshot",
    "RecommendationLog",
    "UnderlyingSnapshot",
    # Logger
    "EvaluationLogger",
    # Metrics
    "EvaluationMetrics",
    "MetricsCalculator",
    # Replay
    "ExpectedDecision",
    "MarketTick",
    "ReplayComparator",
    "ReplayResult",
    "ReplaySession",
    "SessionRecorder",
    "SessionReplayer",
]
