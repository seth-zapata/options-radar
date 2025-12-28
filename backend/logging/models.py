"""Data models for evaluation logging.

These models capture the full state at each recommendation/abstain decision
for later analysis and outcome tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
import uuid


@dataclass
class QuoteSnapshot:
    """Snapshot of quote data at decision time."""
    underlying: str
    expiry: str
    strike: float
    right: str
    bid: float | None
    ask: float | None
    mid: float | None
    spread_percent: float | None
    timestamp: str


@dataclass
class GreeksSnapshot:
    """Snapshot of Greeks data at decision time."""
    underlying: str
    expiry: str
    strike: float
    right: str
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    iv: float | None
    timestamp: str


@dataclass
class UnderlyingSnapshot:
    """Snapshot of underlying data at decision time."""
    symbol: str
    price: float
    iv_rank: float | None
    iv_percentile: float | None
    timestamp: str


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at decision time."""
    total_exposure: float
    open_position_count: int
    cash_available: float
    positions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InputSnapshot:
    """Complete snapshot of all inputs at decision time."""
    quotes: list[QuoteSnapshot]
    greeks: list[GreeksSnapshot]
    underlying: UnderlyingSnapshot
    portfolio: PortfolioSnapshot
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Outcome:
    """Outcome data recorded after the fact.

    Tracks what happened at various intervals after the recommendation
    to evaluate if the recommendation was correct.
    """
    recorded_at: str

    # Underlying price at intervals
    underlying_price_at_15min: float | None = None
    underlying_price_at_1hr: float | None = None
    underlying_price_at_close: float | None = None

    # Option price at intervals
    option_price_at_15min: float | None = None
    option_price_at_1hr: float | None = None
    option_price_at_close: float | None = None

    # Analysis
    would_have_profited: bool | None = None
    theoretical_pnl: float | None = None
    theoretical_pnl_percent: float | None = None

    # For abstains - was this a regret?
    missed_opportunity: bool | None = None
    missed_profit: float | None = None


@dataclass
class RecommendationLog:
    """Complete log entry for a recommendation or abstain decision.

    This captures everything needed to:
    1. Replay the decision later
    2. Evaluate if the decision was correct
    3. Calculate aggregate metrics
    """
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Decision type
    decision_type: Literal["recommendation", "abstain"] = "recommendation"

    # Context
    underlying: str = ""
    underlying_price: float = 0.0

    # The actual decision (either recommendation or abstain details)
    recommendation_id: str | None = None
    recommendation_action: str | None = None
    recommendation_strike: float | None = None
    recommendation_expiry: str | None = None
    recommendation_right: str | None = None
    recommendation_premium: float | None = None
    recommendation_confidence: int | None = None
    recommendation_rationale: str | None = None

    # Abstain details (if abstaining)
    abstain_reason: str | None = None
    abstain_resume_condition: str | None = None
    failed_gates: list[dict[str, Any]] = field(default_factory=list)

    # Full input snapshot
    input_snapshot: InputSnapshot | None = None

    # Outcome (filled in later)
    outcome: Outcome | None = None

    # Session tracking
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "id": self.id,
            "timestamp": self.timestamp,
            "decisionType": self.decision_type,
            "underlying": self.underlying,
            "underlyingPrice": self.underlying_price,
            "sessionId": self.session_id,
        }

        if self.decision_type == "recommendation":
            result["recommendation"] = {
                "id": self.recommendation_id,
                "action": self.recommendation_action,
                "strike": self.recommendation_strike,
                "expiry": self.recommendation_expiry,
                "right": self.recommendation_right,
                "premium": self.recommendation_premium,
                "confidence": self.recommendation_confidence,
                "rationale": self.recommendation_rationale,
            }
        else:
            result["abstain"] = {
                "reason": self.abstain_reason,
                "resumeCondition": self.abstain_resume_condition,
                "failedGates": self.failed_gates,
            }

        if self.outcome:
            result["outcome"] = {
                "recordedAt": self.outcome.recorded_at,
                "underlyingPriceAt15Min": self.outcome.underlying_price_at_15min,
                "underlyingPriceAt1Hr": self.outcome.underlying_price_at_1hr,
                "underlyingPriceAtClose": self.outcome.underlying_price_at_close,
                "optionPriceAt15Min": self.outcome.option_price_at_15min,
                "optionPriceAt1Hr": self.outcome.option_price_at_1hr,
                "optionPriceAtClose": self.outcome.option_price_at_close,
                "wouldHaveProfited": self.outcome.would_have_profited,
                "theoreticalPnl": self.outcome.theoretical_pnl,
                "theoreticalPnlPercent": self.outcome.theoretical_pnl_percent,
                "missedOpportunity": self.outcome.missed_opportunity,
                "missedProfit": self.outcome.missed_profit,
            }

        return result
