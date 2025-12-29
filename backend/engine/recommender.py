"""Signal recommendation generator.

Takes passed gate results and generates actionable trade recommendations.
From spec section 5.1 Stage 5: Explain/Payload.

Recommendations are display-only - no trade execution.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from backend.data.aggregator import AggregatedOptionData
from backend.engine.pipeline import PipelineResult
from backend.models.market_data import UnderlyingData

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GateResultSummary:
    """Summary of a gate result for inclusion in recommendations."""
    name: str
    passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A trade recommendation payload.

    From spec section 4.3 (Recommendation structure).

    Attributes:
        id: Unique recommendation ID (UUID)
        generated_at: ISO timestamp of generation
        underlying: Underlying symbol (e.g., "NVDA")
        action: Trade action (BUY_CALL, BUY_PUT, etc.)
        strike: Option strike price
        expiry: Expiration date (ISO format)
        right: Option type (C or P)
        contracts: Suggested number of contracts
        premium: Current mid price per contract
        total_cost: contracts * premium * 100
        confidence: Confidence score (0-100)
        rationale: Human-readable explanation
        gate_results: Summary of all gate evaluations
        data_freshness: Snapshot of data ages when generated
        valid_until: Recommendation expiry time (short-lived)
    """
    id: str
    generated_at: str
    underlying: str
    action: Literal["BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"]
    strike: float
    expiry: str
    right: str
    contracts: int
    premium: float
    total_cost: float
    confidence: int
    rationale: str
    gate_results: tuple  # tuple of GateResultSummary dicts
    quote_age: float | None
    greeks_age: float | None
    underlying_age: float | None
    valid_until: str


@dataclass
class RecommenderConfig:
    """Configuration for the recommender.

    Attributes:
        default_contracts: Default number of contracts to suggest
        max_contracts: Maximum contracts per recommendation
        min_confidence: Minimum confidence to generate recommendation
        recommendation_ttl_seconds: How long a recommendation is valid
    """
    default_contracts: int = 1
    max_contracts: int = 10
    min_confidence: int = 50
    recommendation_ttl_seconds: int = 300  # 5 minutes


class Recommender:
    """Generates trade recommendations from passed gate results.

    Usage:
        recommender = Recommender()

        if pipeline_result.passed:
            recommendation = recommender.generate(
                result=pipeline_result,
                option=option,
                underlying=underlying,
                action="BUY_CALL",
            )
            if recommendation:
                # Broadcast to clients
                ...
    """

    def __init__(self, config: RecommenderConfig | None = None):
        self.config = config or RecommenderConfig()

    def generate(
        self,
        result: PipelineResult,
        option: AggregatedOptionData,
        underlying: UnderlyingData | None,
        action: Literal["BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"],
        contracts: int | None = None,
    ) -> Recommendation | None:
        """Generate a recommendation from a passed pipeline result.

        Args:
            result: Must be a passed PipelineResult
            option: The option being recommended
            underlying: Current underlying data
            action: Trade action
            contracts: Number of contracts (uses default if not specified)

        Returns:
            Recommendation if confidence is sufficient, None otherwise
        """
        if not result.passed:
            logger.warning("Cannot generate recommendation for failed pipeline")
            return None

        # Check confidence threshold
        if result.confidence_cap < self.config.min_confidence:
            logger.info(
                f"Confidence {result.confidence_cap} below threshold "
                f"{self.config.min_confidence}, not generating recommendation"
            )
            return None

        now = datetime.now(timezone.utc)

        # Determine contracts
        num_contracts = contracts or self.config.default_contracts
        num_contracts = min(num_contracts, self.config.max_contracts)

        # Calculate premium and cost
        premium = option.mid or 0.0
        total_cost = num_contracts * premium * 100

        # Build rationale
        rationale = self._build_rationale(result, option, underlying, action)

        # Calculate valid_until
        valid_until = datetime.fromtimestamp(
            now.timestamp() + self.config.recommendation_ttl_seconds,
            tz=timezone.utc
        )

        # Build gate results summary
        gate_results = tuple(
            {"name": g.gate_name, "passed": g.passed, "message": g.message}
            for g in result.all_results
        )

        return Recommendation(
            id=str(uuid.uuid4()),
            generated_at=now.isoformat(),
            underlying=option.canonical_id.underlying,
            action=action,
            strike=option.canonical_id.strike,
            expiry=option.canonical_id.expiry,
            right=option.canonical_id.right,
            contracts=num_contracts,
            premium=premium,
            total_cost=total_cost,
            confidence=result.confidence_cap,
            rationale=rationale,
            gate_results=gate_results,
            quote_age=option.quote_age_seconds(now),
            greeks_age=option.greeks_age_seconds(now),
            underlying_age=underlying.age_seconds(now) if underlying else None,
            valid_until=valid_until.isoformat(),
        )

    def _build_rationale(
        self,
        result: PipelineResult,
        option: AggregatedOptionData,
        underlying: UnderlyingData | None,
        action: Literal["BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"],
    ) -> str:
        """Build human-readable rationale for the recommendation."""
        parts = []

        # Direction reasoning
        if action.startswith("BUY"):
            direction = "Call" if "CALL" in action else "Put"
            parts.append(f"Buy {direction}")
        else:
            direction = "Call" if "CALL" in action else "Put"
            parts.append(f"Sell {direction}")

        # Strike selection
        strike = option.canonical_id.strike
        if underlying:
            price = underlying.price
            if strike > price:
                pct_otm = ((strike - price) / price) * 100
                parts.append(f"at ${strike} ({pct_otm:.1f}% OTM)")
            elif strike < price:
                pct_itm = ((price - strike) / price) * 100
                parts.append(f"at ${strike} ({pct_itm:.1f}% ITM)")
            else:
                parts.append(f"at ${strike} (ATM)")

        # Delta info
        if option.delta:
            parts.append(f"Delta: {option.delta:.2f}")

        # IV context
        if underlying and underlying.iv_rank:
            if underlying.iv_rank < 30:
                parts.append(f"IV Rank low ({underlying.iv_rank:.0f})")
            elif underlying.iv_rank > 70:
                parts.append(f"IV Rank high ({underlying.iv_rank:.0f})")

        # Confidence qualifier
        if result.confidence_cap >= 90:
            parts.append("High confidence")
        elif result.confidence_cap >= 70:
            parts.append("Moderate confidence")
        else:
            parts.append("Lower confidence")

        # Soft failures
        if result.soft_failures:
            failure_names = [f.gate_name for f in result.soft_failures]
            parts.append(f"(soft warnings: {', '.join(failure_names)})")

        return ". ".join(parts) + "."
