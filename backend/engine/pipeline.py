"""Gating pipeline orchestration.

Implements the 5-stage pipeline from spec section 5.1:
1. Data Freshness - Check all data is current
2. Liquidity Scoring - Verify tradeable conditions
3. Strategy Fit - Match signal to strategy parameters
4. Portfolio Constraints - Check position limits
5. Explain/Payload - Generate recommendation or abstain

Any hard gate failure triggers immediate ABSTAIN.
Soft gate failures reduce confidence but allow continuation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from backend.data.aggregator import AggregatedOptionData
from backend.engine.gates import (
    ALL_GATES,
    DATA_FRESHNESS_GATES,
    LIQUIDITY_GATES,
    PORTFOLIO_CONSTRAINT_GATES,
    STRATEGY_FIT_GATES,
    Gate,
    GateContext,
    GateResult,
    GateSeverity,
)
from backend.models.canonical import CanonicalOptionId
from backend.models.market_data import UnderlyingData

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Pipeline stages in execution order."""
    DATA_FRESHNESS = "data_freshness"
    LIQUIDITY = "liquidity"
    STRATEGY_FIT = "strategy_fit"
    PORTFOLIO_CONSTRAINTS = "portfolio_constraints"
    EXPLAIN = "explain"


class AbstainReason(str, Enum):
    """Reasons why the system abstains from recommending.

    From spec section 4.2.
    """
    STALE_DATA = "STALE_DATA"
    LIQUIDITY_INSUFFICIENT = "LIQUIDITY_INSUFFICIENT"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    FEED_DEGRADED = "FEED_DEGRADED"
    GATES_FAILED = "GATES_FAILED"
    NO_CLEAR_SIGNAL = "NO_CLEAR_SIGNAL"
    PORTFOLIO_CONSTRAINT = "PORTFOLIO_CONSTRAINT"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    NO_DATA = "NO_DATA"


@dataclass(frozen=True, slots=True)
class DataFreshness:
    """Data freshness snapshot for a recommendation.

    From spec section 4.1 dataFreshness.
    """
    quote_age: float | None
    greeks_age: float | None
    underlying_age: float | None
    sentiment_age: float | None = None
    all_fresh: bool = False


@dataclass(frozen=True, slots=True)
class Abstain:
    """Represents a decision to not recommend.

    Generated when any hard gate fails or data is unavailable.
    From spec section 4.2.

    Attributes:
        id: Unique identifier
        generated_at: ISO timestamp of generation
        underlying: Ticker symbol
        reason: Primary abstain reason
        failed_gates: List of gates that failed
        data_freshness: Snapshot of data ages
        resume_condition: What needs to change to reconsider
    """
    id: str
    generated_at: str
    underlying: str
    reason: AbstainReason
    failed_gates: list[GateResult]
    data_freshness: DataFreshness
    resume_condition: str


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Result of running the gating pipeline.

    Contains either an abstain or all gates passed (ready for recommendation).

    Attributes:
        passed: True if all hard gates passed
        abstain: Abstain object if any hard gate failed
        all_results: Results from all evaluated gates
        soft_failures: Soft gates that failed (reduce confidence)
        confidence_cap: Maximum confidence based on soft failures
        stage_reached: Last pipeline stage completed
    """
    passed: bool
    abstain: Abstain | None
    all_results: list[GateResult]
    soft_failures: list[GateResult]
    confidence_cap: int  # 0-100
    stage_reached: PipelineStage


@dataclass
class PortfolioState:
    """Current portfolio state for position sizing gates.

    Attributes:
        available_cash: Cash available for new positions
        portfolio_value: Total portfolio value
        sector_exposures: Dict of sector -> exposure percent
        current_positions: Set of underlying symbols with positions
    """
    available_cash: float = 10000.0
    portfolio_value: float = 10000.0
    sector_exposures: dict[str, float] = field(default_factory=dict)
    current_positions: set[str] = field(default_factory=set)


@dataclass
class GatingPipeline:
    """Orchestrates the gating pipeline for option evaluation.

    Usage:
        pipeline = GatingPipeline()

        # Evaluate an option for a potential BUY_CALL
        result = pipeline.evaluate(
            option=aggregated_option,
            underlying=underlying_data,
            action="BUY_CALL",
            contracts=1,
            portfolio=portfolio_state,
        )

        if result.passed:
            # All hard gates passed, can generate recommendation
            print("Ready for recommendation")
        else:
            # Hard gate failed, abstain
            print(f"Abstain: {result.abstain.reason}")
    """

    # Allow custom gate sets for testing
    data_freshness_gates: list[Gate] = field(default_factory=lambda: list(DATA_FRESHNESS_GATES))
    liquidity_gates: list[Gate] = field(default_factory=lambda: list(LIQUIDITY_GATES))
    strategy_fit_gates: list[Gate] = field(default_factory=lambda: list(STRATEGY_FIT_GATES))
    portfolio_constraint_gates: list[Gate] = field(default_factory=lambda: list(PORTFOLIO_CONSTRAINT_GATES))

    def evaluate(
        self,
        option: AggregatedOptionData,
        underlying: UnderlyingData | None,
        action: Literal["BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"],
        contracts: int = 1,
        portfolio: PortfolioState | None = None,
    ) -> PipelineResult:
        """Run the full gating pipeline for an option.

        Args:
            option: Aggregated option data (quotes + Greeks)
            underlying: Underlying price and IV data
            action: Trade action to evaluate
            contracts: Number of contracts
            portfolio: Current portfolio state

        Returns:
            PipelineResult with pass/fail and details
        """
        if portfolio is None:
            portfolio = PortfolioState()

        # Build gate context
        ctx = self._build_context(option, underlying, action, contracts, portfolio)

        # Track results
        all_results: list[GateResult] = []
        soft_failures: list[GateResult] = []

        # Stage 1: Data Freshness
        stage_result = self._run_stage(
            PipelineStage.DATA_FRESHNESS,
            self.data_freshness_gates,
            ctx,
            all_results,
            soft_failures,
        )
        if stage_result is not None:
            return stage_result

        # Stage 2: Liquidity
        stage_result = self._run_stage(
            PipelineStage.LIQUIDITY,
            self.liquidity_gates,
            ctx,
            all_results,
            soft_failures,
        )
        if stage_result is not None:
            return stage_result

        # Stage 3: Strategy Fit
        stage_result = self._run_stage(
            PipelineStage.STRATEGY_FIT,
            self.strategy_fit_gates,
            ctx,
            all_results,
            soft_failures,
        )
        if stage_result is not None:
            return stage_result

        # Stage 4: Portfolio Constraints
        stage_result = self._run_stage(
            PipelineStage.PORTFOLIO_CONSTRAINTS,
            self.portfolio_constraint_gates,
            ctx,
            all_results,
            soft_failures,
        )
        if stage_result is not None:
            return stage_result

        # Stage 5: Explain (all gates passed)
        confidence_cap = self._calculate_confidence_cap(soft_failures)

        return PipelineResult(
            passed=True,
            abstain=None,
            all_results=all_results,
            soft_failures=soft_failures,
            confidence_cap=confidence_cap,
            stage_reached=PipelineStage.EXPLAIN,
        )

    def _build_context(
        self,
        option: AggregatedOptionData,
        underlying: UnderlyingData | None,
        action: Literal["BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"],
        contracts: int,
        portfolio: PortfolioState,
    ) -> GateContext:
        """Build gate context from option and portfolio data."""
        now = datetime.now(timezone.utc)

        # Calculate premium (use mid price)
        premium = option.mid or 0.0

        # Calculate position as percent of portfolio
        position_value = contracts * premium * 100
        new_position_percent = 0.0
        if portfolio.portfolio_value > 0:
            new_position_percent = (position_value / portfolio.portfolio_value) * 100

        # Get sector exposure (for now, treat each underlying as its own sector)
        underlying_symbol = option.canonical_id.underlying
        current_sector_exposure = portfolio.sector_exposures.get(underlying_symbol, 0.0)

        return GateContext(
            action=action,
            # Data freshness
            underlying_age=underlying.age_seconds(now) if underlying else None,
            quote_age=option.quote_age_seconds(now),
            greeks_age=option.greeks_age_seconds(now),
            # Liquidity
            spread_percent=option.spread_percent,
            open_interest=option.open_interest,
            volume=option.volume,
            bid_size=option.bid_size or 0,
            ask_size=option.ask_size or 0,
            # Greeks
            delta=option.delta,
            gamma=option.gamma,
            theta=option.theta,
            vega=option.vega,
            iv=option.iv,
            # IV metrics
            iv_rank=underlying.iv_rank if underlying else 50.0,
            iv_percentile=underlying.iv_percentile if underlying else 50.0,
            # Position sizing
            contracts=contracts,
            premium=premium,
            available_cash=portfolio.available_cash,
            portfolio_value=portfolio.portfolio_value,
            # Sector
            current_sector_exposure_percent=current_sector_exposure,
            new_position_percent=new_position_percent,
        )

    def _run_stage(
        self,
        stage: PipelineStage,
        gates: list[Gate],
        ctx: GateContext,
        all_results: list[GateResult],
        soft_failures: list[GateResult],
    ) -> PipelineResult | None:
        """Run a pipeline stage and check for hard failures.

        Args:
            stage: Current pipeline stage
            gates: Gates to evaluate
            ctx: Gate context
            all_results: Accumulator for all results
            soft_failures: Accumulator for soft failures

        Returns:
            PipelineResult if hard gate failed, None to continue
        """
        for gate in gates:
            result = gate.evaluate(ctx)
            all_results.append(result)

            if not result.passed:
                if result.severity == GateSeverity.HARD:
                    # Hard failure - generate abstain
                    logger.info(
                        f"Hard gate '{gate.name}' failed: {result.message}"
                    )
                    return self._create_abstain_result(
                        ctx, all_results, soft_failures, result, stage
                    )
                else:
                    # Soft failure - record but continue
                    logger.debug(
                        f"Soft gate '{gate.name}' failed: {result.message}"
                    )
                    soft_failures.append(result)

        return None

    def _create_abstain_result(
        self,
        ctx: GateContext,
        all_results: list[GateResult],
        soft_failures: list[GateResult],
        failed_gate: GateResult,
        stage: PipelineStage,
    ) -> PipelineResult:
        """Create a pipeline result for an abstain."""
        import uuid

        # Determine abstain reason from failed gate
        reason = self._determine_abstain_reason(failed_gate, stage)

        # Build resume condition
        resume_condition = self._build_resume_condition(failed_gate)

        # Build data freshness
        data_freshness = DataFreshness(
            quote_age=ctx.quote_age,
            greeks_age=ctx.greeks_age,
            underlying_age=ctx.underlying_age,
            all_fresh=all(
                r.passed for r in all_results
                if r.gate_name in ("quote_fresh", "greeks_fresh", "underlying_price_fresh")
            ),
        )

        # Collect all failed gates (not just the first one)
        failed_gates = [r for r in all_results if not r.passed]

        abstain = Abstain(
            id=str(uuid.uuid4()),
            generated_at=datetime.now(timezone.utc).isoformat(),
            underlying="",  # Will be set by caller
            reason=reason,
            failed_gates=failed_gates,
            data_freshness=data_freshness,
            resume_condition=resume_condition,
        )

        return PipelineResult(
            passed=False,
            abstain=abstain,
            all_results=all_results,
            soft_failures=soft_failures,
            confidence_cap=0,
            stage_reached=stage,
        )

    def _determine_abstain_reason(
        self,
        failed_gate: GateResult,
        stage: PipelineStage,
    ) -> AbstainReason:
        """Map failed gate to abstain reason."""
        gate_to_reason = {
            "underlying_price_fresh": AbstainReason.STALE_DATA,
            "quote_fresh": AbstainReason.STALE_DATA,
            "greeks_fresh": AbstainReason.STALE_DATA,
            "spread_acceptable": AbstainReason.SPREAD_TOO_WIDE,
            "open_interest_sufficient": AbstainReason.LIQUIDITY_INSUFFICIENT,
            "volume_sufficient": AbstainReason.LIQUIDITY_INSUFFICIENT,
            "delta_in_range": AbstainReason.GATES_FAILED,
            "iv_rank_appropriate": AbstainReason.GATES_FAILED,
            "cash_available": AbstainReason.PORTFOLIO_CONSTRAINT,
            "position_size_limit": AbstainReason.PORTFOLIO_CONSTRAINT,
            "sector_concentration": AbstainReason.PORTFOLIO_CONSTRAINT,
        }

        return gate_to_reason.get(failed_gate.gate_name, AbstainReason.GATES_FAILED)

    def _build_resume_condition(self, failed_gate: GateResult) -> str:
        """Build human-readable resume condition."""
        gate_to_condition = {
            "underlying_price_fresh": "Underlying price must update within 2s",
            "quote_fresh": "Quote must update within 5s",
            "greeks_fresh": "Greeks must update within 90s",
            "spread_acceptable": f"Spread must narrow to < 10% (currently {failed_gate.value}%)"
            if failed_gate.value else "Spread must narrow to < 10%",
            "open_interest_sufficient": f"OI must reach 100 (currently {failed_gate.value})"
            if failed_gate.value else "OI must reach 100",
            "volume_sufficient": f"Volume must reach 50 (currently {failed_gate.value})"
            if failed_gate.value else "Volume must reach 50",
            "delta_in_range": f"Delta must be 0.10-0.80 (currently {failed_gate.value:.2f})"
            if failed_gate.value else "Delta must be 0.10-0.80",
            "iv_rank_appropriate": "IV rank must align with action",
            "cash_available": "Increase cash or reduce position size",
            "position_size_limit": "Reduce position size to < 5% of portfolio",
            "sector_concentration": "Reduce sector exposure to < 25%",
        }

        return gate_to_condition.get(
            failed_gate.gate_name,
            f"Gate '{failed_gate.gate_name}' must pass"
        )

    def _calculate_confidence_cap(self, soft_failures: list[GateResult]) -> int:
        """Calculate maximum confidence based on soft failures.

        Each soft failure reduces confidence cap by a set amount.
        """
        base_confidence = 100

        # Deductions per soft gate failure
        deductions = {
            "volume_sufficient": 15,
            "iv_rank_appropriate": 20,
            "sector_concentration": 10,
        }

        for failure in soft_failures:
            deduction = deductions.get(failure.gate_name, 10)
            base_confidence -= deduction

        return max(0, base_confidence)


def evaluate_option_for_signal(
    option: AggregatedOptionData,
    underlying: UnderlyingData | None,
    action: Literal["BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"],
    contracts: int = 1,
    portfolio: PortfolioState | None = None,
) -> PipelineResult:
    """Convenience function to evaluate an option through the pipeline.

    Args:
        option: Aggregated option data
        underlying: Underlying data
        action: Trade action
        contracts: Number of contracts
        portfolio: Portfolio state

    Returns:
        PipelineResult
    """
    pipeline = GatingPipeline()
    return pipeline.evaluate(option, underlying, action, contracts, portfolio)
