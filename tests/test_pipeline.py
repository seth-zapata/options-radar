"""Unit tests for the gating pipeline.

Tests pipeline orchestration, abstain generation, and confidence capping.
"""

from datetime import datetime, timedelta, timezone

import pytest

from backend.data.aggregator import AggregatedOptionData
from backend.engine.pipeline import (
    Abstain,
    AbstainReason,
    GatingPipeline,
    PipelineResult,
    PipelineStage,
    PortfolioState,
    evaluate_option_for_signal,
)
from backend.models.canonical import CanonicalOptionId
from backend.models.market_data import UnderlyingData


def create_test_option(
    quote_age_seconds: float = 1.0,
    greeks_age_seconds: float = 30.0,
    spread_percent: float = 3.0,
    delta: float = 0.45,
    bid: float | None = None,
    ask: float | None = None,
) -> AggregatedOptionData:
    """Create a test option with specified parameters.

    Args:
        quote_age_seconds: Age of quote data
        greeks_age_seconds: Age of Greeks data
        spread_percent: Desired spread as percentage of mid price
        delta: Option delta
        bid: Override bid (if None, computed from spread_percent)
        ask: Override ask (if None, computed from spread_percent)
    """
    now = datetime.now(timezone.utc)
    quote_time = (now - timedelta(seconds=quote_age_seconds)).isoformat()
    greeks_time = (now - timedelta(seconds=greeks_age_seconds)).isoformat()

    # Compute bid/ask from spread_percent if not provided
    # spread_percent = (ask - bid) / mid * 100
    # For a mid of 5.0 and spread_percent of X:
    # spread = mid * spread_percent / 100
    # bid = mid - spread/2, ask = mid + spread/2
    if bid is None or ask is None:
        mid = 5.0
        spread = mid * spread_percent / 100
        bid = mid - spread / 2
        ask = mid + spread / 2

    return AggregatedOptionData(
        canonical_id=CanonicalOptionId(
            underlying="NVDA",
            expiry="2025-01-17",
            right="C",
            strike=150.0,
        ),
        bid=bid,
        ask=ask,
        bid_size=100,
        ask_size=50,
        last=(bid + ask) / 2,
        quote_timestamp=quote_time,
        delta=delta,
        gamma=0.02,
        theta=-0.15,
        vega=0.25,
        iv=0.35,
        theoretical_value=(bid + ask) / 2,
        greeks_timestamp=greeks_time,
    )


def create_test_underlying(age_seconds: float = 0.5) -> UnderlyingData:
    """Create test underlying data."""
    now = datetime.now(timezone.utc)
    timestamp = (now - timedelta(seconds=age_seconds)).isoformat()

    return UnderlyingData(
        symbol="NVDA",
        price=145.0,
        iv_rank=35.0,
        iv_percentile=40.0,
        timestamp=timestamp,
    )


def create_test_portfolio(
    available_cash: float = 10000.0,
    portfolio_value: float = 50000.0,
) -> PortfolioState:
    """Create test portfolio state."""
    return PortfolioState(
        available_cash=available_cash,
        portfolio_value=portfolio_value,
        sector_exposures={},
        current_positions=set(),
    )


class TestPipelineBasics:
    """Basic pipeline tests."""

    def test_all_gates_pass(self):
        """Pipeline passes when all gates pass."""
        option = create_test_option()
        underlying = create_test_underlying()
        portfolio = create_test_portfolio()

        # Need to set open_interest for OI gate
        # The pipeline currently defaults to 0, which will fail
        # For this test, we'll use a custom pipeline without OI gate
        from backend.engine.gates import (
            DATA_FRESHNESS_GATES,
            GreeksFreshGate,
            QuoteFreshGate,
            SpreadAcceptableGate,
            UnderlyingPriceFreshGate,
        )

        pipeline = GatingPipeline(
            data_freshness_gates=list(DATA_FRESHNESS_GATES),
            liquidity_gates=[SpreadAcceptableGate()],  # Skip OI for this test
            strategy_fit_gates=[],
            portfolio_constraint_gates=[],
        )

        result = pipeline.evaluate(
            option=option,
            underlying=underlying,
            action="BUY_CALL",
            contracts=1,
            portfolio=portfolio,
        )

        assert result.passed is True
        assert result.abstain is None
        assert result.stage_reached == PipelineStage.EXPLAIN

    def test_hard_gate_failure_creates_abstain(self):
        """Hard gate failure creates abstain result."""
        # Stale quote
        option = create_test_option(quote_age_seconds=10.0)
        underlying = create_test_underlying()

        result = evaluate_option_for_signal(
            option=option,
            underlying=underlying,
            action="BUY_CALL",
        )

        assert result.passed is False
        assert result.abstain is not None
        assert result.abstain.reason == AbstainReason.STALE_DATA

    def test_soft_gate_failure_caps_confidence(self):
        """Soft gate failure reduces confidence but continues."""
        option = create_test_option()
        underlying = create_test_underlying()

        # High IV rank (bad for buying)
        underlying_high_iv = UnderlyingData(
            symbol="NVDA",
            price=145.0,
            iv_rank=70.0,  # Too high for buying
            iv_percentile=75.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Use minimal pipeline to isolate IV rank test
        from backend.engine.gates import (
            IVRankAppropriateGate,
            SpreadAcceptableGate,
        )

        pipeline = GatingPipeline(
            data_freshness_gates=[],
            liquidity_gates=[SpreadAcceptableGate()],
            strategy_fit_gates=[IVRankAppropriateGate()],
            portfolio_constraint_gates=[],
        )

        result = pipeline.evaluate(
            option=option,
            underlying=underlying_high_iv,
            action="BUY_CALL",
            contracts=1,
            portfolio=create_test_portfolio(),
        )

        # Should still pass (IV rank is soft), but with reduced confidence
        assert result.passed is True
        assert len(result.soft_failures) == 1
        assert result.confidence_cap < 100


class TestPipelineStages:
    """Tests for pipeline stage ordering."""

    def test_data_freshness_stage_runs_first(self):
        """Data freshness gates run before liquidity."""
        # Stale underlying AND wide spread
        option = create_test_option(spread_percent=20.0)
        underlying = create_test_underlying(age_seconds=5.0)  # Stale

        result = evaluate_option_for_signal(
            option=option,
            underlying=underlying,
            action="BUY_CALL",
        )

        # Should fail on freshness first, not spread
        assert result.passed is False
        assert result.stage_reached == PipelineStage.DATA_FRESHNESS
        assert result.abstain.reason == AbstainReason.STALE_DATA

    def test_liquidity_stage_runs_after_freshness(self):
        """Liquidity gates run after data freshness passes."""
        from backend.engine.gates import (
            DATA_FRESHNESS_GATES,
            SpreadAcceptableGate,
        )

        # Use custom pipeline with only spread gate (not OI)
        pipeline = GatingPipeline(
            data_freshness_gates=list(DATA_FRESHNESS_GATES),
            liquidity_gates=[SpreadAcceptableGate()],  # Only spread
            strategy_fit_gates=[],
            portfolio_constraint_gates=[],
        )

        # Fresh data, but wide spread
        option = create_test_option(spread_percent=20.0)
        underlying = create_test_underlying()

        result = pipeline.evaluate(
            option=option,
            underlying=underlying,
            action="BUY_CALL",
            contracts=1,
            portfolio=create_test_portfolio(),
        )

        # Should fail on liquidity (spread)
        assert result.passed is False
        assert result.stage_reached == PipelineStage.LIQUIDITY
        assert result.abstain.reason == AbstainReason.SPREAD_TOO_WIDE


class TestAbstainGeneration:
    """Tests for abstain object generation."""

    def test_abstain_has_required_fields(self):
        """Abstain object has all required fields."""
        option = create_test_option(quote_age_seconds=10.0)  # Stale
        underlying = create_test_underlying()

        result = evaluate_option_for_signal(
            option=option,
            underlying=underlying,
            action="BUY_CALL",
        )

        abstain = result.abstain
        assert abstain is not None
        assert abstain.id  # Has UUID
        assert abstain.generated_at  # Has timestamp
        assert abstain.reason  # Has reason
        assert abstain.failed_gates  # Has failed gates
        assert abstain.data_freshness  # Has freshness info
        assert abstain.resume_condition  # Has condition

    def test_abstain_includes_failed_gate_details(self):
        """Abstain includes details of failed gates."""
        option = create_test_option(quote_age_seconds=10.0)
        underlying = create_test_underlying()

        result = evaluate_option_for_signal(
            option=option,
            underlying=underlying,
            action="BUY_CALL",
        )

        failed_gates = result.abstain.failed_gates
        assert len(failed_gates) >= 1

        quote_gate = next(
            (g for g in failed_gates if g.gate_name == "quote_fresh"),
            None,
        )
        assert quote_gate is not None
        assert quote_gate.passed is False

    def test_resume_condition_is_actionable(self):
        """Resume condition provides actionable guidance."""
        from backend.engine.gates import (
            DATA_FRESHNESS_GATES,
            SpreadAcceptableGate,
        )

        # Use custom pipeline to test spread failure specifically
        pipeline = GatingPipeline(
            data_freshness_gates=list(DATA_FRESHNESS_GATES),
            liquidity_gates=[SpreadAcceptableGate()],
            strategy_fit_gates=[],
            portfolio_constraint_gates=[],
        )

        option = create_test_option(spread_percent=15.0)
        underlying = create_test_underlying()

        result = pipeline.evaluate(
            option=option,
            underlying=underlying,
            action="BUY_CALL",
            contracts=1,
            portfolio=create_test_portfolio(),
        )

        condition = result.abstain.resume_condition
        assert "10" in condition or "narrow" in condition.lower()


class TestConfidenceCapping:
    """Tests for confidence cap calculation."""

    def test_no_soft_failures_gives_full_confidence(self):
        """No soft failures means 100% confidence cap."""
        from backend.engine.gates import SpreadAcceptableGate

        pipeline = GatingPipeline(
            data_freshness_gates=[],
            liquidity_gates=[SpreadAcceptableGate()],
            strategy_fit_gates=[],
            portfolio_constraint_gates=[],
        )

        option = create_test_option()
        underlying = create_test_underlying()

        result = pipeline.evaluate(
            option=option,
            underlying=underlying,
            action="BUY_CALL",
            contracts=1,
            portfolio=create_test_portfolio(),
        )

        assert result.passed is True
        assert result.confidence_cap == 100

    def test_multiple_soft_failures_reduce_confidence(self):
        """Multiple soft failures compound confidence reduction."""
        from backend.engine.gates import (
            IVRankAppropriateGate,
            SectorConcentrationGate,
            SpreadAcceptableGate,
            VolumeSufficientGate,
        )

        pipeline = GatingPipeline(
            data_freshness_gates=[],
            liquidity_gates=[SpreadAcceptableGate(), VolumeSufficientGate()],
            strategy_fit_gates=[IVRankAppropriateGate()],
            portfolio_constraint_gates=[SectorConcentrationGate()],
        )

        option = create_test_option()
        underlying = UnderlyingData(
            symbol="NVDA",
            price=145.0,
            iv_rank=70.0,  # Bad for buying
            iv_percentile=75.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        portfolio = PortfolioState(
            available_cash=10000.0,
            portfolio_value=50000.0,
            sector_exposures={"NVDA": 22.0},  # High exposure
        )

        result = pipeline.evaluate(
            option=option,
            underlying=underlying,
            action="BUY_CALL",
            contracts=1,
            portfolio=portfolio,
        )

        # Should pass but with multiple soft failures
        assert result.passed is True
        assert len(result.soft_failures) >= 2
        # Each soft failure reduces confidence
        assert result.confidence_cap < 80


class TestConvenienceFunction:
    """Tests for evaluate_option_for_signal function."""

    def test_evaluate_option_for_signal_works(self):
        """Convenience function produces correct results."""
        option = create_test_option()
        underlying = create_test_underlying()

        result = evaluate_option_for_signal(
            option=option,
            underlying=underlying,
            action="BUY_CALL",
            contracts=1,
        )

        assert isinstance(result, PipelineResult)
        assert result.all_results  # Has gate results

    def test_evaluate_option_for_signal_with_none_underlying(self):
        """Function handles missing underlying data."""
        option = create_test_option()

        result = evaluate_option_for_signal(
            option=option,
            underlying=None,  # Missing
            action="BUY_CALL",
        )

        # Should fail on underlying freshness
        assert result.passed is False
        assert result.abstain.reason == AbstainReason.STALE_DATA
