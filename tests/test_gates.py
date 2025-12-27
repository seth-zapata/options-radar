"""Unit tests for gate definitions.

Tests all hard and soft gates with various inputs.
"""

import pytest

from backend.engine.gates import (
    CashAvailableGate,
    DeltaInRangeGate,
    GateContext,
    GateSeverity,
    GreeksFreshGate,
    IVRankAppropriateGate,
    OpenInterestSufficientGate,
    PositionSizeLimitGate,
    QuoteFreshGate,
    SectorConcentrationGate,
    SpreadAcceptableGate,
    UnderlyingPriceFreshGate,
    VolumeSufficientGate,
    get_hard_gates,
    get_soft_gates,
)


class TestDataFreshnessGates:
    """Tests for Stage 1: Data Freshness gates."""

    def test_underlying_price_fresh_passes(self):
        """Fresh underlying price passes."""
        gate = UnderlyingPriceFreshGate()
        ctx = GateContext(action="BUY_CALL", underlying_age=1.0)

        result = gate.evaluate(ctx)

        assert result.passed is True
        assert result.message == "OK"
        assert gate.severity == GateSeverity.HARD

    def test_underlying_price_fresh_fails_when_stale(self):
        """Stale underlying price fails."""
        gate = UnderlyingPriceFreshGate()
        ctx = GateContext(action="BUY_CALL", underlying_age=3.0)

        result = gate.evaluate(ctx)

        assert result.passed is False
        assert "stale" in result.message.lower()
        assert result.value == 3.0
        assert result.threshold == 2.0

    def test_underlying_price_fresh_fails_when_none(self):
        """Missing underlying price fails."""
        gate = UnderlyingPriceFreshGate()
        ctx = GateContext(action="BUY_CALL", underlying_age=None)

        result = gate.evaluate(ctx)

        assert result.passed is False
        assert "no underlying" in result.message.lower()

    def test_quote_fresh_passes(self):
        """Fresh quote passes."""
        gate = QuoteFreshGate()
        ctx = GateContext(action="BUY_CALL", quote_age=2.0)

        result = gate.evaluate(ctx)

        assert result.passed is True
        assert result.message == "OK"

    def test_quote_fresh_fails_when_stale(self):
        """Stale quote fails."""
        gate = QuoteFreshGate()
        ctx = GateContext(action="BUY_CALL", quote_age=6.0)

        result = gate.evaluate(ctx)

        assert result.passed is False
        assert result.threshold == 5.0

    def test_greeks_fresh_passes(self):
        """Fresh Greeks passes."""
        gate = GreeksFreshGate()
        ctx = GateContext(action="BUY_CALL", greeks_age=60.0)

        result = gate.evaluate(ctx)

        assert result.passed is True

    def test_greeks_fresh_fails_when_stale(self):
        """Stale Greeks fails."""
        gate = GreeksFreshGate()
        ctx = GateContext(action="BUY_CALL", greeks_age=100.0)

        result = gate.evaluate(ctx)

        assert result.passed is False
        assert result.threshold == 90.0


class TestLiquidityGates:
    """Tests for Stage 2: Liquidity gates."""

    def test_spread_acceptable_passes(self):
        """Acceptable spread passes."""
        gate = SpreadAcceptableGate()
        ctx = GateContext(action="BUY_CALL", spread_percent=5.0)

        result = gate.evaluate(ctx)

        assert result.passed is True
        assert gate.severity == GateSeverity.HARD

    def test_spread_acceptable_fails_when_wide(self):
        """Wide spread fails."""
        gate = SpreadAcceptableGate()
        ctx = GateContext(action="BUY_CALL", spread_percent=15.0)

        result = gate.evaluate(ctx)

        assert result.passed is False
        assert "15.0%" in result.message
        assert "10" in result.message  # 10.0% or 10%

    def test_open_interest_sufficient_passes(self):
        """Sufficient OI passes."""
        gate = OpenInterestSufficientGate()
        ctx = GateContext(action="BUY_CALL", open_interest=500)

        result = gate.evaluate(ctx)

        assert result.passed is True

    def test_open_interest_sufficient_fails_when_low(self):
        """Low OI fails."""
        gate = OpenInterestSufficientGate()
        ctx = GateContext(action="BUY_CALL", open_interest=50)

        result = gate.evaluate(ctx)

        assert result.passed is False
        assert "50" in result.message

    def test_volume_sufficient_is_soft_gate(self):
        """Volume gate is a soft gate."""
        gate = VolumeSufficientGate()
        ctx = GateContext(action="BUY_CALL", volume=100)

        result = gate.evaluate(ctx)

        assert result.passed is True
        assert gate.severity == GateSeverity.SOFT

    def test_volume_sufficient_fails_when_thin(self):
        """Thin volume fails (soft)."""
        gate = VolumeSufficientGate()
        ctx = GateContext(action="BUY_CALL", volume=20)

        result = gate.evaluate(ctx)

        assert result.passed is False
        assert "thin" in result.message.lower()


class TestStrategyFitGates:
    """Tests for Stage 3: Strategy Fit gates."""

    def test_iv_rank_appropriate_for_buying_low_iv(self):
        """Low IV is good for buying premium."""
        gate = IVRankAppropriateGate()
        ctx = GateContext(action="BUY_CALL", iv_rank=30.0)

        result = gate.evaluate(ctx)

        assert result.passed is True
        assert gate.severity == GateSeverity.SOFT

    def test_iv_rank_appropriate_for_buying_high_iv_fails(self):
        """High IV is bad for buying premium."""
        gate = IVRankAppropriateGate()
        ctx = GateContext(action="BUY_PUT", iv_rank=70.0)

        result = gate.evaluate(ctx)

        assert result.passed is False
        assert "not ideal" in result.message.lower()

    def test_iv_rank_appropriate_for_selling_high_iv(self):
        """High IV is good for selling premium."""
        gate = IVRankAppropriateGate()
        ctx = GateContext(action="SELL_CALL", iv_rank=60.0)

        result = gate.evaluate(ctx)

        assert result.passed is True

    def test_iv_rank_appropriate_for_selling_low_iv_fails(self):
        """Low IV is bad for selling premium."""
        gate = IVRankAppropriateGate()
        ctx = GateContext(action="SELL_PUT", iv_rank=20.0)

        result = gate.evaluate(ctx)

        assert result.passed is False

    def test_delta_in_range_passes(self):
        """Delta in range passes."""
        gate = DeltaInRangeGate()
        ctx = GateContext(action="BUY_CALL", delta=0.45)

        result = gate.evaluate(ctx)

        assert result.passed is True
        assert gate.severity == GateSeverity.HARD

    def test_delta_in_range_with_negative_delta(self):
        """Negative delta (puts) uses absolute value."""
        gate = DeltaInRangeGate()
        ctx = GateContext(action="BUY_PUT", delta=-0.35)

        result = gate.evaluate(ctx)

        assert result.passed is True
        assert result.value == 0.35

    def test_delta_in_range_fails_when_too_low(self):
        """Very low delta fails."""
        gate = DeltaInRangeGate()
        ctx = GateContext(action="BUY_CALL", delta=0.05)

        result = gate.evaluate(ctx)

        assert result.passed is False
        assert "outside" in result.message.lower()

    def test_delta_in_range_fails_when_too_high(self):
        """Very high delta (deep ITM) fails."""
        gate = DeltaInRangeGate()
        ctx = GateContext(action="BUY_CALL", delta=0.95)

        result = gate.evaluate(ctx)

        assert result.passed is False


class TestPortfolioConstraintGates:
    """Tests for Stage 4: Portfolio Constraint gates."""

    def test_cash_available_passes(self):
        """Sufficient cash passes."""
        gate = CashAvailableGate()
        ctx = GateContext(
            action="BUY_CALL",
            contracts=2,
            premium=5.0,
            available_cash=2000.0,
        )

        result = gate.evaluate(ctx)

        # 2 contracts * $5.00 * 100 = $1000 needed
        assert result.passed is True
        assert gate.severity == GateSeverity.HARD

    def test_cash_available_fails_when_insufficient(self):
        """Insufficient cash fails."""
        gate = CashAvailableGate()
        ctx = GateContext(
            action="BUY_CALL",
            contracts=10,
            premium=5.0,
            available_cash=2000.0,
        )

        result = gate.evaluate(ctx)

        # 10 contracts * $5.00 * 100 = $5000 needed
        assert result.passed is False
        assert "5000" in result.message
        assert "2000" in result.message

    def test_position_size_limit_passes(self):
        """Position within limit passes."""
        gate = PositionSizeLimitGate()
        ctx = GateContext(
            action="BUY_CALL",
            contracts=1,
            premium=2.0,
            portfolio_value=10000.0,
        )

        result = gate.evaluate(ctx)

        # 1 contract * $2.00 * 100 = $200 = 2% of $10k
        assert result.passed is True
        assert gate.severity == GateSeverity.HARD

    def test_position_size_limit_fails_when_too_large(self):
        """Position exceeding limit fails."""
        gate = PositionSizeLimitGate()
        ctx = GateContext(
            action="BUY_CALL",
            contracts=10,
            premium=10.0,
            portfolio_value=10000.0,
        )

        result = gate.evaluate(ctx)

        # 10 contracts * $10.00 * 100 = $10000 = 100% of portfolio
        assert result.passed is False
        assert "5" in result.message  # 5% or 5.0%

    def test_sector_concentration_passes(self):
        """Low sector exposure passes."""
        gate = SectorConcentrationGate()
        ctx = GateContext(
            action="BUY_CALL",
            current_sector_exposure_percent=10.0,
            new_position_percent=5.0,
        )

        result = gate.evaluate(ctx)

        # 10% + 5% = 15% < 25%
        assert result.passed is True
        assert gate.severity == GateSeverity.SOFT

    def test_sector_concentration_fails_when_high(self):
        """High sector exposure fails."""
        gate = SectorConcentrationGate()
        ctx = GateContext(
            action="BUY_CALL",
            current_sector_exposure_percent=20.0,
            new_position_percent=10.0,
        )

        result = gate.evaluate(ctx)

        # 20% + 10% = 30% > 25%
        assert result.passed is False
        assert "30.0%" in result.message


class TestGateRegistry:
    """Tests for gate registry functions."""

    def test_get_hard_gates_returns_only_hard(self):
        """Hard gates function returns only hard gates."""
        hard_gates = get_hard_gates()

        for gate in hard_gates:
            assert gate.severity == GateSeverity.HARD

    def test_get_soft_gates_returns_only_soft(self):
        """Soft gates function returns only soft gates."""
        soft_gates = get_soft_gates()

        for gate in soft_gates:
            assert gate.severity == GateSeverity.SOFT

    def test_hard_gates_include_all_critical_gates(self):
        """All critical gates are marked as hard."""
        hard_gate_names = {g.name for g in get_hard_gates()}

        # These must be hard gates per spec
        critical_gates = {
            "underlying_price_fresh",
            "quote_fresh",
            "greeks_fresh",
            "spread_acceptable",
            "open_interest_sufficient",
            "delta_in_range",
            "cash_available",
            "position_size_limit",
        }

        assert critical_gates.issubset(hard_gate_names)

    def test_soft_gates_include_expected_gates(self):
        """Expected soft gates are marked as soft."""
        soft_gate_names = {g.name for g in get_soft_gates()}

        expected_soft = {
            "volume_sufficient",
            "iv_rank_appropriate",
            "sector_concentration",
        }

        assert expected_soft == soft_gate_names
