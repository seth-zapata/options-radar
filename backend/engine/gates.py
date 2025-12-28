"""Gate definitions for the recommendation gating pipeline.

Implements all hard and soft gates from spec section 5.2:
- Data freshness gates (underlying, quote, Greeks)
- Liquidity gates (spread, open interest, volume)
- Strategy fit gates (IV rank, delta range)
- Portfolio constraint gates (cash, position size, sector)

See spec section 5 for full gating pipeline details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


class GateSeverity(str, Enum):
    """Gate severity levels.

    HARD: Failure causes immediate ABSTAIN
    SOFT: Failure caps confidence but allows continuation
    """
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of evaluating a single gate.

    Attributes:
        gate_name: Name of the gate that was evaluated
        passed: True if gate passed, False if failed
        value: The actual value that was evaluated
        threshold: The threshold used for comparison
        message: Human-readable explanation
        severity: Whether this is a hard or soft gate
    """
    gate_name: str
    passed: bool
    value: Any
    threshold: Any
    message: str
    severity: GateSeverity = GateSeverity.HARD


@dataclass
class GateContext:
    """Context containing all data needed for gate evaluation.

    This is the unified input for all gates, populated from
    aggregated market data and portfolio state.

    Attributes:
        # Action info
        action: The type of trade being evaluated

        # Data freshness (seconds)
        underlying_age: Age of underlying price data
        quote_age: Age of option quote data
        greeks_age: Age of Greeks data

        # Liquidity metrics
        spread_percent: Bid-ask spread as percentage of mid
        open_interest: Open interest for the contract
        volume: Daily volume for the contract
        bid_size: Size at bid
        ask_size: Size at ask

        # Greeks
        delta: Option delta
        gamma: Option gamma
        theta: Option theta
        vega: Option vega
        iv: Implied volatility

        # IV metrics
        iv_rank: IV rank (0-100)
        iv_percentile: IV percentile (0-100)

        # Position sizing
        contracts: Number of contracts to trade
        premium: Premium per contract
        available_cash: Cash available in account
        portfolio_value: Total portfolio value

        # Sector exposure
        current_sector_exposure_percent: Current sector exposure %
        new_position_percent: New position as % of portfolio
    """
    # Action
    action: Literal["BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"]

    # Data freshness
    underlying_age: float | None = None
    quote_age: float | None = None
    greeks_age: float | None = None

    # Liquidity
    spread_percent: float | None = None
    open_interest: int = 0
    volume: int = 0
    bid_size: int = 0
    ask_size: int = 0

    # Greeks
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    iv: float | None = None

    # IV metrics
    iv_rank: float = 50.0
    iv_percentile: float = 50.0

    # Position sizing
    contracts: int = 1
    premium: float = 0.0
    available_cash: float = 0.0
    portfolio_value: float = 10000.0  # Default for testing

    # Sector exposure
    current_sector_exposure_percent: float = 0.0
    new_position_percent: float = 0.0

    # Sentiment data (Phase 6) - News + WSB only
    news_sentiment_score: float | None = None  # -100 to 100 (Finnhub)
    wsb_sentiment_score: float | None = None  # -100 to 100 (Quiver WSB)
    combined_sentiment_score: float | None = None  # 50/50 weighted combined score
    wsb_is_trending: bool = False
    news_is_buzzing: bool = False
    sources_aligned: bool = False  # True if news and WSB agree on direction


class Gate(ABC):
    """Abstract base class for gates.

    Each gate evaluates a specific condition and returns a GateResult.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this gate."""
        pass

    @property
    @abstractmethod
    def severity(self) -> GateSeverity:
        """Severity level (hard or soft)."""
        pass

    @abstractmethod
    def evaluate(self, ctx: GateContext) -> GateResult:
        """Evaluate the gate against the context.

        Args:
            ctx: Gate evaluation context

        Returns:
            GateResult indicating pass/fail with details
        """
        pass


# =============================================================================
# Stage 1: Data Freshness Gates (all HARD)
# =============================================================================

class UnderlyingPriceFreshGate(Gate):
    """Ensures underlying price data is fresh (< 2 seconds)."""

    THRESHOLD = 2.0  # seconds

    @property
    def name(self) -> str:
        return "underlying_price_fresh"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.underlying_age is None:
            return GateResult(
                gate_name=self.name,
                passed=False,
                value=None,
                threshold=self.THRESHOLD,
                message="No underlying price data available",
                severity=self.severity,
            )

        passed = ctx.underlying_age <= self.THRESHOLD
        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.underlying_age,
            threshold=self.THRESHOLD,
            message="OK" if passed else f"Underlying price stale ({ctx.underlying_age:.1f}s > {self.THRESHOLD}s)",
            severity=self.severity,
        )


class QuoteFreshGate(Gate):
    """Ensures quote data is fresh (< 5 seconds)."""

    THRESHOLD = 5.0  # seconds

    @property
    def name(self) -> str:
        return "quote_fresh"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.quote_age is None:
            return GateResult(
                gate_name=self.name,
                passed=False,
                value=None,
                threshold=self.THRESHOLD,
                message="No quote data available",
                severity=self.severity,
            )

        passed = ctx.quote_age <= self.THRESHOLD
        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.quote_age,
            threshold=self.THRESHOLD,
            message="OK" if passed else f"Quote data stale ({ctx.quote_age:.1f}s > {self.THRESHOLD}s)",
            severity=self.severity,
        )


class GreeksFreshGate(Gate):
    """Ensures Greeks data is fresh (< 90 seconds)."""

    THRESHOLD = 90.0  # seconds

    @property
    def name(self) -> str:
        return "greeks_fresh"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.greeks_age is None:
            return GateResult(
                gate_name=self.name,
                passed=False,
                value=None,
                threshold=self.THRESHOLD,
                message="No Greeks data available",
                severity=self.severity,
            )

        passed = ctx.greeks_age <= self.THRESHOLD
        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.greeks_age,
            threshold=self.THRESHOLD,
            message="OK" if passed else f"Greeks data stale ({ctx.greeks_age:.1f}s > {self.THRESHOLD}s)",
            severity=self.severity,
        )


# =============================================================================
# Stage 2: Liquidity Gates
# =============================================================================

class SpreadAcceptableGate(Gate):
    """Ensures bid-ask spread is acceptable (< 10%)."""

    THRESHOLD = 10.0  # percent

    @property
    def name(self) -> str:
        return "spread_acceptable"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.spread_percent is None:
            return GateResult(
                gate_name=self.name,
                passed=False,
                value=None,
                threshold=self.THRESHOLD,
                message="Cannot calculate spread (no quote data)",
                severity=self.severity,
            )

        passed = ctx.spread_percent <= self.THRESHOLD
        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.spread_percent,
            threshold=self.THRESHOLD,
            message="OK" if passed else f"Spread {ctx.spread_percent:.1f}% exceeds {self.THRESHOLD}%",
            severity=self.severity,
        )


class OpenInterestSufficientGate(Gate):
    """Ensures open interest is sufficient (>= 100)."""

    THRESHOLD = 100

    @property
    def name(self) -> str:
        return "open_interest_sufficient"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        passed = ctx.open_interest >= self.THRESHOLD
        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.open_interest,
            threshold=self.THRESHOLD,
            message="OK" if passed else f"OI {ctx.open_interest} below {self.THRESHOLD}",
            severity=self.severity,
        )


class VolumeSufficientGate(Gate):
    """Ensures daily volume is sufficient (>= 50). SOFT gate."""

    THRESHOLD = 50

    @property
    def name(self) -> str:
        return "volume_sufficient"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        passed = ctx.volume >= self.THRESHOLD
        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.volume,
            threshold=self.THRESHOLD,
            message="OK" if passed else f"Volume {ctx.volume} is thin (< {self.THRESHOLD})",
            severity=self.severity,
        )


# =============================================================================
# Stage 3: Strategy Fit Gates
# =============================================================================

class IVRankAppropriateGate(Gate):
    """Ensures IV rank is appropriate for the action. SOFT gate.

    For buying premium: IV rank should be < 50 (buy cheap)
    For selling premium: IV rank should be > 30 (collect premium)
    """

    @property
    def name(self) -> str:
        return "iv_rank_appropriate"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        buying_premium = ctx.action in ("BUY_CALL", "BUY_PUT")

        if buying_premium:
            passed = ctx.iv_rank < 50
            threshold = "< 50"
        else:
            passed = ctx.iv_rank > 30
            threshold = "> 30"

        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.iv_rank,
            threshold=threshold,
            message="OK" if passed else f"IV rank {ctx.iv_rank:.1f} not ideal for {ctx.action}",
            severity=self.severity,
        )


class DeltaInRangeGate(Gate):
    """Ensures delta is in tradeable range (0.10 - 0.80)."""

    MIN_DELTA = 0.10
    MAX_DELTA = 0.80

    @property
    def name(self) -> str:
        return "delta_in_range"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.delta is None:
            return GateResult(
                gate_name=self.name,
                passed=False,
                value=None,
                threshold=f"{self.MIN_DELTA} - {self.MAX_DELTA}",
                message="No delta data available",
                severity=self.severity,
            )

        delta_abs = abs(ctx.delta)
        passed = self.MIN_DELTA <= delta_abs <= self.MAX_DELTA

        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=delta_abs,
            threshold=f"{self.MIN_DELTA} - {self.MAX_DELTA}",
            message="OK" if passed else f"Delta {delta_abs:.2f} outside tradeable range",
            severity=self.severity,
        )


# =============================================================================
# Stage 4: Portfolio Constraint Gates
# =============================================================================

class CashAvailableGate(Gate):
    """Ensures sufficient cash is available for the position."""

    @property
    def name(self) -> str:
        return "cash_available"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        required_cash = ctx.contracts * ctx.premium * 100
        passed = ctx.available_cash >= required_cash

        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.available_cash,
            threshold=required_cash,
            message="OK" if passed else f"Need ${required_cash:.0f}, have ${ctx.available_cash:.0f}",
            severity=self.severity,
        )


class PositionSizeLimitGate(Gate):
    """Ensures position size doesn't exceed portfolio limits (20%).

    With a $5,000 portfolio, 20% = $1,000 max per position,
    which aligns with the single position limit in session tracker.
    """

    MAX_PERCENT = 20.0

    @property
    def name(self) -> str:
        return "position_size_limit"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.portfolio_value <= 0:
            return GateResult(
                gate_name=self.name,
                passed=False,
                value=0,
                threshold=self.MAX_PERCENT,
                message="Invalid portfolio value",
                severity=self.severity,
            )

        position_value = ctx.contracts * ctx.premium * 100
        percent_of_portfolio = (position_value / ctx.portfolio_value) * 100
        passed = percent_of_portfolio <= self.MAX_PERCENT

        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=percent_of_portfolio,
            threshold=self.MAX_PERCENT,
            message="OK" if passed else f"Position {percent_of_portfolio:.1f}% exceeds {self.MAX_PERCENT}% limit",
            severity=self.severity,
        )


class SectorConcentrationGate(Gate):
    """Ensures sector exposure stays within limits (25%). SOFT gate."""

    MAX_PERCENT = 25.0

    @property
    def name(self) -> str:
        return "sector_concentration"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        total_exposure = ctx.current_sector_exposure_percent + ctx.new_position_percent
        passed = total_exposure <= self.MAX_PERCENT

        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=total_exposure,
            threshold=self.MAX_PERCENT,
            message="OK" if passed else f"Sector exposure {total_exposure:.1f}% exceeds {self.MAX_PERCENT}%",
            severity=self.severity,
        )


# =============================================================================
# Stage 5: Sentiment Gates (all SOFT - sentiment enhances but doesn't block)
# =============================================================================

class SentimentDirectionGate(Gate):
    """Checks if sentiment aligns with trade direction. SOFT gate.

    For bullish trades (BUY_CALL, SELL_PUT): Combined sentiment should be > -30
    For bearish trades (BUY_PUT, SELL_CALL): Combined sentiment should be < 30
    """

    @property
    def name(self) -> str:
        return "sentiment_direction"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.combined_sentiment_score is None:
            return GateResult(
                gate_name=self.name,
                passed=True,  # Pass if no sentiment data (don't block on missing data)
                value=None,
                threshold="N/A",
                message="No sentiment data available (skipped)",
                severity=self.severity,
            )

        is_bullish_trade = ctx.action in ("BUY_CALL", "SELL_PUT")

        if is_bullish_trade:
            passed = ctx.combined_sentiment_score > -30
            threshold = "> -30"
        else:
            passed = ctx.combined_sentiment_score < 30
            threshold = "< 30"

        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.combined_sentiment_score,
            threshold=threshold,
            message="OK" if passed else f"Sentiment {ctx.combined_sentiment_score:.0f} conflicts with {ctx.action}",
            severity=self.severity,
        )


class RetailMomentumGate(Gate):
    """Checks WSB/retail momentum. SOFT gate.

    If WSB is trending and strongly bullish/bearish, can be a momentum signal.
    This is a "nice to have" confirmation, not a blocker.
    """

    @property
    def name(self) -> str:
        return "retail_momentum"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.wsb_sentiment_score is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                value=None,
                threshold="N/A",
                message="No WSB sentiment data (skipped)",
                severity=self.severity,
            )

        is_bullish_trade = ctx.action in ("BUY_CALL", "SELL_PUT")

        # If WSB is trending, check if direction aligns
        if ctx.wsb_is_trending:
            if is_bullish_trade:
                passed = ctx.wsb_sentiment_score > -20  # Not strongly bearish
            else:
                passed = ctx.wsb_sentiment_score < 20  # Not strongly bullish
        else:
            # Not trending = no strong retail conviction, so pass
            passed = True

        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.wsb_sentiment_score,
            threshold="aligned when trending",
            message="OK" if passed else f"Trending WSB sentiment conflicts with {ctx.action}",
            severity=self.severity,
        )


class SentimentConvergenceGate(Gate):
    """Bonus gate: News and WSB sentiment agree. SOFT gate.

    When both sources point the same direction (bullish or bearish),
    this provides extra confidence in the trade direction.
    """

    @property
    def name(self) -> str:
        return "sentiment_convergence"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        # Need both sources for convergence check
        if ctx.news_sentiment_score is None or ctx.wsb_sentiment_score is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                value=None,
                threshold="both sources needed",
                message="Missing sentiment source (skipped)",
                severity=self.severity,
            )

        # Check if sources are aligned (both bullish or both bearish)
        # This is informational - always passes but message indicates alignment
        if ctx.sources_aligned:
            message = "News and WSB sentiment aligned"
        else:
            message = "News and WSB divergent (mixed signals)"

        return GateResult(
            gate_name=self.name,
            passed=True,  # Informational, not a blocker
            value=ctx.sources_aligned,
            threshold="sources aligned",
            message=message,
            severity=self.severity,
        )


# =============================================================================
# Gate Registry
# =============================================================================

# All gates organized by pipeline stage
SENTIMENT_GATES: list[Gate] = [
    SentimentDirectionGate(),
    RetailMomentumGate(),
    SentimentConvergenceGate(),
]

DATA_FRESHNESS_GATES: list[Gate] = [
    UnderlyingPriceFreshGate(),
    QuoteFreshGate(),
    GreeksFreshGate(),
]

LIQUIDITY_GATES: list[Gate] = [
    SpreadAcceptableGate(),
    OpenInterestSufficientGate(),
    VolumeSufficientGate(),
]

STRATEGY_FIT_GATES: list[Gate] = [
    IVRankAppropriateGate(),
    DeltaInRangeGate(),
]

PORTFOLIO_CONSTRAINT_GATES: list[Gate] = [
    CashAvailableGate(),
    PositionSizeLimitGate(),
    SectorConcentrationGate(),
]

# All gates in pipeline order
ALL_GATES: list[Gate] = (
    DATA_FRESHNESS_GATES +
    LIQUIDITY_GATES +
    STRATEGY_FIT_GATES +
    PORTFOLIO_CONSTRAINT_GATES +
    SENTIMENT_GATES
)


def get_hard_gates() -> list[Gate]:
    """Get all hard gates (failure = immediate ABSTAIN)."""
    return [g for g in ALL_GATES if g.severity == GateSeverity.HARD]


def get_soft_gates() -> list[Gate]:
    """Get all soft gates (failure = cap confidence)."""
    return [g for g in ALL_GATES if g.severity == GateSeverity.SOFT]
