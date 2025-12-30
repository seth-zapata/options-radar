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
    portfolio_value: float = 50000.0  # Default for testing

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
    wsb_mentions: int = 0  # WSB mention count for signal strength
    sentiment_age_hours: float = 0.0  # Age of sentiment data in hours (for recency weighting)

    # Symbol identification (for per-symbol filtering)
    underlying_symbol: str = ""

    # Technical indicators (Phase 7 - from backtest findings)
    rsi: float | None = None  # RSI value (0-100)


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
    """Ensures open interest is sufficient (>= 500).

    Compromise threshold for paper trading:
    - OI >= 500: 71 trades, +965% total, +13.6% avg (47.9% win rate)
    - OI >= 1000: 46 trades, +270% total, +5.9% avg (43.5% win rate)
    - Trades with 500-999 OI: +27.8% avg, 56% win rate

    Can tighten to 1000 for live trading if fills are problematic.
    """

    THRESHOLD = 500

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
            message="OK" if passed else f"OI {ctx.open_interest} below {self.THRESHOLD} (liquidity risk)",
            severity=self.severity,
        )


class VolumeSufficientGate(Gate):
    """Ensures daily volume is sufficient (>= 100). HARD gate.

    Updated from 50 (SOFT) to 100 (HARD) based on backtest validation
    showing low-volume contracts had significant fill issues.
    """

    THRESHOLD = 100

    @property
    def name(self) -> str:
        return "volume_sufficient"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        passed = ctx.volume >= self.THRESHOLD
        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.volume,
            threshold=self.THRESHOLD,
            message="OK" if passed else f"Volume {ctx.volume} below {self.THRESHOLD} (liquidity risk)",
            severity=self.severity,
        )


# =============================================================================
# Stage 3: Strategy Fit Gates
# =============================================================================

class IVRankExtremesGate(Gate):
    """IV Rank gate using extremes framework for sentiment-driven momentum trades.

    Based on backtest of 111 signals (Jan-Dec 2024):
    - IV < 30%: 68.3% accuracy (60 signals) - BEST, cheap premium
    - IV > 60%: 67.7% accuracy (33 signals) - GOOD if strong sentiment
    - IV 30-60%: 55.6% accuracy (18 signals) - WORST, "no man's land"

    This inverts traditional premium-seller logic because we're directional
    buyers riding sentiment waves, not selling volatility. High IV on meme
    stocks often confirms retail excitement rather than warning of overpriced options.

    Confidence modifiers:
    - IV < 30%: +5 (cheap premium, clear value)
    - IV > 60% + strong sentiment (>0.3): +5 (retail excitement confirmed)
    - IV 30-60%: -5 (neutral zone, worst performance)
    """

    LOW_IV_THRESHOLD = 30.0  # Below = cheap premium, +5 boost
    HIGH_IV_THRESHOLD = 60.0  # Above = needs strong sentiment for +5 boost
    STRONG_SENTIMENT_THRESHOLD = 0.3  # |normalized| > 0.3 = strong

    @property
    def name(self) -> str:
        return "iv_rank_extremes"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        iv_rank = ctx.iv_rank

        # Normalize sentiment for strong check
        has_strong_sentiment = False
        if ctx.combined_sentiment_score is not None:
            normalized_sentiment = abs(ctx.combined_sentiment_score / 100.0)
            has_strong_sentiment = normalized_sentiment > self.STRONG_SENTIMENT_THRESHOLD

        # Extremes framework logic
        if iv_rank < self.LOW_IV_THRESHOLD:
            # Low IV = cheap premium, always good
            return GateResult(
                gate_name=self.name,
                passed=True,  # +5 boost applied by pipeline
                value=iv_rank,
                threshold=f"< {self.LOW_IV_THRESHOLD}",
                message=f"Low IV ({iv_rank:.1f}%) - cheap premium (+5 boost)",
                severity=self.severity,
            )
        elif iv_rank > self.HIGH_IV_THRESHOLD:
            # High IV - good only with strong sentiment
            if has_strong_sentiment:
                return GateResult(
                    gate_name=self.name,
                    passed=True,  # +5 boost applied by pipeline
                    value=iv_rank,
                    threshold=f"> {self.HIGH_IV_THRESHOLD} + strong sentiment",
                    message=f"High IV ({iv_rank:.1f}%) + strong sentiment - retail excitement confirmed (+5 boost)",
                    severity=self.severity,
                )
            else:
                # High IV without strong sentiment - neutral (no penalty, no boost)
                return GateResult(
                    gate_name=self.name,
                    passed=True,  # No penalty, just no boost
                    value=iv_rank,
                    threshold=f"> {self.HIGH_IV_THRESHOLD}",
                    message=f"High IV ({iv_rank:.1f}%) without strong sentiment - no boost",
                    severity=self.severity,
                )
        else:
            # Neutral zone (30-60%) = worst performance, apply penalty
            return GateResult(
                gate_name=self.name,
                passed=False,  # -5 penalty applied by pipeline
                value=iv_rank,
                threshold=f"< {self.LOW_IV_THRESHOLD} or > {self.HIGH_IV_THRESHOLD}",
                message=f"IV ({iv_rank:.1f}%) in neutral zone (30-60%) - worst performance (-5 penalty)",
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

    With a $50,000 portfolio, 20% = $10,000 max per position.
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
# Stage 5: Sentiment & Signal Quality Gates
# =============================================================================

class SentimentAlignmentGate(Gate):
    """SOFT gate: Checks if news AND WSB sentiment agree on direction.

    When both professional news sentiment and retail WSB sentiment point
    the same direction, we have higher conviction. Misalignment applies
    a -15 confidence penalty but doesn't block the signal.

    Changed from HARD to SOFT because we lack historical news data to
    validate that alignment improves accuracy. Track aligned vs non-aligned
    signals to validate this hypothesis with live data.

    For bullish trades: Both news and WSB should be >= -10 (not bearish)
    For bearish trades: Both news and WSB should be <= 10 (not bullish)
    """

    @property
    def name(self) -> str:
        return "sentiment_alignment"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        # If missing sentiment, pass with note (don't block on missing data)
        if ctx.news_sentiment_score is None or ctx.wsb_sentiment_score is None:
            return GateResult(
                gate_name=self.name,
                passed=True,  # Pass when data unavailable
                value={"news": ctx.news_sentiment_score, "wsb": ctx.wsb_sentiment_score, "aligned": None},
                threshold="both sources preferred",
                message="[NO_NEWS] Missing news sentiment - WSB only",
                severity=self.severity,
            )

        is_bullish_trade = ctx.action in ("BUY_CALL", "SELL_PUT")

        # Check if both sources agree with trade direction
        if is_bullish_trade:
            # For bullish trades: news and WSB should not be bearish
            news_ok = ctx.news_sentiment_score >= -10  # Allow slight negative
            wsb_ok = ctx.wsb_sentiment_score >= -10
            aligned = news_ok and wsb_ok
            threshold = "news >= -10 AND wsb >= -10"
            if not aligned:
                if not news_ok and not wsb_ok:
                    message = f"[NOT_ALIGNED] Both news ({ctx.news_sentiment_score:.0f}) and WSB ({ctx.wsb_sentiment_score:.0f}) bearish"
                elif not news_ok:
                    message = f"[NOT_ALIGNED] News ({ctx.news_sentiment_score:.0f}) conflicts with bullish trade"
                else:
                    message = f"[NOT_ALIGNED] WSB ({ctx.wsb_sentiment_score:.0f}) conflicts with bullish trade"
        else:
            # For bearish trades: news and WSB should not be bullish
            news_ok = ctx.news_sentiment_score <= 10  # Allow slight positive
            wsb_ok = ctx.wsb_sentiment_score <= 10
            aligned = news_ok and wsb_ok
            threshold = "news <= 10 AND wsb <= 10"
            if not aligned:
                if not news_ok and not wsb_ok:
                    message = f"[NOT_ALIGNED] Both news ({ctx.news_sentiment_score:.0f}) and WSB ({ctx.wsb_sentiment_score:.0f}) bullish"
                elif not news_ok:
                    message = f"[NOT_ALIGNED] News ({ctx.news_sentiment_score:.0f}) conflicts with bearish trade"
                else:
                    message = f"[NOT_ALIGNED] WSB ({ctx.wsb_sentiment_score:.0f}) conflicts with bearish trade"

        if aligned:
            message = f"[ALIGNED] news={ctx.news_sentiment_score:.0f}, WSB={ctx.wsb_sentiment_score:.0f}"

        return GateResult(
            gate_name=self.name,
            passed=aligned,  # False = applies confidence penalty
            value={"news": ctx.news_sentiment_score, "wsb": ctx.wsb_sentiment_score, "aligned": aligned},
            threshold=threshold,
            message=message,
            severity=self.severity,
        )


class MinimumMentionsGate(Gate):
    """HARD gate: Requires minimum WSB mentions for signal reliability.

    A stock mentioned once on WSB is noise. Higher mention counts indicate
    real retail attention and more reliable directional moves.
    """

    MIN_MENTIONS = 5  # Configurable via SignalQualityConfig

    @property
    def name(self) -> str:
        return "minimum_mentions"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        passed = ctx.wsb_mentions >= self.MIN_MENTIONS

        if passed:
            if ctx.wsb_mentions >= 20:
                message = f"High WSB attention ({ctx.wsb_mentions} mentions)"
            else:
                message = f"WSB mentions OK ({ctx.wsb_mentions})"
        else:
            message = f"Insufficient WSB mentions ({ctx.wsb_mentions} < {self.MIN_MENTIONS})"

        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.wsb_mentions,
            threshold=f">= {self.MIN_MENTIONS}",
            message=message,
            severity=self.severity,
        )


class SignalEnabledGate(Gate):
    """HARD gate: Checks if signals are enabled for this symbol.

    Based on backtest results, some symbols (like AMD, AAPL) have ~50% accuracy,
    which is no better than random. These symbols can still be monitored but
    should not generate trade signals.
    """

    # Symbols with disabled signals (configurable via SignalQualityConfig)
    DISABLED_SYMBOLS: tuple[str, ...] = ("AMD", "AAPL")

    @property
    def name(self) -> str:
        return "signal_enabled"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        if not ctx.underlying_symbol:
            return GateResult(
                gate_name=self.name,
                passed=True,
                value=None,
                threshold="symbol required",
                message="No symbol specified (skipped)",
                severity=self.severity,
            )

        passed = ctx.underlying_symbol not in self.DISABLED_SYMBOLS

        return GateResult(
            gate_name=self.name,
            passed=passed,
            value=ctx.underlying_symbol,
            threshold=f"not in {self.DISABLED_SYMBOLS}",
            message="OK" if passed else f"Signals disabled for {ctx.underlying_symbol} (low backtest accuracy)",
            severity=self.severity,
        )


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


class HighMentionsBoostGate(Gate):
    """SOFT gate: Provides confidence boost for high mention counts.

    Based on backtest "extremes framework" analysis:
    - High mentions (>30): 72.2% accuracy
    - Moderate mentions (10-30): 64.4% accuracy
    - Edge: +7.8%

    When a symbol has 30+ WSB mentions, we consider this "high conviction"
    and award a confidence boost (+5).
    """

    HIGH_THRESHOLD = 30  # Updated from 20 based on backtest extremes framework

    @property
    def name(self) -> str:
        return "high_mentions_boost"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        is_high = ctx.wsb_mentions >= self.HIGH_THRESHOLD

        return GateResult(
            gate_name=self.name,
            passed=True,  # Always passes - this is a boost gate
            value=ctx.wsb_mentions,
            threshold=f">= {self.HIGH_THRESHOLD} for boost",
            message=f"High conviction ({ctx.wsb_mentions} mentions)" if is_high else "Normal mention volume",
            severity=self.severity,
        )


class SentimentRecencyGate(Gate):
    """SOFT gate: Adjusts confidence based on sentiment data freshness.

    Fresh sentiment (< 4 hours) gets a confidence boost.
    Stale sentiment (> 24 hours) gets a penalty.
    Based on backtest showing recent data significantly outperforms historical.
    """

    FRESH_HOURS = 4.0  # Sentiment < 4 hours old = fresh
    STALE_HOURS = 24.0  # Sentiment > 24 hours old = stale

    @property
    def name(self) -> str:
        return "sentiment_recency"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        age = ctx.sentiment_age_hours

        if age <= self.FRESH_HOURS:
            message = f"Fresh sentiment ({age:.1f}h old) - confidence boost"
            status = "fresh"
        elif age >= self.STALE_HOURS:
            message = f"Stale sentiment ({age:.1f}h old) - confidence penalty"
            status = "stale"
        else:
            message = f"Sentiment {age:.1f}h old - normal"
            status = "normal"

        return GateResult(
            gate_name=self.name,
            passed=True,  # Always passes - this is a boost/penalty gate
            value=age,
            threshold=f"<{self.FRESH_HOURS}h boost, >{self.STALE_HOURS}h penalty",
            message=message,
            severity=self.severity,
        )


# =============================================================================
# Stage 6: Technical Indicator Gates (from backtest findings)
# =============================================================================

class RSIOverboughtGate(Gate):
    """HARD gate: Blocks RSI > 70 + bullish trades (negative EV).

    Based on backtest of 114 signals (Jan-Dec 2024):
    - RSI > 70 + Bullish: 43.8% accuracy (16 signals) = NEGATIVE EV
    - RSI 30-70 + Bullish: 75.4% accuracy (69 signals)
    - RSI < 30 + Bullish: Traditional reversal opportunity

    Key insight: RSI measures price exhaustion, not conviction. Unlike
    sentiment/IV where extremes outperform, RSI extremes indicate reversal risk.
    """

    OVERBOUGHT_THRESHOLD = 70.0

    @property
    def name(self) -> str:
        return "rsi_overbought"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.HARD

    def evaluate(self, ctx: GateContext) -> GateResult:
        # If no RSI data, pass (don't block on missing data)
        if ctx.rsi is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                value=None,
                threshold=f"< {self.OVERBOUGHT_THRESHOLD} for bullish",
                message="No RSI data available (skipped)",
                severity=self.severity,
            )

        is_bullish_trade = ctx.action in ("BUY_CALL", "SELL_PUT")

        # Only block overbought + bullish (the negative EV combination)
        if is_bullish_trade and ctx.rsi > self.OVERBOUGHT_THRESHOLD:
            return GateResult(
                gate_name=self.name,
                passed=False,
                value=ctx.rsi,
                threshold=f"< {self.OVERBOUGHT_THRESHOLD}",
                message=f"RSI {ctx.rsi:.1f} overbought - bullish trades blocked (43.8% backtest accuracy)",
                severity=self.severity,
            )

        # Bearish trades with high RSI are fine (shorting overbought)
        # Bullish trades with normal/oversold RSI are fine
        return GateResult(
            gate_name=self.name,
            passed=True,
            value=ctx.rsi,
            threshold=f"< {self.OVERBOUGHT_THRESHOLD} for bullish",
            message="OK" if not is_bullish_trade else f"RSI {ctx.rsi:.1f} acceptable for bullish",
            severity=self.severity,
        )


class StrongSentimentBoostGate(Gate):
    """SOFT gate: Provides confidence boost for strong sentiment signals.

    Based on backtest "extremes framework" analysis:
    - Strong sentiment (|score| > 0.3): 83.3% accuracy
    - Moderate sentiment (0.1-0.3): 66.7% accuracy
    - Edge: +16.7%

    For conviction indicators like sentiment, extremes outperform neutral.
    """

    STRONG_THRESHOLD = 0.3  # |normalized_sentiment| > 0.3 = strong

    @property
    def name(self) -> str:
        return "strong_sentiment_boost"

    @property
    def severity(self) -> GateSeverity:
        return GateSeverity.SOFT

    def evaluate(self, ctx: GateContext) -> GateResult:
        # Normalize combined sentiment to -1 to 1 range (from -100 to 100)
        if ctx.combined_sentiment_score is None:
            return GateResult(
                gate_name=self.name,
                passed=True,
                value=None,
                threshold=f"|sentiment| > {self.STRONG_THRESHOLD}",
                message="No sentiment data (skipped)",
                severity=self.severity,
            )

        # Normalize to -1 to 1 range
        normalized = ctx.combined_sentiment_score / 100.0
        is_strong = abs(normalized) > self.STRONG_THRESHOLD

        return GateResult(
            gate_name=self.name,
            passed=True,  # Always passes - this is a boost gate
            value=abs(normalized),
            threshold=f"> {self.STRONG_THRESHOLD} for boost",
            message=f"Strong conviction ({normalized:+.2f})" if is_strong else f"Moderate conviction ({normalized:+.2f})",
            severity=self.severity,
        )


# =============================================================================
# Gate Registry
# =============================================================================

# All gates organized by pipeline stage
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
    IVRankExtremesGate(),  # Extremes framework: < 30% = +5, > 60% + sentiment = +5, 30-60% = -5
    DeltaInRangeGate(),
]

PORTFOLIO_CONSTRAINT_GATES: list[Gate] = [
    CashAvailableGate(),
    PositionSizeLimitGate(),
    SectorConcentrationGate(),
]

# Signal quality gates - HARD gates that filter for high-quality signals
SIGNAL_QUALITY_GATES: list[Gate] = [
    SignalEnabledGate(),      # HARD: Check symbol is allowed to generate signals
    SentimentAlignmentGate(), # HARD: Require news + WSB alignment
    MinimumMentionsGate(),    # HARD: Require minimum WSB mentions
]

# Sentiment enhancement gates - SOFT gates for confidence adjustment
SENTIMENT_GATES: list[Gate] = [
    SentimentDirectionGate(),    # SOFT: Combined sentiment direction
    RetailMomentumGate(),        # SOFT: WSB trending momentum
    HighMentionsBoostGate(),     # SOFT: Confidence boost for 30+ mentions (+5)
    StrongSentimentBoostGate(),  # SOFT: Confidence boost for |sentiment| > 0.3 (+5)
    SentimentRecencyGate(),      # SOFT: Boost/penalty based on data freshness
]

# Technical indicator gates - from backtest findings (Phase 7)
TECHNICAL_GATES: list[Gate] = [
    RSIOverboughtGate(),  # HARD: Block RSI > 70 + bullish (43.8% accuracy = negative EV)
]

# All gates in pipeline order
ALL_GATES: list[Gate] = (
    DATA_FRESHNESS_GATES +
    LIQUIDITY_GATES +
    STRATEGY_FIT_GATES +
    PORTFOLIO_CONSTRAINT_GATES +
    SIGNAL_QUALITY_GATES +
    SENTIMENT_GATES +
    TECHNICAL_GATES
)


def get_hard_gates() -> list[Gate]:
    """Get all hard gates (failure = immediate ABSTAIN)."""
    return [g for g in ALL_GATES if g.severity == GateSeverity.HARD]


def get_soft_gates() -> list[Gate]:
    """Get all soft gates (failure = cap confidence)."""
    return [g for g in ALL_GATES if g.severity == GateSeverity.SOFT]
