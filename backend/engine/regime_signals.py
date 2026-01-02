"""Regime-filtered signal generator for intraday pullback/bounce entries.

Generates BUY_CALL signals during bullish regimes when price pulls back from high.
Generates BUY_PUT signals during bearish regimes when price bounces from low.

This is the core signal logic validated through backtesting:
- 71 trades, 43.7% win rate, +17.4% avg return
- 7-day regime windows, 1.5% pullback threshold, 7 DTE weeklies
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Literal

from backend.engine.regime_detector import RegimeDetector, RegimeType, ActiveRegime
from backend.engine.momentum_signals import MomentumSignalGenerator, MomentumSignalConfig
from backend.data.price_indicators import (
    get_sma,
    get_52_week_high,
    is_death_cross_active,
    count_consecutive_days_below_sma,
)

logger = logging.getLogger(__name__)

# Bear market detection thresholds (same as backtesting)
BEAR_DRAWDOWN_THRESHOLD = 0.80  # 20% below 52-week high
BEAR_SMA_THRESHOLD = 0.95  # 5% below 200-day SMA
BEAR_SMA_DAYS_REQUIRED = 10  # Consecutive days below SMA


class SignalType(Enum):
    """Trade signal types."""
    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT"
    MOMENTUM_PUT = "MOMENTUM_PUT"  # Bear market momentum signal
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class TechnicalIndicators:
    """Technical indicators for confirmation filtering.

    Attributes:
        bb_pct: Bollinger Band percentile (0-1, lower = closer to lower band)
        macd_hist: MACD histogram value (positive = bullish momentum)
        macd_prev_hist: Previous day MACD histogram
        sma_20: 20-day simple moving average
        trend_bullish: True if price > SMA-20
    """
    bb_pct: float | None = None
    macd_hist: float | None = None
    macd_prev_hist: float | None = None
    sma_20: float | None = None
    trend_bullish: bool | None = None


@dataclass
class PriceData:
    """OHLC price data for signal generation.

    Attributes:
        symbol: Stock symbol
        current: Current/close price
        high: Daily high
        low: Daily low
        open: Daily open
        timestamp: When this data was recorded
        technicals: Optional technical indicators for confirmation
    """
    symbol: str
    current: float
    high: float
    low: float
    open: float
    timestamp: datetime
    technicals: TechnicalIndicators | None = None

    @property
    def pullback_pct(self) -> float:
        """Percentage pullback from daily high.

        Formula: (high - current) / high * 100
        Higher = more pullback from high (bullish entry opportunity)
        """
        if self.high <= 0:
            return 0.0
        return (self.high - self.current) / self.high * 100

    @property
    def bounce_pct(self) -> float:
        """Percentage bounce from daily low.

        Formula: (current - low) / low * 100
        Higher = more bounce from low (bearish entry opportunity)
        """
        if self.low <= 0:
            return 0.0
        return (self.current - self.low) / self.low * 100

    def get_technical_confirmations(self, is_bullish: bool) -> int:
        """Count technical confirmations for the given direction.

        For bullish:
        - Bollinger: Price near lower band (bb_pct < 0.3)
        - MACD: Histogram rising (macd_hist > macd_prev_hist)
        - Trend: Price above SMA-20

        For bearish:
        - Bollinger: Price near upper band (bb_pct > 0.7)
        - MACD: Histogram falling (macd_hist < macd_prev_hist)
        - Trend: Price below SMA-20

        Returns:
            Number of technical confirmations (0-3)
        """
        if self.technicals is None:
            return 0

        t = self.technicals
        count = 0

        # Bollinger Band confirmation
        if t.bb_pct is not None:
            if is_bullish and t.bb_pct < 0.3:
                count += 1
            elif not is_bullish and t.bb_pct > 0.7:
                count += 1

        # MACD momentum confirmation
        if t.macd_hist is not None and t.macd_prev_hist is not None:
            if is_bullish and t.macd_hist > t.macd_prev_hist:
                count += 1
            elif not is_bullish and t.macd_hist < t.macd_prev_hist:
                count += 1

        # Trend confirmation
        if t.trend_bullish is not None:
            if is_bullish and t.trend_bullish:
                count += 1
            elif not is_bullish and not t.trend_bullish:
                count += 1

        return count


@dataclass
class TradeSignal:
    """A generated trade signal.

    Attributes:
        signal_type: BUY_CALL, BUY_PUT, or NO_SIGNAL
        symbol: Stock symbol
        generated_at: When signal was generated
        regime_type: The active regime that triggered this
        trigger_reason: Human-readable reason for signal
        trigger_pct: The pullback/bounce percentage that triggered
        entry_price: Suggested stock entry price (current)
    """
    signal_type: SignalType
    symbol: str
    generated_at: datetime
    regime_type: RegimeType
    trigger_reason: str
    trigger_pct: float
    entry_price: float

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "generated_at": self.generated_at.isoformat(),
            "regime_type": self.regime_type.value,
            "trigger_reason": self.trigger_reason,
            "trigger_pct": round(self.trigger_pct, 2),
            "entry_price": round(self.entry_price, 2),
        }


@dataclass
class SignalGeneratorConfig:
    """Configuration for signal generation.

    Validated through backtesting on TSLA 2024-01 to 2025-01.

    Attributes:
        pullback_threshold: Minimum pullback % for bullish entry (default 1.5%)
        bounce_threshold: Minimum bounce % for bearish entry (default 1.5%)
        target_dte: Target days to expiration for options (default 7)
        min_oi: Minimum open interest for liquidity (default 500)
        min_volume: Minimum daily volume (default 100)
        max_concurrent_positions: Max open positions (default 3)
        min_days_between_entries: Cooldown between entries (default 1)
        position_size_pct: Portfolio % per trade (default 10%)
        dual_regime_enabled: Enable momentum PUT signals in bear markets (default True)
    """
    pullback_threshold: float = 1.5
    bounce_threshold: float = 1.5
    target_dte: int = 7
    min_oi: int = 500
    min_volume: int = 100
    max_concurrent_positions: int = 3
    min_days_between_entries: int = 1
    position_size_pct: float = 10.0
    dual_regime_enabled: bool = True


class RegimeSignalGenerator:
    """Generates trade signals based on regime and price action.

    The strategy:
    1. Monitor WSB sentiment to detect regime changes
    2. During bullish regimes, wait for 1.5%+ pullback from daily high
    3. During bearish regimes, wait for 1.5%+ bounce from daily low
    4. Generate BUY_CALL or BUY_PUT signal accordingly

    Cooldown Strategy: "While Open"
    - Block same-direction entries while a position is open
    - Allow new entry immediately after position is closed
    - Validated via backtest: +852% total P&L, 42.6% win rate, +15.8% avg P&L

    Exit is handled by PositionTracker:
    - Take profit: +40%
    - Stop loss: -20%
    - Time exit: DTE < 1

    Usage:
        detector = RegimeDetector()
        generator = RegimeSignalGenerator(detector)

        # Update regime from WSB sentiment
        detector.update_regime("TSLA", wsb_sentiment=0.15)

        # Check for entry signal
        price_data = PriceData(
            symbol="TSLA",
            current=248.50,
            high=252.00,
            low=247.00,
            open=250.00,
            timestamp=datetime.now(timezone.utc)
        )
        signal = generator.check_entry_signal(price_data)

        if signal.signal_type != SignalType.NO_SIGNAL:
            # Generate recommendation
            ...
    """

    def __init__(
        self,
        regime_detector: RegimeDetector,
        config: SignalGeneratorConfig | None = None,
    ):
        self.regime_detector = regime_detector
        self.config = config or SignalGeneratorConfig()
        # Track open positions by symbol and direction: {symbol: {"call": True, "put": False}}
        self._open_positions_by_direction: dict[str, dict[str, bool]] = {}
        self._open_positions: int = 0

        # Initialize momentum signal generator for dual-regime mode
        self._momentum_generator: MomentumSignalGenerator | None = None
        if self.config.dual_regime_enabled:
            self._momentum_generator = MomentumSignalGenerator(MomentumSignalConfig())

    def _detect_bear_market(
        self,
        symbol: str,
        current_price: float,
    ) -> tuple[bool, str]:
        """Detect if bear market conditions exist based on price indicators.

        Same logic as backtesting - blocks CALL signals and enables momentum PUTs.

        Returns:
            (is_bear_market, reason)
        """
        try:
            # Get 50-day SMA for trend confirmation
            sma_50 = get_sma(symbol, 50)
            in_downtrend = sma_50 and current_price and (current_price < sma_50)
            in_recovery = sma_50 and current_price and (current_price >= sma_50 * 1.05)

            # Don't trigger in recovery mode
            if in_recovery:
                return (False, "")

            # Check 1: Death cross active WITH downtrend
            if is_death_cross_active(symbol) and in_downtrend:
                return (True, "Death cross active AND below 50-day SMA")

            # Check 2: 52-week high drawdown WITH downtrend
            high_52w = get_52_week_high(symbol)
            if high_52w and current_price:
                drawdown_ratio = current_price / high_52w
                if drawdown_ratio < BEAR_DRAWDOWN_THRESHOLD and in_downtrend:
                    pct_down = (1 - drawdown_ratio) * 100
                    return (True, f"{pct_down:.1f}% below 52-week high AND downtrend")

            # Check 3: Sustained below 200-day SMA WITH downtrend
            sma_200 = get_sma(symbol, 200)
            if sma_200 and current_price:
                sma_ratio = current_price / sma_200
                if sma_ratio < BEAR_SMA_THRESHOLD:
                    days_below = count_consecutive_days_below_sma(
                        symbol, 200, threshold_ratio=BEAR_SMA_THRESHOLD
                    )
                    if days_below >= BEAR_SMA_DAYS_REQUIRED and in_downtrend:
                        return (True, f"Below 200-day SMA for {days_below} days AND downtrend")

        except Exception as e:
            logger.debug(f"Error checking bear market for {symbol}: {e}")

        return (False, "")

    def check_entry_signal(
        self,
        price_data: PriceData,
    ) -> TradeSignal:
        """Check if price action triggers an entry signal.

        Includes:
        1. Divergence gate: Blocks CALL signals when bear market detected
        2. Independent momentum: Generates PUT signals in bear markets
        3. Sentiment signals: Normal pullback/bounce entries

        Args:
            price_data: Current OHLC price data

        Returns:
            TradeSignal (may be NO_SIGNAL if conditions not met)
        """
        symbol = price_data.symbol
        now = price_data.timestamp

        # FIRST: Check for bear market conditions (price-based, independent of sentiment)
        is_bear_market, bear_reason = self._detect_bear_market(symbol, price_data.current)

        # INDEPENDENT MOMENTUM CHECK: Try momentum signal on ALL bear market days
        if is_bear_market and self.config.dual_regime_enabled and self._momentum_generator:
            # Check if PUT direction is available
            open_dirs = self._open_positions_by_direction.get(symbol, {})
            if not open_dirs.get("put", False):
                # Check max positions
                if self._open_positions < self.config.max_concurrent_positions:
                    momentum_signal = self.check_momentum_signal(symbol, bear_reason, price_data.current)
                    if momentum_signal.signal_type != SignalType.NO_SIGNAL:
                        return momentum_signal

        # Get active sentiment regime
        regime = self.regime_detector.get_active_regime(symbol)

        if not regime or not regime.is_active:
            return self._no_signal(symbol, now, "No active regime")

        # Determine target direction based on sentiment regime
        target_direction = "call" if regime.regime_type.is_bullish else "put"

        # DIVERGENCE GATE: Block CALL signals in bear market
        if regime.regime_type.is_bullish and is_bear_market:
            logger.info(
                f"[DIVERGENCE GATE] Blocking CALL signal for {symbol}: {bear_reason}"
            )
            return self._no_signal(
                symbol, now,
                f"Divergence gate: {bear_reason}"
            )

        # Check "while open" cooldown - block if same-direction position is open
        open_dirs = self._open_positions_by_direction.get(symbol, {})
        if open_dirs.get(target_direction, False):
            return self._no_signal(
                symbol, now,
                f"Position already open ({target_direction})"
            )

        # Check max concurrent positions
        if self._open_positions >= self.config.max_concurrent_positions:
            return self._no_signal(
                symbol, now,
                f"Max positions reached ({self._open_positions})"
            )

        # Check for entry trigger based on regime direction
        if regime.regime_type.is_bullish:
            return self._check_bullish_entry(price_data, regime, now)
        elif regime.regime_type.is_bearish:
            return self._check_bearish_entry(price_data, regime, now)
        else:
            return self._no_signal(symbol, now, "Neutral regime")

    def _check_bullish_entry(
        self,
        price_data: PriceData,
        regime: ActiveRegime,
        now: datetime,
    ) -> TradeSignal:
        """Check for bullish regime pullback entry.

        Requires:
        1. Pullback >= threshold (1.5%)
        2. At least 1 technical confirmation (Bollinger, MACD, or SMA trend)
        """
        pullback = price_data.pullback_pct

        if pullback < self.config.pullback_threshold:
            return self._no_signal(
                price_data.symbol, now,
                f"Pullback {pullback:.1f}% < {self.config.pullback_threshold}% threshold"
            )

        # Check technical confirmation
        tech_confirmations = price_data.get_technical_confirmations(is_bullish=True)
        if tech_confirmations < 1:
            return self._no_signal(
                price_data.symbol, now,
                f"Pullback {pullback:.1f}% met but no technical confirmation"
            )

        reason = (
            f"{pullback:.1f}% pullback from high during {regime.regime_type.value} "
            f"({tech_confirmations} tech confirms)"
        )
        logger.info(
            f"[SIGNAL] {now.strftime('%Y-%m-%d')} {price_data.symbol}: "
            f"BUY_CALL triggered - {reason}"
        )

        return TradeSignal(
            signal_type=SignalType.BUY_CALL,
            symbol=price_data.symbol,
            generated_at=now,
            regime_type=regime.regime_type,
            trigger_reason=reason,
            trigger_pct=pullback,
            entry_price=price_data.current,
        )

    def _check_bearish_entry(
        self,
        price_data: PriceData,
        regime: ActiveRegime,
        now: datetime,
    ) -> TradeSignal:
        """Check for bearish regime bounce entry.

        Requires:
        1. Bounce >= threshold (1.5%)
        2. At least 1 technical confirmation (Bollinger, MACD, or SMA trend)
        """
        bounce = price_data.bounce_pct

        if bounce < self.config.bounce_threshold:
            return self._no_signal(
                price_data.symbol, now,
                f"Bounce {bounce:.1f}% < {self.config.bounce_threshold}% threshold"
            )

        # Check technical confirmation
        tech_confirmations = price_data.get_technical_confirmations(is_bullish=False)
        if tech_confirmations < 1:
            return self._no_signal(
                price_data.symbol, now,
                f"Bounce {bounce:.1f}% met but no technical confirmation"
            )

        reason = (
            f"{bounce:.1f}% bounce from low during {regime.regime_type.value} "
            f"({tech_confirmations} tech confirms)"
        )
        logger.info(
            f"[SIGNAL] {now.strftime('%Y-%m-%d')} {price_data.symbol}: "
            f"BUY_PUT triggered - {reason}"
        )

        return TradeSignal(
            signal_type=SignalType.BUY_PUT,
            symbol=price_data.symbol,
            generated_at=now,
            regime_type=regime.regime_type,
            trigger_reason=reason,
            trigger_pct=bounce,
            entry_price=price_data.current,
        )

    def _no_signal(
        self,
        symbol: str,
        now: datetime,
        reason: str,
    ) -> TradeSignal:
        """Create a NO_SIGNAL result."""
        logger.debug(f"[SIGNAL] {symbol}: No signal - {reason}")

        return TradeSignal(
            signal_type=SignalType.NO_SIGNAL,
            symbol=symbol,
            generated_at=now,
            regime_type=RegimeType.NEUTRAL,
            trigger_reason=reason,
            trigger_pct=0.0,
            entry_price=0.0,
        )

    def check_momentum_signal(
        self,
        symbol: str,
        bear_market_reason: str,
        current_price: float,
    ) -> TradeSignal:
        """Check for momentum PUT signal during bear market.

        Called when the divergence gate blocks a CALL signal. This method
        tries to generate a momentum-based PUT signal instead.

        Args:
            symbol: Stock symbol
            bear_market_reason: Why bear market was detected
            current_price: Current stock price

        Returns:
            TradeSignal (MOMENTUM_PUT if conditions met, NO_SIGNAL otherwise)
        """
        now = datetime.now(timezone.utc)

        if not self._momentum_generator:
            return self._no_signal(symbol, now, "Momentum generator not enabled")

        # Check "while open" cooldown for PUT direction
        open_dirs = self._open_positions_by_direction.get(symbol, {})
        if open_dirs.get("put", False):
            return self._no_signal(symbol, now, "PUT position already open")

        # Check max concurrent positions
        if self._open_positions >= self.config.max_concurrent_positions:
            return self._no_signal(
                symbol, now,
                f"Max positions reached ({self._open_positions})"
            )

        # Try to generate momentum signal
        momentum_signal = self._momentum_generator.generate_signal(
            symbol, bear_market_reason, as_of_date=None  # None = current date
        )

        if momentum_signal:
            reason = f"Momentum PUT: {bear_market_reason} | {', '.join(momentum_signal.reasons[:3])}"
            logger.info(
                f"[MOMENTUM SIGNAL] {now.strftime('%Y-%m-%d')} {symbol}: "
                f"MOMENTUM_PUT triggered - {reason}"
            )

            return TradeSignal(
                signal_type=SignalType.MOMENTUM_PUT,
                symbol=symbol,
                generated_at=now,
                regime_type=RegimeType.STRONG_BEARISH,  # Momentum signals are bearish
                trigger_reason=reason,
                trigger_pct=0.0,  # Momentum signals don't use pullback/bounce %
                entry_price=current_price,
            )

        return self._no_signal(
            symbol, now,
            f"Bear market detected but momentum conditions not met"
        )

    def record_entry(self, symbol: str, direction: str = "call") -> None:
        """Record that an entry was taken (for cooldown tracking).

        Args:
            symbol: Stock symbol
            direction: "call" or "put"
        """
        if symbol not in self._open_positions_by_direction:
            self._open_positions_by_direction[symbol] = {}
        self._open_positions_by_direction[symbol][direction] = True
        self._open_positions += 1
        logger.info(
            f"[ENTRY] {symbol}: {direction.upper()} entry recorded, "
            f"open positions: {self._open_positions}"
        )

    def record_exit(self, symbol: str, direction: str = "call") -> None:
        """Record that a position was closed.

        Args:
            symbol: Stock symbol
            direction: "call" or "put"
        """
        if symbol in self._open_positions_by_direction:
            self._open_positions_by_direction[symbol][direction] = False
        self._open_positions = max(0, self._open_positions - 1)
        logger.info(
            f"[EXIT] {symbol}: {direction.upper()} exit recorded, "
            f"open positions: {self._open_positions}"
        )

    def set_open_positions(self, count: int) -> None:
        """Set the current open position count.

        Args:
            count: Number of open positions
        """
        self._open_positions = max(0, count)

    def get_status(self) -> dict:
        """Get current generator status for logging/display."""
        return {
            "open_positions": self._open_positions,
            "max_positions": self.config.max_concurrent_positions,
            "pullback_threshold": self.config.pullback_threshold,
            "bounce_threshold": self.config.bounce_threshold,
            "target_dte": self.config.target_dte,
            "position_size_pct": self.config.position_size_pct,
            "cooldown_strategy": "while_open",
            "dual_regime_enabled": self.config.dual_regime_enabled,
            "momentum_generator_active": self._momentum_generator is not None,
            "open_by_direction": {
                symbol: {d: v for d, v in dirs.items() if v}
                for symbol, dirs in self._open_positions_by_direction.items()
                if any(dirs.values())
            },
        }


@dataclass
class OptionSelection:
    """Selected option contract for a trade signal.

    Attributes:
        symbol: Underlying symbol
        strike: Strike price
        expiry: Expiration date (ISO format)
        option_type: "call" or "put"
        dte: Days to expiration
        bid: Current bid
        ask: Current ask
        mid: Mid price
        delta: Option delta
        open_interest: Open interest
        volume: Daily volume
    """
    symbol: str
    strike: float
    expiry: str
    option_type: Literal["call", "put"]
    dte: int
    bid: float
    ask: float
    mid: float
    delta: float | None = None
    open_interest: int = 0
    volume: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strike": self.strike,
            "expiry": self.expiry,
            "option_type": self.option_type,
            "dte": self.dte,
            "bid": round(self.bid, 2),
            "ask": round(self.ask, 2),
            "mid": round(self.mid, 2),
            "delta": round(self.delta, 3) if self.delta else None,
            "open_interest": self.open_interest,
            "volume": self.volume,
        }


def select_atm_option(
    signal: TradeSignal,
    available_options: list[dict],
    config: SignalGeneratorConfig,
) -> Optional[OptionSelection]:
    """Select the best ATM option for a signal.

    Finds the closest strike to current price with:
    - DTE closest to target (7 days)
    - Sufficient open interest (>= 500)
    - Sufficient volume (>= 100)

    Args:
        signal: The trade signal to select an option for
        available_options: List of available option contracts
        config: Signal generator configuration

    Returns:
        OptionSelection if suitable option found, None otherwise
    """
    if signal.signal_type == SignalType.NO_SIGNAL:
        return None

    # Determine option type - MOMENTUM_PUT also selects put options
    if signal.signal_type == SignalType.BUY_CALL:
        option_type = "call"
    else:  # BUY_PUT or MOMENTUM_PUT
        option_type = "put"
    target_price = signal.entry_price
    target_dte = config.target_dte

    # Calculate target expiry date
    today = datetime.now(timezone.utc).date()
    target_expiry = today + timedelta(days=target_dte)

    # Filter and score options
    candidates = []
    for opt in available_options:
        # Check option type
        if opt.get("type", "").lower() != option_type:
            continue

        # Check liquidity gates
        oi = opt.get("open_interest", 0) or 0
        vol = opt.get("volume", 0) or 0
        if oi < config.min_oi or vol < config.min_volume:
            continue

        # Check bid/ask
        bid = opt.get("bid", 0) or 0
        ask = opt.get("ask", 0) or 0
        if bid <= 0 or ask <= 0:
            continue

        # Calculate scores
        strike = opt.get("strike", 0)
        expiry_str = opt.get("expiry", "")

        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            dte = (expiry_date - today).days
        except (ValueError, TypeError):
            continue

        # Skip if too short or too long DTE
        if dte < 4 or dte > 14:
            continue

        strike_distance = abs(strike - target_price)
        dte_distance = abs(dte - target_dte)

        # Score: prefer ATM and close to target DTE
        # Lower score = better
        score = strike_distance / target_price * 100 + dte_distance * 2

        candidates.append({
            "option": opt,
            "strike": strike,
            "expiry": expiry_str,
            "dte": dte,
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
            "score": score,
        })

    if not candidates:
        logger.warning(
            f"[OPTION] {signal.symbol}: No suitable {option_type} found "
            f"(target: ${target_price:.2f}, DTE ~{target_dte})"
        )
        return None

    # Select best candidate
    best = min(candidates, key=lambda x: x["score"])

    selection = OptionSelection(
        symbol=signal.symbol,
        strike=best["strike"],
        expiry=best["expiry"],
        option_type=option_type,
        dte=best["dte"],
        bid=best["bid"],
        ask=best["ask"],
        mid=best["mid"],
        delta=best["option"].get("delta"),
        open_interest=best["option"].get("open_interest", 0),
        volume=best["option"].get("volume", 0),
    )

    logger.info(
        f"[OPTION] {signal.symbol}: Selected {option_type.upper()} "
        f"${best['strike']} {best['expiry']} (DTE {best['dte']}, "
        f"bid=${best['bid']:.2f}, ask=${best['ask']:.2f})"
    )

    return selection
