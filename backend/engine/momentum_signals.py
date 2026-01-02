"""Momentum-based signal generator for bear markets.

In bear markets, following sentiment leads to poor outcomes (buying dips that keep dipping).
This module generates PUT signals based on price momentum - "sell the rally" instead of
"buy the dip".

Signal Logic:
1. Bear market confirmed (via Price-Sentiment Divergence Gate detection)
2. Recent bounce detected (2%+ rise from recent low in last 7 days)
3. Bounce exhaustion (gave back 40%+ of gains OR 2+ consecutive red days after peak)
4. Technical confirmation (at least 1 of 3: RSI < 50, MACD < Signal, price < 20 SMA)
5. Not oversold (RSI > 25 - don't short into capitulation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from backend.data.price_indicators import (
    get_sma,
    get_recent_low,
    get_recent_high,
    get_price_n_days_ago,
    get_price_series,
    find_recent_low_index,
    find_peak_after_index,
    count_consecutive_red_days,
    _get_historical_price_data,
    _get_price_data,
)
from backend.data.technicals import calculate_rsi, calculate_macd

logger = logging.getLogger(__name__)


@dataclass
class MomentumSignal:
    """A momentum-based trading signal (PUT in bear markets)."""

    symbol: str
    direction: str  # "PUT" in bear market mode
    strength: str  # "strong", "moderate", "weak"
    confidence: int  # 0-100
    reasons: list[str] = field(default_factory=list)
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class MomentumSignalConfig:
    """Configuration for momentum signal generation."""

    # Bounce detection - identifies relief rally to sell into
    bounce_threshold: float = 0.02  # 2% bounce from low required
    bounce_lookback_days: int = 7  # Days to look back for recent low (extended from 5)

    # Bounce exhaustion - detects when bounce is failing (replaces SMA proximity)
    exhaustion_giveback_pct: float = 0.30  # Bounce gave back 30%+ of gains (relaxed from 40%)
    exhaustion_red_days: int = 2  # Or 2+ consecutive red days after peak

    # RSI thresholds
    oversold_rsi: float = 20.0  # Don't short below this (extreme panic only, relaxed from 25)
    neutral_rsi: float = 50.0  # Above this is bearish for PUTs

    # Technical confirmation requirement
    min_tech_confirmations: int = 2  # Require at least 2/3 for quality signals

    # Enabled flag
    enabled: bool = True


class MomentumSignalGenerator:
    """Generates PUT signals during bear markets based on price momentum.

    Activates only when PriceSentimentDivergenceGate detects bear market.
    Implements "sell the rally" strategy - wait for relief rallies and
    short at resistance levels.
    """

    def __init__(self, config: Optional[MomentumSignalConfig] = None):
        self.config = config or MomentumSignalConfig()

    def _get_price_data_for_date(
        self,
        symbol: str,
        as_of_date: Optional[date] = None
    ) -> tuple[Optional[float], list[float]]:
        """Get current price and historical closes for technical analysis.

        Returns:
            Tuple of (current_price, closes_list_oldest_first)
        """
        # Need about 35 days for MACD (26 + 9) plus some buffer
        calendar_days = 60

        if as_of_date:
            df = _get_historical_price_data(symbol, as_of_date, days_back=calendar_days)
        else:
            df = _get_price_data(symbol, days_back=calendar_days)

        if df.empty:
            return None, []

        current_price = float(df['Close'].iloc[-1])
        closes = df['Close'].tolist()

        return current_price, closes

    def detect_bounce(
        self,
        symbol: str,
        current_price: float,
        as_of_date: Optional[date] = None
    ) -> tuple[bool, float, str]:
        """Detect if price has bounced 3%+ from recent low.

        This identifies the "relief rally" to sell into.

        Returns:
            Tuple of (bounce_detected, bounce_percentage, reason)
        """
        recent_low = get_recent_low(
            symbol,
            as_of_date,
            days=self.config.bounce_lookback_days
        )

        if recent_low is None or recent_low == 0:
            return False, 0.0, "No recent low data"

        bounce_pct = (current_price - recent_low) / recent_low

        if bounce_pct >= self.config.bounce_threshold:
            return (
                True,
                bounce_pct,
                f"Relief rally detected: +{bounce_pct * 100:.1f}% from recent low"
            )

        return (
            False,
            bounce_pct,
            f"No bounce yet ({bounce_pct * 100:.1f}% < {self.config.bounce_threshold * 100:.1f}% threshold)"
        )

    def detect_bounce_exhaustion(
        self,
        symbol: str,
        as_of_date: Optional[date] = None
    ) -> tuple[bool, float, str]:
        """Detect if a recent bounce is failing/exhausted.

        This replaces the SMA proximity check with a more reliable pattern:
        - Bounce gave back 40%+ of gains, OR
        - 2+ consecutive red days after bounce peak

        Returns:
            Tuple of (exhaustion_detected, bounce_pct, reason)
        """
        # Get recent price action (need enough data to find bounce pattern)
        prices = get_price_series(symbol, as_of_date, days=12)
        if len(prices) < 5:
            return False, 0.0, "Insufficient price data"

        # Find recent low within lookback period
        low_idx = find_recent_low_index(prices, lookback=self.config.bounce_lookback_days)
        if low_idx is None:
            return False, 0.0, "No recent low found"

        recent_low = prices[low_idx]

        # Find peak after the low
        peak_idx, bounce_peak = find_peak_after_index(prices, low_idx)
        if peak_idx is None or bounce_peak is None:
            return False, 0.0, "No price action after low"

        current_price = prices[-1]

        # Calculate bounce metrics
        bounce_amount = bounce_peak - recent_low
        if bounce_amount <= 0:
            return False, 0.0, "No bounce detected"

        bounce_pct = bounce_amount / recent_low
        if bounce_pct < self.config.bounce_threshold:
            return False, bounce_pct, f"Bounce too small: {bounce_pct * 100:.1f}%"

        # Check exhaustion criteria
        giveback = bounce_peak - current_price
        giveback_pct = giveback / bounce_amount if bounce_amount > 0 else 0

        # Criterion 1: Gave back 40%+ of bounce
        if giveback_pct >= self.config.exhaustion_giveback_pct:
            return (
                True,
                bounce_pct,
                f"Bounce exhaustion: Gave back {giveback_pct * 100:.0f}% of +{bounce_pct * 100:.1f}% bounce"
            )

        # Criterion 2: Consecutive red days after peak
        if peak_idx < len(prices) - 1:
            prices_after_peak = prices[peak_idx:]
            red_days = count_consecutive_red_days(prices_after_peak)
            if red_days >= self.config.exhaustion_red_days:
                return (
                    True,
                    bounce_pct,
                    f"Bounce failed: {red_days} consecutive red days after +{bounce_pct * 100:.1f}% bounce"
                )

        return (
            False,
            bounce_pct,
            f"Bounce not exhausted yet (gave back {giveback_pct * 100:.0f}%, {count_consecutive_red_days(prices[peak_idx:]) if peak_idx < len(prices) else 0} red days)"
        )

    def get_technical_confirmations(
        self,
        symbol: str,
        current_price: float,
        closes: list[float],
        as_of_date: Optional[date] = None
    ) -> tuple[int, list[str]]:
        """Count technical confirmations for bearish momentum.

        Requires at least 2 of 3:
        1. RSI < 50 (bearish but not oversold, room to fall)
        2. MACD < Signal (bearish momentum)
        3. Price < 20-day SMA (short-term downtrend)

        Returns:
            Tuple of (confirmation_count, reasons)
        """
        confirmations = 0
        reasons = []

        # Check RSI (bearish but not oversold)
        if len(closes) >= 15:  # Need 14 periods + 1
            rsi = calculate_rsi(closes, 14)
            if rsi is not None:
                if self.config.oversold_rsi < rsi < self.config.neutral_rsi:
                    confirmations += 1
                    reasons.append(f"RSI {rsi:.1f} (bearish, room to fall)")
                elif rsi <= self.config.oversold_rsi:
                    # This is a negative - don't count but note it
                    reasons.append(f"RSI {rsi:.1f} OVERSOLD - skip signal")
                    return -1, reasons  # Signal to skip entirely
                else:
                    reasons.append(f"RSI {rsi:.1f} (neutral/bullish)")

        # Check MACD
        if len(closes) >= 35:  # Need 26 + 9
            macd_line, signal_line, histogram = calculate_macd(closes)
            if macd_line is not None and signal_line is not None:
                if macd_line < signal_line:
                    confirmations += 1
                    reasons.append(f"MACD below signal (bearish momentum)")
                else:
                    reasons.append(f"MACD above signal (bullish)")

        # Check 20-day SMA
        sma_20 = get_sma(symbol, 20, as_of_date)
        if sma_20 is not None and current_price is not None:
            if current_price < sma_20:
                confirmations += 1
                reasons.append(f"Price below 20-day SMA (${sma_20:.2f}) - short-term downtrend")
            else:
                reasons.append(f"Price above 20-day SMA (${sma_20:.2f})")

        return confirmations, reasons

    def is_oversold(
        self,
        closes: list[float]
    ) -> bool:
        """Check if RSI indicates oversold conditions.

        Don't short into extreme oversold - bounce is more likely than continuation.
        """
        if len(closes) < 15:
            return False

        rsi = calculate_rsi(closes, 14)
        return rsi is not None and rsi < self.config.oversold_rsi

    def generate_signal(
        self,
        symbol: str,
        bear_market_reason: str,
        as_of_date: Optional[date] = None
    ) -> Optional[MomentumSignal]:
        """Generate momentum-based PUT signal during bear market.

        All conditions must be met:
        1. Bear market confirmed (passed in via bear_market_reason)
        2. Bounce exhaustion detected (relief rally that's now failing)
        3. At least 1/3 technical confirmations
        4. Not oversold (don't short capitulation)

        Args:
            symbol: Stock symbol
            bear_market_reason: Why we're in bear market (from divergence gate)
            as_of_date: Date for backtesting (None = current)

        Returns:
            MomentumSignal if conditions met, None otherwise
        """
        if not self.config.enabled:
            return None

        reasons = [f"Bear market: {bear_market_reason}"]

        # Get price data
        current_price, closes = self._get_price_data_for_date(symbol, as_of_date)
        if current_price is None or len(closes) < 35:
            logger.debug(f"Insufficient price data for {symbol}")
            return None

        # Check for oversold (don't short into capitulation)
        if self.is_oversold(closes):
            logger.debug(f"{symbol} is oversold - skipping momentum signal")
            return None

        # Check for bounce exhaustion (replaces separate bounce + resistance checks)
        exhaustion_detected, bounce_pct, exhaustion_reason = self.detect_bounce_exhaustion(
            symbol, as_of_date
        )
        if not exhaustion_detected:
            logger.debug(f"{symbol}: {exhaustion_reason}")
            return None
        reasons.append(exhaustion_reason)

        # Get technical confirmations
        tech_count, tech_reasons = self.get_technical_confirmations(
            symbol, current_price, closes, as_of_date
        )

        # -1 means RSI was oversold - skip entirely
        if tech_count < 0:
            logger.debug(f"{symbol}: Oversold RSI detected in technicals")
            return None

        if tech_count < self.config.min_tech_confirmations:
            logger.debug(f"{symbol}: Only {tech_count}/3 technical confirmations (need {self.config.min_tech_confirmations})")
            return None

        reasons.extend(tech_reasons)

        # Determine signal strength
        if tech_count == 3 and bounce_pct > 0.05:  # Strong: all technicals + 5%+ bounce
            strength = "strong"
            confidence = 75
        elif tech_count >= 2:  # Moderate: 2+ technicals
            strength = "moderate"
            confidence = 60
        else:
            strength = "weak"
            confidence = 50  # Slightly higher for bounce exhaustion signals

        logger.info(
            f"MOMENTUM PUT SIGNAL: {symbol} - {strength.upper()} (confidence: {confidence}%)\n"
            f"  Reasons: {reasons}"
        )

        return MomentumSignal(
            symbol=symbol,
            direction="PUT",
            strength=strength,
            confidence=confidence,
            reasons=reasons,
        )

    def evaluate_for_date(
        self,
        symbol: str,
        check_date: date,
        bear_market_reason: str
    ) -> Optional[MomentumSignal]:
        """Convenience method for backtesting - same as generate_signal."""
        return self.generate_signal(symbol, bear_market_reason, as_of_date=check_date)
