"""Momentum-based signal generator for bear markets.

In bear markets, following sentiment leads to poor outcomes (buying dips that keep dipping).
This module generates PUT signals based on price momentum - "sell the rally" instead of
"buy the dip".

Signal Logic:
1. Bear market confirmed (via Price-Sentiment Divergence Gate detection)
2. Recent bounce detected (3%+ rise from recent low in last 5 days)
3. Resistance rejection (price within 2% of 50-day SMA but still below)
4. Technical confirmation (at least 2 of 3: RSI < 50, MACD < Signal, price < 20 SMA)
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
    bounce_threshold: float = 0.02  # 2% bounce from low required (was 3%)
    bounce_lookback_days: int = 5  # Days to look back for recent low

    # Resistance detection - price approaching 50-day SMA from below
    resistance_proximity: float = 0.05  # Within 5% of 50-day SMA (was 2%)

    # RSI thresholds
    oversold_rsi: float = 25.0  # Don't short below this (capitulation risk)
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

    def detect_resistance_rejection(
        self,
        symbol: str,
        current_price: float,
        as_of_date: Optional[date] = None
    ) -> tuple[bool, str]:
        """Detect if price is near 50-day SMA resistance.

        In bear markets, the 50-day SMA often acts as resistance.
        This detects when price is testing and likely to reject from it.

        Returns:
            Tuple of (rejection_detected, reason)
        """
        sma_50 = get_sma(symbol, 50, as_of_date)

        if sma_50 is None or sma_50 == 0:
            return False, "No 50-day SMA data"

        # Price must be below SMA (bearish condition)
        if current_price >= sma_50:
            return False, f"Price above 50-day SMA (${current_price:.2f} >= ${sma_50:.2f})"

        # Check if within resistance proximity (close to but below SMA)
        distance_from_sma = (sma_50 - current_price) / sma_50

        if distance_from_sma <= self.config.resistance_proximity:
            return (
                True,
                f"Price within {distance_from_sma * 100:.1f}% of 50-day SMA resistance (${sma_50:.2f})"
            )

        # Check if price touched and reversed (yesterday at/above SMA, today below)
        yesterday_price = get_price_n_days_ago(symbol, as_of_date, n=1)
        if yesterday_price and yesterday_price >= sma_50 * 0.99 and current_price < sma_50:
            return True, "Price rejected at 50-day SMA resistance (touched yesterday, below today)"

        return (
            False,
            f"Not at resistance ({distance_from_sma * 100:.1f}% below 50-day SMA)"
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
        2. Recent bounce detected (relief rally to sell)
        3. At resistance (50-day SMA)
        4. At least 2/3 technical confirmations
        5. Not oversold (don't short capitulation)

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

        # Check for bounce (relief rally)
        bounce_detected, bounce_pct, bounce_reason = self.detect_bounce(
            symbol, current_price, as_of_date
        )
        if not bounce_detected:
            logger.debug(f"{symbol}: {bounce_reason}")
            return None
        reasons.append(bounce_reason)

        # Check for resistance rejection
        rejection_detected, rejection_reason = self.detect_resistance_rejection(
            symbol, current_price, as_of_date
        )
        if not rejection_detected:
            logger.debug(f"{symbol}: {rejection_reason}")
            return None
        reasons.append(rejection_reason)

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
            confidence = 45

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
