"""Technical analysis for scalping signals.

Provides fast technical indicators optimized for intraday scalping:
- VWAP (Volume Weighted Average Price) with bands
- Support/Resistance level detection
- Session high/low tracking
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class VWAPState:
    """VWAP calculation state.

    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    Bands are +/- standard deviation from VWAP.
    """

    cumulative_pv: float = 0.0  # Sum of Price * Volume
    cumulative_volume: int = 0
    vwap: float = 0.0
    upper_band: float = 0.0  # VWAP + 1 std dev
    lower_band: float = 0.0  # VWAP - 1 std dev
    squared_deviations: deque[float] = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class SupportResistance:
    """Support or resistance level.

    Levels are identified from local minima/maxima and strengthened
    each time price touches them without breaking through.
    """

    price: float
    strength: int  # Number of touches
    level_type: Literal["support", "resistance"]
    last_touch: datetime

    def __repr__(self) -> str:
        return (
            f"SupportResistance({self.level_type} @ ${self.price:.2f}, "
            f"strength={self.strength})"
        )


@dataclass
class ScalpTechnicalSignal:
    """Technical signal for scalping.

    Generated when price action meets specific criteria at
    VWAP or support/resistance levels.
    """

    signal_type: Literal["vwap_bounce", "vwap_rejection", "breakout", "momentum_burst"]
    direction: Literal["bullish", "bearish"]
    price: float
    confidence: int  # 0-100
    timestamp: datetime
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ScalpTechnicalSignal({self.signal_type} {self.direction} "
            f"@ ${self.price:.2f}, confidence={self.confidence})"
        )


class TechnicalScalper:
    """Fast technical analysis for scalping decisions.

    Usage:
        scalper = TechnicalScalper("TSLA")

        # Update with each price tick
        scalper.update(420.50, volume=1000, timestamp=now)
        scalper.update(421.00, volume=1500, timestamp=now)

        # Check for VWAP signal
        signal = scalper.check_vwap_signal(421.00, velocity_pct=0.2)
        if signal:
            print(f"Signal: {signal}")

        # Access VWAP values
        print(f"VWAP: ${scalper.vwap:.2f}")
        print(f"Session high: ${scalper.session_high:.2f}")
    """

    def __init__(
        self,
        symbol: str,
        vwap_band_std: float = 1.0,
        sr_threshold_pct: float = 0.3,
        sr_lookback_minutes: int = 60,
        max_sr_levels: int = 5,
    ):
        """Initialize technical scalper.

        Args:
            symbol: Underlying symbol
            vwap_band_std: Standard deviations for VWAP bands
            sr_threshold_pct: Price must be within this % to "touch" a level
            sr_lookback_minutes: How far back to look for S/R levels
            max_sr_levels: Maximum support/resistance levels to track
        """
        self.symbol = symbol
        self.vwap_band_std = vwap_band_std
        self.sr_threshold_pct = sr_threshold_pct
        self.sr_lookback_minutes = sr_lookback_minutes
        self.max_sr_levels = max_sr_levels

        # VWAP state
        self._vwap = VWAPState()

        # Price history for S/R calculation (60 min * 60 sec = 3600 max entries)
        # Using deque with maxlen for O(1) append with automatic cleanup
        self._price_history: deque[tuple[float, datetime]] = deque(maxlen=3600)

        # Detected levels
        self._support_levels: list[SupportResistance] = []
        self._resistance_levels: list[SupportResistance] = []

        # Session tracking
        self._session_high: float = 0.0
        self._session_low: float = float("inf")
        self._session_open: float | None = None
        self._last_price: float | None = None

    def reset_session(self) -> None:
        """Reset for new trading session.

        Call this at market open to start fresh VWAP/S/R calculations.
        """
        self._vwap = VWAPState()
        self._price_history.clear()
        self._support_levels.clear()
        self._resistance_levels.clear()
        self._session_high = 0.0
        self._session_low = float("inf")
        self._session_open = None
        self._last_price = None
        logger.info(f"[{self.symbol}] Technical scalper session reset")

    def update(
        self,
        price: float,
        volume: int,
        timestamp: datetime,
    ) -> None:
        """Update all technical indicators with new tick.

        Args:
            price: Current price
            volume: Associated volume
            timestamp: Time of observation
        """
        # Track session stats
        if self._session_open is None:
            self._session_open = price
        self._session_high = max(self._session_high, price)
        self._session_low = min(self._session_low, price)
        self._last_price = price

        # Update VWAP
        self._update_vwap(price, volume)

        # Store price history (deque auto-removes oldest when full)
        self._price_history.append((price, timestamp))

        # Periodically update S/R levels (every 10 ticks to reduce overhead)
        if len(self._price_history) % 10 == 0:
            self._update_sr_levels(timestamp)

    def _update_vwap(self, price: float, volume: int) -> None:
        """Update VWAP calculation."""
        if volume <= 0:
            return

        self._vwap.cumulative_pv += price * volume
        self._vwap.cumulative_volume += volume

        if self._vwap.cumulative_volume > 0:
            self._vwap.vwap = self._vwap.cumulative_pv / self._vwap.cumulative_volume

            # Track squared deviation for std calculation (deque auto-trims to 100)
            deviation = price - self._vwap.vwap
            self._vwap.squared_deviations.append(deviation**2)

            # Calculate bands
            if len(self._vwap.squared_deviations) >= 2:
                # Standard deviation of price deviations from VWAP
                variance = statistics.mean(self._vwap.squared_deviations)
                std = variance**0.5
                self._vwap.upper_band = self._vwap.vwap + (std * self.vwap_band_std)
                self._vwap.lower_band = self._vwap.vwap - (std * self.vwap_band_std)

    def _update_sr_levels(self, current_time: datetime) -> None:
        """Update support/resistance levels from price history."""
        if len(self._price_history) < 20:
            return

        prices = [p for p, _ in self._price_history]

        # Find local minima (support) and maxima (resistance)
        for i in range(2, len(prices) - 2):
            price = prices[i]

            # Local minimum (potential support)
            if (
                price < prices[i - 1]
                and price < prices[i - 2]
                and price < prices[i + 1]
                and price < prices[i + 2]
            ):
                self._add_or_update_level(price, "support", current_time)

            # Local maximum (potential resistance)
            if (
                price > prices[i - 1]
                and price > prices[i - 2]
                and price > prices[i + 1]
                and price > prices[i + 2]
            ):
                self._add_or_update_level(price, "resistance", current_time)

    def _add_or_update_level(
        self,
        price: float,
        level_type: Literal["support", "resistance"],
        timestamp: datetime,
    ) -> None:
        """Add new S/R level or strengthen existing one."""
        levels = (
            self._support_levels
            if level_type == "support"
            else self._resistance_levels
        )
        threshold = price * (self.sr_threshold_pct / 100)

        # Check if near existing level
        for level in levels:
            if abs(level.price - price) <= threshold:
                level.strength += 1
                level.last_touch = timestamp
                return

        # Add new level
        levels.append(
            SupportResistance(
                price=price,
                strength=1,
                level_type=level_type,
                last_touch=timestamp,
            )
        )

        # Keep only strongest levels
        if level_type == "support":
            self._support_levels = sorted(
                self._support_levels, key=lambda x: -x.strength
            )[: self.max_sr_levels]
        else:
            self._resistance_levels = sorted(
                self._resistance_levels, key=lambda x: -x.strength
            )[: self.max_sr_levels]

    def check_vwap_signal(
        self,
        current_price: float,
        velocity_pct: float,
    ) -> ScalpTechnicalSignal | None:
        """Check for VWAP-based signals.

        Signals:
        - VWAP Bounce: Price touches VWAP from above/below and reverses
        - VWAP Rejection: Price fails to break through VWAP

        Args:
            current_price: Current price
            velocity_pct: Current price velocity (from VelocityTracker)

        Returns:
            ScalpTechnicalSignal if pattern detected, None otherwise
        """
        if self._vwap.vwap <= 0:
            return None

        vwap = self._vwap.vwap
        distance_to_vwap_pct = ((current_price - vwap) / vwap) * 100

        # Must be near VWAP (within 0.2%)
        if abs(distance_to_vwap_pct) > 0.2:
            return None

        timestamp = datetime.now()

        # Bullish bounce: Price at/below VWAP, velocity turning up
        if distance_to_vwap_pct <= 0.1 and velocity_pct > 0.1:
            return ScalpTechnicalSignal(
                signal_type="vwap_bounce",
                direction="bullish",
                price=current_price,
                confidence=70,
                timestamp=timestamp,
                metadata={"vwap": vwap, "distance_pct": distance_to_vwap_pct},
            )

        # Bearish rejection: Price at/above VWAP, velocity turning down
        if distance_to_vwap_pct >= -0.1 and velocity_pct < -0.1:
            return ScalpTechnicalSignal(
                signal_type="vwap_rejection",
                direction="bearish",
                price=current_price,
                confidence=70,
                timestamp=timestamp,
                metadata={"vwap": vwap, "distance_pct": distance_to_vwap_pct},
            )

        return None

    def check_breakout(
        self,
        current_price: float,
        velocity_pct: float,
        volume_ratio: float,
    ) -> ScalpTechnicalSignal | None:
        """Check for support/resistance breakout.

        A breakout requires:
        - Price crossing a level with strength >= 2
        - Momentum in direction of break (velocity)
        - Volume confirmation (ratio > 1.5)

        Args:
            current_price: Current price
            velocity_pct: Current price velocity
            volume_ratio: Current volume vs baseline ratio

        Returns:
            ScalpTechnicalSignal if breakout detected, None otherwise
        """
        timestamp = datetime.now()

        # Check resistance breakout (bullish)
        for level in self._resistance_levels:
            if level.strength < 2:
                continue

            # Price breaking above resistance with momentum and volume
            if (
                current_price > level.price
                and velocity_pct > 0.2
                and volume_ratio > 1.5
            ):
                confidence = 60 + min(level.strength * 5, 20)
                return ScalpTechnicalSignal(
                    signal_type="breakout",
                    direction="bullish",
                    price=current_price,
                    confidence=confidence,
                    timestamp=timestamp,
                    metadata={
                        "level": level.price,
                        "level_strength": level.strength,
                        "volume_ratio": volume_ratio,
                    },
                )

        # Check support breakdown (bearish)
        for level in self._support_levels:
            if level.strength < 2:
                continue

            # Price breaking below support with momentum and volume
            if (
                current_price < level.price
                and velocity_pct < -0.2
                and volume_ratio > 1.5
            ):
                confidence = 60 + min(level.strength * 5, 20)
                return ScalpTechnicalSignal(
                    signal_type="breakout",
                    direction="bearish",
                    price=current_price,
                    confidence=confidence,
                    timestamp=timestamp,
                    metadata={
                        "level": level.price,
                        "level_strength": level.strength,
                        "volume_ratio": volume_ratio,
                    },
                )

        return None

    def get_nearest_support(self, current_price: float) -> SupportResistance | None:
        """Get nearest support level below current price."""
        below = [l for l in self._support_levels if l.price < current_price]
        if not below:
            return None
        return max(below, key=lambda l: l.price)

    def get_nearest_resistance(self, current_price: float) -> SupportResistance | None:
        """Get nearest resistance level above current price."""
        above = [l for l in self._resistance_levels if l.price > current_price]
        if not above:
            return None
        return min(above, key=lambda l: l.price)

    @property
    def vwap(self) -> float:
        """Current VWAP value."""
        return self._vwap.vwap

    @property
    def vwap_upper(self) -> float:
        """VWAP upper band."""
        return self._vwap.upper_band

    @property
    def vwap_lower(self) -> float:
        """VWAP lower band."""
        return self._vwap.lower_band

    @property
    def session_high(self) -> float:
        """Session high price."""
        return self._session_high

    @property
    def session_low(self) -> float:
        """Session low price."""
        return self._session_low if self._session_low != float("inf") else 0.0

    @property
    def session_open(self) -> float | None:
        """Session opening price."""
        return self._session_open

    @property
    def support_levels(self) -> list[SupportResistance]:
        """Current support levels, sorted by strength."""
        return sorted(self._support_levels, key=lambda l: -l.strength)

    @property
    def resistance_levels(self) -> list[SupportResistance]:
        """Current resistance levels, sorted by strength."""
        return sorted(self._resistance_levels, key=lambda l: -l.strength)

    @property
    def last_price(self) -> float | None:
        """Most recent price."""
        return self._last_price
