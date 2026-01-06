"""Price velocity tracker for scalping signals.

Tracks underlying stock price velocity over multiple timeframes
to detect rapid moves that may present scalping opportunities.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """A single price observation."""

    price: float
    timestamp: datetime
    volume: int = 0


@dataclass
class VelocityReading:
    """Result of velocity calculation over a time window.

    Attributes:
        change_pct: Percentage change over window (e.g., 0.5 = +0.5%)
        change_dollars: Absolute dollar change
        direction: 'up', 'down', or 'flat'
        speed: Rate of change per second (absolute)
        window_seconds: Time window used for calculation
        data_points: Number of price points in window
    """

    change_pct: float
    change_dollars: float
    direction: Literal["up", "down", "flat"]
    speed: float
    window_seconds: int
    data_points: int

    @property
    def is_significant(self) -> bool:
        """Check if velocity is significant for trading.

        Requires at least 0.3% move with 3+ data points.
        """
        return abs(self.change_pct) >= 0.3 and self.data_points >= 3

    def __repr__(self) -> str:
        return (
            f"VelocityReading({self.change_pct:+.2f}% {self.direction} "
            f"over {self.window_seconds}s, {self.data_points} pts)"
        )


@dataclass
class SpikeSignal:
    """Detected price spike event.

    A spike is a rapid price move that exceeds a threshold
    within a short time window.
    """

    direction: Literal["up", "down"]
    change_pct: float
    change_dollars: float
    duration_seconds: float
    start_price: float
    end_price: float
    timestamp: datetime

    def __repr__(self) -> str:
        return (
            f"SpikeSignal({self.direction} {self.change_pct:+.2f}% "
            f"${self.start_price:.2f}→${self.end_price:.2f} "
            f"in {self.duration_seconds:.1f}s)"
        )


class PriceVelocityTracker:
    """Tracks price velocity over multiple timeframes.

    Usage:
        tracker = PriceVelocityTracker("TSLA")

        # Add price observations as they arrive
        tracker.add_price(420.50, datetime.now())
        tracker.add_price(421.00, datetime.now())

        # Get velocity over 30-second window
        v = tracker.get_velocity(30)
        if v and v.is_significant:
            print(f"Significant move: {v}")

        # Detect spike (0.5% move in 30 seconds)
        spike = tracker.detect_spike(threshold_pct=0.5, window_seconds=30)
        if spike:
            print(f"Spike detected: {spike}")
    """

    def __init__(
        self,
        symbol: str,
        max_history_seconds: int = 300,
        windows: tuple[int, ...] = (5, 15, 30, 60),
    ):
        """Initialize velocity tracker.

        Args:
            symbol: Underlying symbol being tracked
            max_history_seconds: How long to keep price history (default 5 min)
            windows: Time windows in seconds to calculate velocity for
        """
        self.symbol = symbol
        self.max_history_seconds = max_history_seconds
        self.windows = windows
        self._prices: deque[PricePoint] = deque()
        self._last_spike: SpikeSignal | None = None

    def add_price(
        self,
        price: float,
        timestamp: datetime,
        volume: int = 0,
    ) -> None:
        """Add a new price observation.

        Args:
            price: Current price
            timestamp: Time of observation
            volume: Associated volume (optional)
        """
        self._prices.append(PricePoint(price, timestamp, volume))
        self._cleanup_old_prices(timestamp)

    def _cleanup_old_prices(self, current_time: datetime) -> None:
        """Remove prices older than max_history_seconds."""
        cutoff = current_time - timedelta(seconds=self.max_history_seconds)
        while self._prices and self._prices[0].timestamp < cutoff:
            self._prices.popleft()

    def get_velocity(self, window_seconds: int) -> VelocityReading | None:
        """Calculate price velocity over specified window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            VelocityReading or None if insufficient data
        """
        if len(self._prices) < 2:
            return None

        current_time = self._prices[-1].timestamp
        cutoff = current_time - timedelta(seconds=window_seconds)

        # Get prices within window
        window_prices = [p for p in self._prices if p.timestamp >= cutoff]

        if len(window_prices) < 2:
            return None

        start_price = window_prices[0].price
        end_price = window_prices[-1].price

        if start_price <= 0:
            return None

        change_dollars = end_price - start_price
        change_pct = (change_dollars / start_price) * 100

        # Calculate actual time span
        actual_seconds = (
            window_prices[-1].timestamp - window_prices[0].timestamp
        ).total_seconds()
        speed = abs(change_pct / actual_seconds) if actual_seconds > 0 else 0

        # Determine direction (threshold of 0.05% to avoid noise)
        if change_pct > 0.05:
            direction: Literal["up", "down", "flat"] = "up"
        elif change_pct < -0.05:
            direction = "down"
        else:
            direction = "flat"

        return VelocityReading(
            change_pct=change_pct,
            change_dollars=change_dollars,
            direction=direction,
            speed=speed,
            window_seconds=window_seconds,
            data_points=len(window_prices),
        )

    def get_all_velocities(self) -> dict[int, VelocityReading | None]:
        """Get velocity readings for all configured windows.

        Returns:
            Dict mapping window_seconds to VelocityReading
        """
        return {w: self.get_velocity(w) for w in self.windows}

    def detect_spike(
        self,
        threshold_pct: float = 0.5,
        window_seconds: int = 30,
        cooldown_seconds: float = 60,
    ) -> SpikeSignal | None:
        """Detect if price spiked beyond threshold.

        A spike is a move exceeding threshold_pct within window_seconds.
        After detecting a spike, a cooldown prevents repeated signals.

        Args:
            threshold_pct: Minimum percentage move to trigger (e.g., 0.5 = 0.5%)
            window_seconds: Time window to check
            cooldown_seconds: Minimum time between spike signals

        Returns:
            SpikeSignal if detected, None otherwise
        """
        velocity = self.get_velocity(window_seconds)

        if velocity is None:
            return None

        # Check cooldown from last spike
        if self._last_spike and self._prices:
            elapsed = (
                self._prices[-1].timestamp - self._last_spike.timestamp
            ).total_seconds()
            if elapsed < cooldown_seconds:
                return None

        # Check if move exceeds threshold
        if abs(velocity.change_pct) < threshold_pct:
            return None

        # Build spike signal
        current_time = self._prices[-1].timestamp
        cutoff = current_time - timedelta(seconds=window_seconds)
        window_prices = [p for p in self._prices if p.timestamp >= cutoff]

        if len(window_prices) < 2:
            return None

        spike = SpikeSignal(
            direction="up" if velocity.change_pct > 0 else "down",
            change_pct=velocity.change_pct,
            change_dollars=velocity.change_dollars,
            duration_seconds=(
                window_prices[-1].timestamp - window_prices[0].timestamp
            ).total_seconds(),
            start_price=window_prices[0].price,
            end_price=window_prices[-1].price,
            timestamp=current_time,
        )

        self._last_spike = spike
        logger.info(f"[{self.symbol}] Spike detected: {spike}")
        return spike

    def get_momentum_score(self) -> int:
        """Calculate momentum score from 0-100.

        Combines velocity readings across all windows into
        a single momentum score for quick assessment.

        Returns:
            Score 0-100 where:
            - 0-30: Bearish momentum
            - 30-70: Neutral/mixed
            - 70-100: Bullish momentum
        """
        velocities = self.get_all_velocities()
        valid_readings = [v for v in velocities.values() if v is not None]

        if not valid_readings:
            return 50  # Neutral if no data

        # Weight shorter windows more heavily (more recent = more relevant)
        weighted_sum = 0.0
        total_weight = 0.0

        for v in valid_readings:
            # Shorter windows get higher weight
            weight = 1.0 / (v.window_seconds ** 0.5)
            # Normalize change_pct to -1 to +1 range (assuming max 2% move)
            normalized = max(-1, min(1, v.change_pct / 2))
            weighted_sum += normalized * weight
            total_weight += weight

        if total_weight <= 0:
            return 50

        # Convert weighted average to 0-100 score
        avg = weighted_sum / total_weight  # -1 to +1
        score = int((avg + 1) * 50)  # 0 to 100
        return max(0, min(100, score))

    @property
    def current_price(self) -> float | None:
        """Get most recent price."""
        return self._prices[-1].price if self._prices else None

    @property
    def price_count(self) -> int:
        """Get number of prices in history."""
        return len(self._prices)

    @property
    def last_spike(self) -> SpikeSignal | None:
        """Get most recent spike signal."""
        return self._last_spike

    def clear(self) -> None:
        """Clear all price history and reset state."""
        self._prices.clear()
        self._last_spike = None
