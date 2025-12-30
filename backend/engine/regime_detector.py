"""Regime detection for the validated intraday strategy.

Uses WSB sentiment to identify bullish/bearish regimes with 7-day windows.
Thresholds calibrated from backtested TSLA data (2024-01 to 2025-01).

Performance:
- 71 trades over 13 months
- 43.7% win rate
- +17.4% avg return per trade
- +1238% total return (with 5% position sizing)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime classification based on WSB sentiment."""
    STRONG_BULLISH = "strong_bullish"
    MODERATE_BULLISH = "moderate_bullish"
    NEUTRAL = "neutral"
    MODERATE_BEARISH = "moderate_bearish"
    STRONG_BEARISH = "strong_bearish"

    @property
    def is_bullish(self) -> bool:
        return self in (RegimeType.STRONG_BULLISH, RegimeType.MODERATE_BULLISH)

    @property
    def is_bearish(self) -> bool:
        return self in (RegimeType.STRONG_BEARISH, RegimeType.MODERATE_BEARISH)

    @property
    def is_actionable(self) -> bool:
        """True if regime allows trading (not neutral)."""
        return self != RegimeType.NEUTRAL


@dataclass
class ActiveRegime:
    """An active market regime with window tracking.

    Attributes:
        regime_type: The classified regime
        triggered_date: When the regime signal fired
        trigger_sentiment: The WSB sentiment that triggered the regime
        window_expires: When the 7-day window expires
        days_remaining: Trading days left in window
    """
    regime_type: RegimeType
    triggered_date: datetime
    trigger_sentiment: float
    window_expires: datetime
    symbol: str

    @property
    def days_remaining(self) -> int:
        """Trading days remaining in the window."""
        now = datetime.now(timezone.utc)
        if now >= self.window_expires:
            return 0
        delta = (self.window_expires - now).days
        # Approximate trading days (5/7 of calendar days)
        return max(0, int(delta * 5 / 7))

    @property
    def is_active(self) -> bool:
        """True if regime window is still active."""
        return datetime.now(timezone.utc) < self.window_expires

    def to_dict(self) -> dict:
        return {
            "regime_type": self.regime_type.value,
            "triggered_date": self.triggered_date.isoformat(),
            "trigger_sentiment": round(self.trigger_sentiment, 4),
            "window_expires": self.window_expires.isoformat(),
            "days_remaining": self.days_remaining,
            "is_active": self.is_active,
            "symbol": self.symbol,
        }


@dataclass
class RegimeConfig:
    """Configuration for regime detection.

    Thresholds calibrated from TSLA sentiment distribution:
    - 10th percentile bearish: -0.103
    - 90th percentile bullish: +0.071
    - Strong signals at top 5% (roughly +/-0.12 to 0.15)

    Attributes:
        strong_bullish_threshold: WSB sentiment > this = strong bullish
        moderate_bullish_threshold: WSB sentiment > this = moderate bullish
        moderate_bearish_threshold: WSB sentiment < this = moderate bearish
        strong_bearish_threshold: WSB sentiment < this = strong bearish
        regime_window_days: How many trading days regime stays active
    """
    strong_bullish_threshold: float = 0.12
    moderate_bullish_threshold: float = 0.07
    moderate_bearish_threshold: float = -0.08
    strong_bearish_threshold: float = -0.15
    regime_window_days: int = 7  # Trading days


class RegimeDetector:
    """Detects and tracks market regimes based on WSB sentiment.

    Uses the validated thresholds from backtesting:
    - Strong Bullish: sentiment > +0.12
    - Moderate Bullish: sentiment > +0.07
    - Moderate Bearish: sentiment < -0.08
    - Strong Bearish: sentiment < -0.15

    When a regime triggers, it stays active for 7 trading days,
    allowing entry on pullbacks/bounces during that window.

    Usage:
        detector = RegimeDetector()

        # Check for new regime based on WSB sentiment
        regime = detector.update_regime("TSLA", wsb_sentiment=0.15)

        if regime and regime.is_active:
            # Look for pullback/bounce entry
            ...
    """

    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()
        self._active_regimes: dict[str, ActiveRegime] = {}

    def classify_sentiment(self, sentiment: float) -> RegimeType:
        """Classify WSB sentiment into a regime type.

        Args:
            sentiment: WSB sentiment score (typically -0.5 to +0.5)

        Returns:
            RegimeType classification
        """
        if sentiment > self.config.strong_bullish_threshold:
            return RegimeType.STRONG_BULLISH
        elif sentiment > self.config.moderate_bullish_threshold:
            return RegimeType.MODERATE_BULLISH
        elif sentiment < self.config.strong_bearish_threshold:
            return RegimeType.STRONG_BEARISH
        elif sentiment < self.config.moderate_bearish_threshold:
            return RegimeType.MODERATE_BEARISH
        else:
            return RegimeType.NEUTRAL

    def update_regime(
        self,
        symbol: str,
        wsb_sentiment: float,
    ) -> Optional[ActiveRegime]:
        """Update regime status based on new WSB sentiment.

        If sentiment triggers a new regime (or refreshes an existing one),
        a 7-day window is opened/extended.

        Args:
            symbol: Stock symbol (e.g., "TSLA")
            wsb_sentiment: Current WSB sentiment score

        Returns:
            ActiveRegime if a regime is active, None if neutral
        """
        regime_type = self.classify_sentiment(wsb_sentiment)

        now = datetime.now(timezone.utc)

        # If neutral, no regime is active
        if regime_type == RegimeType.NEUTRAL:
            # Check if we still have an active regime from before
            existing = self._active_regimes.get(symbol)
            if existing and existing.is_active:
                return existing
            # No active regime
            if symbol in self._active_regimes:
                del self._active_regimes[symbol]
            return None

        # Calculate window expiry (7 trading days = ~10 calendar days)
        calendar_days = int(self.config.regime_window_days * 7 / 5) + 1
        window_expires = now + timedelta(days=calendar_days)

        # Check if this is a new regime or refreshing existing
        existing = self._active_regimes.get(symbol)
        is_new = existing is None or not existing.is_active

        # Create/update regime
        regime = ActiveRegime(
            regime_type=regime_type,
            triggered_date=now,
            trigger_sentiment=wsb_sentiment,
            window_expires=window_expires,
            symbol=symbol,
        )
        self._active_regimes[symbol] = regime

        # Log regime change
        if is_new:
            logger.info(
                f"[REGIME] {now.strftime('%Y-%m-%d')} {symbol}: "
                f"{regime_type.value} triggered (sentiment: {wsb_sentiment:+.3f}), "
                f"window expires {window_expires.strftime('%Y-%m-%d')}"
            )
        else:
            logger.debug(
                f"[REGIME] {symbol}: {regime_type.value} refreshed "
                f"(sentiment: {wsb_sentiment:+.3f})"
            )

        return regime

    def get_active_regime(self, symbol: str) -> Optional[ActiveRegime]:
        """Get the currently active regime for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            ActiveRegime if one is active, None otherwise
        """
        regime = self._active_regimes.get(symbol)
        if regime and regime.is_active:
            return regime

        # Clean up expired regimes
        if symbol in self._active_regimes:
            del self._active_regimes[symbol]

        return None

    def get_all_active_regimes(self) -> dict[str, ActiveRegime]:
        """Get all currently active regimes.

        Returns:
            Dict mapping symbol to ActiveRegime
        """
        # Clean up expired and return active only
        active = {}
        expired = []

        for symbol, regime in self._active_regimes.items():
            if regime.is_active:
                active[symbol] = regime
            else:
                expired.append(symbol)

        for symbol in expired:
            del self._active_regimes[symbol]

        return active

    def clear_regime(self, symbol: str) -> None:
        """Manually clear a regime for a symbol."""
        if symbol in self._active_regimes:
            del self._active_regimes[symbol]
            logger.info(f"[REGIME] {symbol}: Manually cleared")

    def is_bullish(self, symbol: str) -> bool:
        """Check if symbol is in a bullish regime."""
        regime = self.get_active_regime(symbol)
        return regime is not None and regime.regime_type.is_bullish

    def is_bearish(self, symbol: str) -> bool:
        """Check if symbol is in a bearish regime."""
        regime = self.get_active_regime(symbol)
        return regime is not None and regime.regime_type.is_bearish

    def get_status_summary(self) -> dict:
        """Get a summary of all regime statuses for logging/display."""
        active = self.get_all_active_regimes()

        return {
            "active_count": len(active),
            "regimes": {
                symbol: regime.to_dict()
                for symbol, regime in active.items()
            },
        }
