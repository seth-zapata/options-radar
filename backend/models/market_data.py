"""Market data models for quotes, Greeks, and underlying prices.

See spec section 3.3 for data contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

from backend.models.canonical import CanonicalOptionId


class DataSource(str, Enum):
    """Data source identifiers."""

    ALPACA = "alpaca"
    TRADIER = "tradier"
    POLYGON = "polygon"
    ORATS = "orats"
    CALCULATED = "calculated"


@dataclass(slots=True)
class QuoteData:
    """Real-time quote data for an option contract.

    Attributes:
        canonical_id: Internal option identifier
        bid: Best bid price
        ask: Best ask price
        bid_size: Size at bid
        ask_size: Size at ask
        last: Last trade price (None if no recent trade)
        timestamp: Source timestamp (ISO 8601 UTC)
        receive_timestamp: Local receive time (ISO 8601 UTC)
        source: Data source identifier
    """

    canonical_id: CanonicalOptionId
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float | None
    timestamp: str
    receive_timestamp: str
    source: Literal["alpaca", "tradier", "polygon"]

    @property
    def mid(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_percent(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = self.mid
        if mid <= 0:
            return float("inf")
        return (self.spread / mid) * 100

    def age_seconds(self, now: datetime | None = None) -> float:
        """Calculate age of quote in seconds.

        Args:
            now: Current time (defaults to datetime.utcnow)

        Returns:
            Age in seconds since timestamp
        """
        if now is None:
            now = datetime.utcnow()
        ts = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        # Make now timezone-aware if ts is
        if ts.tzinfo is not None and now.tzinfo is None:
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds()


@dataclass(slots=True)
class GreeksData:
    """Greeks and implied volatility data.

    Attributes:
        canonical_id: Internal option identifier
        delta: Delta (rate of change vs underlying)
        gamma: Gamma (rate of change of delta)
        theta: Theta (time decay per day)
        vega: Vega (sensitivity to IV)
        rho: Rho (sensitivity to interest rate)
        iv: Implied volatility (decimal, e.g., 0.35 for 35%)
        theoretical_value: Model-computed fair value
        timestamp: Source timestamp (ISO 8601 UTC)
        source: Data source identifier
    """

    canonical_id: CanonicalOptionId
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float
    theoretical_value: float
    timestamp: str
    source: Literal["orats", "calculated"]

    def age_seconds(self, now: datetime | None = None) -> float:
        """Calculate age of Greeks in seconds.

        Args:
            now: Current time (defaults to datetime.utcnow)

        Returns:
            Age in seconds since timestamp
        """
        if now is None:
            now = datetime.utcnow()
        ts = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        if ts.tzinfo is not None and now.tzinfo is None:
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds()


@dataclass(slots=True)
class UnderlyingData:
    """Underlying stock price and IV metrics.

    Attributes:
        symbol: Ticker symbol
        price: Current price
        iv_rank: IV rank (0-100)
        iv_percentile: IV percentile (0-100)
        timestamp: Source timestamp (ISO 8601 UTC)
    """

    symbol: str
    price: float
    iv_rank: float
    iv_percentile: float
    timestamp: str

    def age_seconds(self, now: datetime | None = None) -> float:
        """Calculate age in seconds.

        Args:
            now: Current time (defaults to datetime.utcnow)

        Returns:
            Age in seconds since timestamp
        """
        if now is None:
            now = datetime.utcnow()
        ts = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        if ts.tzinfo is not None and now.tzinfo is None:
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds()


@dataclass(slots=True)
class TradeData:
    """Trade data for an option contract.

    Used for volume tracking and last price updates.
    """

    canonical_id: CanonicalOptionId
    price: float
    size: int
    timestamp: str
    exchange: str
    conditions: list[str] = field(default_factory=list)
