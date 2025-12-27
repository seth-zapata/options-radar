"""Data aggregator for merging quotes and Greeks from multiple sources.

Handles:
- Merging real-time quotes (Alpaca) with Greeks (ORATS)
- Timestamp alignment and staleness tracking
- Unified option data view for the gating engine

See spec section 3 for data freshness requirements.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from backend.config import AppConfig
from backend.models.canonical import CanonicalOptionId
from backend.models.market_data import GreeksData, QuoteData, UnderlyingData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregatedOptionData:
    """Combined quote and Greeks data for a single option.

    Provides a unified view of all data needed for gating decisions.
    """

    canonical_id: CanonicalOptionId

    # Quote data (from Alpaca)
    bid: float | None = None
    ask: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    last: float | None = None
    quote_timestamp: str | None = None

    # Volume/OI data
    open_interest: int = 0
    volume: int = 0

    # Greeks data (from ORATS)
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    iv: float | None = None
    theoretical_value: float | None = None
    greeks_timestamp: str | None = None

    @property
    def mid(self) -> float | None:
        """Calculate mid price if bid/ask available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        """Calculate spread if bid/ask available."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_percent(self) -> float | None:
        """Calculate spread as percentage of mid."""
        mid = self.mid
        spread = self.spread
        if mid and mid > 0 and spread is not None:
            return (spread / mid) * 100
        return None

    @property
    def has_quote(self) -> bool:
        """True if quote data is present."""
        return self.bid is not None and self.ask is not None

    @property
    def has_greeks(self) -> bool:
        """True if Greeks data is present."""
        return self.delta is not None

    def quote_age_seconds(self, now: datetime | None = None) -> float | None:
        """Calculate age of quote data in seconds."""
        if not self.quote_timestamp:
            return None
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            ts = datetime.fromisoformat(self.quote_timestamp.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            return (now - ts).total_seconds()
        except (ValueError, TypeError):
            return None

    def greeks_age_seconds(self, now: datetime | None = None) -> float | None:
        """Calculate age of Greeks data in seconds."""
        if not self.greeks_timestamp:
            return None
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            ts = datetime.fromisoformat(self.greeks_timestamp.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            return (now - ts).total_seconds()
        except (ValueError, TypeError):
            return None


@dataclass
class DataAggregator:
    """Aggregates data from multiple sources into unified option views.

    Maintains in-memory state of latest quotes and Greeks, providing
    a consistent view for the gating engine.

    Usage:
        aggregator = DataAggregator(config)

        # Update from Alpaca quote stream
        aggregator.update_quote(quote)

        # Update from ORATS polling
        aggregator.update_greeks(greeks)

        # Get aggregated data for gating
        option_data = aggregator.get_option("NVDA", "2025-01-17", "C", 150.0)
    """

    config: AppConfig

    # Callbacks for data changes
    on_option_update: Callable[[AggregatedOptionData], None] | None = None
    on_underlying_update: Callable[[UnderlyingData], None] | None = None

    # Internal state
    _quotes: dict[CanonicalOptionId, QuoteData] = field(default_factory=dict, init=False)
    _greeks: dict[CanonicalOptionId, GreeksData] = field(default_factory=dict, init=False)
    _underlyings: dict[str, UnderlyingData] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def update_quote(self, quote: QuoteData) -> None:
        """Update quote data for an option.

        Thread-safe update that merges new quote data.

        Args:
            quote: New quote data from Alpaca
        """
        self._quotes[quote.canonical_id] = quote

        # Notify listener
        if self.on_option_update:
            aggregated = self.get_option_by_id(quote.canonical_id)
            if aggregated:
                try:
                    self.on_option_update(aggregated)
                except Exception as e:
                    logger.exception(f"Error in option update callback: {e}")

    def update_greeks(self, greeks: GreeksData) -> None:
        """Update Greeks data for an option.

        Args:
            greeks: New Greeks data from ORATS
        """
        self._greeks[greeks.canonical_id] = greeks

        # Notify listener
        if self.on_option_update:
            aggregated = self.get_option_by_id(greeks.canonical_id)
            if aggregated:
                try:
                    self.on_option_update(aggregated)
                except Exception as e:
                    logger.exception(f"Error in option update callback: {e}")

    def update_greeks_batch(self, greeks_list: list[GreeksData]) -> None:
        """Update Greeks for multiple options.

        Args:
            greeks_list: List of Greeks data from ORATS
        """
        for greeks in greeks_list:
            self._greeks[greeks.canonical_id] = greeks

        # Batch notify
        if self.on_option_update:
            for greeks in greeks_list:
                aggregated = self.get_option_by_id(greeks.canonical_id)
                if aggregated:
                    try:
                        self.on_option_update(aggregated)
                    except Exception as e:
                        logger.exception(f"Error in option update callback: {e}")

    def update_underlying(self, underlying: UnderlyingData) -> None:
        """Update underlying data (price, IV rank).

        Args:
            underlying: New underlying data
        """
        self._underlyings[underlying.symbol] = underlying

        if self.on_underlying_update:
            try:
                self.on_underlying_update(underlying)
            except Exception as e:
                logger.exception(f"Error in underlying update callback: {e}")

    def get_option_by_id(self, canonical_id: CanonicalOptionId) -> AggregatedOptionData | None:
        """Get aggregated data for an option by ID.

        Args:
            canonical_id: Option identifier

        Returns:
            Aggregated option data or None if not found
        """
        quote = self._quotes.get(canonical_id)
        greeks = self._greeks.get(canonical_id)

        if not quote and not greeks:
            return None

        return AggregatedOptionData(
            canonical_id=canonical_id,
            # Quote fields
            bid=quote.bid if quote else None,
            ask=quote.ask if quote else None,
            bid_size=quote.bid_size if quote else None,
            ask_size=quote.ask_size if quote else None,
            last=quote.last if quote else None,
            quote_timestamp=quote.timestamp if quote else None,
            # Greeks fields
            delta=greeks.delta if greeks else None,
            gamma=greeks.gamma if greeks else None,
            theta=greeks.theta if greeks else None,
            vega=greeks.vega if greeks else None,
            rho=greeks.rho if greeks else None,
            iv=greeks.iv if greeks else None,
            theoretical_value=greeks.theoretical_value if greeks else None,
            greeks_timestamp=greeks.timestamp if greeks else None,
        )

    def get_option(
        self,
        underlying: str,
        expiry: str,
        right: str,
        strike: float,
    ) -> AggregatedOptionData | None:
        """Get aggregated data for an option by components.

        Args:
            underlying: Ticker symbol
            expiry: Expiration date (ISO format)
            right: "C" or "P"
            strike: Strike price

        Returns:
            Aggregated option data or None if not found
        """
        canonical_id = CanonicalOptionId(
            underlying=underlying,
            expiry=expiry,
            right=right,  # type: ignore
            strike=strike,
        )
        return self.get_option_by_id(canonical_id)

    def get_underlying(self, symbol: str) -> UnderlyingData | None:
        """Get underlying data by symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Underlying data or None if not found
        """
        return self._underlyings.get(symbol)

    def get_all_options(self) -> list[AggregatedOptionData]:
        """Get all aggregated option data.

        Returns:
            List of all options with at least quote or Greeks data
        """
        all_ids = set(self._quotes.keys()) | set(self._greeks.keys())
        results = []
        for canonical_id in all_ids:
            data = self.get_option_by_id(canonical_id)
            if data:
                results.append(data)
        return results

    def get_options_for_underlying(self, symbol: str) -> list[AggregatedOptionData]:
        """Get all options for a specific underlying.

        Args:
            symbol: Ticker symbol

        Returns:
            List of aggregated option data for that underlying
        """
        all_ids = set(self._quotes.keys()) | set(self._greeks.keys())
        results = []
        for canonical_id in all_ids:
            if canonical_id.underlying == symbol:
                data = self.get_option_by_id(canonical_id)
                if data:
                    results.append(data)
        return results

    def clear(self) -> None:
        """Clear all cached data."""
        self._quotes.clear()
        self._greeks.clear()
        self._underlyings.clear()

    @property
    def quote_count(self) -> int:
        """Number of options with quote data."""
        return len(self._quotes)

    @property
    def greeks_count(self) -> int:
        """Number of options with Greeks data."""
        return len(self._greeks)
