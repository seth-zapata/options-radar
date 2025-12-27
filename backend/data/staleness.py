"""Staleness detection for market data.

Implements the data freshness requirements from spec section 3.1.

Thresholds:
- Quote data: 5 seconds
- Greeks data: 90 seconds
- Underlying price: 2 seconds
- IV rank: 15 minutes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from backend.data.aggregator import AggregatedOptionData
from backend.models.market_data import UnderlyingData

logger = logging.getLogger(__name__)


class FreshnessLevel(str, Enum):
    """Data freshness classification."""

    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"


@dataclass(frozen=True)
class StalenessThresholds:
    """Configurable staleness thresholds in seconds."""

    quote: float = 5.0
    greeks: float = 90.0
    underlying: float = 2.0
    iv_rank: float = 900.0  # 15 minutes


@dataclass(frozen=True)
class FreshnessReport:
    """Report on data freshness for an option.

    Used by the gating engine to determine if recommendations
    should be made or suppressed.
    """

    quote_age: float | None
    quote_status: FreshnessLevel
    greeks_age: float | None
    greeks_status: FreshnessLevel

    @property
    def all_fresh(self) -> bool:
        """True if all required data is fresh."""
        return (
            self.quote_status == FreshnessLevel.FRESH
            and self.greeks_status == FreshnessLevel.FRESH
        )

    @property
    def any_missing(self) -> bool:
        """True if any required data is missing."""
        return (
            self.quote_status == FreshnessLevel.MISSING
            or self.greeks_status == FreshnessLevel.MISSING
        )

    @property
    def any_stale(self) -> bool:
        """True if any data is stale (but not missing)."""
        return (
            self.quote_status == FreshnessLevel.STALE
            or self.greeks_status == FreshnessLevel.STALE
        )


@dataclass(frozen=True)
class UnderlyingFreshnessReport:
    """Report on underlying data freshness."""

    price_age: float | None
    price_status: FreshnessLevel
    iv_rank_age: float | None
    iv_rank_status: FreshnessLevel

    @property
    def price_fresh(self) -> bool:
        """True if price data is fresh."""
        return self.price_status == FreshnessLevel.FRESH


class StalenessChecker:
    """Checks data freshness against configured thresholds.

    Usage:
        checker = StalenessChecker()
        report = checker.check_option(option_data)

        if not report.all_fresh:
            # Suppress recommendation
            pass
    """

    def __init__(self, thresholds: StalenessThresholds | None = None):
        """Initialize with optional custom thresholds.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or StalenessThresholds()

    def check_option(
        self,
        option: AggregatedOptionData,
        now: datetime | None = None,
    ) -> FreshnessReport:
        """Check freshness of option data.

        Args:
            option: Aggregated option data to check
            now: Current time (defaults to UTC now)

        Returns:
            FreshnessReport with status of each data type
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Check quote freshness
        quote_age = option.quote_age_seconds(now)
        if quote_age is None:
            quote_status = FreshnessLevel.MISSING
        elif quote_age <= self.thresholds.quote:
            quote_status = FreshnessLevel.FRESH
        else:
            quote_status = FreshnessLevel.STALE

        # Check Greeks freshness
        greeks_age = option.greeks_age_seconds(now)
        if greeks_age is None:
            greeks_status = FreshnessLevel.MISSING
        elif greeks_age <= self.thresholds.greeks:
            greeks_status = FreshnessLevel.FRESH
        else:
            greeks_status = FreshnessLevel.STALE

        return FreshnessReport(
            quote_age=quote_age,
            quote_status=quote_status,
            greeks_age=greeks_age,
            greeks_status=greeks_status,
        )

    def check_underlying(
        self,
        underlying: UnderlyingData,
        now: datetime | None = None,
    ) -> UnderlyingFreshnessReport:
        """Check freshness of underlying data.

        Args:
            underlying: Underlying data to check
            now: Current time (defaults to UTC now)

        Returns:
            UnderlyingFreshnessReport with status
        """
        if now is None:
            now = datetime.now(timezone.utc)

        price_age = underlying.age_seconds(now)

        # Price staleness (critical - 2 second threshold)
        if price_age is None:
            price_status = FreshnessLevel.MISSING
        elif price_age <= self.thresholds.underlying:
            price_status = FreshnessLevel.FRESH
        else:
            price_status = FreshnessLevel.STALE

        # IV rank is less critical - 15 minute threshold
        # We use the same timestamp for now, but in practice
        # IV rank might have a different update time
        if price_age is None:
            iv_rank_status = FreshnessLevel.MISSING
        elif price_age <= self.thresholds.iv_rank:
            iv_rank_status = FreshnessLevel.FRESH
        else:
            iv_rank_status = FreshnessLevel.STALE

        return UnderlyingFreshnessReport(
            price_age=price_age,
            price_status=price_status,
            iv_rank_age=price_age,  # Same timestamp for now
            iv_rank_status=iv_rank_status,
        )

    def is_quote_fresh(
        self,
        option: AggregatedOptionData,
        now: datetime | None = None,
    ) -> bool:
        """Quick check if quote data is fresh.

        Args:
            option: Option to check
            now: Current time

        Returns:
            True if quote is fresh
        """
        age = option.quote_age_seconds(now)
        return age is not None and age <= self.thresholds.quote

    def is_greeks_fresh(
        self,
        option: AggregatedOptionData,
        now: datetime | None = None,
    ) -> bool:
        """Quick check if Greeks data is fresh.

        Args:
            option: Option to check
            now: Current time

        Returns:
            True if Greeks are fresh
        """
        age = option.greeks_age_seconds(now)
        return age is not None and age <= self.thresholds.greeks
