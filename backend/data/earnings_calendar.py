"""Earnings calendar integration for blocking trades near earnings events.

Uses Finnhub API (already integrated) to fetch earnings dates.
Provides caching and historical data for backtesting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Historical earnings dates for backtesting (TSLA 2024)
# Source: Finnhub historical data
HISTORICAL_EARNINGS: dict[str, list[str]] = {
    "TSLA": [
        "2024-01-24",  # Q4 2023
        "2024-04-23",  # Q1 2024
        "2024-07-23",  # Q2 2024
        "2024-10-23",  # Q3 2024
        "2025-01-29",  # Q4 2024
    ],
    "NVDA": [
        "2024-02-21",  # Q4 FY2024
        "2024-05-22",  # Q1 FY2025
        "2024-08-28",  # Q2 FY2025
        "2024-11-20",  # Q3 FY2025
        "2025-02-26",  # Q4 FY2025 (estimated)
    ],
    "AAPL": [
        "2024-02-01",  # Q1 FY2024
        "2024-05-02",  # Q2 FY2024
        "2024-08-01",  # Q3 FY2024
        "2024-10-31",  # Q4 FY2024
        "2025-01-30",  # Q1 FY2025
    ],
    "PLTR": [
        "2024-02-05",
        "2024-05-06",
        "2024-08-05",
        "2024-11-04",
        "2025-02-03",  # Estimated
    ],
}


@dataclass
class EarningsInfo:
    """Information about upcoming/recent earnings."""

    symbol: str
    next_earnings_date: Optional[date]
    days_until_earnings: Optional[int]
    days_since_last_earnings: Optional[int]
    last_earnings_date: Optional[date]


class EarningsCalendar:
    """Earnings calendar client with caching.

    Fetches earnings dates from Finnhub and caches them.
    Also provides historical data for backtesting.
    """

    def __init__(
        self,
        finnhub_api_key: str,
        cache_ttl_hours: int = 24,
    ):
        """Initialize earnings calendar.

        Args:
            finnhub_api_key: Finnhub API key
            cache_ttl_hours: How long to cache earnings dates (default 24h)
        """
        self.api_key = finnhub_api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        # Cache: symbol -> (earnings_dates, cache_time)
        self._cache: dict[str, tuple[list[date], datetime]] = {}

    async def fetch_earnings_dates(self, symbol: str) -> list[date]:
        """Fetch earnings dates from Finnhub API.

        Args:
            symbol: Stock symbol (e.g., 'TSLA')

        Returns:
            List of earnings dates (past and future)
        """
        # Check cache first
        if symbol in self._cache:
            dates, cache_time = self._cache[symbol]
            if datetime.now() - cache_time < self.cache_ttl:
                return dates

        # Fetch from API
        try:
            url = f"{self.base_url}/calendar/earnings"
            params = {
                "symbol": symbol,
                "token": self.api_key,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Finnhub earnings API error: {response.status}")
                        return self._get_historical_fallback(symbol)

                    data = await response.json()

            # Parse earnings dates
            earnings_dates = []
            for event in data.get("earningsCalendar", []):
                date_str = event.get("date")
                if date_str:
                    try:
                        earnings_dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())
                    except ValueError:
                        continue

            # Sort and cache
            earnings_dates.sort()
            self._cache[symbol] = (earnings_dates, datetime.now())

            logger.info(f"Fetched {len(earnings_dates)} earnings dates for {symbol}")
            return earnings_dates

        except Exception as e:
            logger.warning(f"Failed to fetch earnings for {symbol}: {e}")
            return self._get_historical_fallback(symbol)

    def _get_historical_fallback(self, symbol: str) -> list[date]:
        """Get historical earnings dates as fallback.

        Args:
            symbol: Stock symbol

        Returns:
            List of known historical earnings dates
        """
        if symbol not in HISTORICAL_EARNINGS:
            return []

        return [
            datetime.strptime(d, "%Y-%m-%d").date()
            for d in HISTORICAL_EARNINGS[symbol]
        ]

    def get_earnings_for_date(
        self,
        symbol: str,
        check_date: date,
    ) -> EarningsInfo:
        """Get earnings info for a specific date (for backtesting).

        Uses historical data only - doesn't make API calls.

        Args:
            symbol: Stock symbol
            check_date: The date to check

        Returns:
            EarningsInfo with days until/since earnings
        """
        historical = self._get_historical_fallback(symbol)

        if not historical:
            return EarningsInfo(
                symbol=symbol,
                next_earnings_date=None,
                days_until_earnings=None,
                days_since_last_earnings=None,
                last_earnings_date=None,
            )

        # Find next and previous earnings relative to check_date
        next_earnings = None
        last_earnings = None

        for earnings_date in historical:
            if earnings_date > check_date:
                if next_earnings is None or earnings_date < next_earnings:
                    next_earnings = earnings_date
            elif earnings_date <= check_date:
                if last_earnings is None or earnings_date > last_earnings:
                    last_earnings = earnings_date

        # Calculate days
        days_until = (next_earnings - check_date).days if next_earnings else None
        days_since = (check_date - last_earnings).days if last_earnings else None

        return EarningsInfo(
            symbol=symbol,
            next_earnings_date=next_earnings,
            days_until_earnings=days_until,
            days_since_last_earnings=days_since,
            last_earnings_date=last_earnings,
        )

    async def get_earnings_info(self, symbol: str) -> EarningsInfo:
        """Get current earnings info for a symbol.

        Fetches from API and calculates days until/since earnings.

        Args:
            symbol: Stock symbol

        Returns:
            EarningsInfo with current earnings status
        """
        earnings_dates = await self.fetch_earnings_dates(symbol)
        today = date.today()

        if not earnings_dates:
            return EarningsInfo(
                symbol=symbol,
                next_earnings_date=None,
                days_until_earnings=None,
                days_since_last_earnings=None,
                last_earnings_date=None,
            )

        # Find next and previous earnings
        next_earnings = None
        last_earnings = None

        for earnings_date in earnings_dates:
            if earnings_date > today:
                if next_earnings is None:
                    next_earnings = earnings_date
                    break
            else:
                last_earnings = earnings_date

        # Calculate days
        days_until = (next_earnings - today).days if next_earnings else None
        days_since = (today - last_earnings).days if last_earnings else None

        return EarningsInfo(
            symbol=symbol,
            next_earnings_date=next_earnings,
            days_until_earnings=days_until,
            days_since_last_earnings=days_since,
            last_earnings_date=last_earnings,
        )

    def add_historical_earnings(self, symbol: str, dates: list[str]) -> None:
        """Add historical earnings dates for a symbol.

        Useful for extending backtest coverage.

        Args:
            symbol: Stock symbol
            dates: List of date strings in YYYY-MM-DD format
        """
        HISTORICAL_EARNINGS[symbol] = dates
        logger.info(f"Added {len(dates)} historical earnings dates for {symbol}")
