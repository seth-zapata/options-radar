"""Dynamic subscription manager for ATM-based option subscriptions.

Handles:
- Fetching underlying price to determine ATM strike
- Generating option symbols for ±10 strikes around ATM
- Managing 2 expiration buckets (nearest weekly + ~45 DTE)
- Re-subscribing when underlying crosses strike boundaries

See spec section 3.2 and 8.1 for rate limit considerations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import httpx

from backend.config import AlpacaConfig
from backend.data.alpaca_client import AlpacaOptionsClient
from backend.models.canonical import CanonicalOptionId, to_alpaca

logger = logging.getLogger(__name__)

# Standard strike intervals by price range
STRIKE_INTERVALS = [
    (0, 50, 1),        # $0-50: $1 strikes
    (50, 100, 2.5),    # $50-100: $2.50 strikes
    (100, 200, 5),     # $100-200: $5 strikes
    (200, 500, 10),    # $200-500: $10 strikes
    (500, 1000, 25),   # $500-1000: $25 strikes
    (1000, float('inf'), 50),  # $1000+: $50 strikes
]


def get_strike_interval(price: float) -> float:
    """Get the standard strike interval for a given price."""
    for low, high, interval in STRIKE_INTERVALS:
        if low <= price < high:
            return interval
    return 50.0


def round_to_strike(price: float, interval: float) -> float:
    """Round price to nearest strike."""
    return round(price / interval) * interval


def get_expiration_buckets(from_date: date | None = None) -> list[date]:
    """Get expiration dates for the two buckets: nearest weekly + ~45 DTE.

    Args:
        from_date: Reference date (defaults to today)

    Returns:
        List of two expiration dates [nearest_weekly, ~45_dte]
    """
    if from_date is None:
        from_date = date.today()

    expirations = []

    # Find nearest Friday (weekly expiration)
    days_until_friday = (4 - from_date.weekday()) % 7
    if days_until_friday == 0:
        # If today is Friday and market is open, use today; otherwise next Friday
        days_until_friday = 7
    nearest_friday = from_date + timedelta(days=days_until_friday)
    expirations.append(nearest_friday)

    # Find ~45 DTE expiration (monthly, typically 3rd Friday)
    target_45dte = from_date + timedelta(days=45)

    # Find the 3rd Friday of the target month
    first_of_month = target_45dte.replace(day=1)
    first_friday = first_of_month + timedelta(days=(4 - first_of_month.weekday()) % 7)
    third_friday = first_friday + timedelta(days=14)

    # If 3rd Friday is before target, move to next month
    if third_friday < target_45dte:
        next_month = (target_45dte.month % 12) + 1
        year = target_45dte.year + (1 if next_month == 1 else 0)
        first_of_next = date(year, next_month, 1)
        first_friday = first_of_next + timedelta(days=(4 - first_of_next.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)

    expirations.append(third_friday)

    return expirations


@dataclass
class SubscriptionManager:
    """Manages dynamic option subscriptions based on ATM strike.

    Subscribes to ±10 strikes around ATM for 2 expiration buckets,
    and re-subscribes when the underlying crosses a strike boundary.

    Usage:
        manager = SubscriptionManager(
            config=alpaca_config,
            client=alpaca_ws_client,
            symbol="NVDA",
        )
        await manager.start()

        # Update underlying price (typically from stock quote stream)
        await manager.update_underlying_price(142.50)

        await manager.stop()
    """

    config: AlpacaConfig
    client: AlpacaOptionsClient
    symbol: str
    strikes_around_atm: int = 10
    on_subscriptions_changed: Callable[[list[str]], None] | None = None

    # Internal state
    _current_atm: float | None = field(default=None, init=False)
    _current_symbols: set[str] = field(default_factory=set, init=False)
    _http_client: httpx.AsyncClient | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)

    async def start(self) -> None:
        """Start the subscription manager.

        Fetches current underlying price and subscribes to initial symbols.
        """
        self._running = True
        self._http_client = httpx.AsyncClient(
            base_url=self.config.data_url,
            headers={
                "APCA-API-KEY-ID": self.config.api_key,
                "APCA-API-SECRET-KEY": self.config.secret_key,
            },
            timeout=10.0,
        )

        # Fetch initial underlying price
        price = await self._fetch_underlying_price()
        if price is not None:
            await self.update_underlying_price(price)
        else:
            logger.warning(f"Could not fetch initial price for {self.symbol}")

    async def stop(self) -> None:
        """Stop the subscription manager and clean up."""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def switch_symbol(self, new_symbol: str) -> None:
        """Switch to a new underlying symbol.

        Unsubscribes from current options and subscribes to new symbol's options.

        Args:
            new_symbol: The new underlying symbol to track
        """
        if new_symbol == self.symbol:
            return

        logger.info(f"Switching from {self.symbol} to {new_symbol}")

        # Unsubscribe from all current symbols
        if self._current_symbols and self.client.is_connected:
            try:
                await self.client.unsubscribe(list(self._current_symbols))
            except Exception as e:
                logger.error(f"Failed to unsubscribe during symbol switch: {e}")

        # Reset state
        self._current_symbols = set()
        self._current_atm = None
        self.symbol = new_symbol

        # Fetch new symbol price and subscribe
        price = await self._fetch_underlying_price()
        if price is not None:
            await self.update_underlying_price(price)
            logger.info(f"Switched to {new_symbol}, subscribed to {len(self._current_symbols)} contracts")
        else:
            logger.warning(f"Could not fetch initial price for {new_symbol}")

    async def _fetch_underlying_price(self) -> float | None:
        """Fetch current underlying price from Alpaca REST API."""
        if not self._http_client:
            return None

        try:
            response = await self._http_client.get(
                f"/v2/stocks/{self.symbol}/quotes/latest"
            )
            response.raise_for_status()
            data = response.json()

            # Use midpoint of bid/ask
            quote = data.get("quote", {})
            bid = quote.get("bp", 0)
            ask = quote.get("ap", 0)

            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return None

        except Exception as e:
            logger.error(f"Failed to fetch underlying price: {e}")
            return None

    async def update_underlying_price(self, price: float) -> None:
        """Update the underlying price and re-subscribe if ATM changed.

        Args:
            price: Current underlying price
        """
        interval = get_strike_interval(price)
        new_atm = round_to_strike(price, interval)

        if new_atm == self._current_atm:
            return  # No change in ATM strike

        logger.info(
            f"{self.symbol} ATM changed: {self._current_atm} -> {new_atm} "
            f"(underlying: ${price:.2f})"
        )
        self._current_atm = new_atm

        # Generate new symbol set
        new_symbols = self._generate_symbols(new_atm, interval)

        # Calculate diff
        to_subscribe = new_symbols - self._current_symbols
        to_unsubscribe = self._current_symbols - new_symbols

        # Apply changes
        if to_unsubscribe and self.client.is_connected:
            try:
                await self.client.unsubscribe(list(to_unsubscribe))
            except Exception as e:
                logger.error(f"Failed to unsubscribe: {e}")

        if to_subscribe and self.client.is_connected:
            try:
                await self.client.subscribe(list(to_subscribe))
            except Exception as e:
                logger.error(f"Failed to subscribe: {e}")

        self._current_symbols = new_symbols

        if self.on_subscriptions_changed:
            try:
                self.on_subscriptions_changed(list(new_symbols))
            except Exception as e:
                logger.exception(f"Error in subscriptions changed callback: {e}")

        logger.info(
            f"Subscriptions updated: +{len(to_subscribe)} -{len(to_unsubscribe)} "
            f"= {len(new_symbols)} total"
        )

    def _generate_symbols(self, atm_strike: float, interval: float) -> set[str]:
        """Generate option symbols for ±N strikes around ATM.

        Args:
            atm_strike: ATM strike price
            interval: Strike interval

        Returns:
            Set of Alpaca-format option symbols
        """
        symbols = set()
        expirations = get_expiration_buckets()

        for expiry in expirations:
            expiry_str = expiry.isoformat()

            for i in range(-self.strikes_around_atm, self.strikes_around_atm + 1):
                strike = atm_strike + (i * interval)
                if strike <= 0:
                    continue

                for right in ["C", "P"]:
                    opt = CanonicalOptionId(
                        underlying=self.symbol,
                        expiry=expiry_str,
                        right=right,  # type: ignore[arg-type]
                        strike=strike,
                    )
                    symbols.add(to_alpaca(opt))

        return symbols

    @property
    def current_atm(self) -> float | None:
        """Current ATM strike price."""
        return self._current_atm

    @property
    def subscribed_count(self) -> int:
        """Number of currently subscribed symbols."""
        return len(self._current_symbols)
