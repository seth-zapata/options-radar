"""Alpaca REST API client for price data.

Handles:
- Current/latest stock prices for outcome tracking
- Historical bar data for backtesting
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from backend.config import AlpacaConfig

logger = logging.getLogger(__name__)


@dataclass
class BarData:
    """OHLCV bar data."""

    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None


@dataclass
class LatestQuote:
    """Latest quote data."""

    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: str


@dataclass
class AlpacaRestClient:
    """REST client for Alpaca market data.

    Usage:
        config = AlpacaConfig(api_key="...", secret_key="...", paper=True)
        client = AlpacaRestClient(config)

        # Get current price
        price = await client.get_latest_price("AAPL")

        # Get historical bars
        bars = await client.get_bars("AAPL", "2024-01-01", "2024-12-01", "1Day")
    """

    config: AlpacaConfig

    # Rate limiting
    _requests_per_second: float = 5.0
    _last_request_time: float = field(default=0, init=False)

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        min_interval = 1.0 / self._requests_per_second

        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self._last_request_time = asyncio.get_event_loop().time()

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make authenticated request to Alpaca data API."""
        await self._rate_limit()

        url = f"{self.config.data_url}/{endpoint}"
        headers = {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.secret_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 401:
                    logger.error("Alpaca API authentication failed")
                    raise Exception("Authentication failed")

                if response.status == 429:
                    logger.warning("Alpaca rate limit hit")
                    raise Exception("Rate limit exceeded")

                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Alpaca API error: {response.status} - {text}")
                    raise Exception(f"API error: {response.status}")

                return await response.json()

    async def get_latest_price(self, symbol: str) -> float | None:
        """Get the latest price for a symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            Latest price or None if unavailable
        """
        try:
            # Use latest trades endpoint
            data = await self._request(f"v2/stocks/{symbol}/trades/latest")

            if data and "trade" in data:
                return float(data["trade"]["p"])

            return None

        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None

    async def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get latest prices for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to price
        """
        try:
            # Use multi-symbol endpoint
            params = {"symbols": ",".join(symbols)}
            data = await self._request("v2/stocks/trades/latest", params)

            prices = {}
            if data and "trades" in data:
                for symbol, trade in data["trades"].items():
                    prices[symbol] = float(trade["p"])

            return prices

        except Exception as e:
            logger.error(f"Error fetching latest prices: {e}")
            return {}

    async def get_latest_quote(self, symbol: str) -> LatestQuote | None:
        """Get the latest quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            LatestQuote or None if unavailable
        """
        try:
            data = await self._request(f"v2/stocks/{symbol}/quotes/latest")

            if data and "quote" in data:
                q = data["quote"]
                return LatestQuote(
                    symbol=symbol,
                    bid=float(q.get("bp", 0)),
                    ask=float(q.get("ap", 0)),
                    bid_size=int(q.get("bs", 0)),
                    ask_size=int(q.get("as", 0)),
                    timestamp=q.get("t", ""),
                )

            return None

        except Exception as e:
            logger.error(f"Error fetching latest quote for {symbol}: {e}")
            return None

    async def get_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1Day",
        limit: int = 1000,
    ) -> list[BarData]:
        """Get historical bars for a symbol.

        Args:
            symbol: Stock symbol
            start: Start date (YYYY-MM-DD or ISO8601)
            end: End date (YYYY-MM-DD or ISO8601)
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            limit: Maximum bars to return per request

        Returns:
            List of BarData
        """
        try:
            all_bars = []
            page_token = None

            while True:
                params = {
                    "start": start,
                    "end": end,
                    "timeframe": timeframe,
                    "limit": limit,
                }

                if page_token:
                    params["page_token"] = page_token

                data = await self._request(f"v2/stocks/{symbol}/bars", params)

                if not data or "bars" not in data or not data["bars"]:
                    break

                for bar in data["bars"]:
                    all_bars.append(BarData(
                        symbol=symbol,
                        timestamp=bar["t"],
                        open=float(bar["o"]),
                        high=float(bar["h"]),
                        low=float(bar["l"]),
                        close=float(bar["c"]),
                        volume=int(bar["v"]),
                        vwap=float(bar.get("vw", 0)) if bar.get("vw") else None,
                    ))

                page_token = data.get("next_page_token")
                if not page_token:
                    break

            logger.info(f"Fetched {len(all_bars)} bars for {symbol}")
            return all_bars

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return []

    async def get_bars_multi(
        self,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str = "1Day",
    ) -> dict[str, list[BarData]]:
        """Get historical bars for multiple symbols.

        Args:
            symbols: List of stock symbols
            start: Start date
            end: End date
            timeframe: Bar timeframe

        Returns:
            Dict mapping symbol to list of BarData
        """
        results = {}

        for symbol in symbols:
            bars = await self.get_bars(symbol, start, end, timeframe)
            if bars:
                results[symbol] = bars

        return results

    async def get_price_at_time(
        self,
        symbol: str,
        target_time: datetime,
    ) -> float | None:
        """Get the price at a specific time (or closest available).

        Args:
            symbol: Stock symbol
            target_time: Target datetime

        Returns:
            Price at or near that time, or None
        """
        try:
            # Get bars around the target time
            start = (target_time - timedelta(hours=1)).isoformat()
            end = (target_time + timedelta(hours=1)).isoformat()

            bars = await self.get_bars(symbol, start, end, "1Min", limit=120)

            if not bars:
                return None

            # Find closest bar
            target_ts = target_time.timestamp()
            closest_bar = min(
                bars,
                key=lambda b: abs(
                    datetime.fromisoformat(b.timestamp.replace("Z", "+00:00")).timestamp()
                    - target_ts
                )
            )

            return closest_bar.close

        except Exception as e:
            logger.error(f"Error fetching price at time for {symbol}: {e}")
            return None

    async def _trading_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make authenticated request to Alpaca Trading API (for options contracts)."""
        await self._rate_limit()

        url = f"{self.config.base_url}/{endpoint}"
        headers = {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.secret_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 401:
                    logger.error("Alpaca API authentication failed")
                    raise Exception("Authentication failed")

                if response.status == 429:
                    logger.warning("Alpaca rate limit hit")
                    raise Exception("Rate limit exceeded")

                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Alpaca Trading API error: {response.status} - {text}")
                    raise Exception(f"API error: {response.status}")

                return await response.json()

    async def get_options_chain_oi(
        self,
        underlying: str,
    ) -> dict[str, Any]:
        """Get options chain with open interest for an underlying.

        Uses Alpaca Trading API's /v2/options/contracts endpoint which includes
        open_interest for each contract.

        Args:
            underlying: Stock symbol (e.g., "TSLA")

        Returns:
            Dict with total_call_oi, total_put_oi, put_call_ratio, contracts
        """
        try:
            # Fetch all active contracts for the underlying
            all_contracts = []
            page_token = None

            while True:
                params: dict[str, Any] = {
                    "underlying_symbols": underlying,
                    "status": "active",
                    "limit": 100,
                }
                if page_token:
                    params["page_token"] = page_token

                data = await self._trading_request("v2/options/contracts", params)

                if not data or "option_contracts" not in data:
                    break

                contracts = data.get("option_contracts", [])
                all_contracts.extend(contracts)

                page_token = data.get("next_page_token")
                if not page_token or len(contracts) < 100:
                    break

                # Limit to 1000 contracts to avoid excessive API calls
                if len(all_contracts) >= 1000:
                    logger.warning(f"Truncating options chain for {underlying} at 1000 contracts")
                    break

            # Calculate totals
            total_call_oi = 0
            total_put_oi = 0

            for contract in all_contracts:
                oi = int(contract.get("open_interest", 0) or 0)
                contract_type = contract.get("type", "").lower()

                if contract_type == "call":
                    total_call_oi += oi
                elif contract_type == "put":
                    total_put_oi += oi

            # Calculate P/C ratio
            put_call_ratio = None
            if total_call_oi > 0:
                put_call_ratio = total_put_oi / total_call_oi

            logger.info(
                f"Alpaca options chain for {underlying}: {len(all_contracts)} contracts, "
                f"call_oi={total_call_oi}, put_oi={total_put_oi}, "
                f"P/C={put_call_ratio:.3f if put_call_ratio else 'N/A'}"
            )

            return {
                "symbol": underlying,
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "put_call_ratio": put_call_ratio,
                "num_contracts": len(all_contracts),
            }

        except Exception as e:
            logger.error(f"Error fetching options chain OI for {underlying}: {e}")
            return {
                "symbol": underlying,
                "total_call_oi": 0,
                "total_put_oi": 0,
                "put_call_ratio": None,
                "num_contracts": 0,
            }
