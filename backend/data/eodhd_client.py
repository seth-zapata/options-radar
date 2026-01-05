"""EODHD client for historical options data.

Fetches options chains with open interest for:
- Put/Call Ratio calculation
- Max Pain calculation

API docs: https://eodhd.com/financial-apis/stock-options-data-api
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from backend.config import EODHDConfig

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Single option contract from EODHD."""

    symbol: str  # Full option symbol (e.g., "TSLA260320P00990000")
    underlying: str  # Underlying symbol (e.g., "TSLA")
    expiry: str  # Expiration date (e.g., "2026-03-20")
    option_type: str  # "put" or "call"
    strike: float
    open_interest: int
    volume: int
    bid: float
    ask: float
    last_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def is_put(self) -> bool:
        return self.option_type == "put"


@dataclass
class OptionsChain:
    """Options chain for a symbol on a given date."""

    symbol: str
    date: str  # Date of the data
    contracts: list[OptionContract]
    timestamp: str

    @property
    def calls(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.is_call]

    @property
    def puts(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.is_put]

    @property
    def total_call_oi(self) -> int:
        return sum(c.open_interest for c in self.calls)

    @property
    def total_put_oi(self) -> int:
        return sum(c.open_interest for c in self.puts)

    @property
    def unique_strikes(self) -> list[float]:
        return sorted(set(c.strike for c in self.contracts))

    @property
    def unique_expiries(self) -> list[str]:
        return sorted(set(c.expiry for c in self.contracts))


@dataclass
class OptionsIndicators:
    """Calculated options indicators for a symbol."""

    symbol: str
    date: str
    put_call_ratio: float | None  # Total Put OI / Total Call OI
    max_pain: float | None  # Strike price where most options expire worthless
    total_call_oi: int
    total_put_oi: int
    num_contracts: int
    timestamp: str

    @property
    def pcr_signal(self) -> str | None:
        """Interpret Put/Call Ratio as a signal.

        P/C > 1.2 = Excessive bearishness (contrarian bullish)
        P/C < 0.6 = Excessive bullishness (contrarian bearish)
        """
        if self.put_call_ratio is None:
            return None
        if self.put_call_ratio > 1.2:
            return "bullish"  # Contrarian: excessive bearishness
        elif self.put_call_ratio < 0.6:
            return "bearish"  # Contrarian: excessive bullishness
        return "neutral"

    def get_directional_modifier(self, is_bullish_signal: bool) -> int:
        """Calculate confidence modifier based on signal direction.

        P/C Ratio is a CONTRARIAN indicator:
        - P/C > 1.2 = Excessive bearishness = contrarian bullish
        - P/C < 0.6 = Excessive bullishness = contrarian bearish

        Args:
            is_bullish_signal: True for BUY_CALL/SELL_PUT, False for BUY_PUT/SELL_CALL

        Returns:
            Confidence modifier: +5 when aligned, 0 otherwise
        """
        modifier = 0

        # P/C Ratio alignment (+5 when contrarian aligned)
        if self.pcr_signal is not None and self.pcr_signal != "neutral":
            if is_bullish_signal:
                if self.pcr_signal == "bullish":  # High P/C ratio (contrarian bullish)
                    modifier += 5
                    logger.debug(
                        f"P/C Ratio aligned: {self.put_call_ratio:.2f} (bullish signal) +5"
                    )
            else:  # Bearish signal
                if self.pcr_signal == "bearish":  # Low P/C ratio (contrarian bearish)
                    modifier += 5
                    logger.debug(
                        f"P/C Ratio aligned: {self.put_call_ratio:.2f} (bearish signal) +5"
                    )

        return modifier

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "date": self.date,
            "putCallRatio": self.put_call_ratio,
            "maxPain": self.max_pain,
            "totalCallOI": self.total_call_oi,
            "totalPutOI": self.total_put_oi,
            "numContracts": self.num_contracts,
            "pcrSignal": self.pcr_signal,
            "timestamp": self.timestamp,
        }


@dataclass
class EODHDClient:
    """Client for EODHD Options API.

    Usage:
        config = EODHDConfig(api_key="...")
        client = EODHDClient(config)

        indicators = await client.get_options_indicators("TSLA")
    """

    config: EODHDConfig

    # Rate limiting (basic)
    _requests_today: int = field(default=0, init=False)

    # Cache
    _chain_cache: dict[tuple[str, str], tuple[OptionsChain, float]] = field(
        default_factory=dict, init=False
    )
    _cache_ttl: float = 3600.0  # 1 hour cache for historical data

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make authenticated request to EODHD API."""
        url = f"{self.config.base_url}/{endpoint}"

        if params is None:
            params = {}
        params["api_token"] = self.config.api_key
        params["fmt"] = "json"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    logger.warning("EODHD rate limit hit")
                    raise Exception("Rate limit exceeded")

                if response.status != 200:
                    text = await response.text()
                    logger.error(f"EODHD API error: {response.status} - {text}")
                    raise Exception(f"API error: {response.status}")

                self._requests_today += 1
                return await response.json()

    async def fetch_options_chain(
        self,
        symbol: str,
        trade_date: str | None = None,
        use_cache: bool = True,
    ) -> OptionsChain | None:
        """Fetch options chain for a symbol.

        Uses the EODHD Marketplace API endpoint:
        /mp/unicornbay/options/contracts with filter[tradetime_eq] for historical data

        Args:
            symbol: Stock symbol (e.g., "TSLA")
            trade_date: Optional date to fetch historical data (YYYY-MM-DD format)
            use_cache: Whether to use cached data

        Returns:
            OptionsChain or None if unavailable
        """
        import asyncio

        now = asyncio.get_event_loop().time()
        cache_key = (symbol, trade_date or "current")

        # Check cache
        if use_cache and cache_key in self._chain_cache:
            cached, cached_time = self._chain_cache[cache_key]
            if now - cached_time < self._cache_ttl:
                logger.debug(f"Using cached options chain for {symbol}")
                return cached

        try:
            # Use marketplace endpoint for historical data
            endpoint = "mp/unicornbay/options/contracts"

            # Build params with filters
            params: dict[str, Any] = {
                "filter[underlying_symbol]": symbol,
                "page[limit]": 1000,  # Max per request
            }
            if trade_date:
                params["filter[tradetime_eq]"] = trade_date

            # Fetch all pages of data
            all_contracts = []
            offset = 0
            total = None

            while True:
                params["page[offset]"] = offset
                data = await self._request(endpoint, params)

                if not data or "data" not in data:
                    if offset == 0:
                        logger.warning(f"No options data for {symbol}")
                        return None
                    break

                # Get total from meta
                if total is None:
                    total = data.get("meta", {}).get("total", 0)

                # Parse contracts from this page
                for item in data["data"]:
                    attrs = item.get("attributes", {})
                    contract = OptionContract(
                        symbol=attrs.get("contract", ""),
                        underlying=attrs.get("underlying_symbol", symbol),
                        expiry=attrs.get("exp_date", ""),
                        option_type=attrs.get("type", "").lower(),
                        strike=float(attrs.get("strike", 0) or 0),
                        open_interest=int(attrs.get("open_interest", 0) or 0),
                        volume=int(attrs.get("volume", 0) or 0),
                        bid=float(attrs.get("bid", 0) or 0),
                        ask=float(attrs.get("ask", 0) or 0),
                        last_price=float(attrs.get("last", 0) or 0),
                        delta=float(attrs.get("delta", 0) or 0),
                        gamma=float(attrs.get("gamma", 0) or 0),
                        theta=float(attrs.get("theta", 0) or 0),
                        vega=float(attrs.get("vega", 0) or 0),
                        iv=float(attrs.get("volatility", 0) or 0),
                    )
                    all_contracts.append(contract)

                # Check if we have all data
                offset += len(data["data"])
                if offset >= total or len(data["data"]) < 1000:
                    break

                # Max 10 pages to avoid rate limits
                if offset >= 10000:
                    logger.warning(f"Truncating options data for {symbol} at 10000 contracts")
                    break

            chain = OptionsChain(
                symbol=symbol,
                date=trade_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                contracts=all_contracts,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Cache the result
            self._chain_cache[cache_key] = (chain, now)

            logger.info(
                f"EODHD options for {symbol} ({trade_date or 'current'}): "
                f"{len(all_contracts)} contracts, "
                f"call_oi={chain.total_call_oi}, put_oi={chain.total_put_oi}"
            )

            return chain

        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return None

    def calculate_put_call_ratio(self, chain: OptionsChain) -> float | None:
        """Calculate Put/Call Ratio from open interest.

        Formula: Total Put OI / Total Call OI

        Interpretation:
        - P/C > 1.2: Excessive bearishness (contrarian bullish signal)
        - P/C < 0.6: Excessive bullishness (contrarian bearish signal)
        - 0.6 - 1.2: Neutral zone

        Args:
            chain: Options chain data

        Returns:
            Put/Call ratio or None if insufficient data
        """
        total_call_oi = chain.total_call_oi
        total_put_oi = chain.total_put_oi

        if total_call_oi == 0:
            logger.warning(f"No call OI for {chain.symbol}, cannot calculate P/C ratio")
            return None

        ratio = total_put_oi / total_call_oi
        logger.debug(f"P/C ratio for {chain.symbol}: {ratio:.3f}")
        return ratio

    def calculate_max_pain(
        self,
        chain: OptionsChain,
        underlying_price: float | None = None,
    ) -> float | None:
        """Calculate Max Pain strike price.

        Max Pain is the strike price where option buyers experience
        maximum loss (option sellers profit most). It's calculated
        by finding the strike that minimizes the total intrinsic value
        of all outstanding options.

        Args:
            chain: Options chain data
            underlying_price: Current underlying price (for context, not used in calc)

        Returns:
            Max Pain strike price or None if insufficient data
        """
        strikes = chain.unique_strikes
        if not strikes:
            logger.warning(f"No strikes for {chain.symbol}, cannot calculate max pain")
            return None

        # Group contracts by strike
        calls_by_strike: dict[float, int] = {}
        puts_by_strike: dict[float, int] = {}

        for contract in chain.contracts:
            if contract.is_call:
                calls_by_strike[contract.strike] = (
                    calls_by_strike.get(contract.strike, 0) + contract.open_interest
                )
            else:
                puts_by_strike[contract.strike] = (
                    puts_by_strike.get(contract.strike, 0) + contract.open_interest
                )

        # Calculate total pain at each potential expiration price
        min_pain = float("inf")
        max_pain_strike = None

        for test_price in strikes:
            total_pain = 0.0

            # Calculate call pain: if price > strike, call is ITM
            for strike, oi in calls_by_strike.items():
                if test_price > strike:
                    # Calls are ITM, buyers profit (pain for sellers)
                    # But for max pain, we want where buyers lose most
                    total_pain += (test_price - strike) * oi

            # Calculate put pain: if price < strike, put is ITM
            for strike, oi in puts_by_strike.items():
                if test_price < strike:
                    # Puts are ITM, buyers profit
                    total_pain += (strike - test_price) * oi

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_price

        if max_pain_strike is not None:
            logger.debug(f"Max pain for {chain.symbol}: ${max_pain_strike}")

        return max_pain_strike

    async def get_options_indicators(
        self,
        symbol: str,
        trade_date: str | None = None,
        underlying_price: float | None = None,
    ) -> OptionsIndicators | None:
        """Get calculated options indicators for a symbol.

        Fetches options chain and calculates:
        - Put/Call Ratio
        - Max Pain

        Args:
            symbol: Stock symbol (e.g., "TSLA")
            trade_date: Optional date for historical data
            underlying_price: Current price for max pain context

        Returns:
            OptionsIndicators or None if data unavailable
        """
        chain = await self.fetch_options_chain(symbol, trade_date)
        if chain is None:
            return None

        pcr = self.calculate_put_call_ratio(chain)
        max_pain = self.calculate_max_pain(chain, underlying_price)

        return OptionsIndicators(
            symbol=symbol,
            date=chain.date,
            put_call_ratio=pcr,
            max_pain=max_pain,
            total_call_oi=chain.total_call_oi,
            total_put_oi=chain.total_put_oi,
            num_contracts=len(chain.contracts),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def get_historical_indicators(
        self,
        symbol: str,
        dates: list[str],
        underlying_prices: dict[str, float] | None = None,
    ) -> dict[str, OptionsIndicators | None]:
        """Get options indicators for multiple dates.

        Args:
            symbol: Stock symbol
            dates: List of dates (YYYY-MM-DD format)
            underlying_prices: Optional dict of date -> price

        Returns:
            Dict mapping date to OptionsIndicators
        """
        results: dict[str, OptionsIndicators | None] = {}

        for date in dates:
            price = underlying_prices.get(date) if underlying_prices else None
            indicators = await self.get_options_indicators(symbol, date, price)
            results[date] = indicators

        return results
