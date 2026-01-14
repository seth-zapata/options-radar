"""ORATS REST client for Greeks and IV data.

Handles:
- Live Greeks polling for option contracts
- IV rank and percentile fetching for underlyings
- Rate limit awareness (100k requests/month budget)

See spec section 3.1-3.2 for cadence and rate limits.
API docs: https://docs.orats.io/datav2-live-api-guide/data.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from backend.config import ORATSConfig
from backend.models.canonical import CanonicalOptionId
from backend.models.market_data import GreeksData, UnderlyingData

logger = logging.getLogger(__name__)


@dataclass
class ORATSClient:
    """REST client for ORATS options data API.

    Fetches live Greeks and IV data for options analysis.

    Usage:
        config = ORATSConfig(api_token="...")
        client = ORATSClient(config)

        async with client:
            greeks = await client.fetch_greeks("NVDA", "2025-01-17", strikes=[140, 145, 150])
            iv_data = await client.fetch_iv_rank("NVDA")

    Attributes:
        config: ORATS API configuration
    """

    config: ORATSConfig

    # Rate limit tracking
    _requests_today: int = field(default=0, init=False)
    _last_reset_date: str = field(default="", init=False)

    # HTTP client (initialized in __aenter__)
    _http: httpx.AsyncClient | None = field(default=None, init=False)

    async def __aenter__(self) -> "ORATSClient":
        """Async context manager entry."""
        self._http = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._http:
            await self._http.aclose()
            self._http = None

    def _check_rate_limit(self) -> None:
        """Check and reset daily rate limit counter."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            self._requests_today = 0
            self._last_reset_date = today

    def _increment_requests(self, count: int = 1) -> None:
        """Increment request counter and log warnings if approaching limit."""
        self._check_rate_limit()
        self._requests_today += count

        # Warn at 80% of daily budget (~2560 requests)
        daily_budget = 3200  # 100k/month ÷ 31 days
        if self._requests_today == int(daily_budget * 0.8):
            logger.warning(
                f"ORATS rate limit warning: {self._requests_today} requests today "
                f"(80% of daily budget)"
            )
        elif self._requests_today >= daily_budget:
            logger.error(
                f"ORATS daily budget exceeded: {self._requests_today} requests"
            )

    @property
    def requests_today(self) -> int:
        """Number of API requests made today."""
        self._check_rate_limit()
        return self._requests_today

    async def fetch_greeks(
        self,
        ticker: str,
        expiry: str,
        strikes: list[float] | None = None,
        dte_min: int | None = None,
        dte_max: int | None = None,
    ) -> list[GreeksData]:
        """Fetch live Greeks for options.

        Args:
            ticker: Underlying symbol (e.g., "NVDA")
            expiry: Expiration date in ISO format (e.g., "2025-01-17")
            strikes: Optional list of specific strikes to fetch
            dte_min: Optional minimum days to expiration
            dte_max: Optional maximum days to expiration

        Returns:
            List of GreeksData for matching options

        Raises:
            httpx.HTTPError: If API request fails
        """
        if not self._http:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        params: dict[str, Any] = {
            "token": self.config.api_token,
            "ticker": ticker,
        }

        if expiry:
            params["expirDate"] = expiry

        if dte_min is not None:
            params["dte"] = f"{dte_min},"
        if dte_max is not None:
            if "dte" in params:
                params["dte"] = f"{dte_min},{dte_max}"
            else:
                params["dte"] = f",{dte_max}"

        response = await self._http.get("/live/strikes", params=params)
        self._increment_requests()
        response.raise_for_status()

        data = response.json()
        timestamp = datetime.now(timezone.utc).isoformat()

        results = []
        for item in data.get("data", []):
            # Filter by strikes if specified
            # Round to 2 decimal places to match Alpaca's integer-based strike parsing
            strike = round(item.get("strike", 0), 2)
            if strikes and strike not in strikes:
                continue

            expiry_date = item.get("expirDate", expiry)
            item_timestamp = item.get("updatedAt", timestamp)
            # ORATS returns smoothed IV (smvVol) which is more stable
            iv = item.get("smvVol", item.get("iv", 0.0))

            # ORATS /live/strikes returns BOTH call and put data per record
            # Fields are prefixed: callDelta/putDelta, callGamma/putGamma, etc.
            # Create two GreeksData entries - one for call, one for put

            # Call option
            call_id = CanonicalOptionId(
                underlying=ticker,
                expiry=expiry_date,
                right="C",
                strike=strike,
            )
            call_greeks = GreeksData(
                canonical_id=call_id,
                delta=item.get("callDelta", item.get("delta", 0.0)),
                gamma=item.get("callGamma", item.get("gamma", 0.0)),
                theta=item.get("callTheta", item.get("theta", 0.0)),
                vega=item.get("callVega", item.get("vega", 0.0)),
                rho=item.get("callRho", item.get("rho", 0.0)),
                iv=iv,
                theoretical_value=item.get("callValue", 0.0),
                timestamp=item_timestamp,
                source="orats",
            )
            results.append(call_greeks)

            # Put option
            put_id = CanonicalOptionId(
                underlying=ticker,
                expiry=expiry_date,
                right="P",
                strike=strike,
            )
            put_greeks = GreeksData(
                canonical_id=put_id,
                delta=item.get("putDelta", item.get("delta", 0.0)),
                gamma=item.get("putGamma", item.get("gamma", 0.0)),
                theta=item.get("putTheta", item.get("theta", 0.0)),
                vega=item.get("putVega", item.get("vega", 0.0)),
                rho=item.get("putRho", item.get("rho", 0.0)),
                iv=iv,
                theoretical_value=item.get("putValue", 0.0),
                timestamp=item_timestamp,
                source="orats",
            )
            results.append(put_greeks)

        logger.debug(f"Fetched {len(results)} Greeks for {ticker} {expiry}")
        return results

    async def fetch_greeks_for_options(
        self,
        options: list[CanonicalOptionId],
    ) -> dict[CanonicalOptionId, GreeksData]:
        """Fetch Greeks for specific option contracts.

        Groups options by ticker/expiry for efficient API calls.

        Args:
            options: List of canonical option IDs

        Returns:
            Dict mapping option IDs to their Greeks data
        """
        # Group by ticker and expiry
        groups: dict[tuple[str, str], list[CanonicalOptionId]] = {}
        for opt in options:
            key = (opt.underlying, opt.expiry)
            if key not in groups:
                groups[key] = []
            groups[key].append(opt)

        results: dict[CanonicalOptionId, GreeksData] = {}

        for (ticker, expiry), opts in groups.items():
            strikes = [opt.strike for opt in opts]
            greeks_list = await self.fetch_greeks(ticker, expiry, strikes=strikes)

            # Match Greeks to options
            for greeks in greeks_list:
                results[greeks.canonical_id] = greeks

        return results

    async def fetch_iv_rank(self, ticker: str) -> UnderlyingData | None:
        """Fetch IV rank and percentile for an underlying.

        Args:
            ticker: Underlying symbol (e.g., "NVDA")

        Returns:
            UnderlyingData with IV metrics, or None if not found

        Raises:
            httpx.HTTPError: If API request fails
        """
        if not self._http:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        params = {
            "token": self.config.api_token,
            "ticker": ticker,
        }

        response = await self._http.get("/ivrank", params=params)
        self._increment_requests()
        response.raise_for_status()

        data = response.json()
        items = data.get("data", [])

        if not items:
            logger.warning(f"No IV rank data for {ticker}")
            return None

        item = items[0]
        timestamp = datetime.now(timezone.utc).isoformat()

        return UnderlyingData(
            symbol=ticker,
            price=0.0,  # IV rank endpoint doesn't include price
            iv_rank=item.get("ivRank1y", item.get("ivRank1m", 0.0)),  # Already 0-100
            iv_percentile=item.get("ivPct1y", item.get("ivPct1m", 0.0)),  # Already 0-100
            timestamp=item.get("updatedAt", timestamp),
        )

    async def fetch_iv_ranks(self, tickers: list[str]) -> dict[str, UnderlyingData]:
        """Fetch IV rank for multiple underlyings.

        Args:
            tickers: List of underlying symbols

        Returns:
            Dict mapping tickers to their IV data
        """
        results = {}
        for ticker in tickers:
            data = await self.fetch_iv_rank(ticker)
            if data:
                results[ticker] = data
        return results
