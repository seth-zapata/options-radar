"""Alpaca REST API client for account and portfolio data.

Read-only access to portfolio for the gating pipeline:
- Account info (cash, buying power, equity)
- Current positions (stocks and options)
- Recent orders (for position sizing context)

This is separate from the WebSocket client for streaming quotes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from backend.config import AlpacaConfig

logger = logging.getLogger(__name__)


@dataclass
class AccountInfo:
    """Alpaca account information."""

    account_id: str
    cash: float
    buying_power: float
    equity: float
    portfolio_value: float
    day_trade_count: int
    pattern_day_trader: bool
    trading_blocked: bool
    account_blocked: bool
    timestamp: str

    @property
    def available_for_options(self) -> float:
        """Cash available for options trading (conservative estimate)."""
        return min(self.cash, self.buying_power)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "accountId": self.account_id,
            "cash": self.cash,
            "buyingPower": self.buying_power,
            "equity": self.equity,
            "portfolioValue": self.portfolio_value,
            "dayTradeCount": self.day_trade_count,
            "patternDayTrader": self.pattern_day_trader,
            "tradingBlocked": self.trading_blocked,
            "accountBlocked": self.account_blocked,
            "availableForOptions": self.available_for_options,
            "timestamp": self.timestamp,
        }


@dataclass
class Position:
    """Single portfolio position."""

    symbol: str
    asset_class: str  # "us_equity" or "us_option"
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_percent: float
    current_price: float
    side: str  # "long" or "short"

    @property
    def is_option(self) -> bool:
        """True if this is an options position."""
        return self.asset_class == "us_option"

    @property
    def is_stock(self) -> bool:
        """True if this is a stock position."""
        return self.asset_class == "us_equity"

    @property
    def underlying(self) -> str:
        """Get underlying symbol for options, or symbol for stocks."""
        if self.is_option:
            # Parse underlying from OCC symbol (e.g., "NVDA250117C00500000" -> "NVDA")
            # Find where the date portion starts (6 digits for YYMMDD)
            for i, c in enumerate(self.symbol):
                if c.isdigit():
                    return self.symbol[:i]
            return self.symbol
        return self.symbol

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "assetClass": self.asset_class,
            "qty": self.qty,
            "avgEntryPrice": self.avg_entry_price,
            "marketValue": self.market_value,
            "unrealizedPl": self.unrealized_pl,
            "unrealizedPlPercent": self.unrealized_pl_percent,
            "currentPrice": self.current_price,
            "side": self.side,
            "isOption": self.is_option,
            "underlying": self.underlying,
        }


@dataclass
class PortfolioSummary:
    """Summary of portfolio for gating pipeline."""

    account: AccountInfo
    positions: list[Position]
    timestamp: str

    @property
    def total_positions(self) -> int:
        """Total number of positions."""
        return len(self.positions)

    @property
    def option_positions(self) -> list[Position]:
        """All option positions."""
        return [p for p in self.positions if p.is_option]

    @property
    def stock_positions(self) -> list[Position]:
        """All stock positions."""
        return [p for p in self.positions if p.is_stock]

    @property
    def sector_exposure(self) -> dict[str, float]:
        """Calculate exposure by underlying symbol."""
        exposure: dict[str, float] = {}
        for pos in self.positions:
            underlying = pos.underlying
            exposure[underlying] = exposure.get(underlying, 0) + abs(pos.market_value)
        return exposure

    def get_symbol_exposure(self, symbol: str) -> float:
        """Get total exposure to a specific symbol.

        Args:
            symbol: The underlying symbol to check

        Returns:
            Total absolute market value exposure
        """
        return sum(
            abs(p.market_value) for p in self.positions
            if p.underlying == symbol
        )

    def get_symbol_exposure_percent(self, symbol: str) -> float:
        """Get exposure to a symbol as percent of portfolio.

        Args:
            symbol: The underlying symbol to check

        Returns:
            Exposure as percentage (0-100)
        """
        if self.account.portfolio_value <= 0:
            return 0.0
        return (self.get_symbol_exposure(symbol) / self.account.portfolio_value) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "account": self.account.to_dict(),
            "positions": [p.to_dict() for p in self.positions],
            "totalPositions": self.total_positions,
            "optionPositions": len(self.option_positions),
            "stockPositions": len(self.stock_positions),
            "sectorExposure": self.sector_exposure,
            "timestamp": self.timestamp,
        }


@dataclass
class AlpacaAccountClient:
    """REST API client for Alpaca account data.

    Read-only access for portfolio context in gating decisions.

    Usage:
        config = AlpacaConfig(api_key="...", secret_key="...", paper=True)
        client = AlpacaAccountClient(config)

        portfolio = await client.get_portfolio()
        print(f"Cash: ${portfolio.account.cash:.2f}")
        print(f"Positions: {len(portfolio.positions)}")
    """

    config: AlpacaConfig

    # Rate limiting
    _requests_per_second: float = 5.0
    _last_request_time: float = field(default=0, init=False)

    # Cache
    _portfolio_cache: tuple[PortfolioSummary, float] | None = field(default=None, init=False)
    _cache_ttl: float = 60.0  # 1 minute cache

    @property
    def _base_url(self) -> str:
        """Get the appropriate base URL for the account."""
        return self.config.base_url

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        min_interval = 1.0 / self._requests_per_second

        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self._last_request_time = asyncio.get_event_loop().time()

    async def _request(self, endpoint: str) -> dict[str, Any] | list[dict[str, Any]]:
        """Make authenticated request to Alpaca Trading API.

        Args:
            endpoint: API endpoint (e.g., "/v2/account")

        Returns:
            JSON response (dict or list)
        """
        await self._rate_limit()

        url = f"{self._base_url}{endpoint}"
        headers = {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.secret_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 401:
                    logger.error("Alpaca API authentication failed")
                    raise Exception("Authentication failed")

                if response.status == 403:
                    logger.error("Alpaca API access forbidden")
                    raise Exception("Access forbidden")

                if response.status == 429:
                    logger.warning("Alpaca rate limit hit")
                    raise Exception("Rate limit exceeded")

                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Alpaca API error: {response.status} - {text}")
                    raise Exception(f"API error: {response.status}")

                return await response.json()

    async def get_account(self) -> AccountInfo:
        """Get account information.

        Returns:
            AccountInfo with cash, buying power, etc.
        """
        data = await self._request("/v2/account")

        if not isinstance(data, dict):
            raise Exception("Unexpected account response format")

        account = AccountInfo(
            account_id=data.get("id", ""),
            cash=float(data.get("cash", 0)),
            buying_power=float(data.get("buying_power", 0)),
            equity=float(data.get("equity", 0)),
            portfolio_value=float(data.get("portfolio_value", 0)),
            day_trade_count=int(data.get("daytrade_count", 0)),
            pattern_day_trader=data.get("pattern_day_trader", False),
            trading_blocked=data.get("trading_blocked", False),
            account_blocked=data.get("account_blocked", False),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            f"Account: cash=${account.cash:.2f}, "
            f"buying_power=${account.buying_power:.2f}, "
            f"portfolio=${account.portfolio_value:.2f}"
        )

        return account

    async def get_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of Position objects
        """
        data = await self._request("/v2/positions")

        if not isinstance(data, list):
            raise Exception("Unexpected positions response format")

        positions = []
        for item in data:
            position = Position(
                symbol=item.get("symbol", ""),
                asset_class=item.get("asset_class", "us_equity"),
                qty=float(item.get("qty", 0)),
                avg_entry_price=float(item.get("avg_entry_price", 0)),
                market_value=float(item.get("market_value", 0)),
                unrealized_pl=float(item.get("unrealized_pl", 0)),
                unrealized_pl_percent=float(item.get("unrealized_plpc", 0)) * 100,
                current_price=float(item.get("current_price", 0)),
                side=item.get("side", "long"),
            )
            positions.append(position)

        logger.info(
            f"Positions: {len(positions)} total, "
            f"{len([p for p in positions if p.is_option])} options, "
            f"{len([p for p in positions if p.is_stock])} stocks"
        )

        return positions

    async def get_portfolio(
        self,
        use_cache: bool = True,
    ) -> PortfolioSummary:
        """Get complete portfolio summary.

        Args:
            use_cache: Whether to use cached data

        Returns:
            PortfolioSummary with account and positions
        """
        now = asyncio.get_event_loop().time()

        # Check cache
        if use_cache and self._portfolio_cache:
            cached, cached_time = self._portfolio_cache
            if now - cached_time < self._cache_ttl:
                logger.debug("Using cached portfolio")
                return cached

        # Fetch account and positions in parallel
        account, positions = await asyncio.gather(
            self.get_account(),
            self.get_positions(),
        )

        portfolio = PortfolioSummary(
            account=account,
            positions=positions,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Cache result
        self._portfolio_cache = (portfolio, now)

        return portfolio

    async def get_position_for_symbol(self, symbol: str) -> Position | None:
        """Get position for a specific symbol.

        Args:
            symbol: Stock or option symbol

        Returns:
            Position or None if no position exists
        """
        try:
            data = await self._request(f"/v2/positions/{symbol}")

            if not isinstance(data, dict):
                return None

            return Position(
                symbol=data.get("symbol", ""),
                asset_class=data.get("asset_class", "us_equity"),
                qty=float(data.get("qty", 0)),
                avg_entry_price=float(data.get("avg_entry_price", 0)),
                market_value=float(data.get("market_value", 0)),
                unrealized_pl=float(data.get("unrealized_pl", 0)),
                unrealized_pl_percent=float(data.get("unrealized_plpc", 0)) * 100,
                current_price=float(data.get("current_price", 0)),
                side=data.get("side", "long"),
            )

        except Exception as e:
            if "404" in str(e):
                return None
            raise
