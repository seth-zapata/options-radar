"""Quiver Quant client for political, insider, and social sentiment data.

Fetches congress trading, insider transactions, and WallStreetBets data.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from backend.config import QuiverConfig

logger = logging.getLogger(__name__)


@dataclass
class CongressTrade:
    """Congressional trading record."""

    representative: str
    party: str
    ticker: str
    transaction_type: str  # "Purchase" or "Sale"
    amount_range: str  # e.g., "$1,001 - $15,000"
    transaction_date: str
    disclosure_date: str

    @property
    def is_buy(self) -> bool:
        """True if this is a purchase."""
        return "purchase" in self.transaction_type.lower()

    @property
    def is_sell(self) -> bool:
        """True if this is a sale."""
        return "sale" in self.transaction_type.lower()

    @property
    def estimated_amount(self) -> float:
        """Estimate middle of amount range."""
        # Parse ranges like "$1,001 - $15,000" or "$15,001 - $50,000"
        try:
            clean = self.amount_range.replace("$", "").replace(",", "")
            if " - " in clean:
                low, high = clean.split(" - ")
                return (float(low) + float(high)) / 2
            return float(clean)
        except (ValueError, AttributeError):
            return 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "representative": self.representative,
            "party": self.party,
            "ticker": self.ticker,
            "transactionType": self.transaction_type,
            "amountRange": self.amount_range,
            "estimatedAmount": self.estimated_amount,
            "isBuy": self.is_buy,
            "transactionDate": self.transaction_date,
            "disclosureDate": self.disclosure_date,
        }


@dataclass
class InsiderTrade:
    """SEC Form 4 insider trading record."""

    ticker: str
    insider_name: str
    insider_title: str
    transaction_type: str  # "P" (purchase), "S" (sale), etc.
    shares: int
    price: float
    value: float
    filing_date: str

    @property
    def is_buy(self) -> bool:
        """True if this is a purchase."""
        return self.transaction_type.upper() == "P"

    @property
    def is_sell(self) -> bool:
        """True if this is a sale."""
        return self.transaction_type.upper() == "S"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "ticker": self.ticker,
            "insiderName": self.insider_name,
            "insiderTitle": self.insider_title,
            "transactionType": self.transaction_type,
            "shares": self.shares,
            "price": self.price,
            "value": self.value,
            "isBuy": self.is_buy,
            "filingDate": self.filing_date,
        }


@dataclass
class WSBSentiment:
    """WallStreetBets social sentiment for a symbol.

    Captures retail investor sentiment from Reddit's r/wallstreetbets.
    This is social/retail sentiment, distinct from institutional data.
    """

    symbol: str
    mentions_24h: int  # Mentions in last 24 hours
    mentions_7d: int  # Mentions in last 7 days
    sentiment: float  # -1 (bearish) to 1 (bullish)
    rank: int  # Rank among all tickers (1 = most mentioned)
    timestamp: str

    @property
    def sentiment_score(self) -> float:
        """Sentiment from -100 to 100."""
        return self.sentiment * 100

    @property
    def is_trending(self) -> bool:
        """True if symbol is trending (top 20 or 50+ mentions)."""
        return self.mentions_24h >= 50 or self.rank <= 20

    @property
    def is_bullish(self) -> bool:
        """True if sentiment is bullish."""
        return self.sentiment > 0.1

    @property
    def is_bearish(self) -> bool:
        """True if sentiment is bearish."""
        return self.sentiment < -0.1

    @property
    def buzz_level(self) -> str:
        """Categorize buzz level."""
        if self.mentions_24h >= 200:
            return "viral"
        elif self.mentions_24h >= 100:
            return "high"
        elif self.mentions_24h >= 50:
            return "moderate"
        elif self.mentions_24h >= 10:
            return "low"
        return "minimal"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "mentions24h": self.mentions_24h,
            "mentions7d": self.mentions_7d,
            "sentiment": round(self.sentiment, 3),
            "sentimentScore": round(self.sentiment_score, 1),
            "rank": self.rank,
            "isTrending": self.is_trending,
            "isBullish": self.is_bullish,
            "isBearish": self.is_bearish,
            "buzzLevel": self.buzz_level,
            "timestamp": self.timestamp,
        }


@dataclass
class PoliticalSentiment:
    """Aggregated political/insider sentiment for a symbol.

    Combines congressional trading and corporate insider (SEC Form 4) data.
    These represent "smart money" with potential information advantages.
    """

    symbol: str
    congress_buys_30d: int
    congress_sells_30d: int
    congress_net_buys: int  # buys - sells
    congress_total_value: float
    insider_buys_30d: int
    insider_sells_30d: int
    insider_net_buys: int
    insider_total_value: float
    timestamp: str

    @property
    def congress_sentiment(self) -> float:
        """Congress sentiment from -100 (all sells) to 100 (all buys)."""
        total = self.congress_buys_30d + self.congress_sells_30d
        if total == 0:
            return 0
        return ((self.congress_buys_30d - self.congress_sells_30d) / total) * 100

    @property
    def insider_sentiment(self) -> float:
        """Insider sentiment from -100 (all sells) to 100 (all buys)."""
        total = self.insider_buys_30d + self.insider_sells_30d
        if total == 0:
            return 0
        return ((self.insider_buys_30d - self.insider_sells_30d) / total) * 100

    @property
    def combined_sentiment(self) -> float:
        """Combined smart money sentiment score.

        Weighting:
        - Congress: 60% (potential information asymmetry)
        - Insider: 40% (direct company knowledge)
        """
        return (self.congress_sentiment * 0.6) + (self.insider_sentiment * 0.4)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "congress": {
                "buys30d": self.congress_buys_30d,
                "sells30d": self.congress_sells_30d,
                "netBuys": self.congress_net_buys,
                "totalValue": self.congress_total_value,
                "sentiment": round(self.congress_sentiment, 1),
            },
            "insider": {
                "buys30d": self.insider_buys_30d,
                "sells30d": self.insider_sells_30d,
                "netBuys": self.insider_net_buys,
                "totalValue": self.insider_total_value,
                "sentiment": round(self.insider_sentiment, 1),
            },
            "combinedSentiment": round(self.combined_sentiment, 1),
            "timestamp": self.timestamp,
        }


@dataclass
class QuiverClient:
    """Client for Quiver Quant API.

    Usage:
        config = QuiverConfig(api_key="...")
        client = QuiverClient(config)

        congress = await client.get_congress_trades("NVDA")
        insiders = await client.get_insider_trades("NVDA")
        sentiment = await client.get_political_sentiment("NVDA")
    """

    config: QuiverConfig

    # Rate limiting
    _requests_per_second: float = 2.0
    _last_request_time: float = field(default=0, init=False)

    # Cache
    _sentiment_cache: dict[str, tuple[PoliticalSentiment, float]] = field(
        default_factory=dict, init=False
    )
    _cache_ttl: float = 600.0  # 10 minute cache

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        min_interval = 1.0 / self._requests_per_second

        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self._last_request_time = asyncio.get_event_loop().time()

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make authenticated request to Quiver API."""
        await self._rate_limit()

        url = f"{self.config.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 401:
                    logger.error("Quiver API authentication failed")
                    raise Exception("Authentication failed")

                if response.status == 429:
                    logger.warning("Quiver rate limit hit")
                    raise Exception("Rate limit exceeded")

                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Quiver API error: {response.status} - {text}")
                    raise Exception(f"API error: {response.status}")

                return await response.json()

    async def get_congress_trades(
        self,
        symbol: str,
        limit: int = 50,
    ) -> list[CongressTrade]:
        """Get recent congress trades for a symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum trades to return

        Returns:
            List of CongressTrade records
        """
        try:
            data = await self._request(f"historical/congresstrading/{symbol}")

            if not data:
                return []

            trades = []
            for item in data[:limit]:
                trades.append(CongressTrade(
                    representative=item.get("Representative", ""),
                    party=item.get("Party", ""),
                    ticker=symbol,
                    transaction_type=item.get("Transaction", ""),
                    amount_range=item.get("Range", ""),
                    transaction_date=item.get("TransactionDate", ""),
                    disclosure_date=item.get("DisclosureDate", ""),
                ))

            logger.info(f"Fetched {len(trades)} congress trades for {symbol}")
            return trades

        except Exception as e:
            logger.error(f"Error fetching congress trades for {symbol}: {e}")
            return []

    async def get_insider_trades(
        self,
        symbol: str,
        limit: int = 50,
    ) -> list[InsiderTrade]:
        """Get recent insider trades for a symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum trades to return

        Returns:
            List of InsiderTrade records
        """
        try:
            data = await self._request("live/insiders", {"ticker": symbol})

            if not data:
                return []

            trades = []
            for item in data[:limit]:
                trades.append(InsiderTrade(
                    ticker=symbol,
                    insider_name=item.get("Name", ""),
                    insider_title=item.get("Title", ""),
                    transaction_type=item.get("TransactionCode", ""),
                    shares=int(item.get("Shares", 0) or 0),
                    price=float(item.get("PricePerShare", 0) or 0),
                    value=float(item.get("Value", 0) or 0),
                    filing_date=item.get("FilingDate", ""),
                ))

            logger.info(f"Fetched {len(trades)} insider trades for {symbol}")
            return trades

        except Exception as e:
            logger.error(f"Error fetching insider trades for {symbol}: {e}")
            return []

    async def get_political_sentiment(
        self,
        symbol: str,
        use_cache: bool = True,
    ) -> PoliticalSentiment:
        """Get aggregated political/insider sentiment for a symbol.

        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data

        Returns:
            PoliticalSentiment with aggregated data
        """
        now = asyncio.get_event_loop().time()

        # Check cache
        if use_cache and symbol in self._sentiment_cache:
            cached, cached_time = self._sentiment_cache[symbol]
            if now - cached_time < self._cache_ttl:
                logger.debug(f"Using cached political sentiment for {symbol}")
                return cached

        # Fetch both congress and insider data
        congress_trades = await self.get_congress_trades(symbol)
        insider_trades = await self.get_insider_trades(symbol)

        # Calculate 30-day window
        now_dt = datetime.now(timezone.utc)
        thirty_days_ago = now_dt.timestamp() - (30 * 24 * 60 * 60)

        # Aggregate congress trades
        congress_buys = 0
        congress_sells = 0
        congress_value = 0.0

        for trade in congress_trades:
            try:
                trade_date = datetime.strptime(
                    trade.transaction_date, "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc)
                if trade_date.timestamp() >= thirty_days_ago:
                    if trade.is_buy:
                        congress_buys += 1
                        congress_value += trade.estimated_amount
                    elif trade.is_sell:
                        congress_sells += 1
                        congress_value -= trade.estimated_amount
            except (ValueError, AttributeError):
                continue

        # Aggregate insider trades
        insider_buys = 0
        insider_sells = 0
        insider_value = 0.0

        for trade in insider_trades:
            try:
                trade_date = datetime.strptime(
                    trade.filing_date[:10], "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc)
                if trade_date.timestamp() >= thirty_days_ago:
                    if trade.is_buy:
                        insider_buys += 1
                        insider_value += trade.value
                    elif trade.is_sell:
                        insider_sells += 1
                        insider_value -= trade.value
            except (ValueError, AttributeError):
                continue

        sentiment = PoliticalSentiment(
            symbol=symbol,
            congress_buys_30d=congress_buys,
            congress_sells_30d=congress_sells,
            congress_net_buys=congress_buys - congress_sells,
            congress_total_value=congress_value,
            insider_buys_30d=insider_buys,
            insider_sells_30d=insider_sells,
            insider_net_buys=insider_buys - insider_sells,
            insider_total_value=insider_value,
            timestamp=now_dt.isoformat(),
        )

        # Cache the result
        self._sentiment_cache[symbol] = (sentiment, now)

        logger.info(
            f"Quiver sentiment for {symbol}: "
            f"congress={sentiment.congress_sentiment:.0f}, "
            f"insider={sentiment.insider_sentiment:.0f}, "
            f"combined={sentiment.combined_sentiment:.0f}"
        )

        return sentiment

    async def get_recent_congress_activity(
        self,
        limit: int = 20,
    ) -> list[CongressTrade]:
        """Get most recent congress trading activity across all symbols.

        Args:
            limit: Maximum trades to return

        Returns:
            List of recent CongressTrade records
        """
        try:
            data = await self._request("live/congresstrading")

            if not data:
                return []

            trades = []
            for item in data[:limit]:
                trades.append(CongressTrade(
                    representative=item.get("Representative", ""),
                    party=item.get("Party", ""),
                    ticker=item.get("Ticker", ""),
                    transaction_type=item.get("Transaction", ""),
                    amount_range=item.get("Range", ""),
                    transaction_date=item.get("TransactionDate", ""),
                    disclosure_date=item.get("DisclosureDate", ""),
                ))

            logger.info(f"Fetched {len(trades)} recent congress trades")
            return trades

        except Exception as e:
            logger.error(f"Error fetching recent congress activity: {e}")
            return []

    async def get_sentiment_batch(
        self,
        symbols: list[str],
    ) -> dict[str, PoliticalSentiment]:
        """Get sentiment for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to PoliticalSentiment
        """
        results = {}

        for symbol in symbols:
            results[symbol] = await self.get_political_sentiment(symbol)

        return results

    # =========================================================================
    # WallStreetBets (WSB) Methods - Social/Retail Sentiment
    # =========================================================================

    async def get_wsb_sentiment(
        self,
        symbol: str,
    ) -> WSBSentiment | None:
        """Get WallStreetBets sentiment for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            WSBSentiment or None if not found
        """
        try:
            # Get 24h data
            data_24h = await self._request("mobile/24hwsbcounts", {"ticker": symbol})

            # Get hottest/ranking data
            data_hot = await self._request("mobile/24hwsbhottest")

            now = datetime.now(timezone.utc)

            # Parse 24h mentions
            mentions_24h = 0
            sentiment = 0.0

            if data_24h and isinstance(data_24h, list) and len(data_24h) > 0:
                for item in data_24h:
                    if item.get("Ticker", "").upper() == symbol.upper():
                        mentions_24h = int(item.get("Count", 0) or 0)
                        sentiment = float(item.get("Sentiment", 0) or 0)
                        break

            # Find rank in hottest
            rank = 999
            if data_hot and isinstance(data_hot, list):
                for i, item in enumerate(data_hot):
                    if item.get("Ticker", "").upper() == symbol.upper():
                        rank = i + 1
                        # Use sentiment from hottest if we didn't get it above
                        if sentiment == 0:
                            sentiment = float(item.get("Sentiment", 0) or 0)
                        break

            # If no data found, return None
            if mentions_24h == 0 and rank == 999:
                logger.debug(f"No WSB data found for {symbol}")
                return None

            wsb = WSBSentiment(
                symbol=symbol,
                mentions_24h=mentions_24h,
                mentions_7d=mentions_24h * 7,  # Estimate (API doesn't give 7d directly)
                sentiment=sentiment,
                rank=rank,
                timestamp=now.isoformat(),
            )

            logger.info(
                f"WSB sentiment for {symbol}: "
                f"mentions={wsb.mentions_24h}, "
                f"sentiment={wsb.sentiment:.2f}, "
                f"rank={wsb.rank}, "
                f"buzz={wsb.buzz_level}"
            )

            return wsb

        except Exception as e:
            logger.error(f"Error fetching WSB sentiment for {symbol}: {e}")
            return None

    async def get_wsb_trending(
        self,
        limit: int = 20,
    ) -> list[WSBSentiment]:
        """Get trending symbols on WallStreetBets.

        Args:
            limit: Maximum symbols to return

        Returns:
            List of WSBSentiment for trending tickers
        """
        try:
            data = await self._request("mobile/24hwsbhottest")

            if not data or not isinstance(data, list):
                return []

            now = datetime.now(timezone.utc)
            results = []

            for i, item in enumerate(data[:limit]):
                ticker = item.get("Ticker", "")
                if not ticker:
                    continue

                wsb = WSBSentiment(
                    symbol=ticker,
                    mentions_24h=int(item.get("Count", 0) or 0),
                    mentions_7d=int(item.get("Count", 0) or 0) * 7,
                    sentiment=float(item.get("Sentiment", 0) or 0),
                    rank=i + 1,
                    timestamp=now.isoformat(),
                )
                results.append(wsb)

            logger.info(f"Fetched {len(results)} trending WSB tickers")
            return results

        except Exception as e:
            logger.error(f"Error fetching WSB trending: {e}")
            return []

    async def get_wsb_sentiment_batch(
        self,
        symbols: list[str],
    ) -> dict[str, WSBSentiment | None]:
        """Get WSB sentiment for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to WSBSentiment (or None)
        """
        results = {}

        for symbol in symbols:
            results[symbol] = await self.get_wsb_sentiment(symbol)

        return results
