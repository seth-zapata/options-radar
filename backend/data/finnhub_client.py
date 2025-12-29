"""Finnhub client for news sentiment data.

Fetches news sentiment scores and company news for trading signals.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from backend.config import FinnhubConfig

logger = logging.getLogger(__name__)


@dataclass
class NewsSentiment:
    """News sentiment data for a symbol."""

    symbol: str
    buzz_articles_week: int  # Number of articles in past week
    buzz_weekly_average: float  # Average articles per week
    company_news_score: float  # -1 to 1, overall sentiment
    sector_average_score: float  # Sector comparison
    bullish_percent: float  # 0-100
    bearish_percent: float  # 0-100
    timestamp: str

    @property
    def sentiment_score(self) -> float:
        """Normalized sentiment score from -100 to 100."""
        return self.company_news_score * 100

    @property
    def relative_buzz(self) -> float:
        """Buzz relative to weekly average (1.0 = normal)."""
        if self.buzz_weekly_average > 0:
            return self.buzz_articles_week / self.buzz_weekly_average
        return 1.0

    @property
    def is_bullish(self) -> bool:
        """True if sentiment is bullish."""
        return self.bullish_percent > self.bearish_percent

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "buzzArticlesWeek": self.buzz_articles_week,
            "buzzWeeklyAverage": self.buzz_weekly_average,
            "companyNewsScore": self.company_news_score,
            "sectorAverageScore": self.sector_average_score,
            "bullishPercent": self.bullish_percent,
            "bearishPercent": self.bearish_percent,
            "sentimentScore": self.sentiment_score,
            "relativeBuzz": round(self.relative_buzz, 2),
            "isBullish": self.is_bullish,
            "timestamp": self.timestamp,
        }


@dataclass
class SocialSentiment:
    """Social media sentiment data for a symbol.

    From Finnhub /stock/social-sentiment endpoint.
    Aggregates hourly Reddit/Twitter mentions and sentiment.
    """

    symbol: str
    total_mentions: int  # Sum of hourly mentions
    avg_positive_score: float  # Average positive score
    avg_negative_score: float  # Average negative score
    net_score: float  # Average net sentiment (-1 to 1)
    hours_of_data: int  # Number of hourly data points
    timestamp: str

    @property
    def sentiment_score(self) -> float:
        """Normalized sentiment score from -100 to 100."""
        return self.net_score * 100

    @property
    def is_bullish(self) -> bool:
        """True if net sentiment is positive."""
        return self.net_score > 0.1

    @property
    def is_bearish(self) -> bool:
        """True if net sentiment is negative."""
        return self.net_score < -0.1

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "totalMentions": self.total_mentions,
            "avgPositiveScore": round(self.avg_positive_score, 3),
            "avgNegativeScore": round(self.avg_negative_score, 3),
            "netScore": round(self.net_score, 3),
            "sentimentScore": round(self.sentiment_score, 1),
            "hoursOfData": self.hours_of_data,
            "isBullish": self.is_bullish,
            "isBearish": self.is_bearish,
            "timestamp": self.timestamp,
        }


@dataclass
class CompanyNews:
    """Single news article for a company."""

    symbol: str
    headline: str
    summary: str
    source: str
    url: str
    category: str
    datetime: int  # Unix timestamp

    @property
    def published_at(self) -> str:
        """ISO timestamp of publication."""
        return datetime.fromtimestamp(self.datetime, tz=timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "category": self.category,
            "publishedAt": self.published_at,
        }


@dataclass
class FinnhubClient:
    """Client for Finnhub API.

    Usage:
        config = FinnhubConfig(api_key="...")
        client = FinnhubClient(config)

        sentiment = await client.get_news_sentiment("NVDA")
        news = await client.get_company_news("NVDA")
    """

    config: FinnhubConfig

    # Rate limiting
    _requests_per_second: float = 1.0  # Free tier limit
    _last_request_time: float = field(default=0, init=False)

    # Cache
    _sentiment_cache: dict[str, tuple[NewsSentiment, float]] = field(
        default_factory=dict, init=False
    )
    _social_cache: dict[str, tuple[SocialSentiment | None, float]] = field(
        default_factory=dict, init=False
    )
    _cache_ttl: float = 300.0  # 5 minute cache

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        min_interval = 1.0 / self._requests_per_second

        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self._last_request_time = asyncio.get_event_loop().time()

    async def _request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated request to Finnhub API."""
        await self._rate_limit()

        url = f"{self.config.base_url}/{endpoint}"
        params["token"] = self.config.api_key

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    logger.warning("Finnhub rate limit hit")
                    raise Exception("Rate limit exceeded")

                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Finnhub API error: {response.status} - {text}")
                    raise Exception(f"API error: {response.status}")

                return await response.json()

    async def get_news_sentiment(
        self,
        symbol: str,
        use_cache: bool = True,
    ) -> NewsSentiment | None:
        """Get news sentiment for a symbol.

        Args:
            symbol: Stock symbol (e.g., "NVDA")
            use_cache: Whether to use cached data

        Returns:
            NewsSentiment or None if unavailable
        """
        now = asyncio.get_event_loop().time()

        # Check cache
        if use_cache and symbol in self._sentiment_cache:
            cached, cached_time = self._sentiment_cache[symbol]
            if now - cached_time < self._cache_ttl:
                logger.debug(f"Using cached sentiment for {symbol}")
                return cached

        try:
            data = await self._request("news-sentiment", {"symbol": symbol})

            if not data or "buzz" not in data:
                logger.warning(f"No sentiment data for {symbol}")
                return None

            sentiment = NewsSentiment(
                symbol=symbol,
                buzz_articles_week=data.get("buzz", {}).get("articlesInLastWeek", 0),
                buzz_weekly_average=data.get("buzz", {}).get("weeklyAverage", 0),
                company_news_score=data.get("companyNewsScore", 0),
                sector_average_score=data.get("sectorAverageNewsScore", 0),
                bullish_percent=data.get("sentiment", {}).get("bullishPercent", 50),
                bearish_percent=data.get("sentiment", {}).get("bearishPercent", 50),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Cache the result
            self._sentiment_cache[symbol] = (sentiment, now)

            logger.info(
                f"Finnhub sentiment for {symbol}: "
                f"score={sentiment.sentiment_score:.1f}, "
                f"bullish={sentiment.bullish_percent:.0f}%"
            )

            return sentiment

        except Exception as e:
            logger.error(f"Error fetching sentiment for {symbol}: {e}")
            return None

    async def get_social_sentiment(
        self,
        symbol: str,
        use_cache: bool = True,
    ) -> SocialSentiment | None:
        """Get social media sentiment for a symbol.

        Fetches from /stock/social-sentiment endpoint which aggregates
        hourly Reddit/Twitter mentions and sentiment scores.

        Args:
            symbol: Stock symbol (e.g., "NVDA")
            use_cache: Whether to use cached data

        Returns:
            SocialSentiment or None if unavailable
        """
        now = asyncio.get_event_loop().time()

        # Check cache
        if use_cache and symbol in self._social_cache:
            cached, cached_time = self._social_cache[symbol]
            if now - cached_time < self._cache_ttl:
                logger.debug(f"Using cached social sentiment for {symbol}")
                return cached

        try:
            data = await self._request("stock/social-sentiment", {"symbol": symbol})

            if not data or "data" not in data or not data["data"]:
                logger.debug(f"No social sentiment data for {symbol}")
                # Cache the None result to avoid repeated API calls
                self._social_cache[symbol] = (None, now)
                return None

            # Aggregate hourly data points
            hourly_data = data["data"]
            total_mentions = sum(h.get("mention", 0) for h in hourly_data)

            # Calculate average scores
            positive_scores = [h.get("positiveScore", 0) for h in hourly_data]
            negative_scores = [h.get("negativeScore", 0) for h in hourly_data]
            net_scores = [h.get("score", 0) for h in hourly_data]

            avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0
            avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0
            net_score = sum(net_scores) / len(net_scores) if net_scores else 0

            social = SocialSentiment(
                symbol=symbol,
                total_mentions=total_mentions,
                avg_positive_score=avg_positive,
                avg_negative_score=avg_negative,
                net_score=net_score,
                hours_of_data=len(hourly_data),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Cache the result
            self._social_cache[symbol] = (social, now)

            logger.info(
                f"Finnhub social for {symbol}: "
                f"mentions={total_mentions}, "
                f"net_score={net_score:.2f}"
            )

            return social

        except Exception as e:
            logger.error(f"Error fetching social sentiment for {symbol}: {e}")
            return None

    async def get_company_news(
        self,
        symbol: str,
        days: int = 7,
    ) -> list[CompanyNews]:
        """Get recent company news.

        Args:
            symbol: Stock symbol
            days: Number of days of news to fetch

        Returns:
            List of CompanyNews items
        """
        try:
            from_date = datetime.now(timezone.utc)
            to_date = from_date
            from_str = (from_date.replace(day=from_date.day - days)).strftime("%Y-%m-%d")
            to_str = to_date.strftime("%Y-%m-%d")

            data = await self._request(
                "company-news",
                {"symbol": symbol, "from": from_str, "to": to_str}
            )

            if not data:
                return []

            news = [
                CompanyNews(
                    symbol=symbol,
                    headline=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    source=item.get("source", ""),
                    url=item.get("url", ""),
                    category=item.get("category", ""),
                    datetime=item.get("datetime", 0),
                )
                for item in data[:20]  # Limit to 20 articles
            ]

            logger.info(f"Fetched {len(news)} news articles for {symbol}")
            return news

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    async def get_sentiment_batch(
        self,
        symbols: list[str],
    ) -> dict[str, NewsSentiment | None]:
        """Get sentiment for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to NewsSentiment
        """
        results = {}

        for symbol in symbols:
            results[symbol] = await self.get_news_sentiment(symbol)
            # Small delay between requests for rate limiting
            await asyncio.sleep(0.5)

        return results
