"""Combined sentiment aggregator for multiple data sources.

Combines two complementary sentiment sources:
1. News sentiment (Finnhub) - Media/analyst sentiment, potential trigger/catalyst
2. Social/Retail sentiment (Quiver WSB) - Retail momentum, confirmation/risk overlay

These sources are complementary:
- News shows the catalyst (earnings, upgrades, breaking news)
- WSB shows retail reaction and momentum
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.config import AppConfig
from backend.data.finnhub_client import FinnhubClient, NewsSentiment
from backend.data.quiver_client import QuiverClient, WSBSentiment

logger = logging.getLogger(__name__)


@dataclass
class CombinedSentiment:
    """Combined sentiment from two complementary sources.

    Two sentiment categories:
    - News: Traditional media/analyst sentiment (catalyst/trigger)
    - Social: WallStreetBets retail sentiment (confirmation/risk overlay)
    """

    symbol: str
    timestamp: str

    # News sentiment (Finnhub)
    news_sentiment: NewsSentiment | None = None

    # Social/Retail sentiment (Quiver WSB)
    wsb_sentiment: WSBSentiment | None = None

    @property
    def news_score(self) -> float:
        """News sentiment score (-100 to 100)."""
        if self.news_sentiment:
            return self.news_sentiment.sentiment_score
        return 0.0

    @property
    def wsb_score(self) -> float:
        """WSB/retail sentiment score (-100 to 100)."""
        if self.wsb_sentiment:
            return self.wsb_sentiment.sentiment_score
        return 0.0

    @property
    def combined_score(self) -> float:
        """Overall combined sentiment score (-100 to 100).

        Dynamic weighting based on data availability:
        - News: 50% (catalyst, professional analysis)
        - WSB/Social: 50% (retail momentum, confirmation)
        """
        weights = {"news": 0.50, "wsb": 0.50}
        scores = {}
        total_weight = 0.0

        if self.news_sentiment is not None:
            scores["news"] = self.news_score
            total_weight += weights["news"]

        if self.wsb_sentiment is not None:
            scores["wsb"] = self.wsb_score
            total_weight += weights["wsb"]

        if total_weight == 0:
            return 0.0

        # Normalize weights and calculate combined score
        weighted_sum = sum(
            scores[k] * weights[k] / total_weight
            for k in scores
        )

        return weighted_sum

    @property
    def signal(self) -> str:
        """Sentiment signal: 'bullish', 'bearish', or 'neutral'."""
        score = self.combined_score
        if score >= 20:
            return "bullish"
        elif score <= -20:
            return "bearish"
        return "neutral"

    @property
    def strength(self) -> str:
        """Signal strength: 'strong', 'moderate', or 'weak'."""
        score = abs(self.combined_score)
        if score >= 50:
            return "strong"
        elif score >= 25:
            return "moderate"
        return "weak"

    @property
    def news_is_buzzing(self) -> bool:
        """True if there's above-average news buzz."""
        if self.news_sentiment:
            return self.news_sentiment.relative_buzz > 1.5
        return False

    @property
    def wsb_is_trending(self) -> bool:
        """True if symbol is trending on WallStreetBets."""
        if self.wsb_sentiment:
            return self.wsb_sentiment.is_trending
        return False

    @property
    def wsb_is_bullish(self) -> bool:
        """True if WSB sentiment is bullish."""
        if self.wsb_sentiment:
            return self.wsb_sentiment.is_bullish
        return False

    @property
    def sources_aligned(self) -> bool:
        """True if both sources agree on direction."""
        if not self.news_sentiment or not self.wsb_sentiment:
            return False

        # Both bullish or both bearish
        news_bullish = self.news_score > 10
        wsb_bullish = self.wsb_score > 10
        news_bearish = self.news_score < -10
        wsb_bearish = self.wsb_score < -10

        return (news_bullish and wsb_bullish) or (news_bearish and wsb_bearish)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "scores": {
                "news": round(self.news_score, 1),
                "wsb": round(self.wsb_score, 1),
                "combined": round(self.combined_score, 1),
            },
            "signal": self.signal,
            "strength": self.strength,
            "flags": {
                "newsBuzzing": self.news_is_buzzing,
                "wsbTrending": self.wsb_is_trending,
                "wsbBullish": self.wsb_is_bullish,
                "sourcesAligned": self.sources_aligned,
            },
            "newsSentiment": self.news_sentiment.to_dict() if self.news_sentiment else None,
            "wsbSentiment": self.wsb_sentiment.to_dict() if self.wsb_sentiment else None,
        }


@dataclass
class SentimentAggregator:
    """Aggregates sentiment from two complementary sources.

    Data sources:
    1. Finnhub - News/media sentiment (catalyst/trigger)
    2. Quiver WSB - WallStreetBets retail sentiment (confirmation)

    Usage:
        config = load_config()
        aggregator = SentimentAggregator(config)

        sentiment = await aggregator.get_sentiment("NVDA")
        print(sentiment.signal)  # "bullish", "bearish", or "neutral"
    """

    config: AppConfig

    # Clients
    _finnhub_client: FinnhubClient | None = field(default=None, init=False)
    _quiver_client: QuiverClient | None = field(default=None, init=False)

    # Cache
    _cache: dict[str, tuple[CombinedSentiment, float]] = field(
        default_factory=dict, init=False
    )
    _cache_ttl: float = 300.0  # 5 minute cache

    def __post_init__(self):
        """Initialize clients based on available API keys."""
        if self.config.finnhub.api_key:
            self._finnhub_client = FinnhubClient(self.config.finnhub)
            logger.info("Finnhub client initialized (news sentiment)")
        else:
            logger.warning("Finnhub API key not configured")

        if self.config.quiver.api_key:
            self._quiver_client = QuiverClient(self.config.quiver)
            logger.info("Quiver client initialized (WSB sentiment)")
        else:
            logger.warning("Quiver API key not configured")

    async def get_sentiment(
        self,
        symbol: str,
        use_cache: bool = True,
    ) -> CombinedSentiment:
        """Get combined sentiment for a symbol.

        Fetches from two sources:
        - Finnhub news sentiment (catalyst/trigger)
        - Quiver WSB sentiment (confirmation/risk overlay)

        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data

        Returns:
            CombinedSentiment with aggregated data
        """
        now = asyncio.get_event_loop().time()

        # Check cache
        if use_cache and symbol in self._cache:
            cached, cached_time = self._cache[symbol]
            if now - cached_time < self._cache_ttl:
                logger.debug(f"Using cached sentiment for {symbol}")
                return cached

        # Build tasks for concurrent fetching
        tasks = []
        task_names = []

        if self._finnhub_client:
            tasks.append(self._finnhub_client.get_news_sentiment(symbol))
            task_names.append("news")

        if self._quiver_client:
            tasks.append(self._quiver_client.get_wsb_sentiment(symbol))
            task_names.append("wsb")

        # Await all tasks concurrently
        results = {}
        if tasks:
            fetched = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(task_names, fetched):
                if isinstance(result, Exception):
                    logger.warning(f"Error fetching {name} sentiment: {result}")
                    results[name] = None
                else:
                    results[name] = result

        # Build combined sentiment
        sentiment = CombinedSentiment(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            news_sentiment=results.get("news"),
            wsb_sentiment=results.get("wsb"),
        )

        # Cache result
        self._cache[symbol] = (sentiment, now)

        # Log summary
        sources = []
        if sentiment.news_sentiment:
            sources.append(f"news={sentiment.news_score:.0f}")
        if sentiment.wsb_sentiment:
            sources.append(f"wsb={sentiment.wsb_score:.0f}")

        logger.info(
            f"Sentiment for {symbol}: "
            f"combined={sentiment.combined_score:.0f} "
            f"({sentiment.signal}/{sentiment.strength}) "
            f"[{', '.join(sources)}]"
        )

        return sentiment

    async def get_sentiment_batch(
        self,
        symbols: list[str],
    ) -> dict[str, CombinedSentiment]:
        """Get sentiment for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to CombinedSentiment
        """
        results = {}

        for symbol in symbols:
            results[symbol] = await self.get_sentiment(symbol)

        return results

    async def get_wsb_trending(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get trending symbols on WallStreetBets.

        Args:
            limit: Maximum symbols to return

        Returns:
            List of trending WSB symbols with sentiment
        """
        if not self._quiver_client:
            return []

        try:
            trending = await self._quiver_client.get_wsb_trending(limit=limit)

            return [
                {
                    "symbol": wsb.symbol,
                    "mentions24h": wsb.mentions_24h,
                    "sentiment": round(wsb.sentiment, 3),
                    "sentimentScore": round(wsb.sentiment_score, 1),
                    "rank": wsb.rank,
                    "buzzLevel": wsb.buzz_level,
                    "isBullish": wsb.is_bullish,
                }
                for wsb in trending
            ]

        except Exception as e:
            logger.error(f"Error fetching WSB trending: {e}")
            return []
