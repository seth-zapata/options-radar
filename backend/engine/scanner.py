"""Daily opportunity scanner for options radar.

Scans the watchlist for high-conviction trading opportunities by combining:
1. Sentiment signals (news, political, social)
2. IV rank conditions
3. Smart money indicators

Use Case: Run daily before market open to identify top opportunities.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.config import AppConfig
from backend.data.sentiment_aggregator import CombinedSentiment, SentimentAggregator

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of scanning a single symbol."""

    symbol: str
    score: float  # Opportunity score (0-100)
    sentiment: CombinedSentiment | None
    signals: list[str]  # Human-readable signal descriptions
    timestamp: str

    @property
    def is_opportunity(self) -> bool:
        """True if this is a noteworthy opportunity."""
        return self.score >= 50

    @property
    def is_strong_opportunity(self) -> bool:
        """True if this is a strong opportunity."""
        return self.score >= 75

    @property
    def direction(self) -> str:
        """Suggested direction: 'bullish', 'bearish', or 'neutral'."""
        if self.sentiment:
            return self.sentiment.signal
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "score": round(self.score, 1),
            "direction": self.direction,
            "isOpportunity": self.is_opportunity,
            "isStrongOpportunity": self.is_strong_opportunity,
            "signals": self.signals,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "timestamp": self.timestamp,
        }


@dataclass
class DailyScanner:
    """Scans watchlist for trading opportunities.

    Usage:
        config = load_config()
        scanner = DailyScanner(config)

        results = await scanner.scan()
        for result in results:
            if result.is_opportunity:
                print(f"{result.symbol}: {result.score} - {result.signals}")
    """

    config: AppConfig
    _sentiment_aggregator: SentimentAggregator | None = field(default=None, init=False)

    def __post_init__(self):
        """Initialize the sentiment aggregator."""
        self._sentiment_aggregator = SentimentAggregator(self.config)

    async def scan_symbol(self, symbol: str) -> ScanResult:
        """Scan a single symbol for opportunity signals.

        Args:
            symbol: Stock symbol to scan

        Returns:
            ScanResult with opportunity score and signals
        """
        now = datetime.now(timezone.utc)
        signals: list[str] = []
        score = 0.0

        # Get sentiment data
        sentiment = None
        if self._sentiment_aggregator:
            try:
                sentiment = await self._sentiment_aggregator.get_sentiment(symbol)
            except Exception as e:
                logger.warning(f"Failed to get sentiment for {symbol}: {e}")

        if sentiment:
            # Score based on combined sentiment strength
            combined_score = sentiment.combined_score
            if abs(combined_score) >= 50:
                score += 35
                signals.append(f"Strong sentiment ({combined_score:.0f})")
            elif abs(combined_score) >= 25:
                score += 20
                signals.append(f"Moderate sentiment ({combined_score:.0f})")

            # WSB trending bonus
            if sentiment.wsb_is_trending:
                if sentiment.wsb_is_bullish:
                    score += 15
                    signals.append("WSB trending bullish")
                else:
                    score += 10
                    signals.append("WSB trending")

            # News buzz bonus (catalyst present)
            if sentiment.news_is_buzzing:
                score += 15
                signals.append("High news buzz (catalyst)")

            # Sources aligned bonus (news + WSB agree)
            if sentiment.sources_aligned:
                score += 20
                signals.append("News + WSB aligned")

        # Cap score at 100
        score = min(score, 100)

        return ScanResult(
            symbol=symbol,
            score=score,
            sentiment=sentiment,
            signals=signals,
            timestamp=now.isoformat(),
        )

    async def scan(
        self,
        symbols: list[str] | None = None,
    ) -> list[ScanResult]:
        """Scan symbols for opportunities.

        Args:
            symbols: Optional list of symbols to scan. Uses enabled_symbols if not provided.

        Returns:
            List of ScanResult sorted by score (highest first)
        """
        if symbols is None:
            # Only scan regime-enabled symbols (TSLA) to avoid wasting API quota
            symbols = list(self.config.regime_strategy.enabled_symbols)

        logger.info(f"Scanning {len(symbols)} symbols for opportunities...")

        # Scan all symbols concurrently with rate limiting
        results = []
        for symbol in symbols:
            try:
                result = await self.scan_symbol(symbol)
                results.append(result)
                logger.info(
                    f"Scanned {symbol}: score={result.score:.0f}, "
                    f"signals={len(result.signals)}"
                )
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        # Sort by score (highest first)
        results.sort(key=lambda r: r.score, reverse=True)

        # Log summary
        opportunities = [r for r in results if r.is_opportunity]
        strong = [r for r in results if r.is_strong_opportunity]
        logger.info(
            f"Scan complete: {len(results)} symbols, "
            f"{len(opportunities)} opportunities, "
            f"{len(strong)} strong opportunities"
        )

        return results

    async def get_top_opportunities(
        self,
        limit: int = 5,
    ) -> list[ScanResult]:
        """Get top N opportunities from the watchlist.

        Args:
            limit: Maximum number of opportunities to return

        Returns:
            List of top ScanResult objects
        """
        results = await self.scan()
        return [r for r in results[:limit] if r.is_opportunity]

    async def get_hot_picks(self) -> dict[str, list[dict[str, Any]]]:
        """Get curated hot picks from various sources.

        Returns:
            Dict with 'wsbTrending' and 'topOpportunities'
        """
        result = {
            "wsbTrending": [],
            "topOpportunities": [],
        }

        if not self._sentiment_aggregator:
            return result

        try:
            # Get WSB trending
            result["wsbTrending"] = await self._sentiment_aggregator.get_wsb_trending(limit=5)
        except Exception as e:
            logger.warning(f"Failed to get WSB trending: {e}")

        try:
            # Get top opportunities from scan
            top = await self.get_top_opportunities(limit=5)
            result["topOpportunities"] = [t.to_dict() for t in top]
        except Exception as e:
            logger.warning(f"Failed to get top opportunities: {e}")

        return result
