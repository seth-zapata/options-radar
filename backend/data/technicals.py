"""Technical analysis indicators for signal confirmation.

Calculates RSI, SMA trend, and volume confirmation from price history.
These are used as soft confidence modifiers, not hard gates.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TechnicalIndicators:
    """Technical analysis indicators for a symbol.

    All indicators are calculated from daily price data.
    """

    symbol: str
    timestamp: str

    # RSI (14-period)
    rsi: float | None  # 0-100, None if insufficient data
    rsi_signal: Literal["oversold", "overbought", "neutral"] | None

    # 20-day SMA trend
    sma_20: float | None
    current_price: float | None
    trend_signal: Literal["above_sma", "below_sma"] | None

    # Volume confirmation
    current_volume: int | None
    avg_volume_20: float | None
    volume_ratio: float | None  # current / avg
    volume_signal: Literal["high_volume", "normal"] | None

    @property
    def confidence_modifier(self) -> int:
        """Calculate total confidence modifier from technicals.

        This is a base modifier that doesn't consider sentiment direction.
        The pipeline should use get_directional_modifier() instead.
        """
        return 0  # Use get_directional_modifier with sentiment direction

    def get_directional_modifier(self, is_bullish_signal: bool) -> int:
        """Calculate confidence modifier based on signal direction.

        Based on backtest of 508 WSB sentiment signals (2021-2024):
        - Overall technical alignment adds ~8% accuracy edge
        - Trend alignment: +8.6% edge (reliable)
        - RSI alignment: inconsistent across timeframes (boost only)
        - Volume: small sample, boost only

        Args:
            is_bullish_signal: True for BUY_CALL/SELL_PUT, False for BUY_PUT/SELL_CALL

        Returns:
            Confidence modifier (-10 to +15)
        """
        modifier = 0

        # RSI alignment - BOOST ONLY (no penalty)
        # RSI edge was inconsistent across timeframes, not reliable enough to penalize
        if self.rsi is not None:
            if is_bullish_signal:
                if self.rsi < 30:  # Oversold + bullish = bounce opportunity
                    modifier += 5
                # No penalty for overbought + bullish
            else:  # Bearish signal
                if self.rsi > 70:  # Overbought + bearish = breakdown
                    modifier += 5
                # No penalty for oversold + bearish

        # Trend alignment - KEEP BOTH BOOST AND PENALTY
        # 508-signal backtest showed +8.6% edge for trend alignment
        if self.trend_signal is not None:
            if is_bullish_signal:
                if self.trend_signal == "above_sma":  # With trend
                    modifier += 5
                else:  # Against trend
                    modifier -= 5
            else:  # Bearish signal
                if self.trend_signal == "below_sma":  # With trend
                    modifier += 5
                else:  # Against trend
                    modifier -= 5

        # Volume confirmation - BOOST ONLY (no penalty)
        if self.volume_ratio is not None and self.volume_ratio > 1.5:
            modifier += 5

        return modifier

    def get_summary_tags(self) -> list[str]:
        """Get summary tags for display."""
        tags = []

        if self.rsi is not None:
            status = "Oversold" if self.rsi < 30 else "Overbought" if self.rsi > 70 else "Neutral"
            tags.append(f"RSI: {self.rsi:.0f} - {status}")

        if self.trend_signal is not None:
            trend = "Above SMA" if self.trend_signal == "above_sma" else "Below SMA"
            tags.append(f"TREND: {trend}")

        if self.volume_ratio is not None:
            tags.append(f"VOLUME: {self.volume_ratio:.1f}x avg")

        return tags

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "rsi": round(self.rsi, 1) if self.rsi else None,
            "rsiSignal": self.rsi_signal,
            "sma20": round(self.sma_20, 2) if self.sma_20 else None,
            "currentPrice": round(self.current_price, 2) if self.current_price else None,
            "trendSignal": self.trend_signal,
            "currentVolume": self.current_volume,
            "avgVolume20": round(self.avg_volume_20, 0) if self.avg_volume_20 else None,
            "volumeRatio": round(self.volume_ratio, 2) if self.volume_ratio else None,
            "volumeSignal": self.volume_signal,
            "summaryTags": self.get_summary_tags(),
        }


def calculate_rsi(closes: list[float], period: int = 14) -> float | None:
    """Calculate RSI from closing prices.

    Args:
        closes: List of closing prices (oldest first)
        period: RSI period (default 14)

    Returns:
        RSI value 0-100, or None if insufficient data
    """
    if len(closes) < period + 1:
        return None

    # Calculate price changes
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Separate gains and losses
    gains = [max(0, c) for c in changes]
    losses = [abs(min(0, c)) for c in changes]

    # Calculate initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Smooth with Wilder's method
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_sma(prices: list[float], period: int = 20) -> float | None:
    """Calculate Simple Moving Average.

    Args:
        prices: List of prices (oldest first)
        period: SMA period

    Returns:
        SMA value or None if insufficient data
    """
    if len(prices) < period:
        return None

    return sum(prices[-period:]) / period


@dataclass
class TechnicalAnalyzer:
    """Fetches price data and calculates technical indicators.

    Uses yfinance for historical data (free, no API key needed).
    Can be extended to use Alpaca for real-time data.
    """

    _cache: dict[str, tuple[TechnicalIndicators, float]] = field(
        default_factory=dict, init=False
    )
    _cache_ttl: float = 300.0  # 5 minutes

    async def get_indicators(
        self,
        symbol: str,
        use_cache: bool = True,
    ) -> TechnicalIndicators | None:
        """Get technical indicators for a symbol.

        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data

        Returns:
            TechnicalIndicators or None if data unavailable
        """
        now = asyncio.get_event_loop().time()

        # Check cache
        if use_cache and symbol in self._cache:
            cached, cached_time = self._cache[symbol]
            if now - cached_time < self._cache_ttl:
                return cached

        try:
            # Fetch data using yfinance (runs in thread pool)
            indicators = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_and_calculate, symbol
            )

            if indicators:
                self._cache[symbol] = (indicators, now)
                logger.info(
                    f"Technicals for {symbol}: "
                    f"RSI={indicators.rsi:.1f if indicators.rsi else 'N/A'}, "
                    f"Trend={indicators.trend_signal}, "
                    f"Vol={indicators.volume_ratio:.1f}x" if indicators.volume_ratio else "Vol=N/A"
                )

            return indicators

        except Exception as e:
            logger.error(f"Error fetching technicals for {symbol}: {e}")
            return None

    def _fetch_and_calculate(self, symbol: str) -> TechnicalIndicators | None:
        """Synchronous fetch and calculate (runs in thread pool)."""
        try:
            import yfinance as yf

            # Fetch 30 days of data (need 20 for SMA + some buffer)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")

            if hist.empty or len(hist) < 20:
                logger.warning(f"Insufficient price data for {symbol}")
                return None

            closes = hist["Close"].tolist()
            volumes = hist["Volume"].tolist()

            # Calculate indicators
            rsi = calculate_rsi(closes, 14)
            sma_20 = calculate_sma(closes, 20)
            current_price = closes[-1]
            current_volume = int(volumes[-1])
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None

            # Determine signals
            rsi_signal = None
            if rsi is not None:
                if rsi < 30:
                    rsi_signal = "oversold"
                elif rsi > 70:
                    rsi_signal = "overbought"
                else:
                    rsi_signal = "neutral"

            trend_signal = None
            if sma_20 is not None and current_price is not None:
                trend_signal = "above_sma" if current_price > sma_20 else "below_sma"

            volume_ratio = None
            volume_signal = None
            if current_volume and avg_volume:
                volume_ratio = current_volume / avg_volume
                volume_signal = "high_volume" if volume_ratio > 1.5 else "normal"

            return TechnicalIndicators(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc).isoformat(),
                rsi=rsi,
                rsi_signal=rsi_signal,
                sma_20=sma_20,
                current_price=current_price,
                trend_signal=trend_signal,
                current_volume=current_volume,
                avg_volume_20=avg_volume,
                volume_ratio=volume_ratio,
                volume_signal=volume_signal,
            )

        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return None
        except Exception as e:
            logger.error(f"Error calculating technicals for {symbol}: {e}")
            return None

    async def get_indicators_batch(
        self,
        symbols: list[str],
    ) -> dict[str, TechnicalIndicators | None]:
        """Get indicators for multiple symbols."""
        results = {}
        for symbol in symbols:
            results[symbol] = await self.get_indicators(symbol)
        return results
