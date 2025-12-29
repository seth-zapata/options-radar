"""Technical analysis indicators for signal confirmation.

Calculates RSI, SMA trend, volume confirmation, Bollinger Bands, and MACD
from price history. These are used as soft confidence modifiers, not hard gates.
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

    # Bollinger Bands (20-day SMA, 2 std dev)
    bb_upper: float | None  # Upper band
    bb_lower: float | None  # Lower band
    bb_signal: Literal["above_upper", "below_lower", "within_bands"] | None

    # MACD (12/26 EMA, 9-day signal)
    macd_line: float | None  # MACD line (12 EMA - 26 EMA)
    macd_signal_line: float | None  # 9-day EMA of MACD
    macd_histogram: float | None  # MACD - Signal
    macd_signal: Literal["bullish_cross", "bearish_cross", "bullish_momentum", "bearish_momentum"] | None

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
            Confidence modifier (-15 to +25)
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

        # Bollinger Bands - MOMENTUM signal (not mean reversion!)
        # 910-signal backtest: momentum following works better than mean reversion
        # Above upper band + bullish = 63.5% accuracy vs below lower + bullish = 40.9%
        if self.bb_signal is not None:
            if is_bullish_signal:
                if self.bb_signal == "above_upper":  # Strong momentum, ride the trend
                    modifier += 5
                elif self.bb_signal == "below_lower":  # Weak momentum, risky for calls
                    modifier -= 5
            else:  # Bearish signal
                if self.bb_signal == "below_lower":  # Strong downward momentum
                    modifier += 5
                elif self.bb_signal == "above_upper":  # Strong upward momentum, risky for puts
                    modifier -= 5

        # MACD momentum alignment
        if self.macd_signal is not None:
            if is_bullish_signal:
                if self.macd_signal in ("bullish_cross", "bullish_momentum"):
                    modifier += 5
                elif self.macd_signal in ("bearish_cross", "bearish_momentum"):
                    modifier -= 5
            else:  # Bearish signal
                if self.macd_signal in ("bearish_cross", "bearish_momentum"):
                    modifier += 5
                elif self.macd_signal in ("bullish_cross", "bullish_momentum"):
                    modifier -= 5

        return modifier

    def get_indicator_alignment(self, is_bullish_signal: bool) -> dict[str, str]:
        """Get alignment status for each indicator (for backtesting).

        Returns dict with indicator names and 'aligned', 'misaligned', or 'neutral'.
        """
        alignment = {}

        # RSI
        if self.rsi is not None:
            if is_bullish_signal:
                if self.rsi < 30:
                    alignment["rsi"] = "aligned"
                elif self.rsi > 70:
                    alignment["rsi"] = "misaligned"
                else:
                    alignment["rsi"] = "neutral"
            else:
                if self.rsi > 70:
                    alignment["rsi"] = "aligned"
                elif self.rsi < 30:
                    alignment["rsi"] = "misaligned"
                else:
                    alignment["rsi"] = "neutral"

        # Trend (SMA)
        if self.trend_signal is not None:
            if is_bullish_signal:
                alignment["trend"] = "aligned" if self.trend_signal == "above_sma" else "misaligned"
            else:
                alignment["trend"] = "aligned" if self.trend_signal == "below_sma" else "misaligned"

        # Volume
        if self.volume_ratio is not None:
            alignment["volume"] = "aligned" if self.volume_ratio > 1.5 else "neutral"

        # Bollinger Bands (momentum-based, not mean reversion)
        if self.bb_signal is not None:
            if is_bullish_signal:
                if self.bb_signal == "above_upper":  # Strong momentum
                    alignment["bollinger"] = "aligned"
                elif self.bb_signal == "below_lower":  # Weak momentum
                    alignment["bollinger"] = "misaligned"
                else:
                    alignment["bollinger"] = "neutral"
            else:  # Bearish
                if self.bb_signal == "below_lower":  # Strong downward momentum
                    alignment["bollinger"] = "aligned"
                elif self.bb_signal == "above_upper":  # Strong upward momentum
                    alignment["bollinger"] = "misaligned"
                else:
                    alignment["bollinger"] = "neutral"

        # MACD
        if self.macd_signal is not None:
            if is_bullish_signal:
                if self.macd_signal in ("bullish_cross", "bullish_momentum"):
                    alignment["macd"] = "aligned"
                elif self.macd_signal in ("bearish_cross", "bearish_momentum"):
                    alignment["macd"] = "misaligned"
                else:
                    alignment["macd"] = "neutral"
            else:
                if self.macd_signal in ("bearish_cross", "bearish_momentum"):
                    alignment["macd"] = "aligned"
                elif self.macd_signal in ("bullish_cross", "bullish_momentum"):
                    alignment["macd"] = "misaligned"
                else:
                    alignment["macd"] = "neutral"

        return alignment

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

        if self.bb_signal is not None:
            bb_status = {
                "above_upper": "Above Upper Band",
                "below_lower": "Below Lower Band",
                "within_bands": "Within Bands",
            }.get(self.bb_signal, "Unknown")
            tags.append(f"BB: {bb_status}")

        if self.macd_signal is not None:
            macd_status = {
                "bullish_cross": "Bullish Cross",
                "bearish_cross": "Bearish Cross",
                "bullish_momentum": "Bullish",
                "bearish_momentum": "Bearish",
            }.get(self.macd_signal, "Unknown")
            tags.append(f"MACD: {macd_status}")

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
            "bbUpper": round(self.bb_upper, 2) if self.bb_upper else None,
            "bbLower": round(self.bb_lower, 2) if self.bb_lower else None,
            "bbSignal": self.bb_signal,
            "macdLine": round(self.macd_line, 4) if self.macd_line else None,
            "macdSignalLine": round(self.macd_signal_line, 4) if self.macd_signal_line else None,
            "macdHistogram": round(self.macd_histogram, 4) if self.macd_histogram else None,
            "macdSignal": self.macd_signal,
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


def calculate_ema(prices: list[float], period: int) -> float | None:
    """Calculate Exponential Moving Average.

    Args:
        prices: List of prices (oldest first)
        period: EMA period

    Returns:
        EMA value or None if insufficient data
    """
    if len(prices) < period:
        return None

    # Start with SMA for initial EMA value
    ema = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)

    # Calculate EMA for remaining prices
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def calculate_bollinger_bands(
    closes: list[float], period: int = 20, num_std: float = 2.0
) -> tuple[float | None, float | None, float | None]:
    """Calculate Bollinger Bands.

    Args:
        closes: List of closing prices (oldest first)
        period: SMA period (default 20)
        num_std: Number of standard deviations (default 2)

    Returns:
        Tuple of (upper_band, middle_band/SMA, lower_band), or (None, None, None) if insufficient data
    """
    if len(closes) < period:
        return None, None, None

    # Calculate SMA (middle band)
    recent = closes[-period:]
    sma = sum(recent) / period

    # Calculate standard deviation
    variance = sum((x - sma) ** 2 for x in recent) / period
    std_dev = variance ** 0.5

    upper_band = sma + (num_std * std_dev)
    lower_band = sma - (num_std * std_dev)

    return upper_band, sma, lower_band


def calculate_macd(
    closes: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[float | None, float | None, float | None]:
    """Calculate MACD indicator.

    Args:
        closes: List of closing prices (oldest first)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram), or (None, None, None) if insufficient data
    """
    # Need enough data for slow EMA + signal period
    if len(closes) < slow_period + signal_period:
        return None, None, None

    # Calculate MACD line values for each point (for signal line calculation)
    macd_values = []
    for i in range(slow_period, len(closes) + 1):
        subset = closes[:i]
        fast_ema = calculate_ema(subset, fast_period)
        slow_ema = calculate_ema(subset, slow_period)
        if fast_ema is not None and slow_ema is not None:
            macd_values.append(fast_ema - slow_ema)

    if len(macd_values) < signal_period:
        return None, None, None

    # Current MACD line
    macd_line = macd_values[-1]

    # Signal line is EMA of MACD values
    signal_line = calculate_ema(macd_values, signal_period)

    if signal_line is None:
        return macd_line, None, None

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


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

            # Fetch 3 months of data (need 35+ for MACD: 26 + 9 = 35)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")

            if hist.empty or len(hist) < 35:
                logger.warning(f"Insufficient price data for {symbol}")
                return None

            closes = hist["Close"].tolist()
            volumes = hist["Volume"].tolist()

            # Calculate basic indicators
            rsi = calculate_rsi(closes, 14)
            sma_20 = calculate_sma(closes, 20)
            current_price = closes[-1]
            current_volume = int(volumes[-1])
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None

            # Calculate Bollinger Bands
            bb_upper, _, bb_lower = calculate_bollinger_bands(closes, 20, 2.0)

            # Calculate MACD
            macd_line, macd_signal_line, macd_histogram = calculate_macd(closes)

            # Also get previous MACD values to detect crossovers
            prev_macd_line, prev_signal_line, _ = calculate_macd(closes[:-1])

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

            # Bollinger Bands signal
            bb_signal = None
            if bb_upper is not None and bb_lower is not None and current_price is not None:
                if current_price > bb_upper:
                    bb_signal = "above_upper"
                elif current_price < bb_lower:
                    bb_signal = "below_lower"
                else:
                    bb_signal = "within_bands"

            # MACD signal - detect crossovers and momentum
            macd_signal = None
            if macd_line is not None and macd_signal_line is not None:
                if prev_macd_line is not None and prev_signal_line is not None:
                    # Check for crossovers
                    prev_diff = prev_macd_line - prev_signal_line
                    curr_diff = macd_line - macd_signal_line
                    if prev_diff < 0 and curr_diff >= 0:
                        macd_signal = "bullish_cross"
                    elif prev_diff > 0 and curr_diff <= 0:
                        macd_signal = "bearish_cross"
                    elif macd_histogram is not None:
                        # No crossover, just momentum
                        macd_signal = "bullish_momentum" if macd_histogram > 0 else "bearish_momentum"
                elif macd_histogram is not None:
                    macd_signal = "bullish_momentum" if macd_histogram > 0 else "bearish_momentum"

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
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_signal=bb_signal,
                macd_line=macd_line,
                macd_signal_line=macd_signal_line,
                macd_histogram=macd_histogram,
                macd_signal=macd_signal,
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
