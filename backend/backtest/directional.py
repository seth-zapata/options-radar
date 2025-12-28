"""Directional backtest using free historical data.

Tests whether WSB sentiment signals predict price direction correctly.
Uses:
- Free historical price data from Alpaca
- Locally collected WSB sentiment data (run sentiment_collector daily)

This doesn't simulate exact options P&L, but validates if the core
directional thesis works: "bullish sentiment -> price goes up"

IMPORTANT: Historical WSB data from Quiver requires premium subscription.
Instead, run the sentiment collector daily to build local history:
    python -m backend.backtest.sentiment_collector

Then run backtest on collected data:
    python -m backend.run_backtest
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from backend.config import AlpacaConfig, QuiverConfig, load_config
from backend.data.alpaca_rest import AlpacaRestClient, BarData
from backend.data.quiver_client import QuiverClient, WSBSentiment
from backend.backtest.sentiment_collector import load_sentiment_history, get_storage_stats

logger = logging.getLogger(__name__)


@dataclass
class SignalEvent:
    """A sentiment signal event for backtesting."""

    date: str  # YYYY-MM-DD
    symbol: str
    sentiment: float  # -1 to 1
    mentions: int
    signal: Literal["bullish", "bearish", "neutral"]

    # Price data (filled in during backtest)
    entry_price: float | None = None
    exit_price_1d: float | None = None
    exit_price_5d: float | None = None

    @property
    def direction_correct_1d(self) -> bool | None:
        """Was the 1-day direction prediction correct?"""
        if self.entry_price is None or self.exit_price_1d is None:
            return None

        if self.signal == "neutral":
            return None

        moved_up = self.exit_price_1d > self.entry_price
        return (self.signal == "bullish" and moved_up) or \
               (self.signal == "bearish" and not moved_up)

    @property
    def direction_correct_5d(self) -> bool | None:
        """Was the 5-day direction prediction correct?"""
        if self.entry_price is None or self.exit_price_5d is None:
            return None

        if self.signal == "neutral":
            return None

        moved_up = self.exit_price_5d > self.entry_price
        return (self.signal == "bullish" and moved_up) or \
               (self.signal == "bearish" and not moved_up)

    @property
    def return_1d(self) -> float | None:
        """1-day return percentage."""
        if self.entry_price is None or self.exit_price_1d is None:
            return None
        return ((self.exit_price_1d - self.entry_price) / self.entry_price) * 100

    @property
    def return_5d(self) -> float | None:
        """5-day return percentage."""
        if self.entry_price is None or self.exit_price_5d is None:
            return None
        return ((self.exit_price_5d - self.entry_price) / self.entry_price) * 100


@dataclass
class BacktestResult:
    """Results from a directional backtest."""

    symbol: str
    start_date: str
    end_date: str
    total_signals: int
    bullish_signals: int
    bearish_signals: int

    # 1-day accuracy
    accuracy_1d: float  # 0-100%
    correct_1d: int
    incorrect_1d: int

    # 5-day accuracy
    accuracy_5d: float  # 0-100%
    correct_5d: int
    incorrect_5d: int

    # Return analysis
    avg_return_when_bullish_1d: float | None
    avg_return_when_bearish_1d: float | None
    avg_return_when_bullish_5d: float | None
    avg_return_when_bearish_5d: float | None

    # Individual signals for analysis
    signals: list[SignalEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "symbol": self.symbol,
            "startDate": self.start_date,
            "endDate": self.end_date,
            "totalSignals": self.total_signals,
            "bullishSignals": self.bullish_signals,
            "bearishSignals": self.bearish_signals,
            "accuracy1d": round(self.accuracy_1d, 1),
            "correct1d": self.correct_1d,
            "incorrect1d": self.incorrect_1d,
            "accuracy5d": round(self.accuracy_5d, 1),
            "correct5d": self.correct_5d,
            "incorrect5d": self.incorrect_5d,
            "avgReturnBullish1d": round(self.avg_return_when_bullish_1d, 2)
            if self.avg_return_when_bullish_1d else None,
            "avgReturnBearish1d": round(self.avg_return_when_bearish_1d, 2)
            if self.avg_return_when_bearish_1d else None,
            "avgReturnBullish5d": round(self.avg_return_when_bullish_5d, 2)
            if self.avg_return_when_bullish_5d else None,
            "avgReturnBearish5d": round(self.avg_return_when_bearish_5d, 2)
            if self.avg_return_when_bearish_5d else None,
        }


@dataclass
class DirectionalBacktest:
    """Run directional backtests using free historical data.

    Usage:
        config = load_config()
        backtest = DirectionalBacktest(config.alpaca, config.quiver)

        result = await backtest.run(
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-01",
        )

        print(f"1-day accuracy: {result.accuracy_1d}%")
        print(f"5-day accuracy: {result.accuracy_5d}%")
    """

    alpaca_config: AlpacaConfig
    quiver_config: QuiverConfig

    # Signal thresholds
    bullish_threshold: float = 0.1  # sentiment > 0.1 = bullish
    bearish_threshold: float = -0.1  # sentiment < -0.1 = bearish
    min_mentions: int = 10  # minimum mentions to consider signal valid

    _alpaca_client: AlpacaRestClient = field(default=None, init=False)
    _quiver_client: QuiverClient = field(default=None, init=False)

    def __post_init__(self):
        self._alpaca_client = AlpacaRestClient(config=self.alpaca_config)
        self._quiver_client = QuiverClient(config=self.quiver_config)

    async def run(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Run a directional backtest for a symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            BacktestResult with accuracy metrics
        """
        logger.info(f"Running directional backtest for {symbol} from {start_date} to {end_date}")

        # Fetch historical sentiment data
        sentiment_data = await self._fetch_historical_sentiment(symbol, start_date, end_date)

        if not sentiment_data:
            logger.warning(f"No sentiment data found for {symbol}")
            return self._empty_result(symbol, start_date, end_date)

        # Fetch historical price data
        # Need extra days at the end for exit prices
        extended_end = (
            datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=10)
        ).strftime("%Y-%m-%d")
        price_data = await self._alpaca_client.get_bars(
            symbol, start_date, extended_end, "1Day"
        )

        if not price_data:
            logger.warning(f"No price data found for {symbol}")
            return self._empty_result(symbol, start_date, end_date)

        # Build price lookup by date
        price_by_date = {
            bar.timestamp[:10]: bar.close for bar in price_data
        }

        # Generate signals from sentiment data
        signals = self._generate_signals(sentiment_data, price_by_date)

        # Calculate results
        return self._calculate_results(symbol, start_date, end_date, signals)

    async def _fetch_historical_sentiment(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Fetch historical WSB sentiment data from local storage.

        Historical data must be collected using the sentiment_collector:
            python -m backend.backtest.sentiment_collector

        Falls back to current sentiment if no local history available.
        """
        # Load from local storage (built up by daily sentiment_collector runs)
        local_history = load_sentiment_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        if local_history:
            # Convert to expected format
            result = []
            for record in local_history:
                result.append({
                    "date": record.get("date"),
                    "sentiment": record.get("sentiment", 0),
                    "mentions": record.get("mentions_24h", 0),
                })
            logger.info(f"Loaded {len(result)} local sentiment records for {symbol}")
            return result

        # No local history - try to get current sentiment as a single data point
        logger.warning(
            f"No local sentiment history for {symbol}. "
            f"Run 'python -m backend.backtest.sentiment_collector' daily to build history."
        )

        try:
            current = await self._quiver_client.get_wsb_sentiment(symbol)
            if current:
                return [{
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "sentiment": current.sentiment,
                    "mentions": current.mentions_24h,
                }]
        except Exception as e:
            logger.debug(f"Could not fetch current sentiment: {e}")

        return []

    def _generate_signals(
        self,
        sentiment_data: list[dict[str, Any]],
        price_by_date: dict[str, float],
    ) -> list[SignalEvent]:
        """Generate trading signals from sentiment data."""
        signals = []
        sorted_dates = sorted(price_by_date.keys())

        for item in sentiment_data:
            date = item["date"]
            sentiment = item["sentiment"]
            mentions = item["mentions"]

            # Skip if below minimum mentions
            if mentions < self.min_mentions:
                continue

            # Determine signal direction
            if sentiment > self.bullish_threshold:
                signal_type = "bullish"
            elif sentiment < self.bearish_threshold:
                signal_type = "bearish"
            else:
                signal_type = "neutral"

            # Get entry price (close of signal day)
            entry_price = price_by_date.get(date)
            if entry_price is None:
                continue

            # Get exit prices
            try:
                date_idx = sorted_dates.index(date)
            except ValueError:
                continue

            exit_1d = price_by_date.get(
                sorted_dates[date_idx + 1] if date_idx + 1 < len(sorted_dates) else None
            )
            exit_5d = price_by_date.get(
                sorted_dates[date_idx + 5] if date_idx + 5 < len(sorted_dates) else None
            )

            signals.append(SignalEvent(
                date=date,
                symbol=item.get("symbol", ""),
                sentiment=sentiment,
                mentions=mentions,
                signal=signal_type,
                entry_price=entry_price,
                exit_price_1d=exit_1d,
                exit_price_5d=exit_5d,
            ))

        return signals

    def _calculate_results(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        signals: list[SignalEvent],
    ) -> BacktestResult:
        """Calculate backtest results from signals."""
        # Count signals
        bullish = [s for s in signals if s.signal == "bullish"]
        bearish = [s for s in signals if s.signal == "bearish"]

        # Calculate 1-day accuracy
        correct_1d = sum(1 for s in signals if s.direction_correct_1d is True)
        incorrect_1d = sum(1 for s in signals if s.direction_correct_1d is False)
        total_1d = correct_1d + incorrect_1d
        accuracy_1d = (correct_1d / total_1d * 100) if total_1d > 0 else 0

        # Calculate 5-day accuracy
        correct_5d = sum(1 for s in signals if s.direction_correct_5d is True)
        incorrect_5d = sum(1 for s in signals if s.direction_correct_5d is False)
        total_5d = correct_5d + incorrect_5d
        accuracy_5d = (correct_5d / total_5d * 100) if total_5d > 0 else 0

        # Calculate average returns
        bullish_returns_1d = [s.return_1d for s in bullish if s.return_1d is not None]
        bearish_returns_1d = [s.return_1d for s in bearish if s.return_1d is not None]
        bullish_returns_5d = [s.return_5d for s in bullish if s.return_5d is not None]
        bearish_returns_5d = [s.return_5d for s in bearish if s.return_5d is not None]

        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_signals=len(signals),
            bullish_signals=len(bullish),
            bearish_signals=len(bearish),
            accuracy_1d=accuracy_1d,
            correct_1d=correct_1d,
            incorrect_1d=incorrect_1d,
            accuracy_5d=accuracy_5d,
            correct_5d=correct_5d,
            incorrect_5d=incorrect_5d,
            avg_return_when_bullish_1d=(
                sum(bullish_returns_1d) / len(bullish_returns_1d)
                if bullish_returns_1d else None
            ),
            avg_return_when_bearish_1d=(
                sum(bearish_returns_1d) / len(bearish_returns_1d)
                if bearish_returns_1d else None
            ),
            avg_return_when_bullish_5d=(
                sum(bullish_returns_5d) / len(bullish_returns_5d)
                if bullish_returns_5d else None
            ),
            avg_return_when_bearish_5d=(
                sum(bearish_returns_5d) / len(bearish_returns_5d)
                if bearish_returns_5d else None
            ),
            signals=signals,
        )

    def _empty_result(self, symbol: str, start: str, end: str) -> BacktestResult:
        """Create an empty result when no data available."""
        return BacktestResult(
            symbol=symbol,
            start_date=start,
            end_date=end,
            total_signals=0,
            bullish_signals=0,
            bearish_signals=0,
            accuracy_1d=0,
            correct_1d=0,
            incorrect_1d=0,
            accuracy_5d=0,
            correct_5d=0,
            incorrect_5d=0,
            avg_return_when_bullish_1d=None,
            avg_return_when_bearish_1d=None,
            avg_return_when_bullish_5d=None,
            avg_return_when_bearish_5d=None,
        )

    async def run_multi(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, BacktestResult]:
        """Run backtest for multiple symbols.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping symbol to BacktestResult
        """
        results = {}

        for symbol in symbols:
            result = await self.run(symbol, start_date, end_date)
            results[symbol] = result

        return results


def summarize_results(results: dict[str, BacktestResult]) -> dict[str, Any]:
    """Summarize backtest results across multiple symbols.

    Args:
        results: Dict of symbol -> BacktestResult

    Returns:
        Summary statistics
    """
    all_signals = sum(r.total_signals for r in results.values())
    all_correct_1d = sum(r.correct_1d for r in results.values())
    all_incorrect_1d = sum(r.incorrect_1d for r in results.values())
    all_correct_5d = sum(r.correct_5d for r in results.values())
    all_incorrect_5d = sum(r.incorrect_5d for r in results.values())

    total_1d = all_correct_1d + all_incorrect_1d
    total_5d = all_correct_5d + all_incorrect_5d

    return {
        "symbolCount": len(results),
        "totalSignals": all_signals,
        "overallAccuracy1d": round(
            all_correct_1d / total_1d * 100, 1
        ) if total_1d > 0 else 0,
        "overallAccuracy5d": round(
            all_correct_5d / total_5d * 100, 1
        ) if total_5d > 0 else 0,
        "bySymbol": {
            symbol: {
                "accuracy1d": round(r.accuracy_1d, 1),
                "accuracy5d": round(r.accuracy_5d, 1),
                "signals": r.total_signals,
            }
            for symbol, r in results.items()
        },
    }


async def run_backtest_cli(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Run backtest from command line.

    Args:
        symbols: Symbols to test (default: QQQ, SPY, AAPL)
        start_date: Start date (default: 6 months ago)
        end_date: End date (default: yesterday)
    """
    config = load_config()

    if not symbols:
        symbols = ["QQQ", "SPY", "AAPL"]

    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if not start_date:
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    # Check local sentiment storage
    stats = get_storage_stats()
    print(f"\nSentiment Data Status:")
    if stats["records"] > 0:
        dr = stats.get("date_range", {})
        print(f"  Records: {stats['records']}")
        print(f"  Date Range: {dr.get('start')} to {dr.get('end')} ({dr.get('days', 0)} days)")
        print(f"  Symbols: {', '.join(stats.get('symbols', []))}")
    else:
        print("  No local sentiment history found!")
        print("")
        print("  To build history, run daily:")
        print("    python -m backend.backtest.sentiment_collector")
        print("")
        print("  For now, will use current sentiment only (single data point).")

    print(f"\nRunning directional backtest...")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print("-" * 50)

    backtest = DirectionalBacktest(
        alpaca_config=config.alpaca,
        quiver_config=config.quiver,
    )

    results = await backtest.run_multi(symbols, start_date, end_date)
    summary = summarize_results(results)

    print(f"\nOverall Results:")
    print(f"  Total Signals: {summary['totalSignals']}")
    print(f"  1-Day Accuracy: {summary['overallAccuracy1d']}%")
    print(f"  5-Day Accuracy: {summary['overallAccuracy5d']}%")

    print(f"\nBy Symbol:")
    for symbol, data in summary["bySymbol"].items():
        print(f"  {symbol}:")
        print(f"    Signals: {data['signals']}")
        print(f"    1-Day: {data['accuracy1d']}%")
        print(f"    5-Day: {data['accuracy5d']}%")

    # Interpretation
    print("\n" + "=" * 50)
    overall = summary["overallAccuracy1d"]
    if overall >= 60:
        print("STRONG SIGNAL: Sentiment appears to predict direction well.")
    elif overall >= 55:
        print("MODERATE SIGNAL: Some predictive value, but with noise.")
    elif overall >= 50:
        print("WEAK SIGNAL: Near random, limited predictive value.")
    else:
        print("CONTRARIAN SIGNAL: Inverse correlation detected!")


if __name__ == "__main__":
    asyncio.run(run_backtest_cli())
