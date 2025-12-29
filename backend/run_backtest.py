#!/usr/bin/env python3
"""Backtest framework using ACTUAL WSB sentiment data from Quiver.

Usage:
    python -m backend.run_backtest --symbols TSLA,NVDA,PLTR --start 2024-01-01

This backtest:
1. Fetches actual historical WSB sentiment from Quiver API
2. Generates bullish signals when WSB sentiment > 0.1
3. Calculates technical indicators at each signal point
4. Compares: Do signals with aligned technicals outperform misaligned?
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import aiohttp
import yfinance as yf

from backend.data.technicals import calculate_rsi, calculate_sma
from backend.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class WSBDataPoint:
    """Single day of WSB data from Quiver."""
    date: str
    mentions: int
    rank: int
    sentiment: float  # -1 to 1


@dataclass
class SignalResult:
    """Result of a sentiment-based signal."""
    symbol: str
    date: str
    signal_type: Literal["BUY_CALL", "BUY_PUT"]
    entry_price: float
    exit_price: float
    price_change_pct: float
    
    # WSB sentiment at signal time
    wsb_sentiment: float
    wsb_mentions: int
    wsb_rank: int
    
    # Technical indicators
    rsi: float | None
    trend_signal: Literal["above_sma", "below_sma"] | None
    volume_ratio: float | None
    
    # Alignment flags
    rsi_aligned: bool
    trend_aligned: bool
    high_volume: bool
    tech_modifier: int
    
    @property
    def was_profitable(self) -> bool:
        """True if signal direction was correct."""
        if self.signal_type == "BUY_CALL":
            return self.price_change_pct > 0
        else:
            return self.price_change_pct < 0


@dataclass
class BacktestStats:
    """Statistics for a backtest run."""
    total_signals: int = 0
    correct_signals: int = 0
    
    # By technical alignment
    aligned_signals: int = 0  # All technicals aligned
    aligned_correct: int = 0
    
    misaligned_signals: int = 0  # At least one technical against
    misaligned_correct: int = 0
    
    # Individual technical factors
    rsi_aligned_correct: int = 0
    rsi_aligned_total: int = 0
    rsi_against_correct: int = 0
    rsi_against_total: int = 0
    
    trend_aligned_correct: int = 0
    trend_aligned_total: int = 0
    trend_against_correct: int = 0
    trend_against_total: int = 0
    
    high_vol_correct: int = 0
    high_vol_total: int = 0
    
    @property
    def accuracy(self) -> float:
        if self.total_signals == 0:
            return 0.0
        return (self.correct_signals / self.total_signals) * 100
    
    @property
    def aligned_accuracy(self) -> float:
        if self.aligned_signals == 0:
            return 0.0
        return (self.aligned_correct / self.aligned_signals) * 100
    
    @property
    def misaligned_accuracy(self) -> float:
        if self.misaligned_signals == 0:
            return 0.0
        return (self.misaligned_correct / self.misaligned_signals) * 100


async def fetch_wsb_history(symbol: str, api_key: str) -> list[WSBDataPoint]:
    """Fetch historical WSB data from Quiver API."""
    url = f"https://api.quiverquant.com/beta/historical/wallstreetbets/{symbol}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                logger.warning(f"Failed to fetch WSB data for {symbol}: {response.status}")
                return []
            
            data = await response.json()
            
            if not data or not isinstance(data, list):
                return []
            
            result = []
            for item in data:
                try:
                    result.append(WSBDataPoint(
                        date=item.get("Date", ""),
                        mentions=int(item.get("Mentions", 0) or 0),
                        rank=int(item.get("Rank", 999) or 999),
                        sentiment=float(item.get("Sentiment", 0) or 0),
                    ))
                except (ValueError, TypeError):
                    continue
            
            logger.info(f"Fetched {len(result)} days of WSB data for {symbol}")
            return result


def get_price_data(symbol: str, start: str, end: str) -> dict:
    """Fetch historical price data from yfinance."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end)
    
    if hist.empty:
        return {}
    
    # Convert to dict with date string keys
    result = {}
    for date, row in hist.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        result[date_str] = {
            "close": row["Close"],
            "volume": row["Volume"],
        }
    
    return result


def calculate_technicals_for_date(
    prices: dict,
    target_date: str,
) -> tuple[float | None, str | None, float | None]:
    """Calculate RSI, trend, volume ratio for a specific date."""
    # Get sorted dates up to target
    all_dates = sorted(prices.keys())
    if target_date not in all_dates:
        return None, None, None
    
    target_idx = all_dates.index(target_date)
    if target_idx < 20:  # Need 20 days for SMA
        return None, None, None
    
    # Get last 30 days of data
    hist_dates = all_dates[max(0, target_idx - 30):target_idx + 1]
    closes = [prices[d]["close"] for d in hist_dates]
    volumes = [prices[d]["volume"] for d in hist_dates]
    
    # Calculate RSI
    rsi = calculate_rsi(closes, 14)
    
    # Calculate SMA and trend
    sma_20 = calculate_sma(closes, 20)
    current_price = closes[-1]
    trend_signal = None
    if sma_20 is not None:
        trend_signal = "above_sma" if current_price > sma_20 else "below_sma"
    
    # Calculate volume ratio
    avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None
    volume_ratio = volumes[-1] / avg_vol if avg_vol and volumes[-1] > 0 else None
    
    return rsi, trend_signal, volume_ratio


def get_tech_modifier(
    rsi: float | None,
    trend_signal: str | None,
    volume_ratio: float | None,
    is_bullish: bool,
) -> tuple[int, bool, bool, bool]:
    """Calculate tech modifier and alignment flags.

    Based on 508-signal backtest (2021-2024):
    - RSI: boost only (no penalty) - inconsistent across timeframes
    - Trend: keep boost and penalty - +8.6% edge
    - Volume: boost only
    """
    modifier = 0
    rsi_aligned = False
    trend_aligned = False
    high_vol = False

    # RSI alignment - BOOST ONLY (no penalty)
    if rsi is not None:
        if is_bullish:
            if rsi < 30:
                modifier += 5
                rsi_aligned = True
            # No penalty for overbought + bullish
        else:
            if rsi > 70:
                modifier += 5
                rsi_aligned = True
            # No penalty for oversold + bearish
    
    # Trend alignment
    if trend_signal is not None:
        if is_bullish:
            if trend_signal == "above_sma":
                modifier += 5
                trend_aligned = True
            else:
                modifier -= 5
        else:
            if trend_signal == "below_sma":
                modifier += 5
                trend_aligned = True
            else:
                modifier -= 5
    
    # Volume confirmation
    if volume_ratio is not None and volume_ratio > 1.5:
        modifier += 5
        high_vol = True
    
    return modifier, rsi_aligned, trend_aligned, high_vol


async def run_backtest(
    symbols: list[str],
    api_key: str,
    start_date: str,
    holding_days: int = 5,
    min_mentions: int = 5,
    sentiment_threshold: float = 0.1,
) -> tuple[BacktestStats, list[SignalResult]]:
    """Run backtest using actual WSB sentiment data."""
    
    stats = BacktestStats()
    all_results: list[SignalResult] = []
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        
        # Fetch WSB history
        wsb_data = await fetch_wsb_history(symbol, api_key)
        if not wsb_data:
            logger.warning(f"No WSB data for {symbol}")
            continue
        
        # Filter to start date
        wsb_data = [w for w in wsb_data if w.date >= start_date]
        
        # Get price data (extend range for technicals + holding period)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)
        end_dt = datetime.now()
        prices = get_price_data(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        
        if not prices:
            logger.warning(f"No price data for {symbol}")
            continue
        
        # Generate signals from WSB data
        for wsb in wsb_data:
            # Skip if not enough mentions
            if wsb.mentions < min_mentions:
                continue
            
            # Skip neutral sentiment
            if abs(wsb.sentiment) < sentiment_threshold:
                continue
            
            # Determine signal direction
            is_bullish = wsb.sentiment > 0
            signal_type = "BUY_CALL" if is_bullish else "BUY_PUT"
            
            # Get entry price
            if wsb.date not in prices:
                continue
            entry_price = prices[wsb.date]["close"]
            
            # Get exit price (holding_days later)
            signal_date = datetime.strptime(wsb.date, "%Y-%m-%d")
            exit_date = signal_date + timedelta(days=holding_days)
            exit_date_str = exit_date.strftime("%Y-%m-%d")
            
            # Find closest available exit date
            sorted_dates = sorted(prices.keys())
            exit_candidates = [d for d in sorted_dates if d >= exit_date_str]
            if not exit_candidates:
                continue
            actual_exit_date = exit_candidates[0]
            exit_price = prices[actual_exit_date]["close"]
            
            # Calculate price change
            price_change_pct = ((exit_price - entry_price) / entry_price) * 100
            
            # Calculate technicals
            rsi, trend_signal, volume_ratio = calculate_technicals_for_date(prices, wsb.date)
            modifier, rsi_aligned, trend_aligned, high_vol = get_tech_modifier(
                rsi, trend_signal, volume_ratio, is_bullish
            )
            
            result = SignalResult(
                symbol=symbol,
                date=wsb.date,
                signal_type=signal_type,
                entry_price=entry_price,
                exit_price=exit_price,
                price_change_pct=price_change_pct,
                wsb_sentiment=wsb.sentiment,
                wsb_mentions=wsb.mentions,
                wsb_rank=wsb.rank,
                rsi=rsi,
                trend_signal=trend_signal,
                volume_ratio=volume_ratio,
                rsi_aligned=rsi_aligned,
                trend_aligned=trend_aligned,
                high_volume=high_vol,
                tech_modifier=modifier,
            )
            all_results.append(result)
            
            # Update stats
            stats.total_signals += 1
            if result.was_profitable:
                stats.correct_signals += 1
            
            # Track alignment
            all_aligned = (modifier > 0) or (rsi_aligned and trend_aligned)
            if all_aligned:
                stats.aligned_signals += 1
                if result.was_profitable:
                    stats.aligned_correct += 1
            else:
                stats.misaligned_signals += 1
                if result.was_profitable:
                    stats.misaligned_correct += 1
            
            # Individual factors
            if rsi is not None:
                if rsi_aligned:
                    stats.rsi_aligned_total += 1
                    if result.was_profitable:
                        stats.rsi_aligned_correct += 1
                elif (is_bullish and rsi > 70) or (not is_bullish and rsi < 30):
                    stats.rsi_against_total += 1
                    if result.was_profitable:
                        stats.rsi_against_correct += 1
            
            if trend_signal is not None:
                if trend_aligned:
                    stats.trend_aligned_total += 1
                    if result.was_profitable:
                        stats.trend_aligned_correct += 1
                else:
                    stats.trend_against_total += 1
                    if result.was_profitable:
                        stats.trend_against_correct += 1
            
            if high_vol:
                stats.high_vol_total += 1
                if result.was_profitable:
                    stats.high_vol_correct += 1
    
    return stats, all_results


def print_results(stats: BacktestStats) -> None:
    """Print backtest results."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS: WSB Sentiment Signals + Technical Alignment")
    print("=" * 70)
    
    print(f"\nTotal Sentiment-Based Signals: {stats.total_signals}")
    print(f"Overall Accuracy: {stats.accuracy:.1f}%")
    
    print("\n" + "-" * 70)
    print("TECHNICAL ALIGNMENT COMPARISON")
    print("-" * 70)
    print(f"  Technicals ALIGNED:    {stats.aligned_accuracy:.1f}% ({stats.aligned_signals} signals)")
    print(f"  Technicals MISALIGNED: {stats.misaligned_accuracy:.1f}% ({stats.misaligned_signals} signals)")
    
    if stats.aligned_signals > 0 and stats.misaligned_signals > 0:
        delta = stats.aligned_accuracy - stats.misaligned_accuracy
        print(f"  EDGE FROM ALIGNMENT:   {delta:+.1f}%")
    
    print("\n" + "-" * 70)
    print("INDIVIDUAL TECHNICAL FACTORS")
    print("-" * 70)
    
    if stats.rsi_aligned_total > 0 or stats.rsi_against_total > 0:
        rsi_aligned_acc = (stats.rsi_aligned_correct / stats.rsi_aligned_total * 100) if stats.rsi_aligned_total > 0 else 0
        rsi_against_acc = (stats.rsi_against_correct / stats.rsi_against_total * 100) if stats.rsi_against_total > 0 else 0
        print(f"\n  RSI Alignment:")
        print(f"    Aligned (oversold+bullish / overbought+bearish): {rsi_aligned_acc:.1f}% ({stats.rsi_aligned_total} signals)")
        print(f"    Against (overbought+bullish / oversold+bearish): {rsi_against_acc:.1f}% ({stats.rsi_against_total} signals)")
        if stats.rsi_aligned_total > 0 and stats.rsi_against_total > 0:
            print(f"    RSI Edge: {rsi_aligned_acc - rsi_against_acc:+.1f}%")
    
    if stats.trend_aligned_total > 0 or stats.trend_against_total > 0:
        trend_aligned_acc = (stats.trend_aligned_correct / stats.trend_aligned_total * 100) if stats.trend_aligned_total > 0 else 0
        trend_against_acc = (stats.trend_against_correct / stats.trend_against_total * 100) if stats.trend_against_total > 0 else 0
        print(f"\n  Trend Alignment (20-day SMA):")
        print(f"    With Trend:    {trend_aligned_acc:.1f}% ({stats.trend_aligned_total} signals)")
        print(f"    Against Trend: {trend_against_acc:.1f}% ({stats.trend_against_total} signals)")
        if stats.trend_aligned_total > 0 and stats.trend_against_total > 0:
            print(f"    Trend Edge: {trend_aligned_acc - trend_against_acc:+.1f}%")
    
    if stats.high_vol_total > 0:
        high_vol_acc = (stats.high_vol_correct / stats.high_vol_total * 100)
        print(f"\n  High Volume (>1.5x avg): {high_vol_acc:.1f}% ({stats.high_vol_total} signals)")
    
    print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(
        description="Backtest WSB sentiment signals with technical alignment"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="TSLA,NVDA,PLTR,AAPL,AMD",
        help="Comma-separated list of symbols",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--holding",
        type=int,
        default=5,
        help="Holding period in days (default: 5)",
    )
    parser.add_argument(
        "--min-mentions",
        type=int,
        default=5,
        help="Minimum WSB mentions to generate signal (default: 5)",
    )
    parser.add_argument(
        "--sentiment-threshold",
        type=float,
        default=0.1,
        help="Minimum sentiment magnitude to generate signal (default: 0.1)",
    )
    
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Load API key from config
    try:
        config = load_config()
        api_key = config.quiver.api_key
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        api_key = os.environ.get("QUIVER_API_KEY", "")
    
    if not api_key:
        logger.error("No Quiver API key found. Set QUIVER_API_KEY or configure in .env")
        return
    
    print(f"Running backtest for {symbols}")
    print(f"Period: {args.start} to today")
    print(f"Holding period: {args.holding} days")
    print(f"Min mentions: {args.min_mentions}")
    print(f"Sentiment threshold: {args.sentiment_threshold}")
    
    stats, results = await run_backtest(
        symbols=symbols,
        api_key=api_key,
        start_date=args.start,
        holding_days=args.holding,
        min_mentions=args.min_mentions,
        sentiment_threshold=args.sentiment_threshold,
    )
    
    print_results(stats)


if __name__ == "__main__":
    asyncio.run(main())
