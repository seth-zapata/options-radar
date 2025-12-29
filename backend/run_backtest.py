#!/usr/bin/env python3
"""Backtest framework using ACTUAL WSB sentiment data from Quiver.

Usage:
    python -m backend.run_backtest --symbols TSLA,NVDA,PLTR --start 2024-01-01
    python -m backend.run_backtest --symbols TSLA,NVDA --start 2023-01-01 --include-options-indicators

This backtest:
1. Fetches actual historical WSB sentiment from Quiver API
2. Generates bullish signals when WSB sentiment > 0.1
3. Calculates technical indicators at each signal point
4. Optionally fetches options data (Put/Call Ratio, Max Pain) from EODHD
5. Compares: Do signals with aligned technicals/options outperform misaligned?
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

from backend.data.technicals import (
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)
from backend.data.eodhd_client import EODHDClient
from backend.data.iv_rank_calculator import IVRankCalculator
from backend.config import load_config, EODHDConfig

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
    bb_signal: Literal["above_upper", "below_lower", "within_bands"] | None
    macd_signal: Literal["bullish_cross", "bearish_cross", "bullish_momentum", "bearish_momentum"] | None

    # Options indicators (from EODHD)
    put_call_ratio: float | None = None
    max_pain: float | None = None
    pcr_signal: Literal["bullish", "bearish", "neutral"] | None = None
    max_pain_signal: Literal["bullish", "bearish", "neutral"] | None = None

    # IV Rank (calculated from EODHD historical IV)
    iv_rank: float | None = None
    current_iv: float | None = None
    iv_rank_pass: bool = False  # True if IV Rank <= 45% for buys

    # Alignment flags
    rsi_aligned: bool = False
    trend_aligned: bool = False
    high_volume: bool = False
    bb_aligned: bool = False
    macd_aligned: bool = False
    pcr_aligned: bool = False
    max_pain_aligned: bool = False
    tech_modifier: int = 0

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

    # Bollinger Bands
    bb_aligned_correct: int = 0
    bb_aligned_total: int = 0
    bb_against_correct: int = 0
    bb_against_total: int = 0

    # MACD
    macd_aligned_correct: int = 0
    macd_aligned_total: int = 0
    macd_against_correct: int = 0
    macd_against_total: int = 0

    # Put/Call Ratio (options indicator)
    pcr_aligned_correct: int = 0
    pcr_aligned_total: int = 0
    pcr_against_correct: int = 0
    pcr_against_total: int = 0

    # Max Pain (options indicator)
    maxpain_aligned_correct: int = 0
    maxpain_aligned_total: int = 0
    maxpain_against_correct: int = 0
    maxpain_against_total: int = 0

    # Options data availability
    options_data_signals: int = 0

    # IV Rank comparison (key validation metric for ORATS subscription)
    iv_rank_pass_correct: int = 0  # IV Rank <= 45% and profitable
    iv_rank_pass_total: int = 0    # IV Rank <= 45%
    iv_rank_fail_correct: int = 0  # IV Rank > 45% and profitable
    iv_rank_fail_total: int = 0    # IV Rank > 45%
    iv_rank_data_signals: int = 0  # Signals with IV Rank data

    # IV Rank "retail confirmation" analysis (alternative framework)
    # High IV + strong sentiment = retail excitement confirmation
    iv_high_strong_sent_correct: int = 0  # IV > 60% + sentiment > 0.3
    iv_high_strong_sent_total: int = 0
    # Low IV = cheap premium opportunity
    iv_low_correct: int = 0               # IV < 30%
    iv_low_total: int = 0
    # Mid IV = neutral
    iv_mid_correct: int = 0               # IV 30-60%
    iv_mid_total: int = 0
    # High IV + weak sentiment (caution - IV high but weak conviction)
    iv_high_weak_sent_correct: int = 0    # IV > 60% + sentiment <= 0.3
    iv_high_weak_sent_total: int = 0

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


@dataclass
class TechnicalData:
    """Technical indicators for a specific date."""
    rsi: float | None = None
    trend_signal: str | None = None
    volume_ratio: float | None = None
    bb_signal: str | None = None
    macd_signal: str | None = None


def calculate_technicals_for_date(
    prices: dict,
    target_date: str,
) -> TechnicalData:
    """Calculate all technical indicators for a specific date."""
    result = TechnicalData()

    # Get sorted dates up to target
    all_dates = sorted(prices.keys())
    if target_date not in all_dates:
        return result

    target_idx = all_dates.index(target_date)
    if target_idx < 35:  # Need 35 days for MACD (26 + 9)
        return result

    # Get last 50 days of data for all indicators
    hist_dates = all_dates[max(0, target_idx - 50):target_idx + 1]
    closes = [prices[d]["close"] for d in hist_dates]
    volumes = [prices[d]["volume"] for d in hist_dates]

    current_price = closes[-1]

    # Calculate RSI
    result.rsi = calculate_rsi(closes, 14)

    # Calculate SMA and trend
    sma_20 = calculate_sma(closes, 20)
    if sma_20 is not None:
        result.trend_signal = "above_sma" if current_price > sma_20 else "below_sma"

    # Calculate volume ratio
    avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None
    result.volume_ratio = volumes[-1] / avg_vol if avg_vol and volumes[-1] > 0 else None

    # Calculate Bollinger Bands
    bb_upper, _, bb_lower = calculate_bollinger_bands(closes, 20, 2.0)
    if bb_upper is not None and bb_lower is not None:
        if current_price > bb_upper:
            result.bb_signal = "above_upper"
        elif current_price < bb_lower:
            result.bb_signal = "below_lower"
        else:
            result.bb_signal = "within_bands"

    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(closes)
    # Also get previous for crossover detection
    if len(closes) > 1:
        prev_macd, prev_signal, _ = calculate_macd(closes[:-1])
        if macd_line is not None and signal_line is not None:
            if prev_macd is not None and prev_signal is not None:
                prev_diff = prev_macd - prev_signal
                curr_diff = macd_line - signal_line
                if prev_diff < 0 and curr_diff >= 0:
                    result.macd_signal = "bullish_cross"
                elif prev_diff > 0 and curr_diff <= 0:
                    result.macd_signal = "bearish_cross"
                elif histogram is not None:
                    result.macd_signal = "bullish_momentum" if histogram > 0 else "bearish_momentum"
            elif histogram is not None:
                result.macd_signal = "bullish_momentum" if histogram > 0 else "bearish_momentum"

    return result


@dataclass
class OptionsData:
    """Options indicators for a specific date."""
    put_call_ratio: float | None = None
    max_pain: float | None = None
    pcr_signal: str | None = None  # "bullish", "bearish", "neutral"
    max_pain_signal: str | None = None


@dataclass
class AlignmentResult:
    """Alignment result for all indicators."""
    modifier: int = 0
    rsi_aligned: bool = False
    trend_aligned: bool = False
    high_vol: bool = False
    bb_aligned: bool = False
    macd_aligned: bool = False
    pcr_aligned: bool = False
    max_pain_aligned: bool = False


def get_tech_modifier(
    tech: TechnicalData,
    is_bullish: bool,
    options: OptionsData | None = None,
    entry_price: float | None = None,
) -> AlignmentResult:
    """Calculate tech modifier and alignment flags.

    Based on 508-signal backtest (2021-2024):
    - RSI: boost only (no penalty) - inconsistent across timeframes
    - Trend: keep boost and penalty - +8.6% edge
    - Volume: boost only
    - BB and MACD: testing both boost and penalty
    - PCR: contrarian (high P/C = bullish, low P/C = bearish)
    - Max Pain: price below max pain = bullish potential
    """
    result = AlignmentResult()

    # RSI alignment - BOOST ONLY (no penalty)
    if tech.rsi is not None:
        if is_bullish:
            if tech.rsi < 30:
                result.modifier += 5
                result.rsi_aligned = True
        else:
            if tech.rsi > 70:
                result.modifier += 5
                result.rsi_aligned = True

    # Trend alignment
    if tech.trend_signal is not None:
        if is_bullish:
            if tech.trend_signal == "above_sma":
                result.modifier += 5
                result.trend_aligned = True
            else:
                result.modifier -= 5
        else:
            if tech.trend_signal == "below_sma":
                result.modifier += 5
                result.trend_aligned = True
            else:
                result.modifier -= 5

    # Volume confirmation
    if tech.volume_ratio is not None and tech.volume_ratio > 1.5:
        result.modifier += 5
        result.high_vol = True

    # Bollinger Bands - MOMENTUM (not mean reversion!)
    # 910-signal backtest showed momentum following beats mean reversion
    if tech.bb_signal is not None:
        if is_bullish:
            if tech.bb_signal == "above_upper":  # Strong momentum, ride the trend
                result.modifier += 5
                result.bb_aligned = True
            elif tech.bb_signal == "below_lower":  # Weak momentum
                result.modifier -= 5
        else:  # Bearish
            if tech.bb_signal == "below_lower":  # Strong downward momentum
                result.modifier += 5
                result.bb_aligned = True
            elif tech.bb_signal == "above_upper":  # Strong upward momentum
                result.modifier -= 5

    # MACD momentum
    if tech.macd_signal is not None:
        if is_bullish:
            if tech.macd_signal in ("bullish_cross", "bullish_momentum"):
                result.modifier += 5
                result.macd_aligned = True
            elif tech.macd_signal in ("bearish_cross", "bearish_momentum"):
                result.modifier -= 5
        else:  # Bearish
            if tech.macd_signal in ("bearish_cross", "bearish_momentum"):
                result.modifier += 5
                result.macd_aligned = True
            elif tech.macd_signal in ("bullish_cross", "bullish_momentum"):
                result.modifier -= 5

    # Options indicators (if available)
    if options is not None:
        # Put/Call Ratio - CONTRARIAN indicator
        # P/C > 1.2 = excessive bearishness = bullish signal
        # P/C < 0.6 = excessive bullishness = bearish signal
        if options.pcr_signal is not None:
            if is_bullish:
                if options.pcr_signal == "bullish":  # High P/C ratio (contrarian bullish)
                    result.modifier += 5
                    result.pcr_aligned = True
                elif options.pcr_signal == "bearish":  # Low P/C ratio
                    result.modifier -= 5
            else:  # Bearish signal
                if options.pcr_signal == "bearish":  # Low P/C ratio (contrarian bearish)
                    result.modifier += 5
                    result.pcr_aligned = True
                elif options.pcr_signal == "bullish":  # High P/C ratio
                    result.modifier -= 5

        # Max Pain alignment
        # If price is below max pain, tends to get pulled up (bullish)
        # If price is above max pain, tends to get pulled down (bearish)
        if options.max_pain is not None and entry_price is not None:
            if is_bullish:
                if entry_price < options.max_pain:  # Price below max pain = bullish
                    result.modifier += 5
                    result.max_pain_aligned = True
                elif entry_price > options.max_pain:  # Price above max pain
                    result.modifier -= 5
            else:  # Bearish signal
                if entry_price > options.max_pain:  # Price above max pain = bearish
                    result.modifier += 5
                    result.max_pain_aligned = True
                elif entry_price < options.max_pain:  # Price below max pain
                    result.modifier -= 5

    return result


async def run_backtest(
    symbols: list[str],
    api_key: str,
    start_date: str,
    holding_days: int = 5,
    min_mentions: int = 5,
    sentiment_threshold: float = 0.1,
    include_options: bool = False,
    include_iv_rank: bool = False,
    eodhd_api_key: str | None = None,
) -> tuple[BacktestStats, list[SignalResult]]:
    """Run backtest using actual WSB sentiment data.

    Args:
        symbols: List of stock symbols
        api_key: Quiver API key
        start_date: Start date (YYYY-MM-DD)
        holding_days: Number of days to hold position
        min_mentions: Minimum WSB mentions for signal
        sentiment_threshold: Minimum sentiment magnitude
        include_options: Whether to fetch options data (P/C ratio, max pain)
        include_iv_rank: Whether to calculate IV Rank from EODHD historical IV
        eodhd_api_key: EODHD API key for options data and IV Rank
    """

    stats = BacktestStats()
    all_results: list[SignalResult] = []

    # Initialize EODHD client if options indicators or IV Rank enabled
    eodhd_client: EODHDClient | None = None
    iv_rank_calculator: IVRankCalculator | None = None
    if (include_options or include_iv_rank) and eodhd_api_key:
        eodhd_client = EODHDClient(config=EODHDConfig(api_key=eodhd_api_key))
        if include_options:
            logger.info("Options indicators enabled (EODHD)")
        if include_iv_rank:
            iv_rank_calculator = IVRankCalculator(eodhd_client)
            logger.info("IV Rank calculation enabled (EODHD historical IV)")

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
            tech = calculate_technicals_for_date(prices, wsb.date)

            # Fetch options data if enabled
            options_data: OptionsData | None = None
            if eodhd_client:
                try:
                    indicators = await eodhd_client.get_options_indicators(
                        symbol, trade_date=wsb.date, underlying_price=entry_price
                    )
                    if indicators:
                        options_data = OptionsData(
                            put_call_ratio=indicators.put_call_ratio,
                            max_pain=indicators.max_pain,
                            pcr_signal=indicators.pcr_signal,
                            max_pain_signal=None,  # Will compute below
                        )
                        stats.options_data_signals += 1
                except Exception as e:
                    logger.debug(f"No options data for {symbol} on {wsb.date}: {e}")

            # Calculate IV Rank if enabled
            iv_rank_result = None
            if iv_rank_calculator:
                try:
                    iv_rank_result = await iv_rank_calculator.get_iv_rank(
                        symbol, wsb.date, entry_price
                    )
                    if iv_rank_result:
                        logger.debug(
                            f"IV Rank for {symbol} on {wsb.date}: {iv_rank_result.iv_rank:.1f}% "
                            f"(IV={iv_rank_result.current_iv:.2f}, {iv_rank_result.data_points} pts)"
                        )
                except Exception as e:
                    logger.debug(f"No IV Rank data for {symbol} on {wsb.date}: {e}")

            alignment = get_tech_modifier(tech, is_bullish, options_data, entry_price)

            # Compute max pain signal based on price vs max pain
            max_pain_signal = None
            if options_data and options_data.max_pain:
                if entry_price < options_data.max_pain:
                    max_pain_signal = "bullish"
                elif entry_price > options_data.max_pain:
                    max_pain_signal = "bearish"
                else:
                    max_pain_signal = "neutral"

            # Compute IV Rank pass/fail (for buy signals, IV Rank <= 45% is favorable)
            iv_rank_pass = False
            if iv_rank_result and iv_rank_result.iv_rank is not None:
                iv_rank_pass = iv_rank_result.iv_rank <= 45.0

            signal_result = SignalResult(
                symbol=symbol,
                date=wsb.date,
                signal_type=signal_type,
                entry_price=entry_price,
                exit_price=exit_price,
                price_change_pct=price_change_pct,
                wsb_sentiment=wsb.sentiment,
                wsb_mentions=wsb.mentions,
                wsb_rank=wsb.rank,
                rsi=tech.rsi,
                trend_signal=tech.trend_signal,
                volume_ratio=tech.volume_ratio,
                bb_signal=tech.bb_signal,
                macd_signal=tech.macd_signal,
                put_call_ratio=options_data.put_call_ratio if options_data else None,
                max_pain=options_data.max_pain if options_data else None,
                pcr_signal=options_data.pcr_signal if options_data else None,
                max_pain_signal=max_pain_signal,
                iv_rank=iv_rank_result.iv_rank if iv_rank_result else None,
                current_iv=iv_rank_result.current_iv if iv_rank_result else None,
                iv_rank_pass=iv_rank_pass,
                rsi_aligned=alignment.rsi_aligned,
                trend_aligned=alignment.trend_aligned,
                high_volume=alignment.high_vol,
                bb_aligned=alignment.bb_aligned,
                macd_aligned=alignment.macd_aligned,
                pcr_aligned=alignment.pcr_aligned,
                max_pain_aligned=alignment.max_pain_aligned,
                tech_modifier=alignment.modifier,
            )
            all_results.append(signal_result)

            # Update stats
            stats.total_signals += 1
            if signal_result.was_profitable:
                stats.correct_signals += 1

            # Track alignment (positive modifier means aligned)
            if alignment.modifier > 0:
                stats.aligned_signals += 1
                if signal_result.was_profitable:
                    stats.aligned_correct += 1
            else:
                stats.misaligned_signals += 1
                if signal_result.was_profitable:
                    stats.misaligned_correct += 1

            # Individual factors - RSI
            if tech.rsi is not None:
                if alignment.rsi_aligned:
                    stats.rsi_aligned_total += 1
                    if signal_result.was_profitable:
                        stats.rsi_aligned_correct += 1
                elif (is_bullish and tech.rsi > 70) or (not is_bullish and tech.rsi < 30):
                    stats.rsi_against_total += 1
                    if signal_result.was_profitable:
                        stats.rsi_against_correct += 1

            # Trend
            if tech.trend_signal is not None:
                if alignment.trend_aligned:
                    stats.trend_aligned_total += 1
                    if signal_result.was_profitable:
                        stats.trend_aligned_correct += 1
                else:
                    stats.trend_against_total += 1
                    if signal_result.was_profitable:
                        stats.trend_against_correct += 1

            # Volume
            if alignment.high_vol:
                stats.high_vol_total += 1
                if signal_result.was_profitable:
                    stats.high_vol_correct += 1

            # Bollinger Bands (momentum-based)
            if tech.bb_signal is not None:
                if alignment.bb_aligned:
                    stats.bb_aligned_total += 1
                    if signal_result.was_profitable:
                        stats.bb_aligned_correct += 1
                elif tech.bb_signal != "within_bands":
                    # Against: bullish but below lower, or bearish but above upper
                    is_against = (is_bullish and tech.bb_signal == "below_lower") or \
                                (not is_bullish and tech.bb_signal == "above_upper")
                    if is_against:
                        stats.bb_against_total += 1
                        if signal_result.was_profitable:
                            stats.bb_against_correct += 1

            # MACD
            if tech.macd_signal is not None:
                if alignment.macd_aligned:
                    stats.macd_aligned_total += 1
                    if signal_result.was_profitable:
                        stats.macd_aligned_correct += 1
                else:
                    # Check if actually against (opposite momentum)
                    macd_against = (is_bullish and tech.macd_signal in ("bearish_cross", "bearish_momentum")) or \
                                  (not is_bullish and tech.macd_signal in ("bullish_cross", "bullish_momentum"))
                    if macd_against:
                        stats.macd_against_total += 1
                        if signal_result.was_profitable:
                            stats.macd_against_correct += 1

            # Put/Call Ratio (options indicator)
            if options_data and options_data.pcr_signal:
                if alignment.pcr_aligned:
                    stats.pcr_aligned_total += 1
                    if signal_result.was_profitable:
                        stats.pcr_aligned_correct += 1
                elif options_data.pcr_signal != "neutral":
                    # Against: bullish signal but low P/C, or bearish signal but high P/C
                    pcr_against = (is_bullish and options_data.pcr_signal == "bearish") or \
                                 (not is_bullish and options_data.pcr_signal == "bullish")
                    if pcr_against:
                        stats.pcr_against_total += 1
                        if signal_result.was_profitable:
                            stats.pcr_against_correct += 1

            # Max Pain (options indicator)
            if options_data and options_data.max_pain:
                if alignment.max_pain_aligned:
                    stats.maxpain_aligned_total += 1
                    if signal_result.was_profitable:
                        stats.maxpain_aligned_correct += 1
                else:
                    # Against: bullish but price above max pain, or bearish but price below
                    if max_pain_signal and max_pain_signal != "neutral":
                        mp_against = (is_bullish and max_pain_signal == "bearish") or \
                                    (not is_bullish and max_pain_signal == "bullish")
                        if mp_against:
                            stats.maxpain_against_total += 1
                            if signal_result.was_profitable:
                                stats.maxpain_against_correct += 1

            # IV Rank (key metric for validating ORATS subscription)
            if signal_result.iv_rank is not None:
                stats.iv_rank_data_signals += 1
                if signal_result.iv_rank_pass:  # IV Rank <= 45%
                    stats.iv_rank_pass_total += 1
                    if signal_result.was_profitable:
                        stats.iv_rank_pass_correct += 1
                else:  # IV Rank > 45%
                    stats.iv_rank_fail_total += 1
                    if signal_result.was_profitable:
                        stats.iv_rank_fail_correct += 1

                # "Retail confirmation" analysis (alternative framework for directional buyers)
                iv_rank = signal_result.iv_rank
                wsb_sent = abs(signal_result.wsb_sentiment)  # Use absolute value for strength

                if iv_rank > 60:
                    # High IV - check sentiment strength
                    if wsb_sent > 0.3:
                        # High IV + strong sentiment = retail excitement confirmation
                        stats.iv_high_strong_sent_total += 1
                        if signal_result.was_profitable:
                            stats.iv_high_strong_sent_correct += 1
                    else:
                        # High IV + weak sentiment = caution
                        stats.iv_high_weak_sent_total += 1
                        if signal_result.was_profitable:
                            stats.iv_high_weak_sent_correct += 1
                elif iv_rank < 30:
                    # Low IV = cheap premium
                    stats.iv_low_total += 1
                    if signal_result.was_profitable:
                        stats.iv_low_correct += 1
                else:
                    # Mid IV (30-60) = neutral
                    stats.iv_mid_total += 1
                    if signal_result.was_profitable:
                        stats.iv_mid_correct += 1

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

    # RSI
    if stats.rsi_aligned_total > 0 or stats.rsi_against_total > 0:
        rsi_aligned_acc = (stats.rsi_aligned_correct / stats.rsi_aligned_total * 100) if stats.rsi_aligned_total > 0 else 0
        rsi_against_acc = (stats.rsi_against_correct / stats.rsi_against_total * 100) if stats.rsi_against_total > 0 else 0
        print(f"\n  RSI Alignment:")
        print(f"    Aligned (oversold+bullish / overbought+bearish): {rsi_aligned_acc:.1f}% ({stats.rsi_aligned_total} signals)")
        print(f"    Against (overbought+bullish / oversold+bearish): {rsi_against_acc:.1f}% ({stats.rsi_against_total} signals)")
        if stats.rsi_aligned_total > 0 and stats.rsi_against_total > 0:
            print(f"    RSI Edge: {rsi_aligned_acc - rsi_against_acc:+.1f}%")

    # Trend/SMA
    if stats.trend_aligned_total > 0 or stats.trend_against_total > 0:
        trend_aligned_acc = (stats.trend_aligned_correct / stats.trend_aligned_total * 100) if stats.trend_aligned_total > 0 else 0
        trend_against_acc = (stats.trend_against_correct / stats.trend_against_total * 100) if stats.trend_against_total > 0 else 0
        print(f"\n  Trend Alignment (20-day SMA):")
        print(f"    With Trend:    {trend_aligned_acc:.1f}% ({stats.trend_aligned_total} signals)")
        print(f"    Against Trend: {trend_against_acc:.1f}% ({stats.trend_against_total} signals)")
        if stats.trend_aligned_total > 0 and stats.trend_against_total > 0:
            print(f"    Trend Edge: {trend_aligned_acc - trend_against_acc:+.1f}%")

    # Volume
    if stats.high_vol_total > 0:
        high_vol_acc = (stats.high_vol_correct / stats.high_vol_total * 100)
        print(f"\n  High Volume (>1.5x avg): {high_vol_acc:.1f}% ({stats.high_vol_total} signals)")

    # Bollinger Bands (momentum-based)
    if stats.bb_aligned_total > 0 or stats.bb_against_total > 0:
        bb_aligned_acc = (stats.bb_aligned_correct / stats.bb_aligned_total * 100) if stats.bb_aligned_total > 0 else 0
        bb_against_acc = (stats.bb_against_correct / stats.bb_against_total * 100) if stats.bb_against_total > 0 else 0
        print(f"\n  Bollinger Bands (Momentum):")
        print(f"    Aligned (above upper+bullish / below lower+bearish): {bb_aligned_acc:.1f}% ({stats.bb_aligned_total} signals)")
        print(f"    Against (below lower+bullish / above upper+bearish): {bb_against_acc:.1f}% ({stats.bb_against_total} signals)")
        if stats.bb_aligned_total > 0 and stats.bb_against_total > 0:
            print(f"    BB Edge: {bb_aligned_acc - bb_against_acc:+.1f}%")

    # MACD
    if stats.macd_aligned_total > 0 or stats.macd_against_total > 0:
        macd_aligned_acc = (stats.macd_aligned_correct / stats.macd_aligned_total * 100) if stats.macd_aligned_total > 0 else 0
        macd_against_acc = (stats.macd_against_correct / stats.macd_against_total * 100) if stats.macd_against_total > 0 else 0
        print(f"\n  MACD Momentum:")
        print(f"    Aligned (bullish momentum+bullish / bearish momentum+bearish): {macd_aligned_acc:.1f}% ({stats.macd_aligned_total} signals)")
        print(f"    Against (bearish momentum+bullish / bullish momentum+bearish): {macd_against_acc:.1f}% ({stats.macd_against_total} signals)")
        if stats.macd_aligned_total > 0 and stats.macd_against_total > 0:
            print(f"    MACD Edge: {macd_aligned_acc - macd_against_acc:+.1f}%")

    # Options Indicators (if data available)
    if stats.options_data_signals > 0:
        print("\n" + "-" * 70)
        print(f"OPTIONS INDICATORS (EODHD) - {stats.options_data_signals} signals with data")
        print("-" * 70)

        # Put/Call Ratio (contrarian)
        if stats.pcr_aligned_total > 0 or stats.pcr_against_total > 0:
            pcr_aligned_acc = (stats.pcr_aligned_correct / stats.pcr_aligned_total * 100) if stats.pcr_aligned_total > 0 else 0
            pcr_against_acc = (stats.pcr_against_correct / stats.pcr_against_total * 100) if stats.pcr_against_total > 0 else 0
            print(f"\n  Put/Call Ratio (Contrarian):")
            print(f"    Aligned (high P/C+bullish / low P/C+bearish): {pcr_aligned_acc:.1f}% ({stats.pcr_aligned_total} signals)")
            print(f"    Against (low P/C+bullish / high P/C+bearish): {pcr_against_acc:.1f}% ({stats.pcr_against_total} signals)")
            if stats.pcr_aligned_total > 0 and stats.pcr_against_total > 0:
                print(f"    P/C Ratio Edge: {pcr_aligned_acc - pcr_against_acc:+.1f}%")

        # Max Pain
        if stats.maxpain_aligned_total > 0 or stats.maxpain_against_total > 0:
            mp_aligned_acc = (stats.maxpain_aligned_correct / stats.maxpain_aligned_total * 100) if stats.maxpain_aligned_total > 0 else 0
            mp_against_acc = (stats.maxpain_against_correct / stats.maxpain_against_total * 100) if stats.maxpain_against_total > 0 else 0
            print(f"\n  Max Pain (Price Magnet):")
            print(f"    Aligned (below max pain+bullish / above max pain+bearish): {mp_aligned_acc:.1f}% ({stats.maxpain_aligned_total} signals)")
            print(f"    Against (above max pain+bullish / below max pain+bearish): {mp_against_acc:.1f}% ({stats.maxpain_against_total} signals)")
            if stats.maxpain_aligned_total > 0 and stats.maxpain_against_total > 0:
                print(f"    Max Pain Edge: {mp_aligned_acc - mp_against_acc:+.1f}%")

    # IV Rank Validation (validates ORATS $199/mo subscription value)
    if stats.iv_rank_data_signals > 0:
        print("\n" + "-" * 70)
        print(f"IV RANK VALIDATION (EODHD Historical IV) - {stats.iv_rank_data_signals} signals with data")
        print("-" * 70)
        print("  This validates whether the ORATS $199/mo IV Rank filter improves signals.")
        print("  Current gate: Block signals when IV Rank > 45%")

        iv_pass_acc = (stats.iv_rank_pass_correct / stats.iv_rank_pass_total * 100) if stats.iv_rank_pass_total > 0 else 0
        iv_fail_acc = (stats.iv_rank_fail_correct / stats.iv_rank_fail_total * 100) if stats.iv_rank_fail_total > 0 else 0

        print(f"\n  IV Rank <= 45% (PASS gate): {iv_pass_acc:.1f}% ({stats.iv_rank_pass_total} signals)")
        print(f"  IV Rank >  45% (FAIL gate): {iv_fail_acc:.1f}% ({stats.iv_rank_fail_total} signals)")

        if stats.iv_rank_pass_total > 0 and stats.iv_rank_fail_total > 0:
            edge = iv_pass_acc - iv_fail_acc
            print(f"\n  IV RANK FILTER EDGE: {edge:+.1f}%")
            if edge > 5:
                print("  --> VERDICT: IV Rank filter provides meaningful edge. ORATS justified.")
            elif edge > 0:
                print("  --> VERDICT: IV Rank filter provides small edge. ORATS marginally useful.")
            else:
                print("  --> VERDICT: IV Rank filter provides NO edge. Consider canceling ORATS.")

        # Alternative framework: "Retail Confirmation" analysis
        print("\n" + "-" * 70)
        print("IV RANK ALTERNATIVE FRAMEWORK: Retail Confirmation (Directional Buyers)")
        print("-" * 70)
        print("  For sentiment-driven momentum trades, high IV may CONFIRM retail excitement.")
        print("  Traditional IV rules are for premium sellers - we're directional buyers.")

        iv_high_strong_acc = (stats.iv_high_strong_sent_correct / stats.iv_high_strong_sent_total * 100) if stats.iv_high_strong_sent_total > 0 else 0
        iv_high_weak_acc = (stats.iv_high_weak_sent_correct / stats.iv_high_weak_sent_total * 100) if stats.iv_high_weak_sent_total > 0 else 0
        iv_low_acc = (stats.iv_low_correct / stats.iv_low_total * 100) if stats.iv_low_total > 0 else 0
        iv_mid_acc = (stats.iv_mid_correct / stats.iv_mid_total * 100) if stats.iv_mid_total > 0 else 0

        print(f"\n  IV Rank > 60% + Strong Sentiment (>0.3):")
        print(f"    Accuracy: {iv_high_strong_acc:.1f}% ({stats.iv_high_strong_sent_total} signals)")
        print(f"    --> \"Retail excitement confirmed\" - high IV validates the hype")

        print(f"\n  IV Rank > 60% + Weak Sentiment (<=0.3):")
        print(f"    Accuracy: {iv_high_weak_acc:.1f}% ({stats.iv_high_weak_sent_total} signals)")
        print(f"    --> Caution: High premium but weak conviction")

        print(f"\n  IV Rank < 30% (Any Sentiment):")
        print(f"    Accuracy: {iv_low_acc:.1f}% ({stats.iv_low_total} signals)")
        print(f"    --> Cheap premium opportunity")

        print(f"\n  IV Rank 30-60% (Neutral Zone):")
        print(f"    Accuracy: {iv_mid_acc:.1f}% ({stats.iv_mid_total} signals)")

        # Calculate best strategy
        print("\n  PROPOSED SOFT GATE MODIFIER:")
        print("    IV Rank > 60% + Strong WSB Sentiment (>0.3): +5 confidence (retail confirmation)")
        print("    IV Rank < 30%: +5 confidence (cheap premium)")
        print("    IV Rank 30-60%: neutral (no modifier)")

        # Show which approach wins
        if stats.iv_high_strong_sent_total >= 3:
            if iv_high_strong_acc > iv_mid_acc + 5:
                print(f"\n  --> FINDING: High IV + strong sentiment OUTPERFORMS by {iv_high_strong_acc - iv_mid_acc:+.1f}%")
                print("      High IV is a POSITIVE signal for sentiment-driven trades!")
            elif iv_high_strong_acc < iv_mid_acc - 5:
                print(f"\n  --> FINDING: High IV + strong sentiment UNDERPERFORMS by {iv_high_strong_acc - iv_mid_acc:+.1f}%")
                print("      Original framework correct: avoid high IV even with strong sentiment.")
            else:
                print(f"\n  --> FINDING: No significant difference ({iv_high_strong_acc - iv_mid_acc:+.1f}%)")

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
    parser.add_argument(
        "--include-options-indicators",
        action="store_true",
        help="Include Put/Call Ratio and Max Pain from EODHD (requires EODHD_API_KEY)",
    )
    parser.add_argument(
        "--include-iv-rank",
        action="store_true",
        help="Calculate IV Rank from EODHD historical IV (validates ORATS subscription value)",
    )

    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Load API keys from config
    needs_eodhd = args.include_options_indicators or args.include_iv_rank
    try:
        config = load_config()
        api_key = config.quiver.api_key
        eodhd_api_key = config.eodhd.api_key if needs_eodhd else None
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        api_key = os.environ.get("QUIVER_API_KEY", "")
        eodhd_api_key = os.environ.get("EODHD_API_KEY", "") if needs_eodhd else None

    if not api_key:
        logger.error("No Quiver API key found. Set QUIVER_API_KEY or configure in .env")
        return

    if args.include_options_indicators and not eodhd_api_key:
        logger.warning("--include-options-indicators specified but no EODHD_API_KEY found. Disabling options indicators.")
        args.include_options_indicators = False

    if args.include_iv_rank and not eodhd_api_key:
        logger.warning("--include-iv-rank specified but no EODHD_API_KEY found. Disabling IV Rank.")
        args.include_iv_rank = False

    print(f"Running backtest for {symbols}")
    print(f"Period: {args.start} to today")
    print(f"Holding period: {args.holding} days")
    print(f"Min mentions: {args.min_mentions}")
    print(f"Sentiment threshold: {args.sentiment_threshold}")
    if args.include_options_indicators:
        print(f"Options indicators: ENABLED (EODHD)")
    if args.include_iv_rank:
        print(f"IV Rank validation: ENABLED (EODHD historical IV)")

    stats, results = await run_backtest(
        symbols=symbols,
        api_key=api_key,
        start_date=args.start,
        holding_days=args.holding,
        min_mentions=args.min_mentions,
        sentiment_threshold=args.sentiment_threshold,
        include_options=args.include_options_indicators,
        include_iv_rank=args.include_iv_rank,
        eodhd_api_key=eodhd_api_key,
    )

    print_results(stats)


if __name__ == "__main__":
    asyncio.run(main())
