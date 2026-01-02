#!/usr/bin/env python3
"""2022 Bear Market Sanity Check for TSLA.

Validates that our regime detection and signal logic would have worked during
the 2022 TSLA bear market (-65% crash) using FREE daily stock data.

We won't have actual options P&L, but we can check directional accuracy.

Data Sources (FREE):
- TSLA Daily Prices: Yahoo Finance (yfinance)
- WSB Sentiment: Quiver Quant (if historical 2022 data available)
- VIX: Yahoo Finance (^VIX)
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional

import pandas as pd
import numpy as np
import aiohttp
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Strategy parameters (same as live system)
REGIME_WINDOW = 7
PULLBACK_THRESHOLD = 1.5
FORWARD_DAYS = 7  # Check direction after 7 days (matching our 7 DTE)

# Regime thresholds (same as live system)
STRONG_BULLISH_THRESHOLD = 0.12
MODERATE_BULLISH_THRESHOLD = 0.07
MODERATE_BEARISH_THRESHOLD = -0.08
STRONG_BEARISH_THRESHOLD = -0.15

# VIX thresholds
VIX_PANIC_THRESHOLD = 35.0
VIX_ELEVATED_THRESHOLD = 25.0

# Earnings blackout
EARNINGS_BLACKOUT_DAYS_BEFORE = 5
EARNINGS_BLACKOUT_DAYS_AFTER = 1

# TSLA 2022 earnings dates
TSLA_EARNINGS_2022 = [
    "2022-01-26",  # Q4 2021
    "2022-04-20",  # Q1 2022
    "2022-07-20",  # Q2 2022
    "2022-10-19",  # Q3 2022
]

# Key 2022 events
KEY_EVENTS_2022 = [
    ("2022-01-26", "Q4 2021 Earnings"),
    ("2022-04-20", "Q1 2022 Earnings"),
    ("2022-07-20", "Q2 2022 Earnings"),
    ("2022-10-19", "Q3 2022 Earnings"),
    ("2022-10-27", "Elon closes Twitter deal"),
    ("2022-11-08", "Midterm elections"),
    ("2022-12-22", "TSLA hits 2022 low"),
]


@dataclass
class Signal:
    """A generated signal."""
    date: str
    signal_type: str  # "CALL" or "PUT"
    regime: str
    tech_confirmations: int
    price_at_signal: float
    price_after_7d: Optional[float] = None
    correct_direction: Optional[bool] = None
    vix_level: float = 0.0
    blocked_by_vix: bool = False
    blocked_by_earnings: bool = False


@dataclass
class DayAnalysis:
    """Analysis for a single day."""
    date: str
    price: float
    regime: str
    sentiment: Optional[float]
    vix: float
    pullback_pct: float
    bounce_pct: float
    tech_confirmations: int
    signal: Optional[Signal] = None


def get_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical stock data from yfinance."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end)
    if hist.empty:
        return pd.DataFrame()
    hist.index = pd.to_datetime(hist.index).strftime("%Y-%m-%d")
    return hist


def get_vix_data(start: str, end: str) -> pd.DataFrame:
    """Fetch historical VIX data."""
    import yfinance as yf
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    if vix.empty:
        return pd.DataFrame()

    # Flatten multi-level columns if present
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [col[0] for col in vix.columns]

    vix.index = pd.to_datetime(vix.index).strftime("%Y-%m-%d")
    return vix


async def fetch_wsb_history(symbol: str, api_key: str) -> dict[str, float]:
    """Fetch historical WSB sentiment data."""
    if not api_key:
        return {}

    url = f"https://api.quiverquant.com/beta/historical/wallstreetbets/{symbol}"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return {}
                data = await response.json()

                wsb_by_date = {}
                for item in data if isinstance(data, list) else []:
                    date = item.get("Date", "")[:10]
                    if date.startswith("2022"):
                        wsb_by_date[date] = item.get("Sentiment", 0)

                return wsb_by_date
    except Exception as e:
        print(f"Warning: Could not fetch WSB data: {e}")
        return {}


def calculate_technicals(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators (same as live system)."""
    df = prices_df.copy()

    # SMA and Bollinger Bands
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['std_20'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
    df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
    df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # MACD
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_prev_hist'] = df['macd_hist'].shift(1)

    # Trend
    df['trend_bullish'] = df['Close'] > df['sma_20']

    # Pullback/Bounce
    df['pullback_pct'] = (df['High'] - df['Close']) / df['High'] * 100
    df['bounce_pct'] = (df['Close'] - df['Low']) / df['Low'] * 100

    return df


def check_technical_confirmation(row: pd.Series, is_bullish: bool) -> int:
    """Count technical confirmations (same as live system)."""
    if pd.isna(row.get('bb_pct')) or pd.isna(row.get('macd_hist')):
        return 0

    count = 0

    # Bollinger Band position
    bb_pct = row['bb_pct']
    if (is_bullish and bb_pct < 0.3) or (not is_bullish and bb_pct > 0.7):
        count += 1

    # MACD momentum
    macd_hist = row['macd_hist']
    macd_prev = row.get('macd_prev_hist', np.nan)
    if not pd.isna(macd_prev):
        if (is_bullish and macd_hist > macd_prev) or (not is_bullish and macd_hist < macd_prev):
            count += 1

    # Trend alignment
    if (is_bullish and row['trend_bullish']) or (not is_bullish and not row['trend_bullish']):
        count += 1

    return count


def classify_regime_with_sentiment(sentiment: float) -> Optional[str]:
    """Classify regime based on WSB sentiment (same as live system)."""
    if sentiment > STRONG_BULLISH_THRESHOLD:
        return "strong_bullish"
    elif sentiment > MODERATE_BULLISH_THRESHOLD:
        return "moderate_bullish"
    elif sentiment < STRONG_BEARISH_THRESHOLD:
        return "strong_bearish"
    elif sentiment < MODERATE_BEARISH_THRESHOLD:
        return "moderate_bearish"
    return None


def classify_regime_technical_only(row: pd.Series) -> Optional[str]:
    """Classify regime using technicals only (fallback if no sentiment)."""
    if pd.isna(row.get('sma_20')) or pd.isna(row.get('macd')):
        return None

    price_above_sma = row['Close'] > row['sma_20']
    macd_bullish = row['macd'] > row['macd_signal']

    if price_above_sma and macd_bullish:
        return "moderate_bullish"
    elif not price_above_sma and not macd_bullish:
        return "moderate_bearish"
    return None


def is_in_earnings_blackout(date: str, earnings_dates: list[str]) -> bool:
    """Check if date is within earnings blackout period."""
    try:
        check_dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return False

    for earnings_date in earnings_dates:
        try:
            earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
            days_diff = (check_dt - earnings_dt).days

            if -EARNINGS_BLACKOUT_DAYS_BEFORE <= days_diff <= EARNINGS_BLACKOUT_DAYS_AFTER:
                return True
        except ValueError:
            continue

    return False


async def run_sanity_check():
    """Run the 2022 bear market sanity check."""

    print("=" * 80)
    print("2022 BEAR MARKET SANITY CHECK - TSLA")
    print("=" * 80)
    print()

    # Fetch data
    print("Fetching data...")

    # Stock prices (get extra days for warmup and forward checking)
    prices_df = get_stock_data("TSLA", "2021-11-01", "2023-02-01")
    if prices_df.empty:
        print("ERROR: Could not fetch TSLA price data")
        return
    print(f"  TSLA prices: {len(prices_df)} days")

    # VIX
    vix_df = get_vix_data("2022-01-01", "2022-12-31")
    vix_by_date = {}
    if not vix_df.empty:
        for idx in vix_df.index:
            vix_by_date[idx] = vix_df.loc[idx, 'Close']
    print(f"  VIX data: {len(vix_by_date)} days")

    # WSB Sentiment
    quiver_api_key = os.getenv("QUIVER_API_KEY")
    wsb_by_date = await fetch_wsb_history("TSLA", quiver_api_key)
    print(f"  WSB sentiment: {len(wsb_by_date)} days")

    use_sentiment = len(wsb_by_date) > 50
    if not use_sentiment:
        print("  WARNING: Insufficient WSB data, using technical-only regime detection")

    print()

    # Calculate technicals
    technicals_df = calculate_technicals(prices_df)

    # Get 2022 trading days only
    trading_days_2022 = [d for d in technicals_df.index if d.startswith("2022")]

    # =========================================================================
    # PART 1: REGIME DETECTION
    # =========================================================================
    print("=" * 80)
    print("PART 1: REGIME DETECTION")
    print("=" * 80)
    print()

    regime_counts = defaultdict(int)
    regime_by_date = {}
    regime_changes = []
    prev_regime = None

    # Build active regimes using regime window
    active_regimes = {}

    for date in trading_days_2022:
        if date not in technicals_df.index:
            continue

        row = technicals_df.loc[date]

        # Determine regime
        if use_sentiment and date in wsb_by_date:
            sentiment = wsb_by_date[date]
            is_bullish = sentiment > 0
            tech_confirms = check_technical_confirmation(row, is_bullish)

            if tech_confirms > 0:
                regime = classify_regime_with_sentiment(sentiment)
            else:
                regime = None
        else:
            regime = classify_regime_technical_only(row)

        # Apply regime window
        if regime:
            try:
                base_dt = datetime.strptime(date, "%Y-%m-%d")
                for i in range(1, REGIME_WINDOW + 1):
                    future_dt = base_dt + timedelta(days=i)
                    future_date = future_dt.strftime("%Y-%m-%d")
                    if future_date.startswith("2022"):
                        active_regimes[future_date] = regime
            except ValueError:
                pass

    # Count regimes
    for date in trading_days_2022:
        regime = active_regimes.get(date, "neutral")
        regime_counts[regime] += 1
        regime_by_date[date] = regime

        if regime != prev_regime and prev_regime is not None:
            regime_changes.append((date, prev_regime, regime))
        prev_regime = regime

    total_days = len(trading_days_2022)
    bullish_days = regime_counts.get("strong_bullish", 0) + regime_counts.get("moderate_bullish", 0)
    bearish_days = regime_counts.get("strong_bearish", 0) + regime_counts.get("moderate_bearish", 0)
    neutral_days = regime_counts.get("neutral", 0)

    print(f"Total trading days: {total_days}")
    print()
    print("Regime Distribution:")
    print(f"  Bullish days:  {bullish_days:>4} ({bullish_days/total_days*100:.1f}%)")
    print(f"    - Strong:    {regime_counts.get('strong_bullish', 0):>4}")
    print(f"    - Moderate:  {regime_counts.get('moderate_bullish', 0):>4}")
    print(f"  Bearish days:  {bearish_days:>4} ({bearish_days/total_days*100:.1f}%)")
    print(f"    - Strong:    {regime_counts.get('strong_bearish', 0):>4}")
    print(f"    - Moderate:  {regime_counts.get('moderate_bearish', 0):>4}")
    print(f"  Neutral days:  {neutral_days:>4} ({neutral_days/total_days*100:.1f}%)")
    print()

    # Show major regime changes
    print("Major Regime Changes:")
    for date, from_regime, to_regime in regime_changes[:15]:
        print(f"  {date}: {from_regime} -> {to_regime}")
    if len(regime_changes) > 15:
        print(f"  ... and {len(regime_changes) - 15} more changes")
    print()

    # =========================================================================
    # PART 2: SIGNAL GENERATION
    # =========================================================================
    print("=" * 80)
    print("PART 2: SIGNAL GENERATION")
    print("=" * 80)
    print()

    signals: list[Signal] = []
    signals_by_month = defaultdict(lambda: {"CALL": 0, "PUT": 0})

    # "While open" cooldown simulation
    open_directions: set[str] = set()

    for date in trading_days_2022:
        if date not in technicals_df.index:
            continue

        row = technicals_df.loc[date]
        regime = regime_by_date.get(date, "neutral")
        vix = vix_by_date.get(date, 20.0)

        # Skip neutral regime
        if regime == "neutral":
            continue

        # Determine direction
        is_bullish = "bullish" in regime
        target_direction = "call" if is_bullish else "put"

        # "While open" cooldown (simplified: assume position lasts 7 days)
        # Skip for sanity check - we want to see all potential signals

        # Check entry conditions
        pullback_pct = row.get('pullback_pct', 0)
        bounce_pct = row.get('bounce_pct', 0)

        entry_trigger = None
        signal_type = None

        if is_bullish and pullback_pct >= PULLBACK_THRESHOLD:
            entry_trigger = "pullback"
            signal_type = "CALL"
        elif not is_bullish and bounce_pct >= PULLBACK_THRESHOLD:
            entry_trigger = "bounce"
            signal_type = "PUT"

        if entry_trigger:
            tech_confirms = check_technical_confirmation(row, is_bullish)

            # Check VIX blocking
            blocked_by_vix = vix >= VIX_PANIC_THRESHOLD

            # Check earnings blackout
            blocked_by_earnings = is_in_earnings_blackout(date, TSLA_EARNINGS_2022)

            # Get price 7 days later for direction check
            price_at_signal = row['Close']
            price_after_7d = None
            correct_direction = None

            try:
                future_dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=7)
                future_date = future_dt.strftime("%Y-%m-%d")

                # Find closest trading day
                for i in range(5):
                    check_date = (future_dt + timedelta(days=i)).strftime("%Y-%m-%d")
                    if check_date in technicals_df.index:
                        price_after_7d = technicals_df.loc[check_date, 'Close']
                        break

                if price_after_7d is not None:
                    if signal_type == "CALL":
                        correct_direction = price_after_7d > price_at_signal
                    else:
                        correct_direction = price_after_7d < price_at_signal
            except (ValueError, KeyError):
                pass

            signal = Signal(
                date=date,
                signal_type=signal_type,
                regime=regime,
                tech_confirmations=tech_confirms,
                price_at_signal=price_at_signal,
                price_after_7d=price_after_7d,
                correct_direction=correct_direction,
                vix_level=vix,
                blocked_by_vix=blocked_by_vix,
                blocked_by_earnings=blocked_by_earnings,
            )
            signals.append(signal)

            month = date[:7]
            signals_by_month[month][signal_type] += 1

    call_signals = [s for s in signals if s.signal_type == "CALL"]
    put_signals = [s for s in signals if s.signal_type == "PUT"]

    print(f"Total Signals Generated: {len(signals)}")
    print(f"  CALL signals: {len(call_signals)}")
    print(f"  PUT signals:  {len(put_signals)}")
    print(f"  PUT/CALL ratio: {len(put_signals)/len(call_signals):.2f}" if call_signals else "  PUT/CALL ratio: N/A")
    print()

    print("Signals by Month:")
    print(f"  {'Month':<10} {'CALL':>6} {'PUT':>6} {'Total':>6}")
    print(f"  {'-'*30}")
    for month in sorted(signals_by_month.keys()):
        counts = signals_by_month[month]
        total = counts["CALL"] + counts["PUT"]
        print(f"  {month:<10} {counts['CALL']:>6} {counts['PUT']:>6} {total:>6}")
    print()

    # =========================================================================
    # PART 3: DIRECTIONAL ACCURACY
    # =========================================================================
    print("=" * 80)
    print("PART 3: DIRECTIONAL ACCURACY (7-day forward)")
    print("=" * 80)
    print()

    # All signals (use == instead of 'is' for numpy bool compatibility)
    call_correct = [s for s in call_signals if s.correct_direction == True]
    call_incorrect = [s for s in call_signals if s.correct_direction == False]
    call_unknown = [s for s in call_signals if s.correct_direction is None]

    put_correct = [s for s in put_signals if s.correct_direction == True]
    put_incorrect = [s for s in put_signals if s.correct_direction == False]
    put_unknown = [s for s in put_signals if s.correct_direction is None]

    call_total_known = len(call_correct) + len(call_incorrect)
    put_total_known = len(put_correct) + len(put_incorrect)

    call_accuracy = len(call_correct) / call_total_known * 100 if call_total_known > 0 else 0
    put_accuracy = len(put_correct) / put_total_known * 100 if put_total_known > 0 else 0

    total_correct = len(call_correct) + len(put_correct)
    total_known = call_total_known + put_total_known
    overall_accuracy = total_correct / total_known * 100 if total_known > 0 else 0

    print("All Signals:")
    print(f"  CALL signals: {len(call_correct)}/{call_total_known} correct ({call_accuracy:.1f}%)")
    print(f"  PUT signals:  {len(put_correct)}/{put_total_known} correct ({put_accuracy:.1f}%)")
    print(f"  Overall:      {total_correct}/{total_known} correct ({overall_accuracy:.1f}%)")
    print()

    # Signals that would NOT have been blocked
    unblocked_signals = [s for s in signals if not s.blocked_by_vix and not s.blocked_by_earnings]
    ub_call = [s for s in unblocked_signals if s.signal_type == "CALL"]
    ub_put = [s for s in unblocked_signals if s.signal_type == "PUT"]

    ub_call_correct = [s for s in ub_call if s.correct_direction == True]
    ub_put_correct = [s for s in ub_put if s.correct_direction == True]
    ub_call_known = len([s for s in ub_call if s.correct_direction is not None])
    ub_put_known = len([s for s in ub_put if s.correct_direction is not None])

    ub_call_acc = len(ub_call_correct) / ub_call_known * 100 if ub_call_known > 0 else 0
    ub_put_acc = len(ub_put_correct) / ub_put_known * 100 if ub_put_known > 0 else 0
    ub_total_correct = len(ub_call_correct) + len(ub_put_correct)
    ub_total_known = ub_call_known + ub_put_known
    ub_overall_acc = ub_total_correct / ub_total_known * 100 if ub_total_known > 0 else 0

    print("After VIX & Earnings Filters:")
    print(f"  CALL signals: {len(ub_call_correct)}/{ub_call_known} correct ({ub_call_acc:.1f}%)")
    print(f"  PUT signals:  {len(ub_put_correct)}/{ub_put_known} correct ({ub_put_acc:.1f}%)")
    print(f"  Overall:      {ub_total_correct}/{ub_total_known} correct ({ub_overall_acc:.1f}%)")
    print()

    # =========================================================================
    # PART 4: VIX OVERLAY
    # =========================================================================
    print("=" * 80)
    print("PART 4: VIX OVERLAY")
    print("=" * 80)
    print()

    vix_values = list(vix_by_date.values())
    if vix_values:
        vix_avg = np.mean(vix_values)
        vix_max = max(vix_values)
        vix_max_date = [d for d, v in vix_by_date.items() if v == vix_max][0]
        vix_min = min(vix_values)

        print(f"VIX Statistics for 2022:")
        print(f"  Average: {vix_avg:.1f}")
        print(f"  Peak:    {vix_max:.1f} ({vix_max_date})")
        print(f"  Low:     {vix_min:.1f}")
        print()

        days_panic = len([v for v in vix_values if v >= VIX_PANIC_THRESHOLD])
        days_elevated = len([v for v in vix_values if VIX_ELEVATED_THRESHOLD <= v < VIX_PANIC_THRESHOLD])
        days_normal = len([v for v in vix_values if v < VIX_ELEVATED_THRESHOLD])

        print(f"VIX Regime Days:")
        print(f"  Panic (>35):    {days_panic:>4} days ({days_panic/len(vix_values)*100:.1f}%)")
        print(f"  Elevated (25-35): {days_elevated:>4} days ({days_elevated/len(vix_values)*100:.1f}%)")
        print(f"  Normal (<25):   {days_normal:>4} days ({days_normal/len(vix_values)*100:.1f}%)")
        print()

    signals_blocked_vix = [s for s in signals if s.blocked_by_vix]
    signals_elevated_vix = [s for s in signals if VIX_ELEVATED_THRESHOLD <= s.vix_level < VIX_PANIC_THRESHOLD]

    print(f"VIX Impact on Signals:")
    print(f"  Blocked by VIX > 35:     {len(signals_blocked_vix)} signals")
    print(f"  Reduced size (VIX 25-35): {len(signals_elevated_vix)} signals")
    print()

    # =========================================================================
    # PART 5: EARNINGS BLACKOUT ANALYSIS
    # =========================================================================
    print("=" * 80)
    print("PART 5: EARNINGS BLACKOUT ANALYSIS")
    print("=" * 80)
    print()

    signals_blocked_earnings = [s for s in signals if s.blocked_by_earnings]

    print(f"Earnings Dates: {', '.join(TSLA_EARNINGS_2022)}")
    print(f"Blackout Window: {EARNINGS_BLACKOUT_DAYS_BEFORE} days before, {EARNINGS_BLACKOUT_DAYS_AFTER} day after")
    print()
    print(f"Signals blocked by earnings blackout: {len(signals_blocked_earnings)}")

    if signals_blocked_earnings:
        print()
        print("Blocked signals:")
        for s in signals_blocked_earnings:
            direction = "Correct" if s.correct_direction else "Wrong" if s.correct_direction is False else "Unknown"
            print(f"  {s.date}: {s.signal_type} ({direction})")
    print()

    # =========================================================================
    # PART 6: KEY EVENT ANALYSIS
    # =========================================================================
    print("=" * 80)
    print("PART 6: KEY EVENT ANALYSIS")
    print("=" * 80)
    print()

    print(f"{'Date':<12} {'Event':<30} {'TSLA Move':>12} {'Signal?':<15}")
    print("-" * 75)

    for event_date, event_name in KEY_EVENTS_2022:
        # Find TSLA move (next day return)
        try:
            if event_date in technicals_df.index:
                price_on_day = technicals_df.loc[event_date, 'Close']

                # Get next trading day
                next_day = None
                event_dt = datetime.strptime(event_date, "%Y-%m-%d")
                for i in range(1, 5):
                    check = (event_dt + timedelta(days=i)).strftime("%Y-%m-%d")
                    if check in technicals_df.index:
                        next_day = check
                        break

                if next_day:
                    price_next = technicals_df.loc[next_day, 'Close']
                    move_pct = (price_next - price_on_day) / price_on_day * 100
                    move_str = f"{move_pct:+.1f}%"
                else:
                    move_str = "N/A"
            else:
                move_str = "N/A"
        except:
            move_str = "N/A"

        # Check if we had a signal around this date
        signals_around = [s for s in signals
                         if abs((datetime.strptime(s.date, "%Y-%m-%d") -
                                datetime.strptime(event_date, "%Y-%m-%d")).days) <= 2]

        if signals_around:
            sig = signals_around[0]
            blocked = ""
            if sig.blocked_by_earnings:
                blocked = " (BLOCKED)"
            elif sig.blocked_by_vix:
                blocked = " (VIX BLOCK)"
            signal_str = f"{sig.signal_type}{blocked}"
        else:
            signal_str = "None"

        print(f"{event_date:<12} {event_name:<30} {move_str:>12} {signal_str:<15}")

    print()

    # =========================================================================
    # SUMMARY DASHBOARD
    # =========================================================================
    print("=" * 80)
    print("SUMMARY DASHBOARD")
    print("=" * 80)
    print()

    # Market context
    if "2022-01-03" in technicals_df.index and "2022-12-30" in technicals_df.index:
        start_price = technicals_df.loc["2022-01-03", 'Close']
        end_price = technicals_df.loc["2022-12-30", 'Close']
        tsla_return = (end_price - start_price) / start_price * 100
    else:
        # Find actual start/end
        prices_2022 = [(d, technicals_df.loc[d, 'Close']) for d in trading_days_2022 if d in technicals_df.index]
        if len(prices_2022) >= 2:
            start_price = prices_2022[0][1]
            end_price = prices_2022[-1][1]
            tsla_return = (end_price - start_price) / start_price * 100
        else:
            tsla_return = -65  # Approximate

    print("MARKET CONTEXT:")
    print(f"  TSLA 2022 Return: {tsla_return:.1f}%")
    if vix_values:
        print(f"  VIX Avg: {vix_avg:.1f} | VIX Peak: {vix_max:.1f} ({vix_max_date})")
    print()

    print("REGIME DETECTION:")
    print(f"  Bullish days: {bullish_days} ({bullish_days/total_days*100:.1f}%)")
    print(f"  Bearish days: {bearish_days} ({bearish_days/total_days*100:.1f}%)")
    print(f"  Neutral days: {neutral_days} ({neutral_days/total_days*100:.1f}%)")
    print()

    print("SIGNAL GENERATION:")
    print(f"  CALL signals: {len(call_signals)}")
    print(f"  PUT signals:  {len(put_signals)}")
    print(f"  PUT/CALL ratio: {len(put_signals)/len(call_signals):.2f}" if call_signals else "  PUT/CALL ratio: N/A")
    print()

    print("DIRECTIONAL ACCURACY (all signals):")
    print(f"  CALL signals: {len(call_correct)}/{call_total_known} correct ({call_accuracy:.1f}%)")
    print(f"  PUT signals:  {len(put_correct)}/{put_total_known} correct ({put_accuracy:.1f}%)")
    print(f"  Overall:      {total_correct}/{total_known} correct ({overall_accuracy:.1f}%)")
    print()

    print("DIRECTIONAL ACCURACY (after filters):")
    print(f"  CALL signals: {len(ub_call_correct)}/{ub_call_known} correct ({ub_call_acc:.1f}%)")
    print(f"  PUT signals:  {len(ub_put_correct)}/{ub_put_known} correct ({ub_put_acc:.1f}%)")
    print(f"  Overall:      {ub_total_correct}/{ub_total_known} correct ({ub_overall_acc:.1f}%)")
    print()

    print("FILTER IMPACT:")
    print(f"  Signals blocked (VIX > 35):    {len(signals_blocked_vix)}")
    print(f"  Signals blocked (earnings):   {len(signals_blocked_earnings)}")
    print(f"  Signals reduced (VIX 25-35):  {len(signals_elevated_vix)}")
    print()

    # Conclusion
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if overall_accuracy >= 55:
        assessment = "POSITIVE - Strategy showed edge in 2022 bear market"
    elif overall_accuracy >= 50:
        assessment = "NEUTRAL - Strategy was roughly breakeven directionally"
    else:
        assessment = "NEGATIVE - Strategy struggled in 2022 bear market"

    put_dominated = len(put_signals) > len(call_signals) * 1.5

    print(f"Assessment: {assessment}")
    print()

    if put_dominated:
        print("The strategy correctly generated more PUT than CALL signals during")
        print("the 2022 bear market, suggesting regime detection worked appropriately.")
    else:
        print("The PUT/CALL ratio suggests the regime detection may not have")
        print("adapted quickly enough to the bearish environment.")

    print()

    if ub_overall_acc > overall_accuracy:
        improvement = ub_overall_acc - overall_accuracy
        print(f"VIX and earnings filters IMPROVED accuracy by {improvement:.1f}%")
    elif ub_overall_acc < overall_accuracy:
        decline = overall_accuracy - ub_overall_acc
        print(f"VIX and earnings filters REDUCED accuracy by {decline:.1f}%")
    else:
        print("Filters had minimal impact on accuracy.")

    print()


if __name__ == "__main__":
    asyncio.run(run_sanity_check())
