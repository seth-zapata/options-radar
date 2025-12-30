#!/usr/bin/env python3
"""Test different regime window lengths.

Uses the best configuration from previous tests:
- 1.5% pullback/bounce threshold
- 7 DTE weeklies
- Include moderate regimes
- Bid/ask realistic fills

Tests window lengths: 5, 7, 10, 14 days
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional

import aiohttp
from dotenv import load_dotenv
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")

CACHE_DB = Path(__file__).parent.parent.parent / "cache" / "options_data.db"

# Fixed parameters (best config from previous tests)
PULLBACK_THRESHOLD = 1.5
TARGET_DTE = 7
INCLUDE_MODERATE = True

# Trade management
STOP_LOSS = -0.20
TAKE_PROFIT = 0.40
MAX_HOLD_DAYS = 10
MIN_OI = 500
MAX_TRADES_PER_REGIME = 5
MIN_DAYS_BETWEEN_ENTRIES = 1


@dataclass
class Trade:
    symbol: str
    entry_date: str
    option_type: str
    regime_type: str
    entry_trigger: str
    entry_price: float = 0.0
    entry_ask: float = 0.0
    exit_price: float = 0.0
    exit_date: str = ""
    exit_reason: str = ""
    pnl_pct: float = 0.0
    holding_days: int = 0
    days_into_regime: int = 0  # How many days into regime window


def calculate_technicals(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Bollinger, MACD, and Trend SMA indicators."""
    df = prices_df.copy()

    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['std_20'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
    df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
    df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_prev_hist'] = df['macd_hist'].shift(1)

    df['trend_bullish'] = df['Close'] > df['sma_20']
    df['pullback_pct'] = (df['High'] - df['Close']) / df['High'] * 100
    df['bounce_pct'] = (df['Close'] - df['Low']) / df['Low'] * 100

    return df


def check_technical_confirmation(row: pd.Series, is_bullish: bool) -> int:
    """Count how many technicals confirm the direction."""
    if pd.isna(row.get('bb_pct')) or pd.isna(row.get('macd_hist')):
        return 0

    count = 0
    bb_pct = row['bb_pct']
    if (is_bullish and bb_pct < 0.3) or (not is_bullish and bb_pct > 0.7):
        count += 1

    macd_hist = row['macd_hist']
    macd_prev = row.get('macd_prev_hist', np.nan)
    if not pd.isna(macd_prev):
        if (is_bullish and macd_hist > macd_prev) or (not is_bullish and macd_hist < macd_prev):
            count += 1

    if (is_bullish and row['trend_bullish']) or (not is_bullish and not row['trend_bullish']):
        count += 1

    return count


def classify_regime(sentiment: float, tech_confirms: int, include_moderate: bool = True) -> Optional[str]:
    """Classify regime based on sentiment and technicals."""
    if tech_confirms == 0:
        return None

    if sentiment > 0.12:
        return "strong_bullish"
    elif sentiment > 0.07 and include_moderate:
        return "moderate_bullish"
    elif sentiment < -0.15:
        return "strong_bearish"
    elif sentiment < -0.08 and include_moderate:
        return "moderate_bearish"
    else:
        return None


async def fetch_wsb_history(symbol: str, api_key: str) -> list[dict]:
    """Fetch historical WSB data from Quiver API."""
    url = f"https://api.quiverquant.com/beta/historical/wallstreetbets/{symbol}"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return []
            data = await response.json()
            return data if isinstance(data, list) else []


def get_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical price data."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end)
    if hist.empty:
        return pd.DataFrame()
    hist.index = hist.index.strftime("%Y-%m-%d")
    return hist


def get_cached_dates(symbol: str) -> list[str]:
    """Get list of cached dates for a symbol."""
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT trade_date FROM cache_metadata WHERE symbol = ? ORDER BY trade_date",
        (symbol,)
    )
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    return dates


def get_option_contract(
    db_path: Path, symbol: str, trade_date: str, stock_price: float,
    option_type: str, target_dte: int
) -> Optional[dict]:
    """Find ATM contract with specific DTE."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    min_expiry = (datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=target_dte - 3)).strftime("%Y-%m-%d")
    max_expiry = (datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=target_dte + 3)).strftime("%Y-%m-%d")

    cursor.execute("""
        SELECT contract_id, strike, expiry, bid, ask, volume, open_interest
        FROM options_contracts
        WHERE symbol = ? AND trade_date = ? AND option_type = ?
        AND expiry >= ? AND expiry <= ?
        AND bid > 0 AND ask > 0 AND open_interest >= ?
        ORDER BY ABS(strike - ?), ABS(julianday(expiry) - julianday(?))
        LIMIT 1
    """, (
        symbol, trade_date, option_type,
        min_expiry, max_expiry, MIN_OI,
        stock_price,
        (datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=target_dte)).strftime("%Y-%m-%d")
    ))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    expiry_dt = datetime.strptime(row[2], "%Y-%m-%d")
    entry_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    dte = (expiry_dt - entry_dt).days

    return {
        "contract_id": row[0],
        "strike": row[1],
        "expiry": row[2],
        "bid": row[3],
        "ask": row[4],
        "mid": (row[3] + row[4]) / 2,
        "dte": dte,
    }


def get_contract_price(db_path: Path, contract_id: str, trade_date: str) -> Optional[dict]:
    """Get price data for a specific contract on a date."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT bid, ask, expiry FROM options_contracts
        WHERE contract_id = ? AND trade_date = ?
    """, (contract_id, trade_date))
    row = cursor.fetchone()
    conn.close()

    if not row or row[0] is None:
        return None

    return {"bid": row[0], "ask": row[1], "mid": (row[0] + row[1]) / 2, "expiry": row[2]}


def get_trading_days_after(start_date: str, num_days: int) -> list[str]:
    """Get list of trading days after start date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    days = []
    current = start + timedelta(days=1)
    while len(days) < num_days:
        if current.weekday() < 5:
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


async def run_backtest_with_window(
    symbol: str,
    start_date: str,
    quiver_api_key: str,
    regime_window: int,  # Variable window length
) -> tuple[list[Trade], dict]:
    """Run backtest with specified regime window length."""

    cached_dates = get_cached_dates(symbol)
    if not cached_dates:
        return [], {"error": "No cached data"}

    wsb_data = await fetch_wsb_history(symbol, quiver_api_key)
    if not wsb_data:
        return [], {"error": "No WSB data"}

    wsb_by_date = {}
    for item in wsb_data:
        date = item.get("Date", "")[:10]
        if date >= start_date:
            wsb_by_date[date] = item.get("Sentiment", 0)

    end_date = datetime.now().strftime("%Y-%m-%d")
    prices_df = get_price_data(symbol, start_date, end_date)
    if prices_df.empty:
        return [], {"error": "No price data"}

    technicals_df = calculate_technicals(prices_df)

    # Build regime windows with day tracking
    # date -> (regime_type, original_sentiment, day_number_in_window)
    active_regimes = {}

    for wsb_date, sentiment in wsb_by_date.items():
        is_bullish = sentiment > 0
        tech_confirms = 0

        if wsb_date in technicals_df.index:
            row = technicals_df.loc[wsb_date]
            tech_confirms = check_technical_confirmation(row, is_bullish)

        regime = classify_regime(sentiment, tech_confirms, INCLUDE_MODERATE)
        if regime:
            try:
                base_dt = datetime.strptime(wsb_date, "%Y-%m-%d")
                current = base_dt
                days_marked = 0
                while days_marked < regime_window:
                    current += timedelta(days=1)
                    if current.weekday() < 5:
                        date_str = current.strftime("%Y-%m-%d")
                        if date_str in technicals_df.index:
                            # Store day number (1 = first day after signal)
                            active_regimes[date_str] = (regime, sentiment, days_marked + 1)
                        days_marked += 1
            except ValueError:
                pass

    all_trades = []
    regime_trades = defaultdict(int)
    last_entry_date = None

    stats = {
        "regime_window": regime_window,
        "regime_days_total": len(active_regimes),
        "trades_by_day": defaultdict(int),  # Track which day in window trades occur
    }

    sorted_dates = sorted(set(cached_dates) & set(technicals_df.index))

    for date in sorted_dates:
        if date not in technicals_df.index:
            continue

        if date not in active_regimes:
            continue

        regime, orig_sentiment, day_in_window = active_regimes[date]
        row = technicals_df.loc[date]

        regime_key = f"{regime}_{date[:7]}"
        if regime_trades[regime_key] >= MAX_TRADES_PER_REGIME:
            continue

        if last_entry_date:
            days_since = (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(last_entry_date, "%Y-%m-%d")).days
            if days_since < MIN_DAYS_BETWEEN_ENTRIES:
                continue

        pullback_pct = row.get('pullback_pct', 0)
        bounce_pct = row.get('bounce_pct', 0)

        entry_trigger = None
        option_type = None

        if "bullish" in regime:
            if pullback_pct >= PULLBACK_THRESHOLD:
                entry_trigger = "pullback"
                option_type = "call"
        else:
            if bounce_pct >= PULLBACK_THRESHOLD:
                entry_trigger = "bounce"
                option_type = "put"

        if not entry_trigger:
            continue

        stock_price = row['Close']
        contract = get_option_contract(CACHE_DB, symbol, date, stock_price, option_type, TARGET_DTE)

        if not contract:
            continue

        trade = Trade(
            symbol=symbol,
            entry_date=date,
            option_type=option_type,
            regime_type=regime,
            entry_trigger=entry_trigger,
            entry_price=contract["mid"],
            entry_ask=contract["ask"],
            days_into_regime=day_in_window,
        )

        entry_ask = contract["ask"]
        tracking_days = get_trading_days_after(date, MAX_HOLD_DAYS + 5)

        for track_date in tracking_days:
            price_data = get_contract_price(CACHE_DB, contract["contract_id"], track_date)
            if not price_data:
                continue

            current_bid = price_data["bid"]
            pnl = (current_bid - entry_ask) / entry_ask

            try:
                expiry_date = datetime.strptime(price_data["expiry"], "%Y-%m-%d")
                current = datetime.strptime(track_date, "%Y-%m-%d")
                dte = (expiry_date - current).days
            except ValueError:
                dte = 7

            exit_reason = None

            if pnl >= TAKE_PROFIT:
                exit_reason = "take_profit"
            if not exit_reason and pnl <= STOP_LOSS:
                exit_reason = "stop_loss"
            if not exit_reason and dte <= 1:
                exit_reason = "dte_exit"

            if exit_reason:
                trade.exit_date = track_date
                trade.exit_price = current_bid
                trade.exit_reason = exit_reason
                trade.pnl_pct = pnl * 100

                entry_dt = datetime.strptime(date, "%Y-%m-%d")
                exit_dt = datetime.strptime(track_date, "%Y-%m-%d")
                trade.holding_days = (exit_dt - entry_dt).days
                break

        if not trade.exit_date:
            for track_date in reversed(tracking_days):
                price_data = get_contract_price(CACHE_DB, contract["contract_id"], track_date)
                if price_data:
                    trade.exit_date = track_date
                    trade.exit_price = price_data["bid"]
                    trade.exit_reason = "max_hold"
                    trade.pnl_pct = ((price_data["bid"] - entry_ask) / entry_ask) * 100
                    break

        if trade.exit_date:
            all_trades.append(trade)
            regime_trades[regime_key] += 1
            last_entry_date = date
            stats["trades_by_day"][day_in_window] += 1

    return all_trades, stats


def calc_stats(trades: list[Trade]) -> dict:
    """Calculate trading statistics."""
    if not trades:
        return {
            "trades": 0, "win_rate": 0, "avg_pnl": 0,
            "total_pnl": 0, "winners": 0, "losers": 0,
        }

    total_pnl = sum(t.pnl_pct for t in trades)
    avg_pnl = total_pnl / len(trades)
    winners = [t for t in trades if t.pnl_pct > 0]
    win_rate = len(winners) / len(trades) * 100

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "winners": len(winners),
        "losers": len(trades) - len(winners),
    }


async def main():
    symbol = "TSLA"
    start_date = "2024-01-01"

    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("=" * 80)
    print("REGIME WINDOW EXTENSION TEST")
    print("=" * 80)
    print()
    print("Fixed Configuration (best from previous tests):")
    print(f"  Pullback/Bounce Threshold: {PULLBACK_THRESHOLD}%")
    print(f"  Target DTE: {TARGET_DTE} days")
    print(f"  Include Moderate Regimes: {INCLUDE_MODERATE}")
    print(f"  Fills: Bid/Ask (entry at ASK, exit at BID)")
    print()

    windows = [5, 7, 10, 14]

    results = []

    print("Testing regime window lengths:", windows)
    print()

    for window in windows:
        print(f"  Running {window}-day window...", end=" ", flush=True)
        trades, stats = await run_backtest_with_window(
            symbol=symbol,
            start_date=start_date,
            quiver_api_key=quiver_api_key,
            regime_window=window,
        )

        trade_stats = calc_stats(trades)

        # Calculate trades per month
        first_date = datetime.strptime(start_date, "%Y-%m-%d")
        last_date = datetime.now()
        months = (last_date.year - first_date.year) * 12 + (last_date.month - first_date.month)
        if months == 0:
            months = 1
        trades_per_month = trade_stats["trades"] / months

        results.append({
            "window": window,
            "trades": trade_stats["trades"],
            "trades_per_month": trades_per_month,
            "win_rate": trade_stats["win_rate"],
            "avg_pnl": trade_stats["avg_pnl"],
            "total_pnl": trade_stats["total_pnl"],
            "trade_list": trades,
            "stats": stats,
        })

        print(f"{trade_stats['trades']} trades, {trade_stats['win_rate']:.1f}% win, {trade_stats['avg_pnl']:+.1f}% avg")

    # Results table
    print()
    print("=" * 80)
    print("RESULTS BY WINDOW LENGTH")
    print("=" * 80)
    print()
    print(f"| {'Window':>8} | {'Trades':>7} | {'T/Month':>7} | {'Win%':>7} | {'Avg P&L':>9} | {'Total P&L':>10} |")
    print(f"|{'-'*10}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*11}|{'-'*12}|")

    for r in results:
        print(f"| {r['window']:>6}d | {r['trades']:>7} | {r['trades_per_month']:>7.1f} | {r['win_rate']:>6.1f}% | {r['avg_pnl']:>+8.1f}% | {r['total_pnl']:>+9.0f}% |")

    print()

    # Analyze trade timing within windows
    print("=" * 80)
    print("TRADE TIMING ANALYSIS: When in the window do trades occur?")
    print("=" * 80)
    print()

    for r in results:
        window = r["window"]
        stats = r["stats"]
        trades_by_day = stats.get("trades_by_day", {})

        if not trades_by_day:
            continue

        print(f"{window}-day window:")
        for day in sorted(trades_by_day.keys()):
            bar = "█" * trades_by_day[day]
            print(f"  Day {day:>2}: {trades_by_day[day]:>3} {bar}")
        print()

    # Quality analysis: Compare early vs late entries
    print("=" * 80)
    print("EDGE DECAY ANALYSIS: Do later entries have worse performance?")
    print("=" * 80)
    print()

    # Use 14-day window for this analysis (most data)
    r14 = next((r for r in results if r["window"] == 14), None)
    if r14 and r14["trade_list"]:
        early_trades = [t for t in r14["trade_list"] if t.days_into_regime <= 5]
        late_trades = [t for t in r14["trade_list"] if t.days_into_regime > 5]

        if early_trades:
            early_avg = sum(t.pnl_pct for t in early_trades) / len(early_trades)
            early_wr = len([t for t in early_trades if t.pnl_pct > 0]) / len(early_trades) * 100
        else:
            early_avg, early_wr = 0, 0

        if late_trades:
            late_avg = sum(t.pnl_pct for t in late_trades) / len(late_trades)
            late_wr = len([t for t in late_trades if t.pnl_pct > 0]) / len(late_trades) * 100
        else:
            late_avg, late_wr = 0, 0

        print("14-day window analysis:")
        print(f"  Days 1-5 (early):  {len(early_trades)} trades, {early_wr:.1f}% win, {early_avg:+.1f}% avg")
        print(f"  Days 6-14 (late):  {len(late_trades)} trades, {late_wr:.1f}% win, {late_avg:+.1f}% avg")
        print()

        if late_trades and early_trades:
            edge_decay = early_avg - late_avg
            print(f"  Edge decay: {edge_decay:+.1f}% (early - late avg P&L)")
            if edge_decay > 5:
                print("  >> SIGNIFICANT DECAY: Longer windows dilute edge quality")
            elif edge_decay > 0:
                print("  >> MILD DECAY: Slight degradation but still profitable")
            else:
                print("  >> NO DECAY: Later entries perform as well or better")

    # Key questions
    print()
    print("=" * 80)
    print("KEY QUESTIONS ANSWERED")
    print("=" * 80)
    print()

    baseline = next((r for r in results if r["window"] == 5), None)

    print("1. Do additional trades from longer windows maintain edge quality (40%+ win rate, 10%+ avg)?")
    for r in results[1:]:  # Skip baseline
        meets_wr = r["win_rate"] >= 40
        meets_avg = r["avg_pnl"] >= 10
        status = "YES" if (meets_wr and meets_avg) else "NO"
        print(f"   {r['window']:>2}d window: {r['win_rate']:.1f}% win, {r['avg_pnl']:+.1f}% avg -> {status}")
    print()

    print("2. Is there a point where the window is too long and signal goes stale?")
    if r14 and r14["trade_list"]:
        # Look at win rate by day
        by_day = defaultdict(list)
        for t in r14["trade_list"]:
            by_day[t.days_into_regime].append(t.pnl_pct)

        stale_day = None
        for day in sorted(by_day.keys()):
            trades = by_day[day]
            wr = len([p for p in trades if p > 0]) / len(trades) * 100 if trades else 0
            if len(trades) >= 3 and wr < 35:  # Significant drop
                stale_day = day
                break

        if stale_day:
            print(f"   Signal appears to go stale around day {stale_day}")
        else:
            print("   No clear staleness point detected within 14 days")
    print()

    print("3. What's the optimal balance between frequency and edge preservation?")
    # Find window that maximizes frequency while keeping avg_pnl >= 12%
    valid_windows = [r for r in results if r["avg_pnl"] >= 12]
    if valid_windows:
        best = max(valid_windows, key=lambda x: x["trades_per_month"])
        print(f"   Optimal window: {best['window']} days")
        print(f"   Trades/month: {best['trades_per_month']:.1f}")
        print(f"   Win rate: {best['win_rate']:.1f}%")
        print(f"   Avg P&L: {best['avg_pnl']:+.1f}%")
    else:
        print("   No window maintains 12%+ avg P&L - stick with 5-day baseline")

    # Success criteria check
    print()
    print("=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)
    print()
    print("Adopt longer window if:")
    print("  - Frequency increases meaningfully (3.5+ trades/month)")
    print("  - Win rate stays >= 40%")
    print("  - Avg P&L stays >= 12%")
    print("  - Total P&L doesn't decrease significantly")
    print()

    best_candidate = None
    for r in results:
        if r["window"] > 5:  # Only look at extensions
            meets_freq = r["trades_per_month"] >= 3.5
            meets_wr = r["win_rate"] >= 40
            meets_avg = r["avg_pnl"] >= 12

            if meets_freq and meets_wr and meets_avg:
                if baseline and r["total_pnl"] >= baseline["total_pnl"] * 0.8:  # Allow 20% drop
                    if best_candidate is None or r["trades_per_month"] > best_candidate["trades_per_month"]:
                        best_candidate = r

    if best_candidate:
        print(f">>> ADOPT {best_candidate['window']}-DAY WINDOW")
        print(f"    Frequency: {best_candidate['trades_per_month']:.1f}/month (target: 3.5+)")
        print(f"    Win Rate:  {best_candidate['win_rate']:.1f}% (target: 40%+)")
        print(f"    Avg P&L:   {best_candidate['avg_pnl']:+.1f}% (target: 12%+)")
        print(f"    Total P&L: {best_candidate['total_pnl']:+.0f}%")
    else:
        print(">>> STICK WITH 5-DAY WINDOW")
        print("    Longer windows don't meet all success criteria")
        if baseline:
            print(f"    Current: {baseline['trades_per_month']:.1f}/month, {baseline['win_rate']:.1f}% win, {baseline['avg_pnl']:+.1f}% avg")


if __name__ == "__main__":
    asyncio.run(main())
