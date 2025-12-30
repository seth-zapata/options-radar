#!/usr/bin/env python3
"""Regime-Filtered Intraday Strategy Backtest.

Uses WSB sentiment to identify directional regimes, then looks for
intraday pullbacks (bullish) or bounces (bearish) as entry points.

Goal: Increase trade frequency while preserving directional edge.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional, Literal
from itertools import product

import aiohttp
from dotenv import load_dotenv
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")

CACHE_DB = Path(__file__).parent.parent.parent / "cache" / "options_data.db"

# Intraday strategy parameters
STOP_LOSS = -0.20       # Tighter than swing (-30%)
TAKE_PROFIT = 0.40      # Quicker exit than swing
MAX_HOLD_DAYS = 10      # Shorter hold period
MIN_OI = 500
MAX_TRADES_PER_REGIME = 5
MIN_DAYS_BETWEEN_ENTRIES = 1


@dataclass
class Regime:
    start_date: str
    end_date: str = ""
    regime_type: str = ""  # strong_bullish, moderate_bullish, strong_bearish, moderate_bearish
    avg_sentiment: float = 0.0
    tech_confirmations: int = 0


@dataclass
class Trade:
    symbol: str
    entry_date: str
    option_type: str  # call or put
    regime_type: str
    entry_trigger: str  # "pullback" or "bounce"

    entry_price: float = 0.0
    entry_ask: float = 0.0
    exit_price: float = 0.0
    exit_date: str = ""
    exit_reason: str = ""
    pnl_pct: float = 0.0
    holding_days: int = 0

    pullback_pct: float = 0.0  # How much pullback/bounce triggered entry
    dte_at_entry: int = 0


def calculate_technicals(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Bollinger, MACD, and Trend SMA indicators."""
    df = prices_df.copy()

    # Bollinger Bands (20-day, 2 std)
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['std_20'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
    df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
    df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # MACD (12, 26, 9)
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_prev_hist'] = df['macd_hist'].shift(1)

    # Trend SMA (price vs 20 SMA)
    df['trend_bullish'] = df['Close'] > df['sma_20']

    # Intraday proxies from OHLC
    df['pullback_pct'] = (df['High'] - df['Close']) / df['High'] * 100  # Pullback from high
    df['bounce_pct'] = (df['Close'] - df['Low']) / df['Low'] * 100      # Bounce from low

    return df


def check_technical_confirmation(row: pd.Series, is_bullish: bool) -> int:
    """Count how many technicals confirm the direction."""
    if pd.isna(row.get('bb_pct')) or pd.isna(row.get('macd_hist')):
        return 0

    count = 0

    # Bollinger: Bullish if price near lower band (<0.3), Bearish if near upper (>0.7)
    bb_pct = row['bb_pct']
    if (is_bullish and bb_pct < 0.3) or (not is_bullish and bb_pct > 0.7):
        count += 1

    # MACD: Bullish if histogram positive/rising, Bearish if negative/falling
    macd_hist = row['macd_hist']
    macd_prev = row.get('macd_prev_hist', np.nan)
    if not pd.isna(macd_prev):
        if (is_bullish and macd_hist > macd_prev) or (not is_bullish and macd_hist < macd_prev):
            count += 1

    # Trend: Bullish if price above SMA, Bearish if below
    if (is_bullish and row['trend_bullish']) or (not is_bullish and not row['trend_bullish']):
        count += 1

    return count


def classify_regime(sentiment: float, tech_confirms: int, include_moderate: bool = True) -> Optional[str]:
    """Classify the current regime based on sentiment and technicals.

    CALIBRATED to match original signal logic (abs(sentiment) >= 0.1):
    - Original PUT trades: -0.104, -0.114, -0.106 (around -0.10)
    - 10th percentile bearish: -0.103
    - 90th percentile bullish: +0.071
    - Distribution is asymmetric (bearish goes lower than bullish goes high)
    """
    if tech_confirms == 0:
        return None  # No technical confirmation = neutral

    # Thresholds calibrated to actual TSLA sentiment distribution
    # Strong = top 5% (roughly ±0.15)
    # Moderate = top 10-20% (roughly ±0.08 to ±0.10)
    if sentiment > 0.12:
        return "strong_bullish"
    elif sentiment > 0.07 and include_moderate:
        return "moderate_bullish"
    elif sentiment < -0.15:
        return "strong_bearish"
    elif sentiment < -0.08 and include_moderate:
        return "moderate_bearish"
    else:
        return None  # Neutral sentiment


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
    """Fetch historical price data from yfinance as DataFrame."""
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
    db_path: Path,
    symbol: str,
    trade_date: str,
    stock_price: float,
    option_type: str,
    target_dte: int
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
        "volume": row[5],
        "open_interest": row[6],
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

    return {
        "bid": row[0],
        "ask": row[1],
        "mid": (row[0] + row[1]) / 2,
        "expiry": row[2],
    }


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


async def run_regime_backtest(
    symbol: str,
    start_date: str,
    quiver_api_key: str,
    pullback_threshold: float,  # e.g., 1.5 for 1.5%
    target_dte: int,            # e.g., 7 or 14
    include_moderate: bool,     # Include moderate regimes or strong-only
) -> tuple[list[Trade], dict]:
    """Run regime-filtered intraday backtest.

    Uses REGIME WINDOW approach:
    - When sentiment exceeds threshold, a regime is active for REGIME_WINDOW days
    - Any pullback/bounce during the window is a valid entry
    - This increases frequency vs requiring sentiment + pullback on same day
    """
    REGIME_WINDOW = 5  # Regime stays active for 5 trading days after signal

    # Get cached dates
    cached_dates = get_cached_dates(symbol)
    if not cached_dates:
        return [], {"error": "No cached data"}

    # Get WSB data
    wsb_data = await fetch_wsb_history(symbol, quiver_api_key)
    if not wsb_data:
        return [], {"error": "No WSB data"}

    # Build date -> sentiment map (forward-fill to trading days)
    wsb_by_date = {}
    for item in wsb_data:
        date = item.get("Date", "")[:10]
        if date >= start_date:
            wsb_by_date[date] = item.get("Sentiment", 0)

    # Get price data and calculate technicals
    end_date = datetime.now().strftime("%Y-%m-%d")
    prices_df = get_price_data(symbol, start_date, end_date)
    if prices_df.empty:
        return [], {"error": "No price data"}

    technicals_df = calculate_technicals(prices_df)

    # Build regime windows: when sentiment exceeds threshold, mark next N days
    active_regimes = {}  # date -> (regime_type, original_sentiment, days_remaining)

    for wsb_date, sentiment in wsb_by_date.items():
        # Determine if this creates a new regime
        is_bullish = sentiment > 0
        tech_confirms = 0

        # Check technicals if the date is in our price data
        if wsb_date in technicals_df.index:
            row = technicals_df.loc[wsb_date]
            tech_confirms = check_technical_confirmation(row, is_bullish)

        regime = classify_regime(sentiment, tech_confirms, include_moderate)
        if regime:
            # Find next trading days and mark regime active
            try:
                base_dt = datetime.strptime(wsb_date, "%Y-%m-%d")
                current = base_dt
                days_marked = 0
                while days_marked < REGIME_WINDOW:
                    current += timedelta(days=1)
                    if current.weekday() < 5:  # Trading day
                        date_str = current.strftime("%Y-%m-%d")
                        if date_str in technicals_df.index:
                            active_regimes[date_str] = (regime, sentiment, REGIME_WINDOW - days_marked)
                        days_marked += 1
            except ValueError:
                pass

    all_trades = []
    regime_trades = defaultdict(int)  # Track trades per regime period
    last_entry_date = None

    stats = {
        "regime_days": defaultdict(int),
        "pullback_triggers": 0,
        "bounce_triggers": 0,
        "valid_options": 0,
        "regime_window_days": len(active_regimes),
    }

    sorted_dates = sorted(set(cached_dates) & set(technicals_df.index))

    for date in sorted_dates:
        if date not in technicals_df.index:
            continue

        # Check if we're in an active regime window
        if date not in active_regimes:
            continue

        regime, orig_sentiment, days_remaining = active_regimes[date]
        stats["regime_days"][regime] += 1

        row = technicals_df.loc[date]

        # Check if we can enter (respect max trades per regime and min days between)
        regime_key = f"{regime}_{date[:7]}"  # Monthly regime key
        if regime_trades[regime_key] >= MAX_TRADES_PER_REGIME:
            continue

        if last_entry_date:
            days_since = (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(last_entry_date, "%Y-%m-%d")).days
            if days_since < MIN_DAYS_BETWEEN_ENTRIES:
                continue

        # Check for pullback/bounce trigger
        pullback_pct = row.get('pullback_pct', 0)
        bounce_pct = row.get('bounce_pct', 0)

        entry_trigger = None
        trigger_pct = 0
        option_type = None

        if "bullish" in regime:
            # Bullish regime - look for pullback from high
            if pullback_pct >= pullback_threshold:
                entry_trigger = "pullback"
                trigger_pct = pullback_pct
                option_type = "call"
                stats["pullback_triggers"] += 1
        else:
            # Bearish regime - look for bounce from low
            if bounce_pct >= pullback_threshold:
                entry_trigger = "bounce"
                trigger_pct = bounce_pct
                option_type = "put"
                stats["bounce_triggers"] += 1

        if not entry_trigger:
            continue

        # Find option contract
        stock_price = row['Close']
        contract = get_option_contract(CACHE_DB, symbol, date, stock_price, option_type, target_dte)

        if not contract:
            continue

        stats["valid_options"] += 1

        # Create trade
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            option_type=option_type,
            regime_type=regime,
            entry_trigger=entry_trigger,
            entry_price=contract["mid"],
            entry_ask=contract["ask"],
            pullback_pct=trigger_pct,
            dte_at_entry=contract["dte"],
        )

        # Track position
        entry_ask = contract["ask"]
        tracking_days = get_trading_days_after(date, MAX_HOLD_DAYS + 5)

        for track_date in tracking_days:
            price_data = get_contract_price(CACHE_DB, contract["contract_id"], track_date)
            if not price_data:
                continue

            current_bid = price_data["bid"]
            pnl = (current_bid - entry_ask) / entry_ask

            # Check DTE
            try:
                expiry_date = datetime.strptime(price_data["expiry"], "%Y-%m-%d")
                current = datetime.strptime(track_date, "%Y-%m-%d")
                dte = (expiry_date - current).days
            except ValueError:
                dte = 7

            exit_reason = None

            # Take profit
            if pnl >= TAKE_PROFIT:
                exit_reason = "take_profit"

            # Stop loss
            if not exit_reason and pnl <= STOP_LOSS:
                exit_reason = "stop_loss"

            # DTE exit
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

        # If no exit, use last available price
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

    return all_trades, stats


def calc_stats(trades: list[Trade]) -> dict:
    """Calculate trading statistics."""
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "total_pnl": 0,
            "winners": 0,
            "losers": 0,
            "max_drawdown": 0,
        }

    total_pnl = sum(t.pnl_pct for t in trades)
    avg_pnl = total_pnl / len(trades)
    winners = [t for t in trades if t.pnl_pct > 0]
    losers = [t for t in trades if t.pnl_pct <= 0]
    win_rate = len(winners) / len(trades) * 100

    # Calculate max drawdown (simplified - cumulative P&L)
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in sorted(trades, key=lambda x: x.entry_date):
        cumulative += t.pnl_pct
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "winners": len(winners),
        "losers": len(losers),
        "max_drawdown": max_dd,
    }


async def main():
    symbol = "TSLA"
    start_date = "2024-01-01"

    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("=" * 80)
    print("REGIME-FILTERED INTRADAY STRATEGY BACKTEST - TSLA")
    print("=" * 80)
    print()
    print("Strategy Logic:")
    print("  - Use WSB sentiment to identify bullish/bearish regimes")
    print("  - Enter on intraday pullbacks (bullish) or bounces (bearish)")
    print("  - Tighter stops (-20%) and quicker profit taking (+40%)")
    print()
    print("Goal: Increase frequency while preserving directional edge")
    print()

    # Test matrix
    thresholds = [1.0, 1.5, 2.0, 3.0]
    dtes = [7, 14]
    moderate_options = [True, False]

    results = []

    print("Running test matrix...")
    print(f"  Thresholds: {thresholds}")
    print(f"  DTEs: {dtes}")
    print(f"  Include moderate regimes: {moderate_options}")
    print()

    for threshold, dte, include_mod in product(thresholds, dtes, moderate_options):
        config_name = f"T{threshold}%_DTE{dte}_{'Mod' if include_mod else 'Strong'}"

        trades, stats = await run_regime_backtest(
            symbol=symbol,
            start_date=start_date,
            quiver_api_key=quiver_api_key,
            pullback_threshold=threshold,
            target_dte=dte,
            include_moderate=include_mod,
        )

        trade_stats = calc_stats(trades)

        # Calculate trades per month
        if trades:
            first_date = datetime.strptime(start_date, "%Y-%m-%d")
            last_date = datetime.now()
            months = (last_date.year - first_date.year) * 12 + (last_date.month - first_date.month)
            if months == 0:
                months = 1
            trades_per_month = trade_stats["trades"] / months
        else:
            trades_per_month = 0

        results.append({
            "config": config_name,
            "threshold": threshold,
            "dte": dte,
            "include_mod": include_mod,
            "trades": trade_stats["trades"],
            "trades_per_month": trades_per_month,
            "win_rate": trade_stats["win_rate"],
            "avg_pnl": trade_stats["avg_pnl"],
            "total_pnl": trade_stats["total_pnl"],
            "max_drawdown": trade_stats["max_drawdown"],
            "trade_list": trades,
            "stats": stats,
        })

    # Sort by a composite score (frequency * profitability)
    for r in results:
        # Score = trades_per_month * avg_pnl (if positive) else 0
        if r["avg_pnl"] > 0 and r["trades"] >= 3:
            r["score"] = r["trades_per_month"] * r["avg_pnl"]
        else:
            r["score"] = 0

    results.sort(key=lambda x: -x["score"])

    # Results table
    print("=" * 80)
    print("TEST MATRIX RESULTS")
    print("=" * 80)
    print()
    print(f"{'Configuration':<25} {'Trades':>7} {'T/Mo':>6} {'Win%':>7} {'Avg P&L':>9} {'Total':>9} {'MaxDD':>8} {'Score':>8}")
    print("-" * 90)

    for r in results:
        print(f"{r['config']:<25} {r['trades']:>7} {r['trades_per_month']:>5.1f} {r['win_rate']:>6.1f}% {r['avg_pnl']:>+8.1f}% {r['total_pnl']:>+8.0f}% {r['max_drawdown']:>7.1f}% {r['score']:>8.1f}")

    print()
    print("-" * 90)
    print("BASELINE (current system): 4 trades, 0.2 T/Mo, 75% win, +119% avg, +476% total")
    print("-" * 90)

    # Find best configuration
    best = results[0] if results else None

    if best and best["score"] > 0:
        print()
        print("=" * 80)
        print(f"BEST CONFIGURATION: {best['config']}")
        print("=" * 80)
        print()
        print(f"Threshold: {best['threshold']}% pullback/bounce")
        print(f"DTE: {best['dte']} days")
        print(f"Moderate regimes: {'Yes' if best['include_mod'] else 'No (strong only)'}")
        print()
        print(f"Trades:           {best['trades']}")
        print(f"Trades/Month:     {best['trades_per_month']:.1f}")
        print(f"Win Rate:         {best['win_rate']:.1f}%")
        print(f"Avg P&L/Trade:    {best['avg_pnl']:+.1f}%")
        print(f"Total P&L:        {best['total_pnl']:+.0f}%")
        print(f"Max Drawdown:     {best['max_drawdown']:.1f}%")

        # Show individual trades
        if best["trade_list"]:
            print()
            print("Individual Trades:")
            print(f"{'Date':<12} {'Type':<6} {'Regime':<18} {'Entry':>8} {'Exit':>8} {'P&L':>9} {'Reason':<12} {'Hold':>5}")
            print("-" * 90)

            for t in sorted(best["trade_list"], key=lambda x: x.entry_date):
                regime_short = t.regime_type.replace("_", " ").title()[:15]
                print(f"{t.entry_date:<12} {t.option_type:<6} {regime_short:<18} ${t.entry_ask:>6.2f} ${t.exit_price:>6.2f} {t.pnl_pct:>+8.1f}% {t.exit_reason:<12} {t.holding_days:>4}d")

    # Analysis by regime type
    print()
    print("=" * 80)
    print("ANALYSIS BY REGIME TYPE (Best Config)")
    print("=" * 80)

    if best and best["trade_list"]:
        by_regime = defaultdict(list)
        for t in best["trade_list"]:
            by_regime[t.regime_type].append(t)

        for regime, trades in sorted(by_regime.items()):
            avg_pnl = sum(t.pnl_pct for t in trades) / len(trades)
            win_rate = len([t for t in trades if t.pnl_pct > 0]) / len(trades) * 100
            print(f"\n{regime}:")
            print(f"  Trades: {len(trades)}, Win Rate: {win_rate:.1f}%, Avg P&L: {avg_pnl:+.1f}%")

    # Exit reason analysis
    print()
    print("=" * 80)
    print("EXIT REASON ANALYSIS (Best Config)")
    print("=" * 80)

    if best and best["trade_list"]:
        by_exit = defaultdict(list)
        for t in best["trade_list"]:
            by_exit[t.exit_reason].append(t)

        for reason, trades in sorted(by_exit.items(), key=lambda x: -len(x[1])):
            avg_pnl = sum(t.pnl_pct for t in trades) / len(trades)
            print(f"  {reason:<15}: {len(trades):>3} trades, {avg_pnl:>+7.1f}% avg P&L")

    # Key questions answers
    print()
    print("=" * 80)
    print("KEY QUESTIONS ANSWERED")
    print("=" * 80)
    print()

    if best:
        baseline_trades = 4
        baseline_tpm = 0.2
        baseline_avg = 119.0
        baseline_wr = 75.0

        freq_increase = best["trades_per_month"] / baseline_tpm if baseline_tpm > 0 else 0

        print(f"1. Does intraday entry increase frequency?")
        print(f"   Baseline: {baseline_tpm:.1f} trades/month")
        print(f"   Best config: {best['trades_per_month']:.1f} trades/month")
        print(f"   Increase: {freq_increase:.1f}x")
        print()

        print(f"2. Do pullbacks during bullish regimes offer better entries?")
        bullish_trades = [t for t in best.get("trade_list", []) if "bullish" in t.regime_type]
        if bullish_trades:
            bull_avg = sum(t.pnl_pct for t in bullish_trades) / len(bullish_trades)
            bull_wr = len([t for t in bullish_trades if t.pnl_pct > 0]) / len(bullish_trades) * 100
            print(f"   Bullish trades: {len(bullish_trades)}, Avg P&L: {bull_avg:+.1f}%, Win Rate: {bull_wr:.0f}%")
        else:
            print(f"   No bullish regime trades")
        print()

        print(f"3. Do bounces during bearish regimes offer better entries?")
        bearish_trades = [t for t in best.get("trade_list", []) if "bearish" in t.regime_type]
        if bearish_trades:
            bear_avg = sum(t.pnl_pct for t in bearish_trades) / len(bearish_trades)
            bear_wr = len([t for t in bearish_trades if t.pnl_pct > 0]) / len(bearish_trades) * 100
            print(f"   Bearish trades: {len(bearish_trades)}, Avg P&L: {bear_avg:+.1f}%, Win Rate: {bear_wr:.0f}%")
        else:
            print(f"   No bearish regime trades")
        print()

        print(f"4. Sweet spot configuration?")
        print(f"   Threshold: {best['threshold']}% pullback/bounce")
        print(f"   DTE: {best['dte']} days")
        print(f"   Include moderate: {'Yes' if best['include_mod'] else 'No'}")
        print()

        print(f"5. Should we include moderate regimes?")
        strong_only = [r for r in results if not r["include_mod"] and r["trades"] > 0]
        with_mod = [r for r in results if r["include_mod"] and r["trades"] > 0]

        if strong_only and with_mod:
            strong_avg = sum(r["avg_pnl"] * r["trades"] for r in strong_only) / sum(r["trades"] for r in strong_only) if sum(r["trades"] for r in strong_only) > 0 else 0
            mod_avg = sum(r["avg_pnl"] * r["trades"] for r in with_mod) / sum(r["trades"] for r in with_mod) if sum(r["trades"] for r in with_mod) > 0 else 0
            print(f"   Strong-only avg P&L: {strong_avg:+.1f}%")
            print(f"   With moderate avg P&L: {mod_avg:+.1f}%")
        print()

    # Final verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    if best and best["trades"] >= 3:
        meets_freq = best["trades_per_month"] >= 1.0
        meets_avg = best["avg_pnl"] >= 15.0
        meets_wr = best["win_rate"] >= 40.0

        if meets_freq and meets_avg and meets_wr:
            print(">>> SUCCESS: New approach meets all criteria!")
            print(f"    Frequency: {best['trades_per_month']:.1f}/month (target: 1+)")
            print(f"    Avg P&L:   {best['avg_pnl']:+.1f}% (target: 15%+)")
            print(f"    Win Rate:  {best['win_rate']:.1f}% (target: 40%+)")
            print()
            print(">>> Recommendation: ADOPT this configuration")
        elif meets_avg or meets_wr:
            print(">>> PARTIAL: Some metrics met, others not")
            print(f"    Frequency: {best['trades_per_month']:.1f}/month {'PASS' if meets_freq else 'FAIL'}")
            print(f"    Avg P&L:   {best['avg_pnl']:+.1f}% {'PASS' if meets_avg else 'FAIL'}")
            print(f"    Win Rate:  {best['win_rate']:.1f}% {'PASS' if meets_wr else 'FAIL'}")
            print()
            print(">>> Consider hybrid approach or stick with baseline")
        else:
            print(">>> FAIL: Edge disappears with higher frequency")
            print(">>> Recommendation: STICK with low-frequency baseline system")
    else:
        print(">>> INSUFFICIENT DATA: Not enough trades to evaluate")
        print(">>> Recommendation: Need more data or adjust parameters")


if __name__ == "__main__":
    asyncio.run(main())
