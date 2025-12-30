#!/usr/bin/env python3
"""Cooldown Strategy Comparison Backtest.

Compares three cooldown approaches for the regime-filtered strategy:
A) 1-day minimum between ANY entries (current backtest)
B) No cooldown - max 3 concurrent positions is the only limit
C) Cooldown while open - block same-direction entries while position is open
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
from enum import Enum

import aiohttp
from dotenv import load_dotenv
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")

CACHE_DB = Path(__file__).parent.parent.parent / "cache" / "options_data.db"

# Fixed strategy parameters (validated from previous backtest)
STOP_LOSS = -0.20
TAKE_PROFIT = 0.40
MAX_HOLD_DAYS = 10
MIN_OI = 500
PULLBACK_THRESHOLD = 1.5
TARGET_DTE = 7
MAX_CONCURRENT = 3


class CooldownStrategy(Enum):
    ONE_DAY = "A) 1-Day Cooldown"
    NO_COOLDOWN = "B) No Cooldown (max 3 concurrent)"
    WHILE_OPEN = "C) Cooldown While Open"


@dataclass
class Trade:
    symbol: str
    entry_date: str
    option_type: str
    regime_type: str
    entry_trigger: str
    contract_id: str = ""
    entry_price: float = 0.0
    entry_ask: float = 0.0
    exit_price: float = 0.0
    exit_date: str = ""
    exit_reason: str = ""
    pnl_pct: float = 0.0
    holding_days: int = 0
    pullback_pct: float = 0.0
    dte_at_entry: int = 0


def calculate_technicals(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
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
    """Count technical confirmations."""
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


def classify_regime(sentiment: float, tech_confirms: int) -> Optional[str]:
    """Classify regime based on sentiment and technicals."""
    if tech_confirms == 0:
        return None

    if sentiment > 0.12:
        return "strong_bullish"
    elif sentiment > 0.07:
        return "moderate_bullish"
    elif sentiment < -0.15:
        return "strong_bearish"
    elif sentiment < -0.08:
        return "moderate_bearish"
    return None


async def fetch_wsb_history(symbol: str, api_key: str) -> list[dict]:
    """Fetch historical WSB data."""
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
    """Get list of cached dates."""
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT trade_date FROM cache_metadata WHERE symbol = ? ORDER BY trade_date",
        (symbol,)
    )
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    return dates


def get_option_contract(db_path: Path, symbol: str, trade_date: str,
                        stock_price: float, option_type: str, target_dte: int) -> Optional[dict]:
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


def can_enter_trade(
    cooldown_strategy: CooldownStrategy,
    date: str,
    direction: str,  # "bullish" or "bearish"
    last_entry_date: Optional[str],
    open_positions: list[Trade],
) -> bool:
    """Check if we can enter a new trade based on cooldown strategy."""

    if cooldown_strategy == CooldownStrategy.ONE_DAY:
        # A) 1-day minimum between ANY entries
        if last_entry_date:
            days_since = (datetime.strptime(date, "%Y-%m-%d") -
                         datetime.strptime(last_entry_date, "%Y-%m-%d")).days
            if days_since < 1:
                return False
        return True

    elif cooldown_strategy == CooldownStrategy.NO_COOLDOWN:
        # B) No cooldown - only max concurrent limit
        return len(open_positions) < MAX_CONCURRENT

    elif cooldown_strategy == CooldownStrategy.WHILE_OPEN:
        # C) Cooldown while open for same direction
        # Can enter if:
        # 1. Under max concurrent positions AND
        # 2. No open position in the same direction
        if len(open_positions) >= MAX_CONCURRENT:
            return False

        for pos in open_positions:
            pos_direction = "bullish" if pos.option_type == "call" else "bearish"
            if pos_direction == direction:
                return False  # Already have an open position in this direction

        return True

    return True


def update_open_positions(
    open_positions: list[Trade],
    date: str,
    cached_dates: list[str],
) -> tuple[list[Trade], list[Trade]]:
    """Update open positions, closing any that hit exit criteria."""
    still_open = []
    closed = []

    for pos in open_positions:
        if pos.exit_date:  # Already closed
            closed.append(pos)
            continue

        # Get current price
        price_data = get_contract_price(CACHE_DB, pos.contract_id, date)
        if not price_data:
            still_open.append(pos)
            continue

        current_bid = price_data["bid"]
        pnl = (current_bid - pos.entry_ask) / pos.entry_ask

        # Check DTE
        try:
            expiry_date = datetime.strptime(price_data["expiry"], "%Y-%m-%d")
            current = datetime.strptime(date, "%Y-%m-%d")
            dte = (expiry_date - current).days
        except ValueError:
            dte = 7

        exit_reason = None

        if pnl >= TAKE_PROFIT:
            exit_reason = "take_profit"
        elif pnl <= STOP_LOSS:
            exit_reason = "stop_loss"
        elif dte <= 1:
            exit_reason = "dte_exit"

        if exit_reason:
            pos.exit_date = date
            pos.exit_price = current_bid
            pos.exit_reason = exit_reason
            pos.pnl_pct = pnl * 100
            entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
            exit_dt = datetime.strptime(date, "%Y-%m-%d")
            pos.holding_days = (exit_dt - entry_dt).days
            closed.append(pos)
        else:
            still_open.append(pos)

    return still_open, closed


async def run_backtest_with_cooldown(
    symbol: str,
    start_date: str,
    quiver_api_key: str,
    cooldown_strategy: CooldownStrategy,
) -> tuple[list[Trade], dict]:
    """Run backtest with specific cooldown strategy."""
    REGIME_WINDOW = 5

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

    # Build regime windows
    active_regimes = {}
    for wsb_date, sentiment in wsb_by_date.items():
        is_bullish = sentiment > 0
        tech_confirms = 0

        if wsb_date in technicals_df.index:
            row = technicals_df.loc[wsb_date]
            tech_confirms = check_technical_confirmation(row, is_bullish)

        regime = classify_regime(sentiment, tech_confirms)
        if regime:
            try:
                base_dt = datetime.strptime(wsb_date, "%Y-%m-%d")
                current = base_dt
                days_marked = 0
                while days_marked < REGIME_WINDOW:
                    current += timedelta(days=1)
                    if current.weekday() < 5:
                        date_str = current.strftime("%Y-%m-%d")
                        if date_str in technicals_df.index:
                            active_regimes[date_str] = (regime, sentiment, REGIME_WINDOW - days_marked)
                        days_marked += 1
            except ValueError:
                pass

    all_trades = []
    open_positions: list[Trade] = []
    last_entry_date = None
    max_concurrent_seen = 0

    stats = {
        "signals_seen": 0,
        "signals_blocked": 0,
        "max_concurrent": 0,
    }

    sorted_dates = sorted(set(cached_dates) & set(technicals_df.index))

    for date in sorted_dates:
        if date not in technicals_df.index:
            continue

        # First, update open positions (check for exits)
        open_positions, newly_closed = update_open_positions(open_positions, date, cached_dates)
        all_trades.extend(newly_closed)

        # Track max concurrent
        if len(open_positions) > max_concurrent_seen:
            max_concurrent_seen = len(open_positions)

        # Check if we're in an active regime
        if date not in active_regimes:
            continue

        regime, orig_sentiment, days_remaining = active_regimes[date]
        row = technicals_df.loc[date]

        # Check for pullback/bounce trigger
        pullback_pct = row.get('pullback_pct', 0)
        bounce_pct = row.get('bounce_pct', 0)

        entry_trigger = None
        trigger_pct = 0
        option_type = None
        direction = None

        if "bullish" in regime:
            if pullback_pct >= PULLBACK_THRESHOLD:
                entry_trigger = "pullback"
                trigger_pct = pullback_pct
                option_type = "call"
                direction = "bullish"
        else:
            if bounce_pct >= PULLBACK_THRESHOLD:
                entry_trigger = "bounce"
                trigger_pct = bounce_pct
                option_type = "put"
                direction = "bearish"

        if not entry_trigger:
            continue

        stats["signals_seen"] += 1

        # Check cooldown
        if not can_enter_trade(cooldown_strategy, date, direction, last_entry_date, open_positions):
            stats["signals_blocked"] += 1
            continue

        # Find option contract
        stock_price = row['Close']
        contract = get_option_contract(CACHE_DB, symbol, date, stock_price, option_type, TARGET_DTE)

        if not contract:
            continue

        # Create trade
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            option_type=option_type,
            regime_type=regime,
            entry_trigger=entry_trigger,
            contract_id=contract["contract_id"],
            entry_price=contract["mid"],
            entry_ask=contract["ask"],
            pullback_pct=trigger_pct,
            dte_at_entry=contract["dte"],
        )

        open_positions.append(trade)
        last_entry_date = date

    # Close any remaining open positions at end
    final_date = sorted_dates[-1] if sorted_dates else None
    if final_date:
        for pos in open_positions:
            if not pos.exit_date:
                price_data = get_contract_price(CACHE_DB, pos.contract_id, final_date)
                if price_data:
                    pos.exit_date = final_date
                    pos.exit_price = price_data["bid"]
                    pos.exit_reason = "end_of_data"
                    pos.pnl_pct = ((price_data["bid"] - pos.entry_ask) / pos.entry_ask) * 100
                    all_trades.append(pos)

    stats["max_concurrent"] = max_concurrent_seen

    return all_trades, stats


def calc_stats(trades: list[Trade]) -> dict:
    """Calculate trading statistics."""
    if not trades:
        return {
            "trades": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0,
            "winners": 0, "losers": 0, "max_drawdown": 0, "sharpe": 0,
        }

    total_pnl = sum(t.pnl_pct for t in trades)
    avg_pnl = total_pnl / len(trades)
    winners = [t for t in trades if t.pnl_pct > 0]
    losers = [t for t in trades if t.pnl_pct <= 0]
    win_rate = len(winners) / len(trades) * 100

    # Sharpe ratio (simplified)
    pnls = [t.pnl_pct for t in trades]
    if len(pnls) > 1:
        std_pnl = np.std(pnls)
        sharpe = (avg_pnl / std_pnl) if std_pnl > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
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
        "sharpe": sharpe,
    }


async def main():
    symbol = "TSLA"
    start_date = "2024-01-01"

    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("=" * 90)
    print("COOLDOWN STRATEGY COMPARISON BACKTEST - TSLA")
    print("=" * 90)
    print()
    print("Testing three cooldown approaches:")
    print("  A) 1-Day Cooldown: Minimum 1 day between ANY entries (current baseline)")
    print("  B) No Cooldown: Max 3 concurrent positions is the only limit")
    print("  C) While Open: Block same-direction entries while position is open")
    print()
    print(f"Fixed parameters: {PULLBACK_THRESHOLD}% pullback, {TARGET_DTE} DTE, +{TAKE_PROFIT*100:.0f}%/-{abs(STOP_LOSS)*100:.0f}% exits")
    print()

    results = {}

    for strategy in CooldownStrategy:
        print(f"Running {strategy.value}...", end=" ", flush=True)
        trades, stats = await run_backtest_with_cooldown(
            symbol=symbol,
            start_date=start_date,
            quiver_api_key=quiver_api_key,
            cooldown_strategy=strategy,
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

        results[strategy] = {
            "trades": trades,
            "stats": trade_stats,
            "trades_per_month": trades_per_month,
            "signals_seen": stats.get("signals_seen", 0),
            "signals_blocked": stats.get("signals_blocked", 0),
            "max_concurrent": stats.get("max_concurrent", 0),
        }

        print(f"{trade_stats['trades']} trades")

    # Results table
    print()
    print("=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)
    print()
    print(f"{'Scenario':<35} {'Trades':>7} {'T/Mo':>6} {'Win%':>7} {'Avg P&L':>9} {'Total':>9} {'MaxConc':>8} {'Sharpe':>7}")
    print("-" * 90)

    for strategy in CooldownStrategy:
        r = results[strategy]
        s = r["stats"]
        print(f"{strategy.value:<35} {s['trades']:>7} {r['trades_per_month']:>5.1f} {s['win_rate']:>6.1f}% {s['avg_pnl']:>+8.1f}% {s['total_pnl']:>+8.0f}% {r['max_concurrent']:>8} {s['sharpe']:>6.2f}")

    # Decision analysis
    print()
    print("=" * 90)
    print("DECISION ANALYSIS")
    print("=" * 90)
    print()

    baseline = results[CooldownStrategy.ONE_DAY]["stats"]

    for strategy in [CooldownStrategy.NO_COOLDOWN, CooldownStrategy.WHILE_OPEN]:
        alt = results[strategy]["stats"]
        r = results[strategy]

        print(f"\n{strategy.value} vs Baseline:")

        if alt["trades"] == 0:
            print("  No trades - cannot compare")
            continue

        trade_diff = alt["trades"] - baseline["trades"]
        pnl_diff = alt["total_pnl"] - baseline["total_pnl"]
        wr_diff = alt["win_rate"] - baseline["win_rate"]
        avg_diff = alt["avg_pnl"] - baseline["avg_pnl"]

        print(f"  Trades:      {alt['trades']:>3} vs {baseline['trades']:>3} ({trade_diff:+d})")
        print(f"  Win Rate:    {alt['win_rate']:>5.1f}% vs {baseline['win_rate']:>5.1f}% ({wr_diff:+.1f}pp)")
        print(f"  Avg P&L:     {alt['avg_pnl']:>+6.1f}% vs {baseline['avg_pnl']:>+6.1f}% ({avg_diff:+.1f}%)")
        print(f"  Total P&L:   {alt['total_pnl']:>+6.0f}% vs {baseline['total_pnl']:>+6.0f}% ({pnl_diff:+.0f}%)")
        print(f"  Max Concurrent: {r['max_concurrent']}")
        print(f"  Signals blocked: {r['signals_blocked']} of {r['signals_seen']}")

        # Decision criteria
        better_total = alt["total_pnl"] > baseline["total_pnl"]
        acceptable_wr = alt["win_rate"] >= baseline["win_rate"] - 5
        acceptable_avg = alt["avg_pnl"] >= 10

        if better_total and acceptable_wr and acceptable_avg:
            print(f"  >>> BETTER: Higher total P/L with acceptable metrics")
        elif better_total:
            print(f"  >>> MIXED: Higher total P/L but some metrics worse")
        else:
            print(f"  >>> WORSE: Lower total P/L")

    # Final recommendation
    print()
    print("=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)
    print()

    best_strategy = max(results.keys(), key=lambda s: results[s]["stats"]["total_pnl"])
    best = results[best_strategy]

    print(f"Best strategy by Total P&L: {best_strategy.value}")
    print(f"  Total P&L: {best['stats']['total_pnl']:+.0f}%")
    print(f"  Trades: {best['stats']['trades']}")
    print(f"  Win Rate: {best['stats']['win_rate']:.1f}%")
    print()

    if best_strategy == CooldownStrategy.ONE_DAY:
        print(">>> KEEP current 1-day cooldown - it produces the best results")
    else:
        wr_drop = baseline["win_rate"] - best["stats"]["win_rate"]
        if wr_drop <= 5 and best["stats"]["avg_pnl"] >= 10:
            print(f">>> ADOPT {best_strategy.value}")
            print(f"    Win rate drop ({wr_drop:.1f}pp) is acceptable")
            print(f"    Average P&L ({best['stats']['avg_pnl']:+.1f}%) is above 10% threshold")
        else:
            print(">>> KEEP current 1-day cooldown")
            print(f"    Alternative has better total but worse risk-adjusted metrics")

    # Show individual trades for best config
    print()
    print("=" * 90)
    print(f"INDIVIDUAL TRADES - {best_strategy.value}")
    print("=" * 90)
    print()
    print(f"{'Date':<12} {'Type':<6} {'Regime':<18} {'Entry':>8} {'Exit':>8} {'P&L':>9} {'Reason':<12} {'Hold':>5}")
    print("-" * 90)

    for t in sorted(best["trades"], key=lambda x: x.entry_date)[:20]:
        regime_short = t.regime_type.replace("_", " ").title()[:15]
        print(f"{t.entry_date:<12} {t.option_type:<6} {regime_short:<18} ${t.entry_ask:>6.2f} ${t.exit_price:>6.2f} {t.pnl_pct:>+8.1f}% {t.exit_reason:<12} {t.holding_days:>4}d")

    if len(best["trades"]) > 20:
        print(f"... and {len(best['trades']) - 20} more trades")


if __name__ == "__main__":
    asyncio.run(main())
