#!/usr/bin/env python3
"""Out-of-sample validation backtest with FROZEN parameters.

Tests strategy on tickers it was NEVER optimized for (AMD, GME).
Uses IDENTICAL parameters to in-sample backtest - no tweaking allowed.

Parameters frozen from in-sample optimization:
- ATM calls/puts based on WSB sentiment
- ~30 DTE target
- OI >= 500 filter
- Trailing stop: +30% activation, 15% trail, -30% hard stop
- Bid/Ask fills (ask entry, bid exit)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import aiohttp
from dotenv import load_dotenv
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")

CACHE_DB = Path(__file__).parent.parent.parent / "cache" / "options_data.db"

# FROZEN PARAMETERS - DO NOT CHANGE
# These were optimized on TSLA, NVDA, PLTR
TRAILING_ACTIVATION = 0.30  # Activate at +30%
TRAILING_DISTANCE = 0.15    # Trail 15% from high water
STOP_LOSS = -0.30           # Hard stop at -30%
MIN_DTE = 7
MAX_HOLD_DAYS = 45
MIN_OI = 500  # Compromise threshold
TARGET_DTE = 30


@dataclass
class Trade:
    symbol: str
    signal_date: str
    signal_type: str

    # Entry
    entry_date: str
    entry_contract: str
    entry_bid: float
    entry_ask: float
    entry_mid: float
    open_interest: int
    volume: int

    # Exit (filled during tracking)
    exit_date: str = ""
    exit_bid: float = 0.0
    exit_ask: float = 0.0
    exit_mid: float = 0.0
    exit_reason: str = ""

    # P&L calculated both ways
    pnl_mid: float = 0.0
    pnl_bidask: float = 0.0
    holding_days: int = 0
    spread_at_entry_pct: float = 0.0


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


def get_price_data(symbol: str, start: str, end: str) -> dict[str, float]:
    """Fetch historical price data from yfinance."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end)
    if hist.empty:
        return {}
    return {
        date.strftime("%Y-%m-%d"): row["Close"]
        for date, row in hist.iterrows()
    }


def get_atm_contract(
    db_path: Path,
    symbol: str,
    trade_date: str,
    underlying_price: float,
    option_type: str = "call",
    target_dte: int = TARGET_DTE,
) -> dict | None:
    """Find ATM contract closest to target DTE with bid/ask data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT contract_id, expiry, strike, bid, ask, delta, iv, open_interest, volume
        FROM options_contracts
        WHERE symbol = ?
          AND trade_date = ?
          AND option_type = ?
          AND strike BETWEEN ? AND ?
          AND bid > 0 AND ask > 0
        ORDER BY strike, expiry
    """, (
        symbol, trade_date, option_type,
        underlying_price * 0.95, underlying_price * 1.05
    ))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    try:
        td = datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        return None

    best = None
    best_score = float('inf')

    for row in rows:
        contract_id, expiry, strike, bid, ask, delta, iv, oi, vol = row

        try:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            dte = (expiry_date - td).days
        except ValueError:
            continue

        if dte < 14 or dte > 60:
            continue

        strike_dist = abs(strike - underlying_price) / underlying_price
        dte_dist = abs(dte - target_dte) / target_dte
        score = strike_dist + dte_dist * 0.5

        if score < best_score:
            best_score = score
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else 0
            best = {
                "contract_id": contract_id,
                "expiry": expiry,
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct": spread_pct,
                "delta": delta,
                "iv": iv,
                "dte": dte,
                "open_interest": oi or 0,
                "volume": vol or 0,
            }

    return best


def get_contract_price(db_path: Path, contract_id: str, trade_date: str) -> dict | None:
    """Get contract price with bid/ask on a specific date."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT bid, ask, expiry
        FROM options_contracts
        WHERE contract_id = ? AND trade_date = ?
    """, (contract_id, trade_date))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    bid, ask, expiry = row
    if bid <= 0 or ask <= 0:
        return None

    mid = (bid + ask) / 2
    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "expiry": expiry,
    }


def get_trading_days_after(start_date: str, days: int) -> list[str]:
    """Get list of trading days after a start date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    result = []
    current = start + timedelta(days=1)

    while len(result) < days:
        if current.weekday() < 5:
            result.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return result


async def run_backtest(
    symbols: list[str],
    start_date: str,
    quiver_api_key: str,
) -> list[Trade]:
    """Run backtest with FROZEN parameters from in-sample optimization."""

    all_trades: list[Trade] = []

    for symbol in symbols:
        print(f"Processing {symbol}...")

        # Check cache
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache_metadata WHERE symbol = ?", (symbol,))
        count = cursor.fetchone()[0]
        if count == 0:
            print(f"  No cached data for {symbol}")
            conn.close()
            continue
        print(f"  Found {count} cached dates")
        conn.close()

        # Get WSB signals
        wsb_data = await fetch_wsb_history(symbol, quiver_api_key)
        if not wsb_data:
            print(f"  No WSB data for {symbol}")
            continue
        print(f"  Found {len(wsb_data)} WSB data points")

        wsb_by_date = {}
        for item in wsb_data:
            date = item.get("Date", "")[:10]
            if date >= start_date:
                wsb_by_date[date] = {
                    "sentiment": float(item.get("Sentiment", 0) or 0),
                    "mentions": int(item.get("Mentions", 0) or 0),
                }

        # Get prices
        end_date = datetime.now().strftime("%Y-%m-%d")
        prices = get_price_data(symbol, start_date, end_date)
        if not prices:
            print(f"  No price data for {symbol}")
            continue
        print(f"  Found {len(prices)} price data points")

        # Track filtering
        signals_with_volume = 0
        signals_with_options = 0
        signals_with_oi = 0

        # Process signals
        for date, wsb in sorted(wsb_by_date.items()):
            sentiment = wsb["sentiment"]
            mentions = wsb["mentions"]

            if mentions < 5 or abs(sentiment) < 0.1:
                continue
            signals_with_volume += 1

            underlying_price = prices.get(date)
            if not underlying_price:
                continue

            is_bullish = sentiment > 0
            signal_type = "BUY_CALL" if is_bullish else "BUY_PUT"
            option_type = "call" if is_bullish else "put"

            contract = get_atm_contract(CACHE_DB, symbol, date, underlying_price, option_type)
            if not contract:
                continue
            signals_with_options += 1

            # Apply OI filter (FROZEN parameter)
            if contract["open_interest"] < MIN_OI:
                continue
            signals_with_oi += 1

            # Create trade
            trade = Trade(
                symbol=symbol,
                signal_date=date,
                signal_type=signal_type,
                entry_date=date,
                entry_contract=contract["contract_id"],
                entry_bid=contract["bid"],
                entry_ask=contract["ask"],
                entry_mid=contract["mid"],
                open_interest=contract["open_interest"],
                volume=contract["volume"],
                spread_at_entry_pct=contract["spread_pct"],
            )

            # Track position with trailing stop (FROZEN parameters)
            entry_ask = contract["ask"]  # Realistic entry
            entry_mid = contract["mid"]
            high_water_mark_bidask = 0.0
            trailing_active_bidask = False

            tracking_days = get_trading_days_after(date, MAX_HOLD_DAYS + 10)

            for track_date in tracking_days:
                price_data = get_contract_price(CACHE_DB, contract["contract_id"], track_date)
                if not price_data:
                    continue

                current_mid = price_data["mid"]
                current_bid = price_data["bid"]

                # Calculate P&L both ways
                pnl_mid = (current_mid - entry_mid) / entry_mid
                pnl_bidask = (current_bid - entry_ask) / entry_ask

                # Update high water mark
                if pnl_bidask > high_water_mark_bidask:
                    high_water_mark_bidask = pnl_bidask

                # Check trailing stop activation
                if not trailing_active_bidask and pnl_bidask >= TRAILING_ACTIVATION:
                    trailing_active_bidask = True

                # Calculate DTE
                try:
                    expiry_date = datetime.strptime(contract["expiry"], "%Y-%m-%d")
                    current = datetime.strptime(track_date, "%Y-%m-%d")
                    dte = (expiry_date - current).days
                except ValueError:
                    dte = 30

                # Check exit conditions
                exit_reason = None

                # 1. Trailing stop (if activated)
                if trailing_active_bidask:
                    trailing_stop_level = high_water_mark_bidask - TRAILING_DISTANCE
                    if pnl_bidask <= trailing_stop_level:
                        exit_reason = "trailing_stop"

                # 2. Hard stop loss
                if not exit_reason and pnl_bidask <= STOP_LOSS:
                    exit_reason = "stop_loss"

                # 3. DTE exit
                if not exit_reason and dte < MIN_DTE:
                    exit_reason = "dte_exit"

                if exit_reason:
                    trade.exit_date = track_date
                    trade.exit_bid = price_data["bid"]
                    trade.exit_ask = price_data["ask"]
                    trade.exit_mid = current_mid
                    trade.exit_reason = exit_reason
                    trade.pnl_mid = pnl_mid * 100
                    trade.pnl_bidask = pnl_bidask * 100

                    try:
                        entry_dt = datetime.strptime(date, "%Y-%m-%d")
                        exit_dt = datetime.strptime(track_date, "%Y-%m-%d")
                        trade.holding_days = (exit_dt - entry_dt).days
                    except ValueError:
                        trade.holding_days = 0

                    break

            # If no exit, use last available price
            if not trade.exit_date:
                for track_date in reversed(tracking_days):
                    price_data = get_contract_price(CACHE_DB, contract["contract_id"], track_date)
                    if price_data:
                        trade.exit_date = track_date
                        trade.exit_bid = price_data["bid"]
                        trade.exit_ask = price_data["ask"]
                        trade.exit_mid = price_data["mid"]
                        trade.exit_reason = "max_hold"
                        trade.pnl_mid = ((price_data["mid"] - entry_mid) / entry_mid) * 100
                        trade.pnl_bidask = ((price_data["bid"] - entry_ask) / entry_ask) * 100
                        break

            if trade.exit_date:
                all_trades.append(trade)

        print(f"  Signals with volume: {signals_with_volume}")
        print(f"  Signals with options: {signals_with_options}")
        print(f"  Signals passing OI filter: {signals_with_oi}")
        print(f"  Completed trades: {len([t for t in all_trades if t.symbol == symbol])}")

    return all_trades


def calc_stats(trades, pnl_attr):
    """Calculate stats for a set of trades."""
    if not trades:
        return {
            "total_pnl": 0,
            "avg_pnl": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "winners": 0,
            "losers": 0,
        }
    total_pnl = sum(getattr(t, pnl_attr) for t in trades)
    avg_pnl = total_pnl / len(trades)
    winners = [t for t in trades if getattr(t, pnl_attr) > 0]
    losers = [t for t in trades if getattr(t, pnl_attr) <= 0]
    win_rate = len(winners) / len(trades) * 100
    avg_win = sum(getattr(t, pnl_attr) for t in winners) / len(winners) if winners else 0
    avg_loss = sum(getattr(t, pnl_attr) for t in losers) / len(losers) if losers else 0
    return {
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "winners": len(winners),
        "losers": len(losers),
    }


async def main():
    parser = argparse.ArgumentParser(description="Out-of-sample validation backtest")
    parser.add_argument(
        "--symbols",
        type=str,
        default="AMD,GME",
        help="Comma-separated list of OUT-OF-SAMPLE symbols (default: AMD,GME)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION BACKTEST")
    print("=" * 70)
    print()
    print(">>> CRITICAL: These tickers were NEVER used in parameter optimization!")
    print()
    print("FROZEN Parameters (from in-sample optimization):")
    print(f"  - Entry: ATM calls/puts based on WSB sentiment")
    print(f"  - Target DTE: {TARGET_DTE} days")
    print(f"  - OI Filter: >= {MIN_OI}")
    print(f"  - Trailing Stop: +{int(TRAILING_ACTIVATION*100)}% activation, {int(TRAILING_DISTANCE*100)}% trail")
    print(f"  - Hard Stop: {int(STOP_LOSS*100)}%")
    print(f"  - Fill Method: Bid/Ask (ask entry, bid exit)")
    print()
    print(f"Out-of-Sample Tickers: {symbols}")
    print()

    trades = await run_backtest(
        symbols=symbols,
        start_date=args.start,
        quiver_api_key=quiver_api_key,
    )

    print(f"\nTotal out-of-sample trades: {len(trades)}")

    if not trades:
        print("\nNo trades to analyze. Check if:")
        print("  1. Options data is cached for these symbols")
        print("  2. WSB data exists for these symbols")
        return

    # Calculate out-of-sample stats
    oos_stats = calc_stats(trades, "pnl_bidask")

    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE RESULTS (Bid/Ask Fills)")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'Total Trades':<25} {len(trades):>15}")
    print(f"{'Total P&L':<25} {oos_stats['total_pnl']:>+14.0f}%")
    print(f"{'Avg P&L per trade':<25} {oos_stats['avg_pnl']:>+14.1f}%")
    print(f"{'Win Rate':<25} {oos_stats['win_rate']:>14.1f}%")
    print(f"{'Winners':<25} {oos_stats['winners']:>15}")
    print(f"{'Losers':<25} {oos_stats['losers']:>15}")
    print(f"{'Avg Winner':<25} {oos_stats['avg_win']:>+14.1f}%")
    print(f"{'Avg Loser':<25} {oos_stats['avg_loss']:>+14.1f}%")

    # By symbol breakdown
    print("\n" + "=" * 70)
    print("BY SYMBOL BREAKDOWN")
    print("=" * 70)

    for symbol in symbols:
        symbol_trades = [t for t in trades if t.symbol == symbol]
        if not symbol_trades:
            print(f"\n{symbol}: No trades")
            continue

        s = calc_stats(symbol_trades, "pnl_bidask")
        avg_spread = sum(t.spread_at_entry_pct for t in symbol_trades) / len(symbol_trades)

        print(f"\n{symbol} ({len(symbol_trades)} trades):")
        print(f"  Total P&L:    {s['total_pnl']:>+8.0f}%")
        print(f"  Avg P&L:      {s['avg_pnl']:>+8.1f}%")
        print(f"  Win Rate:     {s['win_rate']:>8.1f}%")
        print(f"  Avg Spread:   {avg_spread:>8.1f}%")

    # Compare to in-sample benchmarks
    print("\n" + "=" * 70)
    print("COMPARISON TO IN-SAMPLE RESULTS")
    print("=" * 70)
    print()
    print("In-Sample Results (TSLA, NVDA, PLTR - bid/ask fills):")
    print("  TSLA: +46.5% avg per trade, 67% win rate (6 trades)")
    print("  NVDA: +11.8% avg per trade, 50% win rate (44 trades)")
    print("  PLTR: +5.7% avg per trade, 35% win rate (21 trades)")
    print("  Combined: +11.9% avg per trade, 48% win rate (71 trades)")
    print()
    print(f"Out-of-Sample Results ({', '.join(symbols)}):")
    print(f"  Combined: {oos_stats['avg_pnl']:+.1f}% avg per trade, {oos_stats['win_rate']:.0f}% win rate ({len(trades)} trades)")

    # Verdict
    print("\n" + "=" * 70)
    print("VALIDATION VERDICT")
    print("=" * 70)

    in_sample_avg = 11.9  # From previous backtest
    in_sample_win_rate = 48

    if oos_stats['avg_pnl'] > 0 and oos_stats['win_rate'] > 40:
        print("\n>>> PASSED: Strategy is profitable on out-of-sample tickers")
        if oos_stats['avg_pnl'] >= in_sample_avg * 0.7:
            print(">>> Performance is comparable to in-sample (within 30%)")
        else:
            print(f">>> Performance degraded but still positive ({oos_stats['avg_pnl']:.1f}% vs {in_sample_avg:.1f}% in-sample)")
    elif oos_stats['avg_pnl'] > 0:
        print("\n>>> MARGINAL: Strategy is profitable but win rate is concerning")
        print(f"    Out-of-sample: {oos_stats['win_rate']:.0f}% win rate")
        print(f"    In-sample:     {in_sample_win_rate}% win rate")
    else:
        print("\n>>> FAILED: Strategy is NOT profitable on out-of-sample tickers")
        print(">>> This suggests overfitting to in-sample data")

    # Exit reasons breakdown
    print("\n" + "=" * 70)
    print("EXIT REASONS")
    print("=" * 70)
    exit_reasons = defaultdict(lambda: {"count": 0, "pnl": 0})
    for t in trades:
        exit_reasons[t.exit_reason]["count"] += 1
        exit_reasons[t.exit_reason]["pnl"] += t.pnl_bidask

    for reason, data in sorted(exit_reasons.items(), key=lambda x: -x[1]["count"]):
        avg = data["pnl"] / data["count"] if data["count"] > 0 else 0
        print(f"  {reason:<15}: {data['count']:>3} trades, {avg:>+7.1f}% avg P&L")

    # Monthly breakdown
    print("\n" + "=" * 70)
    print("MONTHLY BREAKDOWN (Out-of-Sample)")
    print("=" * 70)

    monthly = defaultdict(list)
    for t in trades:
        month = t.signal_date[:7]
        monthly[month].append(t.pnl_bidask)

    print(f"\n{'Month':<10} {'Trades':>8} {'Total P&L':>12} {'Avg P&L':>10}")
    print("-" * 50)
    for month in sorted(monthly.keys()):
        pnls = monthly[month]
        total = sum(pnls)
        avg = total / len(pnls)
        print(f"{month:<10} {len(pnls):>8} {total:>+11.0f}% {avg:>+9.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
