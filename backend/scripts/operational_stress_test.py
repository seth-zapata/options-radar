#!/usr/bin/env python3
"""Operational stress tests for the trading strategy.

Tests:
1. Missed Entry: 20% of signals randomly skipped
2. Delayed Exit: Stop losses execute 1 day late
3. Partial Fills: Only 80% of position size filled

These simulate real-world operational issues.
"""

from __future__ import annotations

import asyncio
import os
import random
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

# Final config
TRAILING_ACTIVATION = 0.30
TRAILING_DISTANCE = 0.15
STOP_LOSS = -0.30
MIN_DTE = 7
MAX_HOLD_DAYS = 45
MIN_OI = 500


@dataclass
class Trade:
    symbol: str
    signal_date: str
    entry_ask: float
    exit_bid: float
    pnl_pct: float
    exit_reason: str


async def fetch_wsb_history(symbol: str, api_key: str) -> list[dict]:
    url = f"https://api.quiverquant.com/beta/historical/wallstreetbets/{symbol}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return []
            data = await response.json()
            return data if isinstance(data, list) else []


def get_price_data(symbol: str, start: str, end: str) -> dict[str, float]:
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end)
    if hist.empty:
        return {}
    return {date.strftime("%Y-%m-%d"): row["Close"] for date, row in hist.iterrows()}


def get_atm_contract(db_path, symbol, trade_date, underlying_price, option_type="call"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT contract_id, expiry, strike, bid, ask, open_interest
        FROM options_contracts
        WHERE symbol = ? AND trade_date = ? AND option_type = ?
          AND strike BETWEEN ? AND ? AND bid > 0 AND ask > 0
        ORDER BY strike, expiry
    """, (symbol, trade_date, option_type, underlying_price * 0.95, underlying_price * 1.05))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    td = datetime.strptime(trade_date, "%Y-%m-%d")
    best = None
    best_score = float('inf')

    for row in rows:
        contract_id, expiry, strike, bid, ask, oi = row
        try:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            dte = (expiry_date - td).days
        except ValueError:
            continue

        if dte < 14 or dte > 60:
            continue
        if (oi or 0) < MIN_OI:
            continue

        strike_dist = abs(strike - underlying_price) / underlying_price
        dte_dist = abs(dte - 30) / 30
        score = strike_dist + dte_dist * 0.5

        if score < best_score:
            best_score = score
            best = {
                "contract_id": contract_id,
                "expiry": expiry,
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "dte": dte,
            }
    return best


def get_contract_price(db_path, contract_id, trade_date):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT bid, ask, expiry FROM options_contracts
        WHERE contract_id = ? AND trade_date = ?
    """, (contract_id, trade_date))
    row = cursor.fetchone()
    conn.close()
    if not row or row[0] <= 0 or row[1] <= 0:
        return None
    return {"bid": row[0], "ask": row[1], "expiry": row[2]}


def get_trading_days_after(start_date: str, days: int) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    result = []
    current = start + timedelta(days=1)
    while len(result) < days:
        if current.weekday() < 5:
            result.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return result


async def run_baseline_backtest(symbols, start_date, quiver_api_key):
    """Run baseline backtest, return list of trade dicts with full tracking data."""
    all_trades = []

    for symbol in symbols:
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache_metadata WHERE symbol = ?", (symbol,))
        if cursor.fetchone()[0] == 0:
            conn.close()
            continue
        conn.close()

        wsb_data = await fetch_wsb_history(symbol, quiver_api_key)
        if not wsb_data:
            continue

        wsb_by_date = {}
        for item in wsb_data:
            date = item.get("Date", "")[:10]
            if date >= start_date:
                wsb_by_date[date] = {
                    "sentiment": float(item.get("Sentiment", 0) or 0),
                    "mentions": int(item.get("Mentions", 0) or 0),
                }

        end_date = datetime.now().strftime("%Y-%m-%d")
        prices = get_price_data(symbol, start_date, end_date)
        if not prices:
            continue

        for date, wsb in sorted(wsb_by_date.items()):
            if wsb["mentions"] < 5 or abs(wsb["sentiment"]) < 0.1:
                continue

            underlying_price = prices.get(date)
            if not underlying_price:
                continue

            option_type = "call" if wsb["sentiment"] > 0 else "put"
            contract = get_atm_contract(CACHE_DB, symbol, date, underlying_price, option_type)
            if not contract:
                continue

            # Track position
            entry_ask = contract["ask"]
            high_water = 0.0
            trailing_active = False

            tracking_days = get_trading_days_after(date, MAX_HOLD_DAYS + 10)
            exit_data = None

            for i, track_date in enumerate(tracking_days):
                price_data = get_contract_price(CACHE_DB, contract["contract_id"], track_date)
                if not price_data:
                    continue

                current_bid = price_data["bid"]
                pnl = (current_bid - entry_ask) / entry_ask

                if pnl > high_water:
                    high_water = pnl
                if not trailing_active and pnl >= TRAILING_ACTIVATION:
                    trailing_active = True

                try:
                    expiry_date = datetime.strptime(contract["expiry"], "%Y-%m-%d")
                    current = datetime.strptime(track_date, "%Y-%m-%d")
                    dte = (expiry_date - current).days
                except ValueError:
                    dte = 30

                exit_reason = None
                if trailing_active and pnl <= high_water - TRAILING_DISTANCE:
                    exit_reason = "trailing_stop"
                elif pnl <= STOP_LOSS:
                    exit_reason = "stop_loss"
                elif dte < MIN_DTE:
                    exit_reason = "dte_exit"

                if exit_reason:
                    exit_data = {
                        "exit_date": track_date,
                        "exit_day_index": i,
                        "exit_bid": current_bid,
                        "exit_reason": exit_reason,
                        "pnl_pct": pnl * 100,
                    }
                    break

            if exit_data:
                all_trades.append({
                    "symbol": symbol,
                    "signal_date": date,
                    "contract_id": contract["contract_id"],
                    "entry_ask": entry_ask,
                    "tracking_days": tracking_days,
                    **exit_data,
                })

    return all_trades


def simulate_missed_entries(trades, miss_rate=0.20, seed=42):
    """Simulate missing miss_rate fraction of entries randomly."""
    random.seed(seed)
    kept = [t for t in trades if random.random() > miss_rate]
    return kept


def simulate_delayed_exits(trades, delay_days=1):
    """Simulate stop losses executing delay_days late."""
    delayed_trades = []

    for t in trades:
        if t["exit_reason"] != "stop_loss":
            # Non-stop-loss exits aren't affected
            delayed_trades.append(t)
            continue

        # Try to get price 1 day later
        exit_idx = t["exit_day_index"]
        tracking_days = t["tracking_days"]

        if exit_idx + delay_days < len(tracking_days):
            new_exit_date = tracking_days[exit_idx + delay_days]
            price_data = get_contract_price(CACHE_DB, t["contract_id"], new_exit_date)

            if price_data:
                new_pnl = (price_data["bid"] - t["entry_ask"]) / t["entry_ask"] * 100
                delayed_trades.append({
                    **t,
                    "exit_date": new_exit_date,
                    "exit_bid": price_data["bid"],
                    "pnl_pct": new_pnl,
                    "delayed": True,
                })
            else:
                delayed_trades.append(t)
        else:
            delayed_trades.append(t)

    return delayed_trades


def calc_stats(trades):
    if not trades:
        return {"count": 0, "total_pnl": 0, "avg_pnl": 0, "win_rate": 0}
    total = sum(t["pnl_pct"] for t in trades)
    winners = sum(1 for t in trades if t["pnl_pct"] > 0)
    return {
        "count": len(trades),
        "total_pnl": total,
        "avg_pnl": total / len(trades),
        "win_rate": winners / len(trades) * 100,
    }


async def main():
    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("=" * 70)
    print("OPERATIONAL STRESS TESTS")
    print("=" * 70)
    print("Testing strategy robustness to real-world operational issues\n")

    # Run baseline
    print("Running baseline backtest...")
    trades = await run_baseline_backtest(
        symbols=["TSLA", "NVDA", "PLTR"],
        start_date="2024-01-01",
        quiver_api_key=quiver_api_key,
    )

    baseline = calc_stats(trades)
    print(f"Baseline: {baseline['count']} trades, {baseline['total_pnl']:+.0f}% total, {baseline['avg_pnl']:+.1f}% avg\n")

    # Test 1: Missed Entries (20%)
    print("=" * 70)
    print("TEST 1: MISSED ENTRIES (20% of signals randomly skipped)")
    print("=" * 70)

    # Run multiple seeds for robustness
    missed_results = []
    for seed in range(10):
        missed_trades = simulate_missed_entries(trades, miss_rate=0.20, seed=seed)
        stats = calc_stats(missed_trades)
        missed_results.append(stats)

    avg_count = sum(r["count"] for r in missed_results) / len(missed_results)
    avg_total = sum(r["total_pnl"] for r in missed_results) / len(missed_results)
    avg_avg = sum(r["avg_pnl"] for r in missed_results) / len(missed_results)
    avg_wr = sum(r["win_rate"] for r in missed_results) / len(missed_results)

    print(f"\nBaseline:    {baseline['count']} trades, {baseline['total_pnl']:>+7.0f}% total, {baseline['avg_pnl']:>+6.1f}% avg, {baseline['win_rate']:.1f}% WR")
    print(f"20% Missed:  {avg_count:.0f} trades, {avg_total:>+7.0f}% total, {avg_avg:>+6.1f}% avg, {avg_wr:.1f}% WR")
    print(f"Impact:      {avg_count - baseline['count']:.0f} trades, {avg_total - baseline['total_pnl']:>+7.0f}% total")

    # Expected: proportional reduction in total P&L
    expected_reduction = baseline['total_pnl'] * 0.20
    actual_reduction = baseline['total_pnl'] - avg_total
    print(f"\nExpected P&L loss: {expected_reduction:+.0f}%")
    print(f"Actual P&L loss: {actual_reduction:+.0f}%")
    print(f">>> Avg P&L per trade should be ~unchanged: {baseline['avg_pnl']:+.1f}% -> {avg_avg:+.1f}%")

    # Test 2: Delayed Exits (1 day late on stop losses)
    print("\n" + "=" * 70)
    print("TEST 2: DELAYED EXITS (stop losses execute 1 day late)")
    print("=" * 70)

    delayed_trades = simulate_delayed_exits(trades, delay_days=1)
    delayed = calc_stats(delayed_trades)

    # Count affected trades
    stop_loss_trades = [t for t in trades if t["exit_reason"] == "stop_loss"]
    delayed_stops = [t for t in delayed_trades if t.get("delayed")]

    print(f"\nStop loss trades: {len(stop_loss_trades)}")
    print(f"Successfully delayed: {len(delayed_stops)}")

    if delayed_stops:
        original_pnl = sum(t["pnl_pct"] for t in stop_loss_trades)
        delayed_pnl = sum(t["pnl_pct"] for t in delayed_stops)
        print(f"\nStop loss trades - original avg P&L: {original_pnl/len(stop_loss_trades):+.1f}%")
        print(f"Stop loss trades - delayed avg P&L:  {delayed_pnl/len(delayed_stops):+.1f}%")
        print(f"Avg slippage per stop loss: {(delayed_pnl - original_pnl)/len(delayed_stops):+.1f}%")

    print(f"\nBaseline:    {baseline['count']} trades, {baseline['total_pnl']:>+7.0f}% total, {baseline['avg_pnl']:>+6.1f}% avg")
    print(f"1-Day Delay: {delayed['count']} trades, {delayed['total_pnl']:>+7.0f}% total, {delayed['avg_pnl']:>+6.1f}% avg")
    print(f"Impact:                    {delayed['total_pnl'] - baseline['total_pnl']:>+7.0f}% total, {delayed['avg_pnl'] - baseline['avg_pnl']:>+6.1f}% avg")

    # Test 3: Partial Fills (80% of position)
    print("\n" + "=" * 70)
    print("TEST 3: PARTIAL FILLS (only 80% of position size filled)")
    print("=" * 70)

    # Partial fills don't change % returns, just total capital at risk
    # Simulate by scaling total P&L by 0.8
    partial_total = baseline['total_pnl'] * 0.80

    print(f"\nBaseline total P&L: {baseline['total_pnl']:+.0f}%")
    print(f"80% fills total P&L: {partial_total:+.0f}%")
    print(f"Impact: {partial_total - baseline['total_pnl']:+.0f}% ({(partial_total - baseline['total_pnl'])/baseline['total_pnl']*100:.0f}% reduction)")
    print(f"\n>>> Avg P&L per trade unchanged: {baseline['avg_pnl']:+.1f}%")
    print(">>> Only affects total capital deployed, not strategy edge")

    # Summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)

    print(f"\n{'Scenario':<25} {'Total P&L':>12} {'Avg P&L':>10} {'Impact':>12}")
    print("-" * 60)
    print(f"{'Baseline':<25} {baseline['total_pnl']:>+11.0f}% {baseline['avg_pnl']:>+9.1f}% {'-':>12}")
    print(f"{'20% Missed Entries':<25} {avg_total:>+11.0f}% {avg_avg:>+9.1f}% {avg_total - baseline['total_pnl']:>+11.0f}%")
    print(f"{'1-Day Delayed Stops':<25} {delayed['total_pnl']:>+11.0f}% {delayed['avg_pnl']:>+9.1f}% {delayed['total_pnl'] - baseline['total_pnl']:>+11.0f}%")
    print(f"{'80% Partial Fills':<25} {partial_total:>+11.0f}% {baseline['avg_pnl']:>+9.1f}% {partial_total - baseline['total_pnl']:>+11.0f}%")

    # Combined worst case
    worst_case = avg_total * 0.80  # Missed entries + partial fills
    # Add delayed exit slippage
    if delayed_stops:
        slippage_total = delayed['total_pnl'] - baseline['total_pnl']
        worst_case += slippage_total

    print(f"\n{'WORST CASE (all 3)':<25} {worst_case:>+11.0f}% {'-':>10} {worst_case - baseline['total_pnl']:>+11.0f}%")

    print("\n" + "-" * 60)
    if worst_case > 0:
        print(f">>> Strategy remains PROFITABLE under worst-case operations: {worst_case:+.0f}%")
    else:
        print(f">>> WARNING: Strategy becomes unprofitable under worst-case: {worst_case:+.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
