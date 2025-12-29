#!/usr/bin/env python3
"""Compare strategy WITH vs WITHOUT sentiment reversal exit.

For the 44 trades that exited via sentiment reversal:
- What would have happened if we held until profit target or stop loss?
- Total P&L comparison
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.scripts.pnl_backtest import (
    run_backtest, TradeResult, CACHE_DB,
    PROFIT_TARGET, STOP_LOSS, MAX_HOLD_DAYS,
    get_contract_price, get_trading_days_after
)

load_dotenv(Path(__file__).parent.parent.parent / ".env")


def simulate_no_sentiment_exit(trade: TradeResult) -> dict:
    """Simulate what would have happened without sentiment reversal exit.

    Continue tracking until profit target, stop loss, or max hold.
    """
    # Start from the day AFTER the sentiment reversal exit
    # and track until we hit profit target or stop loss

    entry_price = trade.entry_price
    contract_id = trade.entry_contract

    # Get all trading days from entry to max hold
    tracking_days = get_trading_days_after(trade.entry_date, MAX_HOLD_DAYS + 10)

    exit_date = None
    exit_price = None
    exit_reason = None

    for track_date in tracking_days:
        price_data = get_contract_price(CACHE_DB, contract_id, track_date)

        if not price_data or price_data["mid"] <= 0:
            continue

        current_price = price_data["mid"]
        pnl_pct = (current_price - entry_price) / entry_price

        # Calculate DTE
        try:
            expiry_date = datetime.strptime(trade.entry_expiry, "%Y-%m-%d")
            current = datetime.strptime(track_date, "%Y-%m-%d")
            dte = (expiry_date - current).days
        except ValueError:
            dte = 30

        # Check exits (NO sentiment reversal)

        # 1. Profit target
        if pnl_pct >= PROFIT_TARGET:
            exit_date = track_date
            exit_price = current_price
            exit_reason = "profit_target"
            break

        # 2. Stop loss
        if pnl_pct <= STOP_LOSS:
            exit_date = track_date
            exit_price = current_price
            exit_reason = "stop_loss"
            break

        # 3. DTE exit (keep this - it's a hard constraint)
        if dte < 7:
            exit_date = track_date
            exit_price = current_price
            exit_reason = "dte_exit"
            break

    # If no exit, use max hold
    if exit_date is None:
        for track_date in reversed(tracking_days):
            price_data = get_contract_price(CACHE_DB, contract_id, track_date)
            if price_data and price_data["mid"] > 0:
                exit_date = track_date
                exit_price = price_data["mid"]
                exit_reason = "max_hold"
                break

    if exit_date is None:
        return {
            "original_exit": trade.exit_reason,
            "original_pnl": trade.pnl_pct,
            "simulated_exit": "no_data",
            "simulated_pnl": trade.pnl_pct,  # Same as original
            "difference": 0,
        }

    simulated_pnl = (exit_price - entry_price) / entry_price * 100

    try:
        entry_dt = datetime.strptime(trade.entry_date, "%Y-%m-%d")
        exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
        holding_days = (exit_dt - entry_dt).days
    except:
        holding_days = 0

    return {
        "trade": trade,
        "original_exit": trade.exit_reason,
        "original_pnl": trade.pnl_pct,
        "original_days": trade.holding_days,
        "simulated_exit": exit_reason,
        "simulated_pnl": simulated_pnl,
        "simulated_days": holding_days,
        "simulated_exit_price": exit_price,
        "difference": simulated_pnl - trade.pnl_pct,
    }


async def main():
    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("Running backtest to get trade data...")
    stats, trades = await run_backtest(
        symbols=["TSLA", "NVDA", "PLTR"],
        start_date="2024-01-01",
        quiver_api_key=quiver_api_key,
    )

    # Get sentiment reversal trades
    sent_trades = [t for t in trades if t.exit_reason == "sentiment_reversal"]
    other_trades = [t for t in trades if t.exit_reason != "sentiment_reversal"]

    print(f"\nFound {len(sent_trades)} sentiment reversal trades to simulate")
    print("=" * 70)

    # Simulate each trade without sentiment exit
    simulations = []
    for i, trade in enumerate(sent_trades):
        result = simulate_no_sentiment_exit(trade)
        simulations.append(result)
        if (i + 1) % 10 == 0:
            print(f"  Simulated {i+1}/{len(sent_trades)} trades...")

    print("\n" + "=" * 70)
    print("COMPARISON: WITH vs WITHOUT SENTIMENT REVERSAL EXIT")
    print("=" * 70)

    # Summary stats
    original_pnl = sum(s["original_pnl"] for s in simulations)
    simulated_pnl = sum(s["simulated_pnl"] for s in simulations)

    print(f"\nSentiment Reversal Trades ({len(simulations)} trades):")
    print(f"  WITH sentiment reversal exit:    {original_pnl:+.1f}% total ({original_pnl/len(simulations):+.1f}% avg)")
    print(f"  WITHOUT sentiment reversal exit: {simulated_pnl:+.1f}% total ({simulated_pnl/len(simulations):+.1f}% avg)")
    print(f"  DIFFERENCE:                      {simulated_pnl - original_pnl:+.1f}%")

    # Where did they end up?
    by_new_exit = defaultdict(list)
    for s in simulations:
        by_new_exit[s["simulated_exit"]].append(s)

    print(f"\n  What would have happened without sentiment exit:")
    for exit_type, items in sorted(by_new_exit.items(), key=lambda x: -len(x[1])):
        count = len(items)
        avg_orig = sum(i["original_pnl"] for i in items) / count
        avg_sim = sum(i["simulated_pnl"] for i in items) / count
        diff = avg_sim - avg_orig
        print(f"    {exit_type:<20}: {count:>3} trades | orig avg: {avg_orig:+.1f}% -> sim avg: {avg_sim:+.1f}% (diff: {diff:+.1f}%)")

    # Full strategy comparison
    print("\n" + "-" * 70)
    print("FULL STRATEGY COMPARISON")
    print("-" * 70)

    # Current strategy total P&L (all trades)
    current_total = sum(t.pnl_pct for t in trades)
    current_avg = current_total / len(trades)

    # Simulated strategy: other trades same + simulated sentiment trades
    other_pnl = sum(t.pnl_pct for t in other_trades)
    new_total = other_pnl + simulated_pnl
    new_avg = new_total / len(trades)

    print(f"\n  {'Strategy':<35} {'Total P&L':>15} {'Avg P&L':>12} {'Win Rate':>10}")
    print(f"  {'-' * 35} {'-' * 15} {'-' * 12} {'-' * 10}")

    # Current
    current_wins = sum(1 for t in trades if t.was_profitable)
    print(f"  {'WITH sentiment reversal exit':<35} {current_total:>+14.1f}% {current_avg:>+11.1f}% {current_wins/len(trades)*100:>9.1f}%")

    # Simulated
    sim_wins = sum(1 for t in other_trades if t.was_profitable) + sum(1 for s in simulations if s["simulated_pnl"] > 0)
    print(f"  {'WITHOUT sentiment reversal exit':<35} {new_total:>+14.1f}% {new_avg:>+11.1f}% {sim_wins/len(trades)*100:>9.1f}%")

    print(f"\n  DIFFERENCE: {new_total - current_total:+.1f}% total P&L")

    if new_total > current_total:
        print(f"\n  *** REMOVING sentiment reversal exit would IMPROVE returns by {new_total - current_total:.1f}% ***")
    else:
        print(f"\n  *** KEEPING sentiment reversal exit PROTECTS returns by {current_total - new_total:.1f}% ***")

    # Show individual trade comparisons
    print("\n" + "-" * 70)
    print("TRADE-BY-TRADE COMPARISON (sorted by difference)")
    print("-" * 70)

    # Sort by difference (biggest improvement first)
    sorted_sims = sorted(simulations, key=lambda x: -x["difference"])

    print("\n  Trades that would have been BETTER without sentiment exit:")
    print(f"  {'Date':<12} {'Symbol':<6} {'Orig Exit':<12} {'Orig P&L':>10} {'Sim Exit':<15} {'Sim P&L':>10} {'Diff':>10}")
    print("  " + "-" * 80)

    improved = [s for s in sorted_sims if s["difference"] > 1]
    for s in improved[:15]:
        t = s["trade"]
        print(f"  {t.signal_date:<12} {t.symbol:<6} sent_rev      {s['original_pnl']:>+9.1f}% {s['simulated_exit']:<15} {s['simulated_pnl']:>+9.1f}% {s['difference']:>+9.1f}%")

    print(f"\n  Trades that would have been WORSE without sentiment exit:")
    worse = [s for s in reversed(sorted_sims) if s["difference"] < -1]
    for s in worse[:15]:
        t = s["trade"]
        print(f"  {t.signal_date:<12} {t.symbol:<6} sent_rev      {s['original_pnl']:>+9.1f}% {s['simulated_exit']:<15} {s['simulated_pnl']:>+9.1f}% {s['difference']:>+9.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    improved_count = sum(1 for s in simulations if s["difference"] > 0)
    worse_count = sum(1 for s in simulations if s["difference"] < 0)

    print(f"\n  Of {len(simulations)} sentiment reversal exits:")
    print(f"    {improved_count} would have been BETTER without it")
    print(f"    {worse_count} would have been WORSE without it")

    if new_total > current_total:
        print(f"\n  RECOMMENDATION: REMOVE sentiment reversal as exit trigger")
        print(f"  Expected improvement: +{new_total - current_total:.1f}% total P&L")
    else:
        print(f"\n  RECOMMENDATION: KEEP sentiment reversal as exit trigger")
        print(f"  Protection value: +{current_total - new_total:.1f}% total P&L")


if __name__ == "__main__":
    asyncio.run(main())
