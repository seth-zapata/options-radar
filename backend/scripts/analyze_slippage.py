#!/usr/bin/env python3
"""Analyze stop loss slippage and DTE exits."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.scripts.pnl_backtest import run_backtest, TradeResult

load_dotenv(Path(__file__).parent.parent.parent / ".env")


async def main():
    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("Running backtest...")
    stats, trades = await run_backtest(
        symbols=["TSLA", "NVDA", "PLTR"],
        start_date="2024-01-01",
        quiver_api_key=quiver_api_key,
    )

    # Q2: DTE Exit Analysis
    print("\n" + "=" * 70)
    print("Q2: DTE EXIT ANALYSIS")
    print("=" * 70)

    dte_exits = [t for t in trades if t.exit_reason == "dte_exit"]
    print(f"\nDTE exits: {len(dte_exits)} out of {len(trades)} trades")

    if not dte_exits:
        print("\nWhy no DTE exits?")
        print("  - Target DTE at entry: 30 days")
        print("  - DTE exit triggers at: <7 days")
        print("  - Most trades hit profit target (+50%) or stop loss (-30%) before 23+ days pass")

        # Show holding days distribution
        profit_trades = [t for t in trades if t.exit_reason == "profit_target"]
        stop_trades = [t for t in trades if t.exit_reason == "stop_loss"]

        if profit_trades:
            avg_hold_profit = sum(t.holding_days for t in profit_trades) / len(profit_trades)
            max_hold_profit = max(t.holding_days for t in profit_trades)
            print(f"\n  Profit target trades:")
            print(f"    Avg holding days: {avg_hold_profit:.1f}")
            print(f"    Max holding days: {max_hold_profit}")

        if stop_trades:
            avg_hold_stop = sum(t.holding_days for t in stop_trades) / len(stop_trades)
            max_hold_stop = max(t.holding_days for t in stop_trades)
            print(f"\n  Stop loss trades:")
            print(f"    Avg holding days: {avg_hold_stop:.1f}")
            print(f"    Max holding days: {max_hold_stop}")

    # Q3: Stop Loss Slippage Analysis
    print("\n" + "=" * 70)
    print("Q3: STOP LOSS SLIPPAGE ANALYSIS")
    print("=" * 70)

    stop_trades = [t for t in trades if t.exit_reason == "stop_loss"]
    print(f"\nStop loss trades: {len(stop_trades)}")
    print(f"Stop loss trigger: -30%")
    print(f"Average actual loss: {sum(t.pnl_pct for t in stop_trades) / len(stop_trades):.1f}%")

    # Distribution
    print("\n  P&L distribution for stop loss exits:")
    ranges = [(-100, -70), (-70, -50), (-50, -40), (-40, -35), (-35, -30), (-30, -25)]
    for low, high in ranges:
        count = sum(1 for t in stop_trades if low <= t.pnl_pct < high)
        pct = count / len(stop_trades) * 100 if stop_trades else 0
        bar = "*" * count
        print(f"    {low:>4}% to {high:>4}%: {count:>3} ({pct:>5.1f}%) {bar}")

    # Why the slippage?
    print("\n  Why is actual loss > 30%?")
    print("  The backtest checks END OF DAY prices only.")
    print("  If option gaps down overnight or drops fast intraday,")
    print("  the first available price might already be below -30%.")

    # Show worst slippage examples
    worst = sorted(stop_trades, key=lambda t: t.pnl_pct)[:10]
    print("\n  Worst slippage examples:")
    print(f"  {'Date':<12} {'Symbol':<6} {'P&L':>10} {'Days':>6} {'Entry':>8} {'Exit':>8}")
    print("  " + "-" * 60)
    for t in worst:
        print(f"  {t.signal_date:<12} {t.symbol:<6} {t.pnl_pct:>+9.1f}% {t.holding_days:>6} ${t.entry_price:>7.2f} ${t.exit_price:>7.2f}")

    # Calculate slippage stats
    slippage = [t.pnl_pct - (-30) for t in stop_trades]  # How much worse than -30%
    avg_slippage = sum(slippage) / len(slippage)
    max_slippage = min(slippage)  # Most negative = worst slippage

    print(f"\n  Slippage stats:")
    print(f"    Average slippage: {avg_slippage:.1f}% beyond -30% trigger")
    print(f"    Worst slippage: {max_slippage:.1f}% beyond -30% trigger")

    # Trades that exited close to -30%
    clean_stops = [t for t in stop_trades if -35 <= t.pnl_pct <= -30]
    print(f"\n  Clean stops (-35% to -30%): {len(clean_stops)} ({len(clean_stops)/len(stop_trades)*100:.0f}%)")
    print(f"  Slippage stops (< -35%): {len(stop_trades) - len(clean_stops)} ({(len(stop_trades) - len(clean_stops))/len(stop_trades)*100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
