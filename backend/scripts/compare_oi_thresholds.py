#!/usr/bin/env python3
"""Compare OI threshold impact on trade count and P&L."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.scripts.pnl_backtest import run_backtest

load_dotenv(Path(__file__).parent.parent.parent / ".env")


async def main():
    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("Running backtest to get all trades with OI data...")
    stats, trades = await run_backtest(
        symbols=["TSLA", "NVDA", "PLTR"],
        start_date="2024-01-01",
        quiver_api_key=quiver_api_key,
    )

    print(f"\nTotal trades with OI data: {len(trades)}")

    # Test different OI thresholds
    thresholds = [100, 500, 1000]

    print("\n" + "=" * 70)
    print("OI THRESHOLD COMPARISON")
    print("=" * 70)
    print(f"\n{'Threshold':<12} {'Trades':<10} {'Blocked':<10} {'Total P&L':>12} {'Avg P&L':>10} {'Win Rate':>10}")
    print("-" * 70)

    baseline_trades = len(trades)
    baseline_pnl = sum(t.pnl_pct for t in trades)

    for threshold in thresholds:
        # Filter trades by OI threshold
        filtered = [t for t in trades if t.open_interest >= threshold]
        blocked = baseline_trades - len(filtered)

        if filtered:
            total_pnl = sum(t.pnl_pct for t in filtered)
            avg_pnl = total_pnl / len(filtered)
            winners = sum(1 for t in filtered if t.pnl_pct > 0)
            win_rate = winners / len(filtered) * 100
        else:
            total_pnl = avg_pnl = win_rate = 0

        print(f"OI >= {threshold:<6} {len(filtered):<10} {blocked:<10} {total_pnl:>+11.0f}% {avg_pnl:>+9.1f}% {win_rate:>9.1f}%")

    # Detailed breakdown for OI >= 500
    print("\n" + "=" * 70)
    print("DETAILED BREAKDOWN: OI >= 500 (Compromise)")
    print("=" * 70)

    oi_500 = [t for t in trades if t.open_interest >= 500]
    oi_1000 = [t for t in trades if t.open_interest >= 1000]

    # Trades that would be added by using 500 instead of 1000
    extra_trades = [t for t in trades if 500 <= t.open_interest < 1000]

    print(f"\nTrades with 500 <= OI < 1000: {len(extra_trades)}")
    if extra_trades:
        extra_pnl = sum(t.pnl_pct for t in extra_trades)
        extra_avg = extra_pnl / len(extra_trades)
        extra_winners = sum(1 for t in extra_trades if t.pnl_pct > 0)
        extra_win_rate = extra_winners / len(extra_trades) * 100

        print(f"  Total P&L: {extra_pnl:+.0f}%")
        print(f"  Avg P&L: {extra_avg:+.1f}%")
        print(f"  Win rate: {extra_win_rate:.1f}%")

        # Are these extra trades good or bad?
        if oi_1000:
            oi_1000_avg = sum(t.pnl_pct for t in oi_1000) / len(oi_1000)
            print(f"\n  Comparison:")
            print(f"    OI >= 1000 avg: {oi_1000_avg:+.1f}%")
            print(f"    500-999 OI avg: {extra_avg:+.1f}%")

            if extra_avg >= oi_1000_avg * 0.8:  # Within 20% is acceptable
                print(f"    → GOOD: Medium-OI trades perform comparably")
            else:
                print(f"    → CAUTION: Medium-OI trades underperform by {(oi_1000_avg - extra_avg):.1f}%")

    # Monthly breakdown for OI >= 500
    print("\n" + "-" * 50)
    print("Monthly P&L (OI >= 500)")
    print("-" * 50)

    from collections import defaultdict
    monthly = defaultdict(list)
    for t in oi_500:
        month = t.signal_date[:7]
        monthly[month].append(t.pnl_pct)

    for month in sorted(monthly.keys()):
        trades_list = monthly[month]
        total = sum(trades_list)
        print(f"  {month}: {len(trades_list):>3} trades, {total:>+7.0f}%")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if extra_trades:
        if extra_avg > 0:
            print(f"\n✓ OI >= 500 is a GOOD compromise:")
            print(f"  - Adds {len(extra_trades)} trades with positive avg P&L ({extra_avg:+.1f}%)")
            print(f"  - Total opportunity: {len(oi_500)} trades vs {len(oi_1000)} at OI >= 1000")
            print(f"  - Can tighten to 1000 for live trading if fills are problematic")
        else:
            print(f"\n⚠ OI >= 500 adds {len(extra_trades)} trades with NEGATIVE avg P&L ({extra_avg:+.1f}%)")
            print(f"  - Consider keeping OI >= 1000 threshold")


if __name__ == "__main__":
    asyncio.run(main())
