#!/usr/bin/env python3
"""Deep analysis of P&L backtest results.

Answers:
1. Average P&L breakdown BY EXIT TYPE
2. Trades that were directionally correct but lost money
3. Sentiment reversal analysis - helping or hurting?
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.scripts.pnl_backtest import run_backtest, TradeResult

# Load env
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def analyze_by_exit_type(trades: list[TradeResult]) -> None:
    """Q1: Average P&L breakdown by exit type."""
    print("\n" + "=" * 70)
    print("Q1: P&L BREAKDOWN BY EXIT TYPE")
    print("=" * 70)

    by_exit = defaultdict(list)
    for t in trades:
        by_exit[t.exit_reason].append(t)

    print(f"\n{'Exit Type':<25} {'Count':>8} {'Wins':>8} {'Win%':>8} {'Avg P&L':>12} {'Total P&L':>12}")
    print("-" * 73)

    for exit_type in ["profit_target", "stop_loss", "sentiment_reversal", "dte_exit", "max_hold"]:
        exit_trades = by_exit.get(exit_type, [])
        if not exit_trades:
            continue

        count = len(exit_trades)
        wins = sum(1 for t in exit_trades if t.was_profitable)
        win_pct = wins / count * 100
        avg_pnl = sum(t.pnl_pct for t in exit_trades) / count
        total_pnl = sum(t.pnl_pct for t in exit_trades)

        print(f"{exit_type:<25} {count:>8} {wins:>8} {win_pct:>7.1f}% {avg_pnl:>+11.1f}% {total_pnl:>+11.1f}%")

    print("-" * 73)
    total = len(trades)
    total_wins = sum(1 for t in trades if t.was_profitable)
    total_avg = sum(t.pnl_pct for t in trades) / total
    total_sum = sum(t.pnl_pct for t in trades)
    print(f"{'TOTAL':<25} {total:>8} {total_wins:>8} {total_wins/total*100:>7.1f}% {total_avg:>+11.1f}% {total_sum:>+11.1f}%")

    # Why is avg win > 50%?
    profit_target_trades = by_exit.get("profit_target", [])
    if profit_target_trades:
        avg_pt = sum(t.pnl_pct for t in profit_target_trades) / len(profit_target_trades)
        max_pt = max(t.pnl_pct for t in profit_target_trades)
        min_pt = min(t.pnl_pct for t in profit_target_trades)
        print(f"\n  Profit target trades: avg={avg_pt:+.1f}%, min={min_pt:+.1f}%, max={max_pt:+.1f}%")

        # The profit target should trigger at 50%, but if price gaps up overnight
        # or we check intraday highs, we could exit higher
        over_50 = [t for t in profit_target_trades if t.pnl_pct > 55]
        print(f"  Trades that exited >55%: {len(over_50)}")
        if over_50[:3]:
            print("  Examples of large profit target exits:")
            for t in over_50[:3]:
                print(f"    {t.signal_date} {t.symbol}: +{t.pnl_pct:.1f}% "
                      f"(entry ${t.entry_price:.2f} -> exit ${t.exit_price:.2f})")

    sentiment_trades = by_exit.get("sentiment_reversal", [])
    if sentiment_trades:
        wins = [t for t in sentiment_trades if t.was_profitable]
        losses = [t for t in sentiment_trades if not t.was_profitable]
        avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
        print(f"\n  Sentiment reversal: {len(wins)} wins (avg +{avg_win:.1f}%), {len(losses)} losses (avg {avg_loss:.1f}%)")


def analyze_directional_losses(trades: list[TradeResult]) -> None:
    """Q2: Trades directionally correct but lost money."""
    print("\n" + "=" * 70)
    print("Q2: DIRECTIONALLY CORRECT BUT LOST MONEY")
    print("=" * 70)

    # We need stock price data to check if the move happened
    import yfinance as yf
    from datetime import datetime, timedelta

    # For each losing trade, check if the stock actually moved in the predicted direction
    directional_correct_losses = []

    for t in trades:
        if t.was_profitable:
            continue  # Only look at losses

        # Get stock prices from signal date to exit date
        try:
            start = datetime.strptime(t.signal_date, "%Y-%m-%d")
            end = datetime.strptime(t.exit_date, "%Y-%m-%d") + timedelta(days=1)

            ticker = yf.Ticker(t.symbol)
            hist = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

            if len(hist) < 2:
                continue

            entry_stock_price = hist.iloc[0]["Close"]
            exit_stock_price = hist.iloc[-1]["Close"]
            stock_move = (exit_stock_price - entry_stock_price) / entry_stock_price * 100

            # Was direction correct?
            is_call = t.signal_type == "BUY_CALL"
            stock_went_up = stock_move > 0

            direction_correct = (is_call and stock_went_up) or (not is_call and not stock_went_up)

            if direction_correct:
                directional_correct_losses.append({
                    "trade": t,
                    "stock_move": stock_move,
                    "entry_stock": entry_stock_price,
                    "exit_stock": exit_stock_price,
                })
        except Exception as e:
            continue

    print(f"\nTotal losing trades: {sum(1 for t in trades if not t.was_profitable)}")
    print(f"Directionally correct but lost: {len(directional_correct_losses)}")

    if not directional_correct_losses:
        print("\nNo directionally correct losses found.")
        return

    # Analyze these trades
    print(f"\nThese trades predicted the stock direction correctly but still lost on the option:")
    print()

    by_exit = defaultdict(list)
    for item in directional_correct_losses:
        by_exit[item["trade"].exit_reason].append(item)

    print(f"{'Exit Reason':<25} {'Count':>8} {'Avg Option Loss':>15} {'Avg Stock Move':>15}")
    print("-" * 65)

    for exit_type, items in sorted(by_exit.items(), key=lambda x: -len(x[1])):
        avg_option_loss = sum(i["trade"].pnl_pct for i in items) / len(items)
        avg_stock_move = sum(abs(i["stock_move"]) for i in items) / len(items)
        print(f"{exit_type:<25} {len(items):>8} {avg_option_loss:>+14.1f}% {avg_stock_move:>+14.1f}%")

    print("\n" + "-" * 70)
    print("DETAILED EXAMPLES: Direction correct but option lost money")
    print("-" * 70)

    # Sort by most painful (largest loss despite correct direction)
    directional_correct_losses.sort(key=lambda x: x["trade"].pnl_pct)

    for item in directional_correct_losses[:10]:
        t = item["trade"]
        print(f"\n  {t.signal_date} {t.symbol} {t.signal_type}")
        print(f"    Stock: ${item['entry_stock']:.2f} -> ${item['exit_stock']:.2f} ({item['stock_move']:+.1f}%)")
        print(f"    Option: ${t.entry_price:.2f} -> ${t.exit_price:.2f} ({t.pnl_pct:+.1f}%)")
        print(f"    Exit: {t.exit_reason} after {t.holding_days} days")
        print(f"    Strike: ${t.entry_strike:.2f}, Delta: {t.entry_delta:.3f}, IV: {t.entry_iv:.1f}%")

        # What happened?
        if t.exit_reason == "stop_loss":
            print(f"    STOPPED OUT before move completed")
        elif t.exit_reason == "sentiment_reversal":
            print(f"    SENTIMENT REVERSED while underwater")


def analyze_sentiment_reversals(trades: list[TradeResult]) -> None:
    """Q3: Are sentiment reversal exits helping or hurting?"""
    print("\n" + "=" * 70)
    print("Q3: SENTIMENT REVERSAL ANALYSIS")
    print("=" * 70)

    sentiment_trades = [t for t in trades if t.exit_reason == "sentiment_reversal"]
    other_trades = [t for t in trades if t.exit_reason != "sentiment_reversal"]

    if not sentiment_trades:
        print("\nNo sentiment reversal exits found.")
        return

    # Split into wins and losses
    sent_wins = [t for t in sentiment_trades if t.was_profitable]
    sent_losses = [t for t in sentiment_trades if not t.was_profitable]

    print(f"\nSentiment reversal exits: {len(sentiment_trades)} ({len(sentiment_trades)/len(trades)*100:.1f}% of all trades)")
    print(f"  Wins: {len(sent_wins)} ({len(sent_wins)/len(sentiment_trades)*100:.1f}%)")
    print(f"  Losses: {len(sent_losses)} ({len(sent_losses)/len(sentiment_trades)*100:.1f}%)")

    avg_win = sum(t.pnl_pct for t in sent_wins) / len(sent_wins) if sent_wins else 0
    avg_loss = sum(t.pnl_pct for t in sent_losses) / len(sent_losses) if sent_losses else 0
    total_pnl = sum(t.pnl_pct for t in sentiment_trades)
    avg_pnl = total_pnl / len(sentiment_trades)

    print(f"\n  Avg P&L on wins: +{avg_win:.1f}%")
    print(f"  Avg P&L on losses: {avg_loss:.1f}%")
    print(f"  Overall avg P&L: {avg_pnl:+.1f}%")
    print(f"  Total contribution: {total_pnl:+.1f}%")

    # Compare to other exits
    other_pnl = sum(t.pnl_pct for t in other_trades)
    other_avg = other_pnl / len(other_trades) if other_trades else 0

    print(f"\n  Comparison to other exits:")
    print(f"    Sentiment reversal avg: {avg_pnl:+.1f}%")
    print(f"    Other exits avg: {other_avg:+.1f}%")

    if avg_pnl > other_avg:
        print(f"    --> Sentiment reversals are OUTPERFORMING by {avg_pnl - other_avg:.1f}%")
    else:
        print(f"    --> Sentiment reversals are UNDERPERFORMING by {other_avg - avg_pnl:.1f}%")

    # The big question: are we cutting winners or saving from losers?
    print("\n" + "-" * 70)
    print("CUTTING WINNERS OR SAVING FROM LOSERS?")
    print("-" * 70)

    # For winners: were we cutting too early? Would have hit profit target?
    cutting_winners = []
    saving_from_losers = []
    PROFIT_TARGET = 50  # 50% as percentage

    # We'd need to track what would have happened without the exit
    # For now, just look at the trades that were positive when we exited

    for t in sentiment_trades:
        if t.pnl_pct > 0 and t.pnl_pct < PROFIT_TARGET:
            # Positive but below profit target - we cut a potential winner
            cutting_winners.append(t)
        elif t.pnl_pct < 0:
            # Negative - did sentiment reversal save us from a bigger loss?
            # The trade would have continued to stop loss or worse
            saving_from_losers.append(t)

    print(f"\n  Cutting potential winners (exited 0-50%): {len(cutting_winners)}")
    if cutting_winners:
        avg_cut = sum(t.pnl_pct for t in cutting_winners) / len(cutting_winners)
        print(f"    Avg P&L when cut: +{avg_cut:.1f}%")
        print(f"    These might have hit +50% profit target if we held")

    print(f"\n  Exited underwater (saved from worse?): {len(saving_from_losers)}")
    if saving_from_losers:
        avg_saved = sum(t.pnl_pct for t in saving_from_losers) / len(saving_from_losers)
        print(f"    Avg P&L when saved: {avg_saved:.1f}%")
        print(f"    Would have hit -30% stop loss if not for reversal exit")

    # Show distribution
    print("\n  P&L distribution for sentiment reversal exits:")
    ranges = [(-100, -30), (-30, -10), (-10, 0), (0, 10), (10, 30), (30, 50), (50, 200)]
    for low, high in ranges:
        count = sum(1 for t in sentiment_trades if low <= t.pnl_pct < high)
        bar = "*" * count
        print(f"    {low:>4} to {high:>4}%: {count:>3} {bar}")

    # Show example trades
    print("\n" + "-" * 70)
    print("EXAMPLE SENTIMENT REVERSAL EXITS")
    print("-" * 70)

    print("\n  WINS (caught early profit):")
    for t in sorted(sent_wins, key=lambda x: -x.pnl_pct)[:5]:
        print(f"    {t.signal_date} {t.symbol}: {t.pnl_pct:+.1f}% in {t.holding_days} days")

    print("\n  LOSSES (exited underwater):")
    for t in sorted(sent_losses, key=lambda x: x.pnl_pct)[:5]:
        print(f"    {t.signal_date} {t.symbol}: {t.pnl_pct:+.1f}% in {t.holding_days} days")


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

    print(f"\nLoaded {len(trades)} trades for analysis")

    # Run all analyses
    analyze_by_exit_type(trades)
    analyze_directional_losses(trades)
    analyze_sentiment_reversals(trades)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
