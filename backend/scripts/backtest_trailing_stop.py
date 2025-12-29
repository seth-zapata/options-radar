#!/usr/bin/env python3
"""Backtest trailing stop variant vs current strategy.

Trailing Stop Logic:
- Once position reaches +30% profit, activate trailing stop
- Trailing stop = highest profit reached minus 15%
- If trailing stop never activates, use existing -30% hard stop

Compare against current strategy:
- 50% profit target
- 30% stop loss
- DTE < 7 exit
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from collections import defaultdict

import aiohttp
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.scripts.pnl_backtest import (
    CACHE_DB, PROFIT_TARGET, STOP_LOSS, MIN_DTE, MAX_HOLD_DAYS,
    get_atm_contract, get_contract_price, get_trading_days_after,
    fetch_wsb_history, get_price_data
)

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Trailing stop parameters
TRAILING_ACTIVATION = 0.30  # Activate at +30%
TRAILING_DISTANCE = 0.15    # Trail by 15%


@dataclass
class TradeResultV2:
    """Trade result with strategy comparison."""
    symbol: str
    signal_date: str
    signal_type: str
    entry_price: float
    entry_strike: float

    # Current strategy results
    current_exit_date: str
    current_exit_price: float
    current_exit_reason: str
    current_pnl_pct: float
    current_holding_days: int

    # Trailing stop results
    trailing_exit_date: str
    trailing_exit_price: float
    trailing_exit_reason: str
    trailing_pnl_pct: float
    trailing_holding_days: int
    trailing_activated: bool
    trailing_high_water: float  # Highest P&L reached

    @property
    def pnl_difference(self) -> float:
        return self.trailing_pnl_pct - self.current_pnl_pct


def simulate_both_strategies(
    entry_price: float,
    contract_id: str,
    entry_date: str,
    expiry: str,
) -> tuple[dict, dict]:
    """Simulate both current and trailing stop strategies on same trade."""

    tracking_days = get_trading_days_after(entry_date, MAX_HOLD_DAYS + 10)

    # Current strategy state
    current_exit = None
    current_price = None
    current_reason = None

    # Trailing stop state
    trailing_exit = None
    trailing_price = None
    trailing_reason = None
    trailing_activated = False
    high_water_mark = 0.0  # Highest P&L percentage reached

    for track_date in tracking_days:
        price_data = get_contract_price(CACHE_DB, contract_id, track_date)

        if not price_data or price_data["mid"] <= 0:
            continue

        current_mid = price_data["mid"]
        pnl_pct = (current_mid - entry_price) / entry_price

        # Calculate DTE
        try:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            current = datetime.strptime(track_date, "%Y-%m-%d")
            dte = (expiry_date - current).days
        except ValueError:
            dte = 30

        # === CURRENT STRATEGY ===
        if current_exit is None:
            # Profit target
            if pnl_pct >= PROFIT_TARGET:
                current_exit = track_date
                current_price = current_mid
                current_reason = "profit_target"

            # Stop loss
            elif pnl_pct <= STOP_LOSS:
                current_exit = track_date
                current_price = current_mid
                current_reason = "stop_loss"

            # DTE exit
            elif dte < MIN_DTE:
                current_exit = track_date
                current_price = current_mid
                current_reason = "dte_exit"

        # === TRAILING STOP STRATEGY ===
        if trailing_exit is None:
            # Update high water mark
            if pnl_pct > high_water_mark:
                high_water_mark = pnl_pct

            # Check if trailing stop should activate
            if not trailing_activated and pnl_pct >= TRAILING_ACTIVATION:
                trailing_activated = True

            # If trailing stop is active, check if triggered
            if trailing_activated:
                trailing_stop_level = high_water_mark - TRAILING_DISTANCE

                if pnl_pct <= trailing_stop_level:
                    trailing_exit = track_date
                    trailing_price = current_mid
                    trailing_reason = "trailing_stop"

            # Still use hard stop if trailing never activated
            if not trailing_activated and pnl_pct <= STOP_LOSS:
                trailing_exit = track_date
                trailing_price = current_mid
                trailing_reason = "hard_stop"

            # DTE exit applies to both
            if dte < MIN_DTE and trailing_exit is None:
                trailing_exit = track_date
                trailing_price = current_mid
                trailing_reason = "dte_exit"

        # If both have exited, we're done
        if current_exit and trailing_exit:
            break

    # Handle max hold if no exit
    for track_date in reversed(tracking_days):
        price_data = get_contract_price(CACHE_DB, contract_id, track_date)
        if price_data and price_data["mid"] > 0:
            if current_exit is None:
                current_exit = track_date
                current_price = price_data["mid"]
                current_reason = "max_hold"
            if trailing_exit is None:
                trailing_exit = track_date
                trailing_price = price_data["mid"]
                trailing_reason = "max_hold"
            break

    # Calculate final P&L
    if current_exit and current_price:
        current_pnl = (current_price - entry_price) / entry_price * 100
    else:
        current_pnl = 0
        current_exit = entry_date
        current_price = entry_price
        current_reason = "no_data"

    if trailing_exit and trailing_price:
        trailing_pnl = (trailing_price - entry_price) / entry_price * 100
    else:
        trailing_pnl = 0
        trailing_exit = entry_date
        trailing_price = entry_price
        trailing_reason = "no_data"

    # Calculate holding days
    try:
        entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
        current_days = (datetime.strptime(current_exit, "%Y-%m-%d") - entry_dt).days
        trailing_days = (datetime.strptime(trailing_exit, "%Y-%m-%d") - entry_dt).days
    except:
        current_days = 0
        trailing_days = 0

    return (
        {
            "exit_date": current_exit,
            "exit_price": current_price,
            "exit_reason": current_reason,
            "pnl_pct": current_pnl,
            "holding_days": current_days,
        },
        {
            "exit_date": trailing_exit,
            "exit_price": trailing_price,
            "exit_reason": trailing_reason,
            "pnl_pct": trailing_pnl,
            "holding_days": trailing_days,
            "activated": trailing_activated,
            "high_water": high_water_mark * 100,  # As percentage
        }
    )


async def run_comparison(
    symbols: list[str],
    start_date: str,
    quiver_api_key: str,
    min_mentions: int = 5,
    sentiment_threshold: float = 0.1,
) -> list[TradeResultV2]:
    """Run both strategies on all trades."""

    all_trades: list[TradeResultV2] = []

    for symbol in symbols:
        print(f"Processing {symbol}...")

        # Check cache
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM cache_metadata WHERE symbol = ?",
            (symbol,)
        )
        cached_dates = cursor.fetchone()[0]
        conn.close()

        if cached_dates == 0:
            print(f"  No cached data for {symbol}, skipping")
            continue

        print(f"  {symbol}: {cached_dates} cached dates")

        # Fetch WSB data
        wsb_data = await fetch_wsb_history(symbol, quiver_api_key)
        if not wsb_data:
            print(f"  No WSB data for {symbol}")
            continue

        # Build lookup
        wsb_by_date = {}
        for item in wsb_data:
            date = item.get("Date", "")[:10]
            if date >= start_date:
                wsb_by_date[date] = {
                    "sentiment": float(item.get("Sentiment", 0) or 0),
                    "mentions": int(item.get("Mentions", 0) or 0),
                }

        # Get stock prices
        end_date = datetime.now().strftime("%Y-%m-%d")
        prices = get_price_data(symbol, start_date, end_date)

        if not prices:
            print(f"  No price data for {symbol}")
            continue

        # Process signals
        symbol_trades = 0
        for date, wsb in sorted(wsb_by_date.items()):
            sentiment = wsb["sentiment"]
            mentions = wsb["mentions"]

            if mentions < min_mentions or abs(sentiment) < sentiment_threshold:
                continue

            underlying_price = prices.get(date)
            if not underlying_price:
                continue

            is_bullish = sentiment > 0
            signal_type = "BUY_CALL" if is_bullish else "BUY_PUT"
            option_type = "call" if is_bullish else "put"

            contract = get_atm_contract(
                CACHE_DB, symbol, date, underlying_price, option_type
            )

            if not contract or contract["mid"] <= 0:
                continue

            # Simulate both strategies
            current_result, trailing_result = simulate_both_strategies(
                entry_price=contract["mid"],
                contract_id=contract["contract_id"],
                entry_date=date,
                expiry=contract["expiry"],
            )

            trade = TradeResultV2(
                symbol=symbol,
                signal_date=date,
                signal_type=signal_type,
                entry_price=contract["mid"],
                entry_strike=contract["strike"],
                current_exit_date=current_result["exit_date"],
                current_exit_price=current_result["exit_price"],
                current_exit_reason=current_result["exit_reason"],
                current_pnl_pct=current_result["pnl_pct"],
                current_holding_days=current_result["holding_days"],
                trailing_exit_date=trailing_result["exit_date"],
                trailing_exit_price=trailing_result["exit_price"],
                trailing_exit_reason=trailing_result["exit_reason"],
                trailing_pnl_pct=trailing_result["pnl_pct"],
                trailing_holding_days=trailing_result["holding_days"],
                trailing_activated=trailing_result["activated"],
                trailing_high_water=trailing_result["high_water"],
            )

            all_trades.append(trade)
            symbol_trades += 1

        print(f"  {symbol}: {symbol_trades} trades processed")

    return all_trades


def print_comparison(trades: list[TradeResultV2]) -> None:
    """Print detailed comparison of both strategies."""

    print("\n" + "=" * 80)
    print("TRAILING STOP BACKTEST COMPARISON")
    print("=" * 80)

    # === OVERALL COMPARISON ===
    print("\n" + "-" * 80)
    print("OVERALL COMPARISON")
    print("-" * 80)

    # Current strategy stats
    current_total_pnl = sum(t.current_pnl_pct for t in trades)
    current_avg_pnl = current_total_pnl / len(trades)
    current_wins = sum(1 for t in trades if t.current_pnl_pct > 0)
    current_win_rate = current_wins / len(trades) * 100
    current_winners = [t.current_pnl_pct for t in trades if t.current_pnl_pct > 0]
    current_losers = [t.current_pnl_pct for t in trades if t.current_pnl_pct <= 0]
    current_avg_win = sum(current_winners) / len(current_winners) if current_winners else 0
    current_avg_loss = sum(current_losers) / len(current_losers) if current_losers else 0

    # Trailing stop stats
    trailing_total_pnl = sum(t.trailing_pnl_pct for t in trades)
    trailing_avg_pnl = trailing_total_pnl / len(trades)
    trailing_wins = sum(1 for t in trades if t.trailing_pnl_pct > 0)
    trailing_win_rate = trailing_wins / len(trades) * 100
    trailing_winners = [t.trailing_pnl_pct for t in trades if t.trailing_pnl_pct > 0]
    trailing_losers = [t.trailing_pnl_pct for t in trades if t.trailing_pnl_pct <= 0]
    trailing_avg_win = sum(trailing_winners) / len(trailing_winners) if trailing_winners else 0
    trailing_avg_loss = sum(trailing_losers) / len(trailing_losers) if trailing_losers else 0

    print(f"\n{'Metric':<25} {'Current Strategy':>20} {'Trailing Stop':>20} {'Difference':>15}")
    print("-" * 80)
    print(f"{'Total P&L':<25} {current_total_pnl:>+19.1f}% {trailing_total_pnl:>+19.1f}% {trailing_total_pnl - current_total_pnl:>+14.1f}%")
    print(f"{'Avg P&L per Trade':<25} {current_avg_pnl:>+19.1f}% {trailing_avg_pnl:>+19.1f}% {trailing_avg_pnl - current_avg_pnl:>+14.1f}%")
    print(f"{'Win Rate':<25} {current_win_rate:>19.1f}% {trailing_win_rate:>19.1f}% {trailing_win_rate - current_win_rate:>+14.1f}%")
    print(f"{'Avg Winner':<25} {current_avg_win:>+19.1f}% {trailing_avg_win:>+19.1f}% {trailing_avg_win - current_avg_win:>+14.1f}%")
    print(f"{'Avg Loser':<25} {current_avg_loss:>+19.1f}% {trailing_avg_loss:>+19.1f}% {trailing_avg_loss - current_avg_loss:>+14.1f}%")

    # === EXIT BREAKDOWN ===
    print("\n" + "-" * 80)
    print("EXIT BREAKDOWN")
    print("-" * 80)

    # Current strategy exits
    current_exits = defaultdict(int)
    for t in trades:
        current_exits[t.current_exit_reason] += 1

    # Trailing stop exits
    trailing_exits = defaultdict(int)
    for t in trades:
        trailing_exits[t.trailing_exit_reason] += 1

    print(f"\n{'Exit Reason':<20} {'Current':>15} {'Trailing Stop':>15}")
    print("-" * 50)
    all_reasons = set(current_exits.keys()) | set(trailing_exits.keys())
    for reason in sorted(all_reasons):
        print(f"{reason:<20} {current_exits.get(reason, 0):>15} {trailing_exits.get(reason, 0):>15}")

    # How many activated trailing stop?
    activated_count = sum(1 for t in trades if t.trailing_activated)
    print(f"\nTrailing stop activated: {activated_count} trades ({activated_count/len(trades)*100:.1f}%)")

    # === TRAILING STOP ANALYSIS ===
    print("\n" + "-" * 80)
    print("TRAILING STOP DETAILED ANALYSIS")
    print("-" * 80)

    # Trades where trailing stop activated
    activated_trades = [t for t in trades if t.trailing_activated]
    non_activated = [t for t in trades if not t.trailing_activated]

    if activated_trades:
        print(f"\nTrades with trailing stop activated: {len(activated_trades)}")

        # What was their exit?
        ts_exits = defaultdict(list)
        for t in activated_trades:
            ts_exits[t.trailing_exit_reason].append(t)

        print("\n  By trailing stop exit reason:")
        for reason, reason_trades in sorted(ts_exits.items(), key=lambda x: -len(x[1])):
            avg_current = sum(t.current_pnl_pct for t in reason_trades) / len(reason_trades)
            avg_trailing = sum(t.trailing_pnl_pct for t in reason_trades) / len(reason_trades)
            avg_high = sum(t.trailing_high_water for t in reason_trades) / len(reason_trades)
            print(f"    {reason}: {len(reason_trades)} trades")
            print(f"      Current strategy avg: {avg_current:+.1f}%")
            print(f"      Trailing stop avg:    {avg_trailing:+.1f}%")
            print(f"      Avg high water mark:  {avg_high:+.1f}%")
            print(f"      Difference:           {avg_trailing - avg_current:+.1f}%")

    # === TRADES WHERE TRAILING STOP HELPED ===
    print("\n" + "-" * 80)
    print("TRADES WHERE TRAILING STOP HELPED (improved P&L)")
    print("-" * 80)

    helped = [t for t in trades if t.trailing_pnl_pct > t.current_pnl_pct + 5]  # At least 5% better
    helped.sort(key=lambda t: -t.pnl_difference)

    print(f"\nTrades improved by >5%: {len(helped)}")
    if helped:
        print(f"\n{'Date':<12} {'Symbol':<6} {'Current':>12} {'Trailing':>12} {'Diff':>10} {'Reason':<15}")
        print("-" * 70)
        for t in helped[:15]:
            print(f"{t.signal_date:<12} {t.symbol:<6} {t.current_pnl_pct:>+11.1f}% {t.trailing_pnl_pct:>+11.1f}% {t.pnl_difference:>+9.1f}% {t.trailing_exit_reason:<15}")

    # === TRADES WHERE TRAILING STOP HURT ===
    print("\n" + "-" * 80)
    print("TRADES WHERE TRAILING STOP HURT (reduced P&L)")
    print("-" * 80)

    hurt = [t for t in trades if t.trailing_pnl_pct < t.current_pnl_pct - 5]  # At least 5% worse
    hurt.sort(key=lambda t: t.pnl_difference)

    print(f"\nTrades worsened by >5%: {len(hurt)}")
    if hurt:
        print(f"\n{'Date':<12} {'Symbol':<6} {'Current':>12} {'Trailing':>12} {'Diff':>10} {'High Water':>12}")
        print("-" * 75)
        for t in hurt[:15]:
            print(f"{t.signal_date:<12} {t.symbol:<6} {t.current_pnl_pct:>+11.1f}% {t.trailing_pnl_pct:>+11.1f}% {t.pnl_difference:>+9.1f}% {t.trailing_high_water:>+11.1f}%")

        # Analyze the hurt trades
        print("\n  Analysis of hurt trades:")
        for t in hurt[:5]:
            print(f"\n    {t.signal_date} {t.symbol}:")
            print(f"      Current: exited at {t.current_exit_reason} = {t.current_pnl_pct:+.1f}%")
            print(f"      Trailing: hit +{t.trailing_high_water:.0f}%, trailed to {t.trailing_pnl_pct:+.1f}%")
            print(f"      Would have been better to: ", end="")
            if t.current_exit_reason == "profit_target":
                print("take the +50% profit target")
            else:
                print("use current strategy")

    # === VERDICT ===
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    improvement = trailing_total_pnl - current_total_pnl
    if improvement > 0:
        print(f"\n  TRAILING STOP IMPROVES P&L by {improvement:+.1f}%")
        print(f"  Avg P&L improvement: {trailing_avg_pnl - current_avg_pnl:+.1f}% per trade")
        print("\n  RECOMMENDATION: Implement trailing stop in live position tracker")
    else:
        print(f"\n  TRAILING STOP REDUCES P&L by {improvement:.1f}%")
        print(f"  Avg P&L reduction: {trailing_avg_pnl - current_avg_pnl:.1f}% per trade")
        print("\n  RECOMMENDATION: Keep current strategy (profit target + hard stop)")

    # Show the math
    print("\n  The math:")
    print(f"    Trades where trailing helped (>5%): {len(helped)} trades, total gain: {sum(t.pnl_difference for t in helped):+.1f}%")
    print(f"    Trades where trailing hurt (>5%):   {len(hurt)} trades, total loss: {sum(t.pnl_difference for t in hurt):+.1f}%")
    print(f"    Net impact: {sum(t.pnl_difference for t in trades):+.1f}%")


async def main():
    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("=" * 80)
    print("TRAILING STOP BACKTEST")
    print("=" * 80)
    print("\nCurrent Strategy:")
    print("  - 50% profit target")
    print("  - 30% stop loss")
    print("  - DTE < 7 exit")
    print("\nTrailing Stop Variant:")
    print("  - Activate at +30% profit")
    print("  - Trail by 15% from high water mark")
    print("  - Hard stop at -30% if never activated")
    print()

    trades = await run_comparison(
        symbols=["TSLA", "NVDA", "PLTR"],
        start_date="2024-01-01",
        quiver_api_key=quiver_api_key,
    )

    print(f"\nTotal trades analyzed: {len(trades)}")

    print_comparison(trades)


if __name__ == "__main__":
    asyncio.run(main())
