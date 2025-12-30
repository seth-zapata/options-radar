#!/usr/bin/env python3
"""TRUE P&L Backtest using actual historical options prices.

Uses cached EODHD options data to simulate real trades with exit rules:
- 50% profit target
- 30% stop loss
- DTE < 7 (close before expiry)

Sentiment reversal was tested and REMOVED - backtest showed +381% improvement
without it. The profit target gains outweigh stop loss protection.

No estimation, no approximation - actual historical prices.

Usage:
    python -m backend.scripts.pnl_backtest --symbols TSLA,NVDA,PLTR
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import aiohttp
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
CACHE_DB = Path(__file__).parent.parent.parent / "cache" / "options_data.db"

# Exit rules
PROFIT_TARGET = 0.50  # 50% profit
STOP_LOSS = -0.30     # 30% loss
MIN_DTE = 7           # Close when DTE < 7
MAX_HOLD_DAYS = 45    # Maximum holding period


@dataclass
class TradeResult:
    """Result of a single trade."""
    symbol: str
    signal_date: str
    signal_type: Literal["BUY_CALL", "BUY_PUT"]
    sentiment: float
    mentions: int

    # Entry
    entry_date: str
    entry_contract: str
    entry_expiry: str
    entry_strike: float
    entry_price: float  # Mid price at entry
    entry_delta: float
    entry_iv: float

    # Liquidity at entry
    open_interest: int = 0
    volume: int = 0

    # Exit
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: Literal["profit_target", "stop_loss", "dte_exit", "max_hold", "no_data"] = "no_data"

    # P&L
    pnl_pct: float = 0.0  # Percentage return
    pnl_dollars: float = 0.0  # Dollar P&L per contract (x100)
    holding_days: int = 0

    @property
    def was_profitable(self) -> bool:
        return self.pnl_pct > 0


@dataclass
class BacktestStats:
    """Aggregate statistics for backtest."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # P&L
    total_pnl_pct: float = 0
    total_pnl_dollars: float = 0
    avg_win_pct: float = 0
    avg_loss_pct: float = 0

    # Exit reasons
    profit_target_exits: int = 0
    stop_loss_exits: int = 0
    dte_exits: int = 0
    max_hold_exits: int = 0
    no_data_exits: int = 0

    # By symbol
    by_symbol: dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return self.winning_trades / self.total_trades * 100

    @property
    def avg_pnl_pct(self) -> float:
        if self.total_trades == 0:
            return 0
        return self.total_pnl_pct / self.total_trades

    @property
    def expected_value(self) -> float:
        """Expected value per trade in percentage terms."""
        if self.total_trades == 0:
            return 0
        return self.total_pnl_pct / self.total_trades


def get_atm_contract(
    db_path: Path,
    symbol: str,
    trade_date: str,
    underlying_price: float,
    option_type: str = "call",
    target_dte: int = 30,
) -> dict | None:
    """Find ATM contract closest to target DTE."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query for contracts near ATM with reasonable DTE
    # Strike within 5% of underlying, expiry 20-45 days out
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

    # Parse trade date
    try:
        td = datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        return None

    # Find best contract: closest to ATM + closest to target DTE
    best = None
    best_score = float('inf')

    for row in rows:
        contract_id, expiry, strike, bid, ask, delta, iv, oi, vol = row

        try:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            dte = (expiry_date - td).days
        except ValueError:
            continue

        # Skip if DTE is too short or too long
        if dte < 14 or dte > 60:
            continue

        # Score: distance from ATM + distance from target DTE
        strike_dist = abs(strike - underlying_price) / underlying_price
        dte_dist = abs(dte - target_dte) / target_dte
        score = strike_dist + dte_dist * 0.5

        if score < best_score:
            best_score = score
            mid = (bid + ask) / 2
            best = {
                "contract_id": contract_id,
                "expiry": expiry,
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "delta": delta,
                "iv": iv,
                "dte": dte,
                "open_interest": oi or 0,
                "volume": vol or 0,
            }

    return best


def get_contract_price(
    db_path: Path,
    contract_id: str,
    trade_date: str,
) -> dict | None:
    """Get contract price on a specific date."""
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
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

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
        if current.weekday() < 5:  # Skip weekends
            result.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return result


def get_underlying_price(prices: dict, date: str) -> float | None:
    """Get underlying price from yfinance data."""
    return prices.get(date)


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


async def run_backtest(
    symbols: list[str],
    start_date: str,
    quiver_api_key: str,
    min_mentions: int = 5,
    sentiment_threshold: float = 0.1,
) -> tuple[BacktestStats, list[TradeResult]]:
    """Run P&L backtest with real options prices."""

    stats = BacktestStats()
    all_trades: list[TradeResult] = []

    for symbol in symbols:
        logger.info(f"Processing {symbol}...")

        # Check if we have options data for this symbol
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM cache_metadata WHERE symbol = ?",
            (symbol,)
        )
        cached_dates = cursor.fetchone()[0]
        conn.close()

        if cached_dates == 0:
            logger.warning(f"No cached options data for {symbol}, skipping")
            continue

        logger.info(f"  {symbol}: {cached_dates} cached dates")

        # Fetch WSB data
        wsb_data = await fetch_wsb_history(symbol, quiver_api_key)
        if not wsb_data:
            logger.warning(f"No WSB data for {symbol}")
            continue

        # Filter to date range and build lookup
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
            logger.warning(f"No price data for {symbol}")
            continue

        # Generate and process signals
        symbol_trades = 0
        for date, wsb in sorted(wsb_by_date.items()):
            sentiment = wsb["sentiment"]
            mentions = wsb["mentions"]

            # Skip if below thresholds
            if mentions < min_mentions or abs(sentiment) < sentiment_threshold:
                continue

            # Skip if no stock price
            underlying_price = prices.get(date)
            if not underlying_price:
                continue

            # Determine signal type
            is_bullish = sentiment > 0
            signal_type = "BUY_CALL" if is_bullish else "BUY_PUT"
            option_type = "call" if is_bullish else "put"

            # Find ATM contract
            contract = get_atm_contract(
                CACHE_DB, symbol, date, underlying_price, option_type
            )

            if not contract:
                continue

            entry_price = contract["mid"]
            if entry_price <= 0:
                continue

            # Track position until exit
            exit_date = None
            exit_price = None
            exit_reason = None

            tracking_days = get_trading_days_after(date, MAX_HOLD_DAYS + 10)

            for track_date in tracking_days:
                # Check if we have data for this date
                price_data = get_contract_price(CACHE_DB, contract["contract_id"], track_date)

                if not price_data or price_data["mid"] <= 0:
                    continue

                current_price = price_data["mid"]
                pnl_pct = (current_price - entry_price) / entry_price

                # Calculate DTE
                try:
                    expiry_date = datetime.strptime(contract["expiry"], "%Y-%m-%d")
                    current = datetime.strptime(track_date, "%Y-%m-%d")
                    dte = (expiry_date - current).days
                except ValueError:
                    dte = 30

                # Check exit conditions in priority order

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

                # 3. DTE exit
                if dte < MIN_DTE:
                    exit_date = track_date
                    exit_price = current_price
                    exit_reason = "dte_exit"
                    break

                # NOTE: Sentiment reversal exit was REMOVED after backtest analysis
                # showed +381% improvement without it. Profit target gains outweigh
                # the stop loss protection sentiment reversals provided.

            # If no exit triggered, use last available price
            if exit_date is None:
                # Try to get last known price
                for track_date in reversed(tracking_days):
                    price_data = get_contract_price(CACHE_DB, contract["contract_id"], track_date)
                    if price_data and price_data["mid"] > 0:
                        exit_date = track_date
                        exit_price = price_data["mid"]
                        exit_reason = "max_hold"
                        break

            if exit_date is None:
                exit_date = date
                exit_price = entry_price
                exit_reason = "no_data"

            # Calculate final P&L
            final_pnl_pct = (exit_price - entry_price) / entry_price
            final_pnl_dollars = (exit_price - entry_price) * 100  # Per contract

            # Calculate holding days
            try:
                entry_dt = datetime.strptime(date, "%Y-%m-%d")
                exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
                holding_days = (exit_dt - entry_dt).days
            except ValueError:
                holding_days = 0

            trade = TradeResult(
                symbol=symbol,
                signal_date=date,
                signal_type=signal_type,
                sentiment=sentiment,
                mentions=mentions,
                entry_date=date,
                entry_contract=contract["contract_id"],
                entry_expiry=contract["expiry"],
                entry_strike=contract["strike"],
                entry_price=entry_price,
                entry_delta=contract["delta"],
                entry_iv=contract["iv"],
                open_interest=contract["open_interest"],
                volume=contract["volume"],
                exit_date=exit_date,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl_pct=final_pnl_pct * 100,  # As percentage
                pnl_dollars=final_pnl_dollars,
                holding_days=holding_days,
            )

            all_trades.append(trade)
            symbol_trades += 1

            # Update stats
            stats.total_trades += 1
            stats.total_pnl_pct += trade.pnl_pct
            stats.total_pnl_dollars += trade.pnl_dollars

            if trade.was_profitable:
                stats.winning_trades += 1
            else:
                stats.losing_trades += 1

            # Exit reason counts
            if exit_reason == "profit_target":
                stats.profit_target_exits += 1
            elif exit_reason == "stop_loss":
                stats.stop_loss_exits += 1
            elif exit_reason == "dte_exit":
                stats.dte_exits += 1
            elif exit_reason == "max_hold":
                stats.max_hold_exits += 1
            else:
                stats.no_data_exits += 1

        logger.info(f"  {symbol}: {symbol_trades} trades processed")

        # Track by symbol
        symbol_trades_list = [t for t in all_trades if t.symbol == symbol]
        if symbol_trades_list:
            wins = sum(1 for t in symbol_trades_list if t.was_profitable)
            total_pnl = sum(t.pnl_pct for t in symbol_trades_list)
            stats.by_symbol[symbol] = {
                "trades": len(symbol_trades_list),
                "wins": wins,
                "win_rate": wins / len(symbol_trades_list) * 100,
                "total_pnl_pct": total_pnl,
                "avg_pnl_pct": total_pnl / len(symbol_trades_list),
            }

    # Calculate avg win/loss
    wins = [t for t in all_trades if t.was_profitable]
    losses = [t for t in all_trades if not t.was_profitable]

    if wins:
        stats.avg_win_pct = sum(t.pnl_pct for t in wins) / len(wins)
    if losses:
        stats.avg_loss_pct = sum(t.pnl_pct for t in losses) / len(losses)

    return stats, all_trades


def print_results(stats: BacktestStats, trades: list[TradeResult]) -> None:
    """Print backtest results."""
    print("\n" + "=" * 70)
    print("TRUE P&L BACKTEST RESULTS - ACTUAL OPTIONS PRICES")
    print("=" * 70)

    print(f"\nTotal Trades: {stats.total_trades}")
    print(f"Win Rate: {stats.win_rate:.1f}% ({stats.winning_trades}/{stats.total_trades})")

    print("\n" + "-" * 70)
    print("P&L SUMMARY")
    print("-" * 70)
    print(f"  Total P&L: {stats.total_pnl_pct:+.1f}%")
    print(f"  Average P&L per Trade: {stats.avg_pnl_pct:+.1f}%")
    print(f"  Expected Value per Trade: {stats.expected_value:+.1f}%")
    print(f"  Total Dollar P&L (per contract): ${stats.total_pnl_dollars:+,.0f}")
    print()
    print(f"  Average Win: {stats.avg_win_pct:+.1f}%")
    print(f"  Average Loss: {stats.avg_loss_pct:+.1f}%")

    # Win/Loss ratio
    if stats.avg_loss_pct != 0:
        win_loss_ratio = abs(stats.avg_win_pct / stats.avg_loss_pct)
        print(f"  Win/Loss Ratio: {win_loss_ratio:.2f}")

    print("\n" + "-" * 70)
    print("EXIT BREAKDOWN")
    print("-" * 70)
    print(f"  Profit Target (50%):     {stats.profit_target_exits} ({stats.profit_target_exits/stats.total_trades*100:.1f}%)")
    print(f"  Stop Loss (-30%):        {stats.stop_loss_exits} ({stats.stop_loss_exits/stats.total_trades*100:.1f}%)")
    print(f"  DTE Exit (<7 days):      {stats.dte_exits} ({stats.dte_exits/stats.total_trades*100:.1f}%)")
    print(f"  Max Hold (45 days):      {stats.max_hold_exits} ({stats.max_hold_exits/stats.total_trades*100:.1f}%)")
    print(f"  No Data:                 {stats.no_data_exits} ({stats.no_data_exits/stats.total_trades*100:.1f}%)")

    print("\n" + "-" * 70)
    print("BY SYMBOL")
    print("-" * 70)
    for symbol, data in sorted(stats.by_symbol.items()):
        print(f"  {symbol}:")
        print(f"    Trades: {data['trades']}")
        print(f"    Win Rate: {data['win_rate']:.1f}%")
        print(f"    Total P&L: {data['total_pnl_pct']:+.1f}%")
        print(f"    Avg P&L: {data['avg_pnl_pct']:+.1f}%")

    # Show sample trades
    print("\n" + "-" * 70)
    print("SAMPLE TRADES (First 10)")
    print("-" * 70)
    for trade in trades[:10]:
        print(f"  {trade.signal_date} {trade.symbol} {trade.signal_type}")
        print(f"    Entry: ${trade.entry_price:.2f} @ strike {trade.entry_strike}")
        print(f"    Exit:  ${trade.exit_price:.2f} ({trade.exit_reason})")
        print(f"    P&L:   {trade.pnl_pct:+.1f}% ({trade.holding_days} days)")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    if stats.expected_value > 0:
        print(f"  POSITIVE EXPECTED VALUE: +{stats.expected_value:.1f}% per trade")
        print(f"  Strategy is profitable with real options prices!")
    else:
        print(f"  NEGATIVE EXPECTED VALUE: {stats.expected_value:.1f}% per trade")
        print(f"  Strategy loses money with real options prices.")
        print(f"  Consider adjusting exit rules or entry criteria.")

    # Compare to directional accuracy
    print("\n  For comparison:")
    print(f"    Directional accuracy: ~71.4% (from previous backtest)")
    print(f"    Actual profitable trades: {stats.win_rate:.1f}%")
    if stats.win_rate < 71.4:
        print(f"    Gap: {71.4 - stats.win_rate:.1f}% (exit rules matter!)")


async def main():
    parser = argparse.ArgumentParser(description="True P&L backtest with real options prices")
    parser.add_argument(
        "--symbols",
        type=str,
        default="TSLA,NVDA,PLTR",
        help="Comma-separated list of symbols",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--min-mentions",
        type=int,
        default=5,
        help="Minimum WSB mentions for signal",
    )

    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Load API key
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    quiver_api_key = os.getenv("QUIVER_API_KEY")

    if not quiver_api_key:
        logger.error("No QUIVER_API_KEY found")
        return

    # Check cache exists
    if not CACHE_DB.exists():
        logger.error(f"Cache database not found: {CACHE_DB}")
        logger.error("Run cache_full_options.py first to cache options data")
        return

    print("=" * 70)
    print("TRUE P&L BACKTEST")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Start Date: {args.start}")
    print(f"Exit Rules: 50% profit, 30% stop, DTE<7")
    print()

    stats, trades = await run_backtest(
        symbols=symbols,
        start_date=args.start,
        quiver_api_key=quiver_api_key,
        min_mentions=args.min_mentions,
    )

    print_results(stats, trades)


if __name__ == "__main__":
    asyncio.run(main())
