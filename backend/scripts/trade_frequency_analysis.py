#!/usr/bin/env python3
"""Trade Frequency Sensitivity Analysis + Portfolio Simulation.

Part 1: Tests what happens when we loosen entry criteria for TSLA
Part 2: Runs portfolio simulation with compounding

Technical indicators:
- Bollinger Bands: Price near lower band = bullish, upper = bearish
- MACD: Signal line crossover
- Trend SMA: Price vs 20-day SMA direction
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional

import aiohttp
from dotenv import load_dotenv
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")

CACHE_DB = Path(__file__).parent.parent.parent / "cache" / "options_data.db"

# FROZEN PARAMETERS
TRAILING_ACTIVATION = 0.30
TRAILING_DISTANCE = 0.15
STOP_LOSS = -0.30
MIN_DTE = 7
MAX_HOLD_DAYS = 45
MIN_OI = 500
TARGET_DTE = 30


@dataclass
class Trade:
    symbol: str
    signal_date: str
    signal_type: str
    entry_date: str
    entry_contract: str
    entry_ask: float
    entry_bid: float
    entry_mid: float
    open_interest: int
    exit_date: str = ""
    exit_bid: float = 0.0
    exit_reason: str = ""
    pnl_bidask: float = 0.0
    holding_days: int = 0
    # Technical confirmation
    bollinger_confirm: bool = False
    macd_confirm: bool = False
    trend_confirm: bool = False
    technicals_passed: int = 0


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

    return df


def check_technical_confirmation(
    technicals_df: pd.DataFrame,
    date: str,
    signal_type: str
) -> dict:
    """Check which technicals confirm the signal."""
    if date not in technicals_df.index:
        # Try to find nearest prior date
        prior_dates = [d for d in technicals_df.index if d < date]
        if not prior_dates:
            return {"bollinger": False, "macd": False, "trend": False, "count": 0}
        date = max(prior_dates)

    row = technicals_df.loc[date]

    # Skip if not enough data for indicators
    if pd.isna(row.get('bb_pct')) or pd.isna(row.get('macd_hist')):
        return {"bollinger": False, "macd": False, "trend": False, "count": 0}

    is_bullish = signal_type == "call"

    # Bollinger: Bullish if price near lower band (<0.3), Bearish if near upper (>0.7)
    bb_pct = row['bb_pct']
    bollinger_confirm = (is_bullish and bb_pct < 0.3) or (not is_bullish and bb_pct > 0.7)

    # MACD: Bullish if histogram crossing up, Bearish if crossing down
    macd_hist = row['macd_hist']
    macd_prev = row['macd_prev_hist']
    if pd.isna(macd_prev):
        macd_confirm = False
    else:
        macd_confirm = (is_bullish and macd_hist > macd_prev) or (not is_bullish and macd_hist < macd_prev)

    # Trend: Bullish if price above SMA, Bearish if below
    trend_bullish = row['trend_bullish']
    trend_confirm = (is_bullish and trend_bullish) or (not is_bullish and not trend_bullish)

    count = sum([bollinger_confirm, macd_confirm, trend_confirm])

    return {
        "bollinger": bollinger_confirm,
        "macd": macd_confirm,
        "trend": trend_confirm,
        "count": count
    }


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


def get_atm_contract(db_path: Path, symbol: str, trade_date: str, stock_price: float, option_type: str) -> Optional[dict]:
    """Find ATM contract with ~30 DTE."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    target_expiry = (datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=TARGET_DTE)).strftime("%Y-%m-%d")

    cursor.execute("""
        SELECT contract_id, strike, expiry, bid, ask, volume, open_interest
        FROM options_contracts
        WHERE symbol = ? AND trade_date = ? AND option_type = ?
        AND expiry >= ? AND expiry <= ?
        AND bid > 0 AND ask > 0
        ORDER BY ABS(strike - ?), ABS(julianday(expiry) - julianday(?))
        LIMIT 1
    """, (
        symbol, trade_date, option_type,
        (datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=MIN_DTE)).strftime("%Y-%m-%d"),
        (datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=MAX_HOLD_DAYS)).strftime("%Y-%m-%d"),
        stock_price, target_expiry
    ))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "contract_id": row[0],
        "strike": row[1],
        "expiry": row[2],
        "bid": row[3],
        "ask": row[4],
        "mid": (row[3] + row[4]) / 2,
        "volume": row[5],
        "open_interest": row[6],
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


async def run_backtest_scenario(
    symbol: str,
    start_date: str,
    quiver_api_key: str,
    min_technicals_required: int,
    scenario_name: str,
) -> list[Trade]:
    """Run backtest with specific technical confirmation requirement."""

    print(f"\n  Running {scenario_name}...")

    # Get cached dates
    cached_dates = get_cached_dates(symbol)
    if not cached_dates:
        print(f"    No cached data for {symbol}")
        return []

    # Get WSB data
    wsb_data = await fetch_wsb_history(symbol, quiver_api_key)
    if not wsb_data:
        print(f"    No WSB data for {symbol}")
        return []

    # Build date -> sentiment map
    wsb_by_date = {}
    for item in wsb_data:
        date = item.get("Date", "")[:10]
        if date >= start_date:
            wsb_by_date[date] = {
                "sentiment": item.get("Sentiment", 0),
                "mentions": item.get("Mentions", 0),
            }

    # Get price data and calculate technicals
    end_date = datetime.now().strftime("%Y-%m-%d")
    prices_df = get_price_data(symbol, start_date, end_date)
    if prices_df.empty:
        print(f"    No price data for {symbol}")
        return []

    technicals_df = calculate_technicals(prices_df)
    price_dict = prices_df['Close'].to_dict()

    all_trades = []
    signals_found = 0
    signals_with_options = 0
    signals_with_technicals = 0

    for date in sorted(wsb_by_date.keys()):
        if date not in cached_dates:
            continue
        if date not in price_dict:
            continue

        wsb = wsb_by_date[date]
        sentiment = wsb["sentiment"]
        mentions = wsb["mentions"]

        # Need positive or negative sentiment with volume
        if abs(sentiment) < 0.1 or mentions < 10:
            continue

        signals_found += 1
        signal_type = "call" if sentiment > 0 else "put"
        stock_price = price_dict[date]

        # Check technical confirmation
        tech_result = check_technical_confirmation(technicals_df, date, signal_type)

        if tech_result["count"] < min_technicals_required:
            continue

        signals_with_technicals += 1

        # Find ATM contract
        contract = get_atm_contract(CACHE_DB, symbol, date, stock_price, signal_type)
        if not contract:
            continue

        if contract["open_interest"] < MIN_OI:
            continue

        signals_with_options += 1

        # Create trade
        trade = Trade(
            symbol=symbol,
            signal_date=date,
            signal_type=signal_type,
            entry_date=date,
            entry_contract=contract["contract_id"],
            entry_ask=contract["ask"],
            entry_bid=contract["bid"],
            entry_mid=contract["mid"],
            open_interest=contract["open_interest"],
            bollinger_confirm=tech_result["bollinger"],
            macd_confirm=tech_result["macd"],
            trend_confirm=tech_result["trend"],
            technicals_passed=tech_result["count"],
        )

        # Track with trailing stop
        entry_ask = contract["ask"]
        high_water_mark = 0.0
        trailing_active = False

        tracking_days = get_trading_days_after(date, MAX_HOLD_DAYS + 10)

        for track_date in tracking_days:
            price_data = get_contract_price(CACHE_DB, contract["contract_id"], track_date)
            if not price_data:
                continue

            current_bid = price_data["bid"]
            pnl_bidask = (current_bid - entry_ask) / entry_ask

            if pnl_bidask > high_water_mark:
                high_water_mark = pnl_bidask

            if not trailing_active and pnl_bidask >= TRAILING_ACTIVATION:
                trailing_active = True

            # Check DTE
            try:
                expiry_date = datetime.strptime(price_data["expiry"], "%Y-%m-%d")
                current = datetime.strptime(track_date, "%Y-%m-%d")
                dte = (expiry_date - current).days
            except ValueError:
                dte = 30

            exit_reason = None

            if trailing_active:
                if pnl_bidask <= high_water_mark - TRAILING_DISTANCE:
                    exit_reason = "trailing_stop"

            if not exit_reason and pnl_bidask <= STOP_LOSS:
                exit_reason = "stop_loss"

            if not exit_reason and dte < MIN_DTE:
                exit_reason = "dte_exit"

            if exit_reason:
                trade.exit_date = track_date
                trade.exit_bid = current_bid
                trade.exit_reason = exit_reason
                trade.pnl_bidask = pnl_bidask * 100

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
                    trade.exit_bid = price_data["bid"]
                    trade.exit_reason = "max_hold"
                    trade.pnl_bidask = ((price_data["bid"] - entry_ask) / entry_ask) * 100
                    break

        if trade.exit_date:
            all_trades.append(trade)

    print(f"    Signals found: {signals_found}")
    print(f"    Passing tech filter: {signals_with_technicals}")
    print(f"    With valid options: {signals_with_options}")
    print(f"    Completed trades: {len(all_trades)}")

    return all_trades


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
        }

    total_pnl = sum(t.pnl_bidask for t in trades)
    avg_pnl = total_pnl / len(trades)
    winners = [t for t in trades if t.pnl_bidask > 0]
    losers = [t for t in trades if t.pnl_bidask <= 0]
    win_rate = len(winners) / len(trades) * 100

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "winners": len(winners),
        "losers": len(losers),
    }


def run_portfolio_simulation(
    trades: list[Trade],
    starting_capital: float = 100000,
    position_pct: float = 0.05,
) -> dict:
    """Run portfolio simulation with compounding."""

    if not trades:
        return {
            "final_value": starting_capital,
            "total_return_pct": 0,
            "max_drawdown_pct": 0,
            "max_drawdown_dollars": 0,
            "monthly_equity": {},
            "capital_deployed_pct": 0,
        }

    # Sort trades by entry date
    sorted_trades = sorted(trades, key=lambda t: t.entry_date)

    portfolio_value = starting_capital
    peak_value = starting_capital
    max_drawdown_pct = 0
    max_drawdown_dollars = 0

    # Track daily equity (simplified - on trade dates)
    equity_curve = {sorted_trades[0].entry_date: starting_capital}

    # Track capital deployment
    days_with_position = set()

    for trade in sorted_trades:
        # Position size based on current portfolio value
        position_size = portfolio_value * position_pct

        # Calculate P&L in dollars
        pnl_pct = trade.pnl_bidask / 100  # Convert from percentage
        pnl_dollars = position_size * pnl_pct

        # Update portfolio
        portfolio_value += pnl_dollars

        # Track equity at exit
        equity_curve[trade.exit_date] = portfolio_value

        # Track days with position
        entry = datetime.strptime(trade.entry_date, "%Y-%m-%d")
        exit_dt = datetime.strptime(trade.exit_date, "%Y-%m-%d")
        current = entry
        while current <= exit_dt:
            if current.weekday() < 5:
                days_with_position.add(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        # Update peak and drawdown
        if portfolio_value > peak_value:
            peak_value = portfolio_value

        drawdown_pct = (peak_value - portfolio_value) / peak_value * 100
        drawdown_dollars = peak_value - portfolio_value

        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct
            max_drawdown_dollars = drawdown_dollars

    # Calculate monthly equity snapshots
    monthly_equity = {}
    for date, value in sorted(equity_curve.items()):
        month = date[:7]
        monthly_equity[month] = value  # Last value of month

    # Calculate capital deployment
    start_dt = datetime.strptime(sorted_trades[0].entry_date, "%Y-%m-%d")
    end_dt = datetime.strptime(sorted_trades[-1].exit_date, "%Y-%m-%d")
    total_trading_days = 0
    current = start_dt
    while current <= end_dt:
        if current.weekday() < 5:
            total_trading_days += 1
        current += timedelta(days=1)

    capital_deployed_pct = len(days_with_position) / total_trading_days * 100 if total_trading_days > 0 else 0

    total_return_pct = (portfolio_value - starting_capital) / starting_capital * 100

    return {
        "final_value": portfolio_value,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_dollars": max_drawdown_dollars,
        "monthly_equity": monthly_equity,
        "capital_deployed_pct": capital_deployed_pct,
        "peak_value": peak_value,
    }


async def main():
    symbol = "TSLA"
    start_date = "2024-01-01"

    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("=" * 70)
    print("TRADE FREQUENCY SENSITIVITY ANALYSIS - TSLA")
    print("=" * 70)
    print()
    print("Testing entry criteria scenarios:")
    print("  A) BASELINE: Sentiment + ALL 3 technicals")
    print("  B) Sentiment + 2 technicals")
    print("  C) Sentiment + 1 technical")
    print("  D) SENTIMENT ONLY: No technical confirmation")
    print()
    print("Technicals: Bollinger Bands, MACD, 20-day Trend SMA")
    print()

    # Run all scenarios
    scenarios = [
        (3, "A) BASELINE: All 3 technicals"),
        (2, "B) At least 2 technicals"),
        (1, "C) At least 1 technical"),
        (0, "D) SENTIMENT ONLY"),
    ]

    results = {}

    for min_tech, name in scenarios:
        trades = await run_backtest_scenario(
            symbol=symbol,
            start_date=start_date,
            quiver_api_key=quiver_api_key,
            min_technicals_required=min_tech,
            scenario_name=name,
        )
        results[name] = {
            "trades": trades,
            "stats": calc_stats(trades),
        }

    # Calculate time period for trades/month
    end_date = datetime.now()
    start = datetime.strptime(start_date, "%Y-%m-%d")
    months = (end_date.year - start.year) * 12 + (end_date.month - start.month)
    if months == 0:
        months = 1

    # Part 1 Results
    print("\n" + "=" * 70)
    print("PART 1: TRADE FREQUENCY SENSITIVITY RESULTS")
    print("=" * 70)
    print()
    print(f"{'Scenario':<35} {'Trades':>7} {'Win%':>7} {'Avg P&L':>10} {'Total P&L':>11} {'Trades/Mo':>10}")
    print("-" * 80)

    for name, data in results.items():
        s = data["stats"]
        trades_per_month = s["trades"] / months
        print(f"{name:<35} {s['trades']:>7} {s['win_rate']:>6.1f}% {s['avg_pnl']:>+9.1f}% {s['total_pnl']:>+10.0f}% {trades_per_month:>10.1f}")

    # Determine best scenario
    best_scenario = None
    best_score = float('-inf')

    for name, data in results.items():
        s = data["stats"]
        if s["trades"] >= 3:  # Need at least 3 trades
            # Score = avg_pnl * sqrt(trades) to balance profitability with sample size
            score = s["avg_pnl"] * (s["trades"] ** 0.5)
            if score > best_score:
                best_score = score
                best_scenario = name

    if not best_scenario:
        best_scenario = list(results.keys())[-1]  # Default to sentiment only

    print()
    print(f"Best scenario (by avg P&L * sqrt(trades)): {best_scenario}")

    # Part 2: Portfolio Simulation
    print("\n" + "=" * 70)
    print("PART 2: PORTFOLIO SIMULATION")
    print("=" * 70)
    print()

    best_trades = results[best_scenario]["trades"]

    print(f"Using: {best_scenario}")
    print(f"Starting Capital: $100,000")
    print(f"Position Size: 5% per trade (half-Kelly)")
    print(f"Compounding: Yes")
    print()

    sim = run_portfolio_simulation(best_trades, 100000, 0.05)

    print("Portfolio Statistics:")
    print(f"  Final Value:        ${sim['final_value']:>12,.0f}")
    print(f"  Total Return:       {sim['total_return_pct']:>+12.1f}%")
    print(f"  Max Drawdown:       {sim['max_drawdown_pct']:>12.1f}%")
    print(f"  Max Drawdown ($):   ${sim['max_drawdown_dollars']:>12,.0f}")
    print(f"  Peak Value:         ${sim['peak_value']:>12,.0f}")
    print(f"  Capital Deployed:   {sim['capital_deployed_pct']:>12.1f}% of time")

    print("\nMonthly Equity Curve:")
    print(f"{'Month':<12} {'Portfolio Value':>18} {'Return from Start':>18}")
    print("-" * 50)

    for month, value in sorted(sim['monthly_equity'].items()):
        ret = (value - 100000) / 100000 * 100
        print(f"{month:<12} ${value:>17,.0f} {ret:>+17.1f}%")

    # Also show each trade if reasonable number
    if len(best_trades) <= 50:
        print("\n" + "=" * 70)
        print("INDIVIDUAL TRADES")
        print("=" * 70)
        print()
        print(f"{'Date':<12} {'Type':<6} {'Entry':>8} {'Exit':>8} {'P&L':>10} {'Reason':<15} {'Hold':>5}")
        print("-" * 75)

        for t in sorted(best_trades, key=lambda x: x.entry_date):
            print(f"{t.entry_date:<12} {t.signal_type:<6} ${t.entry_ask:>6.2f} ${t.exit_bid:>6.2f} {t.pnl_bidask:>+9.1f}% {t.exit_reason:<15} {t.holding_days:>4}d")

    # Summary comparison table
    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON SUMMARY")
    print("=" * 70)
    print()

    for name, data in results.items():
        trades = data["trades"]
        if not trades:
            continue

        sim = run_portfolio_simulation(trades, 100000, 0.05)
        s = data["stats"]

        print(f"{name}:")
        print(f"  Trades: {s['trades']}, Win Rate: {s['win_rate']:.1f}%, Avg P&L: {s['avg_pnl']:+.1f}%")
        print(f"  Portfolio: ${sim['final_value']:,.0f} ({sim['total_return_pct']:+.1f}%), Max DD: {sim['max_drawdown_pct']:.1f}%")
        print()


if __name__ == "__main__":
    asyncio.run(main())
