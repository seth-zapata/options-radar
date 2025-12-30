#!/usr/bin/env python3
"""Portfolio Simulation with Compounding.

Uses the validated regime-filtered intraday strategy:
- 7-day regime window
- 1.5% pullback/bounce threshold
- 7 DTE weeklies
- Bid/ask realistic fills

Tests different position sizing scenarios with compounding.
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
import math

import aiohttp
from dotenv import load_dotenv
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")

CACHE_DB = Path(__file__).parent.parent.parent / "cache" / "options_data.db"

# Strategy parameters (validated configuration)
REGIME_WINDOW = 7
PULLBACK_THRESHOLD = 1.5
TARGET_DTE = 7
INCLUDE_MODERATE = True

# Trade management
STOP_LOSS = -0.20
TAKE_PROFIT = 0.40
MAX_HOLD_DAYS = 10
MIN_OI = 500
MAX_TRADES_PER_REGIME = 5
MIN_DAYS_BETWEEN_ENTRIES = 1

# Portfolio parameters
STARTING_CAPITAL = 100_000
MAX_CONCURRENT_POSITIONS = 3


@dataclass
class Position:
    """An open position."""
    entry_date: str
    option_type: str
    regime_type: str
    contract_id: str
    entry_ask: float
    contracts: int
    position_value: float  # Initial position value
    current_value: float = 0.0
    exit_date: str = ""
    exit_reason: str = ""
    pnl_dollar: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class EquityPoint:
    """Point on the equity curve."""
    date: str
    portfolio_value: float
    cash: float
    positions_value: float
    num_positions: int


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


def classify_regime(sentiment: float, tech_confirms: int, include_moderate: bool = True) -> Optional[str]:
    """Classify regime based on sentiment and technicals."""
    if tech_confirms == 0:
        return None
    if sentiment > 0.12:
        return "strong_bullish"
    elif sentiment > 0.07 and include_moderate:
        return "moderate_bullish"
    elif sentiment < -0.15:
        return "strong_bearish"
    elif sentiment < -0.08 and include_moderate:
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
    """Get cached dates for a symbol."""
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT trade_date FROM cache_metadata WHERE symbol = ? ORDER BY trade_date",
        (symbol,)
    )
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    return dates


def get_option_contract(
    db_path: Path, symbol: str, trade_date: str, stock_price: float,
    option_type: str, target_dte: int
) -> Optional[dict]:
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
    """, (symbol, trade_date, option_type, min_expiry, max_expiry, MIN_OI,
          stock_price, (datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=target_dte)).strftime("%Y-%m-%d")))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "contract_id": row[0], "strike": row[1], "expiry": row[2],
        "bid": row[3], "ask": row[4], "mid": (row[3] + row[4]) / 2,
    }


def get_contract_price(db_path: Path, contract_id: str, trade_date: str) -> Optional[dict]:
    """Get price for a contract on a date."""
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


async def run_portfolio_simulation(
    symbol: str,
    start_date: str,
    quiver_api_key: str,
    position_size_pct: float,  # e.g., 0.05 for 5%
) -> dict:
    """Run portfolio simulation with compounding."""

    cached_dates = get_cached_dates(symbol)
    if not cached_dates:
        return {"error": "No cached data"}

    wsb_data = await fetch_wsb_history(symbol, quiver_api_key)
    if not wsb_data:
        return {"error": "No WSB data"}

    wsb_by_date = {}
    for item in wsb_data:
        date = item.get("Date", "")[:10]
        if date >= start_date:
            wsb_by_date[date] = item.get("Sentiment", 0)

    end_date = datetime.now().strftime("%Y-%m-%d")
    prices_df = get_price_data(symbol, start_date, end_date)
    if prices_df.empty:
        return {"error": "No price data"}

    technicals_df = calculate_technicals(prices_df)

    # Build regime windows
    active_regimes = {}
    for wsb_date, sentiment in wsb_by_date.items():
        is_bullish = sentiment > 0
        tech_confirms = 0
        if wsb_date in technicals_df.index:
            row = technicals_df.loc[wsb_date]
            tech_confirms = check_technical_confirmation(row, is_bullish)
        regime = classify_regime(sentiment, tech_confirms, INCLUDE_MODERATE)
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
                            active_regimes[date_str] = (regime, sentiment, days_marked + 1)
                        days_marked += 1
            except ValueError:
                pass

    # Portfolio state
    cash = STARTING_CAPITAL
    open_positions: list[Position] = []
    closed_positions: list[Position] = []
    equity_curve: list[EquityPoint] = []

    regime_trades = defaultdict(int)
    last_entry_date = None

    sorted_dates = sorted(set(cached_dates) & set(technicals_df.index))

    for date in sorted_dates:
        if date not in technicals_df.index:
            continue

        row = technicals_df.loc[date]
        stock_price = row['Close']

        # Update open positions
        positions_to_close = []
        for i, pos in enumerate(open_positions):
            price_data = get_contract_price(CACHE_DB, pos.contract_id, date)
            if not price_data:
                continue

            current_bid = price_data["bid"]
            pos.current_value = current_bid * pos.contracts * 100
            pnl_pct = (current_bid - pos.entry_ask) / pos.entry_ask

            # Check DTE
            try:
                expiry_date = datetime.strptime(price_data["expiry"], "%Y-%m-%d")
                current_dt = datetime.strptime(date, "%Y-%m-%d")
                dte = (expiry_date - current_dt).days
            except ValueError:
                dte = 7

            exit_reason = None
            if pnl_pct >= TAKE_PROFIT:
                exit_reason = "take_profit"
            elif pnl_pct <= STOP_LOSS:
                exit_reason = "stop_loss"
            elif dte <= 1:
                exit_reason = "dte_exit"

            # Check max hold
            entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
            current_dt = datetime.strptime(date, "%Y-%m-%d")
            hold_days = (current_dt - entry_dt).days
            if hold_days >= MAX_HOLD_DAYS:
                exit_reason = "max_hold"

            if exit_reason:
                pos.exit_date = date
                pos.exit_reason = exit_reason
                pos.pnl_pct = pnl_pct * 100
                exit_value = current_bid * pos.contracts * 100
                pos.pnl_dollar = exit_value - pos.position_value
                cash += exit_value
                positions_to_close.append(i)

        # Close positions
        for i in reversed(positions_to_close):
            closed_positions.append(open_positions.pop(i))

        # Check for new entry
        if date in active_regimes and len(open_positions) < MAX_CONCURRENT_POSITIONS:
            regime, orig_sentiment, day_in_window = active_regimes[date]

            regime_key = f"{regime}_{date[:7]}"
            if regime_trades[regime_key] < MAX_TRADES_PER_REGIME:
                # Check min days between entries
                can_enter = True
                if last_entry_date:
                    days_since = (datetime.strptime(date, "%Y-%m-%d") -
                                  datetime.strptime(last_entry_date, "%Y-%m-%d")).days
                    if days_since < MIN_DAYS_BETWEEN_ENTRIES:
                        can_enter = False

                if can_enter:
                    pullback_pct = row.get('pullback_pct', 0)
                    bounce_pct = row.get('bounce_pct', 0)

                    entry_trigger = None
                    option_type = None

                    if "bullish" in regime and pullback_pct >= PULLBACK_THRESHOLD:
                        entry_trigger = "pullback"
                        option_type = "call"
                    elif "bearish" in regime and bounce_pct >= PULLBACK_THRESHOLD:
                        entry_trigger = "bounce"
                        option_type = "put"

                    if entry_trigger:
                        contract = get_option_contract(
                            CACHE_DB, symbol, date, stock_price, option_type, TARGET_DTE
                        )

                        if contract:
                            # Calculate position size based on current portfolio value
                            portfolio_value = cash + sum(p.current_value for p in open_positions)
                            position_budget = portfolio_value * position_size_pct

                            # Calculate number of contracts (each contract is 100 shares)
                            contract_cost = contract["ask"] * 100
                            num_contracts = max(1, int(position_budget / contract_cost))
                            actual_cost = num_contracts * contract_cost

                            if actual_cost <= cash:
                                pos = Position(
                                    entry_date=date,
                                    option_type=option_type,
                                    regime_type=regime,
                                    contract_id=contract["contract_id"],
                                    entry_ask=contract["ask"],
                                    contracts=num_contracts,
                                    position_value=actual_cost,
                                    current_value=actual_cost,
                                )
                                open_positions.append(pos)
                                cash -= actual_cost
                                regime_trades[regime_key] += 1
                                last_entry_date = date

        # Record equity point
        positions_value = sum(p.current_value for p in open_positions)
        equity_curve.append(EquityPoint(
            date=date,
            portfolio_value=cash + positions_value,
            cash=cash,
            positions_value=positions_value,
            num_positions=len(open_positions),
        ))

    # Close any remaining positions at last available price
    for pos in open_positions:
        pos.exit_date = sorted_dates[-1] if sorted_dates else ""
        pos.exit_reason = "end_of_backtest"
        pos.pnl_dollar = pos.current_value - pos.position_value
        pos.pnl_pct = (pos.pnl_dollar / pos.position_value) * 100 if pos.position_value > 0 else 0
        cash += pos.current_value
        closed_positions.append(pos)

    # Calculate metrics
    final_value = cash
    total_return = (final_value - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    # Max drawdown
    peak = STARTING_CAPITAL
    max_dd_dollar = 0
    max_dd_pct = 0
    drawdown_start = None
    drawdown_peak = None
    worst_drawdown = {"peak_date": "", "peak_value": 0, "trough_date": "", "trough_value": 0}

    for eq in equity_curve:
        if eq.portfolio_value > peak:
            peak = eq.portfolio_value
            drawdown_start = eq.date
            drawdown_peak = peak

        dd_dollar = peak - eq.portfolio_value
        dd_pct = dd_dollar / peak * 100 if peak > 0 else 0

        if dd_dollar > max_dd_dollar:
            max_dd_dollar = dd_dollar
            max_dd_pct = dd_pct
            worst_drawdown = {
                "peak_date": drawdown_start,
                "peak_value": drawdown_peak,
                "trough_date": eq.date,
                "trough_value": eq.portfolio_value,
            }

    # Monthly returns
    monthly_returns = defaultdict(float)
    monthly_start = {}
    for eq in equity_curve:
        month = eq.date[:7]
        if month not in monthly_start:
            monthly_start[month] = eq.portfolio_value
        monthly_returns[month] = eq.portfolio_value

    monthly_pct = {}
    prev_value = STARTING_CAPITAL
    for month in sorted(monthly_returns.keys()):
        end_value = monthly_returns[month]
        monthly_pct[month] = (end_value - prev_value) / prev_value * 100 if prev_value > 0 else 0
        prev_value = end_value

    # Calculate Sharpe ratio (annualized)
    if len(monthly_pct) > 1:
        monthly_rets = list(monthly_pct.values())
        avg_monthly = np.mean(monthly_rets)
        std_monthly = np.std(monthly_rets)
        sharpe = (avg_monthly * 12) / (std_monthly * np.sqrt(12)) if std_monthly > 0 else 0
    else:
        sharpe = 0

    # Longest losing streak in dollars
    losing_streak = 0
    max_losing_streak = 0
    losing_streak_dollars = 0
    max_losing_dollars = 0

    for pos in sorted(closed_positions, key=lambda x: x.entry_date):
        if pos.pnl_dollar < 0:
            losing_streak += 1
            losing_streak_dollars += pos.pnl_dollar
        else:
            if losing_streak_dollars < max_losing_dollars:
                max_losing_dollars = losing_streak_dollars
                max_losing_streak = losing_streak
            losing_streak = 0
            losing_streak_dollars = 0

    # Check final streak
    if losing_streak_dollars < max_losing_dollars:
        max_losing_dollars = losing_streak_dollars
        max_losing_streak = losing_streak

    # Capital utilization
    total_days = len(equity_curve)
    days_with_positions = sum(1 for eq in equity_curve if eq.num_positions > 0)
    capital_utilization = days_with_positions / total_days * 100 if total_days > 0 else 0

    return {
        "position_size_pct": position_size_pct * 100,
        "starting_capital": STARTING_CAPITAL,
        "final_value": final_value,
        "total_return": total_return,
        "max_dd_dollar": max_dd_dollar,
        "max_dd_pct": max_dd_pct,
        "sharpe": sharpe,
        "capital_utilization": capital_utilization,
        "trades": len(closed_positions),
        "winners": len([p for p in closed_positions if p.pnl_dollar > 0]),
        "losers": len([p for p in closed_positions if p.pnl_dollar <= 0]),
        "avg_pnl_dollar": np.mean([p.pnl_dollar for p in closed_positions]) if closed_positions else 0,
        "avg_pnl_pct": np.mean([p.pnl_pct for p in closed_positions]) if closed_positions else 0,
        "monthly_returns": dict(sorted(monthly_pct.items())),
        "worst_drawdown": worst_drawdown,
        "max_losing_streak": max_losing_streak,
        "max_losing_dollars": abs(max_losing_dollars),
        "equity_curve": equity_curve,
        "positions": closed_positions,
    }


async def main():
    symbol = "TSLA"
    start_date = "2024-01-01"

    quiver_api_key = os.getenv("QUIVER_API_KEY")
    if not quiver_api_key:
        print("No QUIVER_API_KEY found")
        return

    print("=" * 80)
    print("PORTFOLIO SIMULATION - MONTHLY COMPARISON")
    print("=" * 80)
    print()
    print("Strategy Configuration:")
    print(f"  Ticker: {symbol}")
    print(f"  Regime window: {REGIME_WINDOW} days")
    print(f"  Pullback threshold: {PULLBACK_THRESHOLD}%")
    print(f"  DTE: {TARGET_DTE} days")
    print(f"  Take profit: +{int(TAKE_PROFIT*100)}%")
    print(f"  Stop loss: {int(STOP_LOSS*100)}%")
    print(f"  Max concurrent positions: {MAX_CONCURRENT_POSITIONS}")
    print()
    print("Portfolio Parameters:")
    print(f"  Starting capital: ${STARTING_CAPITAL:,}")
    print(f"  Compounding: Yes")
    print()

    # Test three position sizes
    position_sizes = [0.025, 0.05, 0.10]  # 2.5%, 5%, 10%

    results = []
    for pct in position_sizes:
        print(f"Running {pct*100:.1f}% position size simulation...", end=" ", flush=True)
        result = await run_portfolio_simulation(symbol, start_date, quiver_api_key, pct)
        results.append(result)
        print(f"Done - Final: ${result['final_value']:,.0f}")

    # Summary table
    print()
    print("=" * 80)
    print("SUMMARY METRICS")
    print("=" * 80)
    print()
    print(f"| {'Metric':<22} | {'2.5%':>12} | {'5%':>12} | {'10%':>12} |")
    print(f"|{'-'*24}|{'-'*14}|{'-'*14}|{'-'*14}|")

    r25, r50, r100 = results

    print(f"| {'Final Value':<22} | ${r25['final_value']:>10,.0f} | ${r50['final_value']:>10,.0f} | ${r100['final_value']:>10,.0f} |")
    print(f"| {'Total Return':<22} | {r25['total_return']:>11.1f}% | {r50['total_return']:>11.1f}% | {r100['total_return']:>11.1f}% |")
    print(f"| {'Max Drawdown $':<22} | ${r25['max_dd_dollar']:>10,.0f} | ${r50['max_dd_dollar']:>10,.0f} | ${r100['max_dd_dollar']:>10,.0f} |")
    print(f"| {'Max Drawdown %':<22} | {r25['max_dd_pct']:>11.1f}% | {r50['max_dd_pct']:>11.1f}% | {r100['max_dd_pct']:>11.1f}% |")
    print(f"| {'Sharpe Ratio':<22} | {r25['sharpe']:>12.2f} | {r50['sharpe']:>12.2f} | {r100['sharpe']:>12.2f} |")
    print(f"| {'Capital Utilization':<22} | {r25['capital_utilization']:>11.1f}% | {r50['capital_utilization']:>11.1f}% | {r100['capital_utilization']:>11.1f}% |")
    print(f"| {'Trades':<22} | {r25['trades']:>12} | {r50['trades']:>12} | {r100['trades']:>12} |")
    print(f"| {'Win Rate':<22} | {r25['winners']/r25['trades']*100 if r25['trades'] else 0:>11.1f}% | {r50['winners']/r50['trades']*100 if r50['trades'] else 0:>11.1f}% | {r100['winners']/r100['trades']*100 if r100['trades'] else 0:>11.1f}% |")
    print(f"| {'Avg P&L/Trade $':<22} | ${r25['avg_pnl_dollar']:>10,.0f} | ${r50['avg_pnl_dollar']:>10,.0f} | ${r100['avg_pnl_dollar']:>10,.0f} |")

    # Build monthly data for all three
    def get_monthly_data(result):
        monthly_values = {}
        for eq in result['equity_curve']:
            month = eq.date[:7]
            monthly_values[month] = eq.portfolio_value

        monthly_returns = {}
        prev_value = STARTING_CAPITAL
        for month in sorted(monthly_values.keys()):
            value = monthly_values[month]
            ret = (value - prev_value) / prev_value * 100 if prev_value > 0 else 0
            monthly_returns[month] = ret
            prev_value = value

        return monthly_values, monthly_returns

    mv25, mr25 = get_monthly_data(r25)
    mv50, mr50 = get_monthly_data(r50)
    mv100, mr100 = get_monthly_data(r100)

    # Get all months (use one that has all months)
    all_months = sorted(mv50.keys())

    # Filter to only months with trading activity (before 2025 data ends)
    active_months = [m for m in all_months if m <= "2025-01"]

    # Monthly Equity Curve Comparison
    print()
    print("=" * 80)
    print("MONTHLY EQUITY CURVE COMPARISON")
    print("=" * 80)
    print()
    print(f"| {'Month':<10} | {'2.5% Size':>12} | {'5% Size':>12} | {'10% Size':>12} |")
    print(f"|{'-'*12}|{'-'*14}|{'-'*14}|{'-'*14}|")

    for month in active_months:
        v25 = mv25.get(month, 0)
        v50 = mv50.get(month, 0)
        v100 = mv100.get(month, 0)
        print(f"| {month:<10} | ${v25:>10,.0f} | ${v50:>10,.0f} | ${v100:>10,.0f} |")

    # Monthly Returns Comparison
    print()
    print("=" * 80)
    print("MONTHLY RETURNS COMPARISON (%)")
    print("=" * 80)
    print()
    print(f"| {'Month':<10} | {'2.5%':>10} | {'5%':>10} | {'10%':>10} |")
    print(f"|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|")

    for month in active_months:
        ret25 = mr25.get(month, 0)
        ret50 = mr50.get(month, 0)
        ret100 = mr100.get(month, 0)
        print(f"| {month:<10} | {ret25:>+9.1f}% | {ret50:>+9.1f}% | {ret100:>+9.1f}% |")

    # Monthly Statistics
    print()
    print("=" * 80)
    print("MONTHLY RETURN STATISTICS")
    print("=" * 80)
    print()

    active_rets_25 = [mr25.get(m, 0) for m in active_months if m != "2024-01"]
    active_rets_50 = [mr50.get(m, 0) for m in active_months if m != "2024-01"]
    active_rets_100 = [mr100.get(m, 0) for m in active_months if m != "2024-01"]

    print(f"| {'Statistic':<22} | {'2.5%':>10} | {'5%':>10} | {'10%':>10} |")
    print(f"|{'-'*24}|{'-'*12}|{'-'*12}|{'-'*12}|")

    print(f"| {'Avg Monthly Return':<22} | {np.mean(active_rets_25):>+9.1f}% | {np.mean(active_rets_50):>+9.1f}% | {np.mean(active_rets_100):>+9.1f}% |")
    print(f"| {'Best Month':<22} | {max(active_rets_25):>+9.1f}% | {max(active_rets_50):>+9.1f}% | {max(active_rets_100):>+9.1f}% |")
    print(f"| {'Worst Month':<22} | {min(active_rets_25):>+9.1f}% | {min(active_rets_50):>+9.1f}% | {min(active_rets_100):>+9.1f}% |")
    print(f"| {'Monthly Std Dev':<22} | {np.std(active_rets_25):>9.1f}% | {np.std(active_rets_50):>9.1f}% | {np.std(active_rets_100):>9.1f}% |")
    print(f"| {'Negative Months':<22} | {len([r for r in active_rets_25 if r < 0]):>10} | {len([r for r in active_rets_50 if r < 0]):>10} | {len([r for r in active_rets_100 if r < 0]):>10} |")
    print(f"| {'Positive Months':<22} | {len([r for r in active_rets_25 if r > 0]):>10} | {len([r for r in active_rets_50 if r > 0]):>10} | {len([r for r in active_rets_100 if r > 0]):>10} |")

    # Worst drawdown period
    print()
    print("=" * 80)
    print("WORST DRAWDOWN PERIOD (5% Position Sizing)")
    print("=" * 80)
    print()

    wd = r50['worst_drawdown']
    if wd['peak_date']:
        print(f"Peak date:   {wd['peak_date']}")
        print(f"Peak value:  ${wd['peak_value']:,.0f}")
        print(f"Trough date: {wd['trough_date']}")
        print(f"Trough value: ${wd['trough_value']:,.0f}")
        print(f"Drawdown:    ${wd['peak_value'] - wd['trough_value']:,.0f} ({r50['max_dd_pct']:.1f}%)")

        # Calculate duration
        peak_dt = datetime.strptime(wd['peak_date'], "%Y-%m-%d")
        trough_dt = datetime.strptime(wd['trough_date'], "%Y-%m-%d")
        duration = (trough_dt - peak_dt).days
        print(f"Duration:    {duration} days")

    # Longest losing streak
    print()
    print("=" * 80)
    print("LONGEST LOSING STREAK (5% Position Sizing)")
    print("=" * 80)
    print()
    print(f"Consecutive losses: {r50['max_losing_streak']} trades")
    print(f"Total lost:         ${r50['max_losing_dollars']:,.0f}")

    # Individual trade breakdown for 5%
    print()
    print("=" * 80)
    print("TRADE LOG (5% Position Sizing)")
    print("=" * 80)
    print()
    print(f"{'Entry':<12} {'Type':<6} {'Regime':<18} {'Contracts':>9} {'P&L $':>10} {'P&L %':>8} {'Exit Reason':<12}")
    print("-" * 85)

    for pos in sorted(r50['positions'], key=lambda x: x.entry_date):
        regime_short = pos.regime_type.replace("_", " ").title()[:15]
        print(f"{pos.entry_date:<12} {pos.option_type:<6} {regime_short:<18} {pos.contracts:>9} ${pos.pnl_dollar:>+9,.0f} {pos.pnl_pct:>+7.1f}% {pos.exit_reason:<12}")

    # Recommendation
    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    # Find sizing that keeps max DD under 25%
    for r in results:
        pct = r['position_size_pct']
        dd = r['max_dd_pct']
        ret = r['total_return']

        if dd <= 25:
            recommended = r
            break
    else:
        recommended = results[0]  # Default to most conservative

    print(f"Target: Max drawdown should not exceed 25% for psychological comfort.")
    print()

    for r in results:
        pct = r['position_size_pct']
        dd = r['max_dd_pct']
        status = "PASS" if dd <= 25 else "FAIL"
        marker = " <-- RECOMMENDED" if r == recommended else ""
        print(f"  {pct:>4.1f}%: Max DD = {dd:>5.1f}% [{status}] Return = {r['total_return']:>6.1f}%{marker}")

    print()
    print(f"RECOMMENDATION: Use {recommended['position_size_pct']:.1f}% position sizing")
    print()
    print(f"  Expected annual return: ~{recommended['total_return']:.0f}% (over {len(recommended['monthly_returns'])} months)")
    print(f"  Maximum drawdown: {recommended['max_dd_pct']:.1f}% (${recommended['max_dd_dollar']:,.0f})")
    print(f"  Sharpe ratio: {recommended['sharpe']:.2f}")
    print(f"  Worst losing streak: {recommended['max_losing_streak']} trades (${recommended['max_losing_dollars']:,.0f})")
    print()

    if recommended['max_dd_pct'] <= 15:
        print("Risk Level: CONSERVATIVE - Can increase position size if comfortable")
    elif recommended['max_dd_pct'] <= 25:
        print("Risk Level: MODERATE - Good balance of risk and return")
    else:
        print("Risk Level: AGGRESSIVE - Consider reducing position size for comfort")


if __name__ == "__main__":
    asyncio.run(main())
