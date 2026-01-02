#!/usr/bin/env python3
"""Portfolio Simulation with Compounding.

Mirrors the live trading system behavior:
- 7-day regime window
- 1.5% pullback/bounce threshold
- 7 DTE weeklies
- Bid/ask realistic fills
- Exit rules: +40% take profit, -20% stop loss, DTE < 1
- Cooldown: "while open" (block same direction while position open)

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

# Trade management (matches live trading system)
STOP_LOSS = -0.20       # -20% stop loss
TAKE_PROFIT = 0.40      # +40% take profit
MIN_DTE_EXIT = 1        # Exit when DTE < 1 (matches live)
MIN_OI = 500

# Cooldown strategy: "while open" (matches live trading)
# - Block same-direction entries while a position is open
# - Allow new entry immediately after position closes
# - Validated via backtest: +852% total P&L, 42.6% win rate

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

    # "While open" cooldown: track which directions have open positions
    # This matches the live trading system behavior
    open_directions: set[str] = set()  # Contains "call" and/or "put"

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
            elif dte <= MIN_DTE_EXIT:
                exit_reason = "dte_exit"

            if exit_reason:
                pos.exit_date = date
                pos.exit_reason = exit_reason
                pos.pnl_pct = pnl_pct * 100
                exit_value = current_bid * pos.contracts * 100
                pos.pnl_dollar = exit_value - pos.position_value
                cash += exit_value
                positions_to_close.append(i)

        # Close positions and update open_directions
        for i in reversed(positions_to_close):
            closed_pos = open_positions.pop(i)
            # Remove direction from cooldown tracking (allows re-entry)
            open_directions.discard(closed_pos.option_type)
            closed_positions.append(closed_pos)

        # Check for new entry
        if date in active_regimes and len(open_positions) < MAX_CONCURRENT_POSITIONS:
            regime, orig_sentiment, day_in_window = active_regimes[date]

            # Determine target direction based on regime
            if "bullish" in regime:
                target_direction = "call"
            elif "bearish" in regime:
                target_direction = "put"
            else:
                continue

            # "While open" cooldown: block if same direction already open
            if target_direction in open_directions:
                continue

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
                        # Track direction for "while open" cooldown
                        open_directions.add(option_type)

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
    print("Strategy Configuration (mirrors live trading):")
    print(f"  Ticker: {symbol}")
    print(f"  Regime window: {REGIME_WINDOW} days")
    print(f"  Pullback threshold: {PULLBACK_THRESHOLD}%")
    print(f"  DTE: {TARGET_DTE} days")
    print(f"  Take profit: +{int(TAKE_PROFIT*100)}%")
    print(f"  Stop loss: {int(STOP_LOSS*100)}%")
    print(f"  Time exit: DTE < {MIN_DTE_EXIT}")
    print(f"  Cooldown: 'while open' (block same direction while position open)")
    print(f"  Max concurrent positions: {MAX_CONCURRENT_POSITIONS}")
    print()
    print("Portfolio Parameters:")
    print(f"  Starting capital: ${STARTING_CAPITAL:,}")
    print(f"  Compounding: Yes")
    print()

    # Test five position sizes (2.5% to 20%)
    position_sizes = [0.025, 0.05, 0.10, 0.15, 0.20]  # 2.5%, 5%, 10%, 15%, 20%

    results = []
    for pct in position_sizes:
        print(f"Running {pct*100:.1f}% position size simulation...", end=" ", flush=True)
        result = await run_portfolio_simulation(symbol, start_date, quiver_api_key, pct)
        results.append(result)
        print(f"Done - Final: ${result['final_value']:,.0f}")

    # Helper to build dynamic table headers and rows
    def format_pct(pct: float) -> str:
        """Format position size percentage for display."""
        return f"{pct:.1f}%" if pct < 10 else f"{pct:.0f}%"

    col_width = 12
    num_cols = len(results)

    # Build header row
    def print_table_header(metric_label: str):
        header = f"| {metric_label:<22} |"
        for r in results:
            header += f" {format_pct(r['position_size_pct']):>{col_width}} |"
        print(header)

    def print_table_separator():
        sep = f"|{'-'*24}|"
        for _ in results:
            sep += f"{'-'*(col_width+2)}|"
        print(sep)

    # Summary table
    print()
    print("=" * 100)
    print("SUMMARY METRICS")
    print("=" * 100)
    print()
    print_table_header("Metric")
    print_table_separator()

    # Final Value
    row = f"| {'Final Value':<22} |"
    for r in results:
        row += f" ${r['final_value']:>{col_width-1},.0f} |"
    print(row)

    # Total Return
    row = f"| {'Total Return':<22} |"
    for r in results:
        row += f" {r['total_return']:>{col_width-1}.1f}% |"
    print(row)

    # Max Drawdown $
    row = f"| {'Max Drawdown $':<22} |"
    for r in results:
        row += f" ${r['max_dd_dollar']:>{col_width-1},.0f} |"
    print(row)

    # Max Drawdown %
    row = f"| {'Max Drawdown %':<22} |"
    for r in results:
        row += f" {r['max_dd_pct']:>{col_width-1}.1f}% |"
    print(row)

    # Sharpe Ratio
    row = f"| {'Sharpe Ratio':<22} |"
    for r in results:
        row += f" {r['sharpe']:>{col_width}.2f} |"
    print(row)

    # Capital Utilization
    row = f"| {'Capital Utilization':<22} |"
    for r in results:
        row += f" {r['capital_utilization']:>{col_width-1}.1f}% |"
    print(row)

    # Trades
    row = f"| {'Trades':<22} |"
    for r in results:
        row += f" {r['trades']:>{col_width}} |"
    print(row)

    # Win Rate
    row = f"| {'Win Rate':<22} |"
    for r in results:
        wr = r['winners'] / r['trades'] * 100 if r['trades'] else 0
        row += f" {wr:>{col_width-1}.1f}% |"
    print(row)

    # Avg P&L/Trade
    row = f"| {'Avg P&L/Trade $':<22} |"
    for r in results:
        row += f" ${r['avg_pnl_dollar']:>{col_width-1},.0f} |"
    print(row)

    # Build monthly data for all results
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

    monthly_data = [get_monthly_data(r) for r in results]

    # Get all months from first result
    all_months = sorted(monthly_data[0][0].keys())
    active_months = [m for m in all_months if m <= "2025-01"]

    # Monthly Equity Curve Comparison
    print()
    print("=" * 100)
    print("MONTHLY EQUITY CURVE COMPARISON")
    print("=" * 100)
    print()

    header = f"| {'Month':<10} |"
    for r in results:
        header += f" {format_pct(r['position_size_pct']):>{col_width}} |"
    print(header)

    sep = f"|{'-'*12}|"
    for _ in results:
        sep += f"{'-'*(col_width+2)}|"
    print(sep)

    for month in active_months:
        row = f"| {month:<10} |"
        for mv, _ in monthly_data:
            v = mv.get(month, 0)
            row += f" ${v:>{col_width-1},.0f} |"
        print(row)

    # Monthly Returns Comparison
    print()
    print("=" * 100)
    print("MONTHLY RETURNS COMPARISON (%)")
    print("=" * 100)
    print()

    header = f"| {'Month':<10} |"
    for r in results:
        header += f" {format_pct(r['position_size_pct']):>{col_width}} |"
    print(header)
    print(sep)

    for month in active_months:
        row = f"| {month:<10} |"
        for _, mr in monthly_data:
            ret = mr.get(month, 0)
            row += f" {ret:>+{col_width-1}.1f}% |"
        print(row)

    # Monthly Statistics
    print()
    print("=" * 100)
    print("MONTHLY RETURN STATISTICS")
    print("=" * 100)
    print()

    # Calculate active returns for each result
    active_returns = []
    for _, mr in monthly_data:
        rets = [mr.get(m, 0) for m in active_months if m != "2024-01"]
        active_returns.append(rets)

    print_table_header("Statistic")
    print_table_separator()

    # Avg Monthly Return
    row = f"| {'Avg Monthly Return':<22} |"
    for rets in active_returns:
        row += f" {np.mean(rets):>+{col_width-1}.1f}% |"
    print(row)

    # Best Month
    row = f"| {'Best Month':<22} |"
    for rets in active_returns:
        row += f" {max(rets):>+{col_width-1}.1f}% |"
    print(row)

    # Worst Month
    row = f"| {'Worst Month':<22} |"
    for rets in active_returns:
        row += f" {min(rets):>+{col_width-1}.1f}% |"
    print(row)

    # Monthly Std Dev
    row = f"| {'Monthly Std Dev':<22} |"
    for rets in active_returns:
        row += f" {np.std(rets):>{col_width-1}.1f}% |"
    print(row)

    # Negative Months
    row = f"| {'Negative Months':<22} |"
    for rets in active_returns:
        row += f" {len([r for r in rets if r < 0]):>{col_width}} |"
    print(row)

    # Positive Months
    row = f"| {'Positive Months':<22} |"
    for rets in active_returns:
        row += f" {len([r for r in rets if r > 0]):>{col_width}} |"
    print(row)

    # Use 10% position size for detailed analysis (middle ground)
    ref_idx = 2 if len(results) > 2 else 0  # 10% is at index 2
    ref_result = results[ref_idx]
    ref_pct = format_pct(ref_result['position_size_pct'])

    # Worst drawdown period
    print()
    print("=" * 100)
    print(f"WORST DRAWDOWN PERIOD ({ref_pct} Position Sizing)")
    print("=" * 100)
    print()

    wd = ref_result['worst_drawdown']
    if wd['peak_date']:
        print(f"Peak date:   {wd['peak_date']}")
        print(f"Peak value:  ${wd['peak_value']:,.0f}")
        print(f"Trough date: {wd['trough_date']}")
        print(f"Trough value: ${wd['trough_value']:,.0f}")
        print(f"Drawdown:    ${wd['peak_value'] - wd['trough_value']:,.0f} ({ref_result['max_dd_pct']:.1f}%)")

        peak_dt = datetime.strptime(wd['peak_date'], "%Y-%m-%d")
        trough_dt = datetime.strptime(wd['trough_date'], "%Y-%m-%d")
        duration = (trough_dt - peak_dt).days
        print(f"Duration:    {duration} days")

    # Longest losing streak
    print()
    print("=" * 100)
    print(f"LONGEST LOSING STREAK ({ref_pct} Position Sizing)")
    print("=" * 100)
    print()
    print(f"Consecutive losses: {ref_result['max_losing_streak']} trades")
    print(f"Total lost:         ${ref_result['max_losing_dollars']:,.0f}")

    # Individual trade breakdown
    print()
    print("=" * 100)
    print(f"TRADE LOG ({ref_pct} Position Sizing)")
    print("=" * 100)
    print()
    print(f"{'Entry':<12} {'Type':<6} {'Regime':<18} {'Contracts':>9} {'P&L $':>10} {'P&L %':>8} {'Exit Reason':<12}")
    print("-" * 85)

    for pos in sorted(ref_result['positions'], key=lambda x: x.entry_date):
        regime_short = pos.regime_type.replace("_", " ").title()[:15]
        print(f"{pos.entry_date:<12} {pos.option_type:<6} {regime_short:<18} {pos.contracts:>9} ${pos.pnl_dollar:>+9,.0f} {pos.pnl_pct:>+7.1f}% {pos.exit_reason:<12}")

    # Recommendation
    print()
    print("=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print()

    # Find sizing that keeps max DD under 25%
    recommended = None
    for r in results:
        if r['max_dd_pct'] <= 25:
            recommended = r
            break
    if recommended is None:
        recommended = results[0]  # Default to most conservative

    print("Target: Max drawdown should not exceed 25% for psychological comfort.")
    print()

    for r in results:
        pct = r['position_size_pct']
        dd = r['max_dd_pct']
        status = "PASS" if dd <= 25 else "FAIL"
        marker = " <-- RECOMMENDED" if r == recommended else ""
        print(f"  {pct:>5.1f}%: Max DD = {dd:>5.1f}% [{status}] Return = {r['total_return']:>7.1f}% Sharpe = {r['sharpe']:.2f}{marker}")

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
