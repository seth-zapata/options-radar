#!/usr/bin/env python3
"""Run the directional backtest from command line.

Usage:
    # Default (QQQ, SPY, AAPL for last 6 months)
    python -m backend.run_backtest

    # Custom symbols and dates
    python -m backend.run_backtest --symbols NVDA,AMD,TSLA --start 2024-01-01 --end 2024-12-01
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta

from backend.backtest.directional import run_backtest_cli


def main():
    parser = argparse.ArgumentParser(
        description="Run directional backtest using WSB sentiment"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: QQQ,SPY,AAPL)",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date YYYY-MM-DD (default: 6 months ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date YYYY-MM-DD (default: yesterday)",
    )

    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None

    try:
        asyncio.run(run_backtest_cli(
            symbols=symbols,
            start_date=args.start,
            end_date=args.end,
        ))
    except ValueError as e:
        print(f"Error: {e}")
        print("\nMake sure you have configured API keys in .env:")
        print("  ALPACA_API_KEY=...")
        print("  ALPACA_SECRET_KEY=...")
        print("  QUIVER_API_KEY=...")
        sys.exit(1)
    except Exception as e:
        print(f"Error running backtest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
