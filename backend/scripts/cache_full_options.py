#!/usr/bin/env python3
"""Cache COMPLETE historical options data from EODHD.

Caches ALL dates, ALL strikes, ALL expiries for specified symbols.
This gives us ground-truth entry AND exit prices for P&L backtesting.

Usage:
    python -m backend.scripts.cache_full_options --symbols TSLA,NVDA,PLTR
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import load_config, EODHDConfig
from backend.data.eodhd_client import EODHDClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Cache database path
CACHE_DB = Path(__file__).parent.parent.parent / "cache" / "options_data.db"


def init_db(db_path: Path) -> None:
    """Initialize SQLite database with schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Main contracts table - stores ALL contract data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS options_contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            contract_id TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            expiry TEXT NOT NULL,
            option_type TEXT NOT NULL,
            strike REAL NOT NULL,
            bid REAL,
            ask REAL,
            last REAL,
            volume INTEGER,
            open_interest INTEGER,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL,
            iv REAL,
            created_at TEXT NOT NULL,
            UNIQUE(contract_id, trade_date)
        )
    """)

    # Indexes for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_options_symbol_date
        ON options_contracts (symbol, trade_date)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_options_contract_date
        ON options_contracts (contract_id, trade_date)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_options_strike_expiry
        ON options_contracts (symbol, trade_date, strike, expiry, option_type)
    """)

    # Cache metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache_metadata (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            num_contracts INTEGER,
            cached_at TEXT NOT NULL,
            PRIMARY KEY (symbol, trade_date)
        )
    """)

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")


def is_date_cached(db_path: Path, symbol: str, trade_date: str) -> bool:
    """Check if we have cached data for a symbol/date."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM cache_metadata WHERE symbol = ? AND trade_date = ?",
        (symbol, trade_date)
    )
    result = cursor.fetchone() is not None
    conn.close()
    return result


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    """Generate list of trading days (weekdays only)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    days = []
    current = start
    while current <= end:
        # Skip weekends
        if current.weekday() < 5:
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return days


async def fetch_and_cache_chain(
    session: aiohttp.ClientSession,
    api_key: str,
    symbol: str,
    trade_date: str,
    db_path: Path,
) -> int:
    """Fetch options chain for a date and cache to database."""
    base_url = "https://eodhd.com/api"
    endpoint = "mp/unicornbay/options/contracts"

    all_contracts = []
    offset = 0
    total = None

    while True:
        params = {
            "api_token": api_key,
            "fmt": "json",
            "filter[underlying_symbol]": symbol,
            "filter[tradetime_eq]": trade_date,
            "page[limit]": 1000,
            "page[offset]": offset,
        }

        try:
            async with session.get(f"{base_url}/{endpoint}", params=params) as response:
                if response.status == 429:
                    logger.warning("Rate limit hit, waiting 5s...")
                    await asyncio.sleep(5)
                    continue

                if response.status != 200:
                    logger.warning(f"API error {response.status} for {symbol} {trade_date}")
                    return 0

                data = await response.json()

                if not data or "data" not in data:
                    if offset == 0:
                        return 0  # No data for this date
                    break

                # Get total from meta
                if total is None:
                    total = data.get("meta", {}).get("total", 0)

                # Parse contracts
                for item in data["data"]:
                    attrs = item.get("attributes", {})
                    all_contracts.append({
                        "contract_id": attrs.get("contract", ""),
                        "expiry": attrs.get("exp_date", ""),
                        "option_type": attrs.get("type", "").lower(),
                        "strike": float(attrs.get("strike", 0) or 0),
                        "bid": float(attrs.get("bid", 0) or 0),
                        "ask": float(attrs.get("ask", 0) or 0),
                        "last": float(attrs.get("last", 0) or 0),
                        "volume": int(attrs.get("volume", 0) or 0),
                        "open_interest": int(attrs.get("open_interest", 0) or 0),
                        "delta": float(attrs.get("delta", 0) or 0),
                        "gamma": float(attrs.get("gamma", 0) or 0),
                        "theta": float(attrs.get("theta", 0) or 0),
                        "vega": float(attrs.get("vega", 0) or 0),
                        "iv": float(attrs.get("volatility", 0) or 0),
                    })

                offset += len(data["data"])
                if offset >= total or len(data["data"]) < 1000:
                    break

                # Small delay between pages
                await asyncio.sleep(0.2)

        except Exception as e:
            logger.error(f"Error fetching {symbol} {trade_date}: {e}")
            return 0

    if not all_contracts:
        return 0

    # Store in database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    for c in all_contracts:
        cursor.execute("""
            INSERT OR REPLACE INTO options_contracts (
                symbol, contract_id, trade_date, expiry, option_type, strike,
                bid, ask, last, volume, open_interest,
                delta, gamma, theta, vega, iv, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, c["contract_id"], trade_date, c["expiry"], c["option_type"], c["strike"],
            c["bid"], c["ask"], c["last"], c["volume"], c["open_interest"],
            c["delta"], c["gamma"], c["theta"], c["vega"], c["iv"], now
        ))

    # Update metadata
    cursor.execute("""
        INSERT OR REPLACE INTO cache_metadata (symbol, trade_date, num_contracts, cached_at)
        VALUES (?, ?, ?, ?)
    """, (symbol, trade_date, len(all_contracts), now))

    conn.commit()
    conn.close()

    return len(all_contracts)


async def cache_symbol(
    api_key: str,
    symbol: str,
    start_date: str,
    end_date: str,
    db_path: Path,
) -> dict:
    """Cache all options data for a symbol."""
    trading_days = get_trading_days(start_date, end_date)

    # Filter out already cached dates
    dates_to_fetch = []
    for date in trading_days:
        if not is_date_cached(db_path, symbol, date):
            dates_to_fetch.append(date)

    cached_count = len(trading_days) - len(dates_to_fetch)
    logger.info(f"{symbol}: {len(dates_to_fetch)} dates to fetch ({cached_count} already cached)")

    if not dates_to_fetch:
        return {"symbol": symbol, "dates_fetched": 0, "contracts": 0, "skipped": cached_count}

    total_contracts = 0
    fetched = 0
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        for i, date in enumerate(dates_to_fetch):
            contracts = await fetch_and_cache_chain(session, api_key, symbol, date, db_path)
            total_contracts += contracts
            fetched += 1

            # Progress update every 20 dates
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = fetched / elapsed if elapsed > 0 else 0
                remaining = len(dates_to_fetch) - fetched
                eta_mins = remaining / rate / 60 if rate > 0 else 0
                logger.info(
                    f"  {symbol}: {fetched}/{len(dates_to_fetch)} dates "
                    f"({total_contracts:,} contracts) - ETA: {eta_mins:.1f} min"
                )

            # Rate limiting
            await asyncio.sleep(0.3)

    elapsed = time.time() - start_time
    logger.info(
        f"{symbol} COMPLETE: {fetched} dates, {total_contracts:,} contracts in {elapsed/60:.1f} min"
    )

    return {
        "symbol": symbol,
        "dates_fetched": fetched,
        "contracts": total_contracts,
        "skipped": cached_count,
        "time_mins": elapsed / 60,
    }


def get_cache_stats(db_path: Path) -> dict:
    """Get cache statistics."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM options_contracts")
    total_contracts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cache_metadata")
    total_dates = cursor.fetchone()[0]

    cursor.execute("SELECT symbol, COUNT(*) FROM cache_metadata GROUP BY symbol")
    by_symbol = dict(cursor.fetchall())

    cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM cache_metadata")
    min_date, max_date = cursor.fetchone()

    db_size = db_path.stat().st_size / 1024 / 1024  # MB

    conn.close()

    return {
        "total_contracts": total_contracts,
        "total_dates": total_dates,
        "by_symbol": by_symbol,
        "date_range": (min_date, max_date),
        "db_size_mb": db_size,
    }


async def main():
    parser = argparse.ArgumentParser(description="Cache historical options data from EODHD")
    parser.add_argument(
        "--symbols",
        type=str,
        default="TSLA,NVDA,PLTR",
        help="Comma-separated list of symbols",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-02",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (default: today)",
    )

    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")

    # Load API key
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    api_key = os.getenv("EODHD_API_KEY")

    if not api_key:
        logger.error("No EODHD_API_KEY found")
        return

    print("=" * 60)
    print("EODHD OPTIONS DATA CACHING")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print(f"Date Range: {args.start} to {end_date}")
    print(f"Database: {CACHE_DB}")
    print()

    # Initialize database
    init_db(CACHE_DB)

    # Cache each symbol
    start_time = time.time()
    results = []

    for symbol in symbols:
        result = await cache_symbol(api_key, symbol, args.start, end_date, CACHE_DB)
        results.append(result)
        print()

    total_time = time.time() - start_time

    # Summary
    print("=" * 60)
    print("CACHING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print()

    for r in results:
        print(f"  {r['symbol']}: {r['dates_fetched']} dates, {r['contracts']:,} contracts")

    # Show cache stats
    stats = get_cache_stats(CACHE_DB)
    print()
    print(f"Cache Statistics:")
    print(f"  Total contracts: {stats.get('total_contracts', 0):,}")
    print(f"  Total (symbol, date) pairs: {stats.get('total_dates', 0)}")
    print(f"  Database size: {stats.get('db_size_mb', 0):.1f} MB")
    print(f"  Date range: {stats.get('date_range', ('', ''))[0]} to {stats.get('date_range', ('', ''))[1]}")


if __name__ == "__main__":
    asyncio.run(main())
