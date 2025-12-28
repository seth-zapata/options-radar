"""Sentiment data collector for building local historical data.

Since Quiver's historical WSB endpoint requires a premium subscription,
we collect and store sentiment data locally over time to enable backtesting.

Run daily to build up historical sentiment data:
    python -m backend.backtest.sentiment_collector

Data is stored in: ./data/sentiment_history.jsonl
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import load_config
from backend.data.quiver_client import QuiverClient

logger = logging.getLogger(__name__)

# Default storage location
DEFAULT_STORAGE_PATH = Path("./data/sentiment_history.jsonl")

# Symbols to track
DEFAULT_SYMBOLS = ["QQQ", "SPY", "AAPL", "NVDA", "AMD", "TSLA", "GOOGL", "AMZN", "META", "MSFT"]


async def collect_sentiment(
    symbols: list[str] | None = None,
    storage_path: Path | None = None,
) -> dict[str, Any]:
    """Collect current WSB sentiment for symbols and store locally.

    Args:
        symbols: Symbols to collect (default: common watchlist)
        storage_path: Path to store data (default: ./data/sentiment_history.jsonl)

    Returns:
        Dict with collection results
    """
    symbols = symbols or DEFAULT_SYMBOLS
    storage_path = storage_path or DEFAULT_STORAGE_PATH

    # Ensure storage directory exists
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config()
    client = QuiverClient(config=config.quiver)

    collected = []
    errors = []
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    for symbol in symbols:
        try:
            sentiment = await client.get_wsb_sentiment(symbol)

            if sentiment:
                record = {
                    "date": date_str,
                    "timestamp": now.isoformat(),
                    "symbol": symbol,
                    "sentiment": sentiment.sentiment,
                    "mentions_24h": sentiment.mentions_24h,
                    "rank": sentiment.rank,
                    "buzz_level": sentiment.buzz_level,
                }
                collected.append(record)
                logger.info(f"Collected sentiment for {symbol}: {sentiment.sentiment:.2f}")
            else:
                logger.debug(f"No WSB data for {symbol}")

        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})
            logger.error(f"Error collecting {symbol}: {e}")

    # Append to storage file (JSONL format - one JSON per line)
    if collected:
        with open(storage_path, "a") as f:
            for record in collected:
                f.write(json.dumps(record) + "\n")
        logger.info(f"Stored {len(collected)} sentiment records to {storage_path}")

    return {
        "date": date_str,
        "collected": len(collected),
        "errors": len(errors),
        "symbols": [r["symbol"] for r in collected],
    }


def load_sentiment_history(
    storage_path: Path | None = None,
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Load sentiment history from local storage.

    Args:
        storage_path: Path to storage file
        symbol: Filter by symbol (optional)
        start_date: Filter by start date YYYY-MM-DD (optional)
        end_date: Filter by end date YYYY-MM-DD (optional)

    Returns:
        List of sentiment records
    """
    storage_path = storage_path or DEFAULT_STORAGE_PATH

    if not storage_path.exists():
        logger.warning(f"No sentiment history found at {storage_path}")
        return []

    records = []
    with open(storage_path) as f:
        for line in f:
            try:
                record = json.loads(line.strip())

                # Apply filters
                if symbol and record.get("symbol") != symbol:
                    continue
                if start_date and record.get("date", "") < start_date:
                    continue
                if end_date and record.get("date", "") > end_date:
                    continue

                records.append(record)

            except json.JSONDecodeError:
                continue

    # Remove duplicates (keep latest per symbol per date)
    seen = {}
    for record in records:
        key = (record.get("symbol"), record.get("date"))
        seen[key] = record

    return list(seen.values())


def get_storage_stats(storage_path: Path | None = None) -> dict[str, Any]:
    """Get statistics about stored sentiment data.

    Returns:
        Dict with storage statistics
    """
    storage_path = storage_path or DEFAULT_STORAGE_PATH

    if not storage_path.exists():
        return {
            "exists": False,
            "path": str(storage_path),
            "records": 0,
            "symbols": [],
            "date_range": None,
        }

    records = load_sentiment_history(storage_path)

    if not records:
        return {
            "exists": True,
            "path": str(storage_path),
            "records": 0,
            "symbols": [],
            "date_range": None,
        }

    symbols = sorted(set(r.get("symbol") for r in records))
    dates = sorted(r.get("date") for r in records)

    return {
        "exists": True,
        "path": str(storage_path),
        "records": len(records),
        "symbols": symbols,
        "date_range": {
            "start": dates[0] if dates else None,
            "end": dates[-1] if dates else None,
            "days": len(set(dates)),
        },
    }


async def main():
    """Run sentiment collection from command line."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Collect WSB sentiment data")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--stats", action="store_true", help="Show storage stats only")

    args = parser.parse_args()

    if args.stats:
        stats = get_storage_stats()
        print(f"\nSentiment Storage Stats:")
        print(f"  Path: {stats['path']}")
        print(f"  Exists: {stats['exists']}")
        print(f"  Records: {stats['records']}")
        print(f"  Symbols: {', '.join(stats['symbols']) if stats['symbols'] else 'none'}")
        if stats.get("date_range"):
            dr = stats["date_range"]
            print(f"  Date Range: {dr['start']} to {dr['end']} ({dr['days']} days)")
        return

    symbols = args.symbols.split(",") if args.symbols else None

    try:
        result = await collect_sentiment(symbols)
        print(f"\nCollection complete:")
        print(f"  Date: {result['date']}")
        print(f"  Collected: {result['collected']} symbols")
        print(f"  Errors: {result['errors']}")

        stats = get_storage_stats()
        print(f"\nStorage now has {stats['records']} records across {stats['date_range']['days'] if stats.get('date_range') else 0} days")

    except ValueError as e:
        print(f"Error: {e}")
        print("\nMake sure QUIVER_API_KEY is set in .env")


if __name__ == "__main__":
    asyncio.run(main())
