"""Cache system for historical stock bars from Alpaca.

Fetches and caches 1-minute and 1-second bars for backtesting, so we don't
need to re-fetch from Alpaca on every backtest run.

1-second bars are aggregated from trade data since Alpaca doesn't provide
sub-minute bar timeframes directly.
"""

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp

from backend.config import AlpacaConfig
from backend.data.alpaca_rest import AlpacaRestClient


def _get_alpaca_config() -> AlpacaConfig:
    """Get Alpaca config from environment."""
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

    if not api_key or not secret_key:
        # Try loading from .env file
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ALPACA_API_KEY="):
                        api_key = line.split("=", 1)[1].strip('"\'')
                    elif line.startswith("ALPACA_SECRET_KEY="):
                        secret_key = line.split("=", 1)[1].strip('"\'')

    return AlpacaConfig(api_key=api_key, secret_key=secret_key, paper=paper)


@dataclass
class CachedBar:
    """A single cached bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None


class BarCache:
    """Cache for historical stock bars.

    Stores bars in JSON files organized by symbol and date.
    Cache structure: cache/bars/{symbol}/{YYYY-MM-DD}.json
    """

    def __init__(self, cache_dir: str = "cache/bars"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[datetime, CachedBar]] = {}

    def _get_cache_path(self, symbol: str, target_date: date) -> Path:
        """Get cache file path for a symbol/date."""
        symbol_dir = self.cache_dir / symbol.upper()
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / f"{target_date.isoformat()}.json"

    def _has_cached_date(self, symbol: str, target_date: date) -> bool:
        """Check if we have cached data for a date."""
        return self._get_cache_path(symbol, target_date).exists()

    def _load_cached_date(self, symbol: str, target_date: date) -> list[CachedBar]:
        """Load cached bars for a date."""
        cache_path = self._get_cache_path(symbol, target_date)
        if not cache_path.exists():
            return []

        with open(cache_path) as f:
            data = json.load(f)

        bars = []
        for bar in data.get("bars", []):
            bars.append(CachedBar(
                timestamp=datetime.fromisoformat(bar["timestamp"]),
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                vwap=bar.get("vwap"),
            ))
        return bars

    def _save_cached_date(self, symbol: str, target_date: date, bars: list[CachedBar]) -> None:
        """Save bars to cache for a date."""
        cache_path = self._get_cache_path(symbol, target_date)

        data = {
            "symbol": symbol,
            "date": target_date.isoformat(),
            "fetched_at": datetime.now().isoformat(),
            "bar_count": len(bars),
            "bars": [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                }
                for bar in bars
            ]
        }

        with open(cache_path, "w") as f:
            json.dump(data, f)

    async def fetch_and_cache_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        force_refresh: bool = False,
    ) -> int:
        """Fetch and cache bars for a date range.

        Args:
            symbol: Stock symbol (e.g., "TSLA")
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            force_refresh: If True, re-fetch even if cached

        Returns:
            Number of dates fetched (not cached)
        """
        config = _get_alpaca_config()
        client = AlpacaRestClient(config)
        dates_fetched = 0

        # Find dates that need fetching
        current = start_date
        dates_to_fetch: list[date] = []

        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:  # Mon-Fri
                if force_refresh or not self._has_cached_date(symbol, current):
                    dates_to_fetch.append(current)
            current += timedelta(days=1)

        if not dates_to_fetch:
            print(f"All {symbol} bars already cached for {start_date} to {end_date}")
            return 0

        print(f"Fetching {len(dates_to_fetch)} days of {symbol} 1-minute bars...")

        # Fetch in batches to avoid rate limits
        # Alpaca allows fetching multiple days at once
        batch_size = 30  # ~30 days at a time

        for i in range(0, len(dates_to_fetch), batch_size):
            batch = dates_to_fetch[i:i + batch_size]
            batch_start = batch[0]
            batch_end = batch[-1]

            # Fetch 1-minute bars for the batch
            # Add 1 day to end to make it inclusive
            bars = await client.get_bars(
                symbol,
                batch_start.isoformat(),
                (batch_end + timedelta(days=1)).isoformat(),
                timeframe="1Min",
                limit=10000,  # ~390 bars per day * 30 days
            )

            if not bars:
                print(f"  Warning: No bars returned for {batch_start} to {batch_end}")
                continue

            # Group bars by date
            bars_by_date: dict[date, list[CachedBar]] = {}
            for bar in bars:
                # Parse timestamp
                ts = bar.timestamp
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

                bar_date = ts.date()
                if bar_date not in bars_by_date:
                    bars_by_date[bar_date] = []

                bars_by_date[bar_date].append(CachedBar(
                    timestamp=ts,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    vwap=bar.vwap,
                ))

            # Save each date to cache
            for bar_date, date_bars in bars_by_date.items():
                self._save_cached_date(symbol, bar_date, date_bars)
                dates_fetched += 1

            print(f"  Cached {len(bars_by_date)} days ({batch_start} to {batch_end})")

            # Small delay to be nice to the API
            if i + batch_size < len(dates_to_fetch):
                await asyncio.sleep(0.5)

        return dates_fetched

    def load_range_to_memory(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> dict[datetime, CachedBar]:
        """Load cached bars into memory for fast lookup.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping timestamp to bar
        """
        cache_key = f"{symbol}:{start_date}:{end_date}"

        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        bars_by_time: dict[datetime, CachedBar] = {}
        current = start_date

        while current <= end_date:
            if current.weekday() < 5:  # Mon-Fri
                day_bars = self._load_cached_date(symbol, current)
                for bar in day_bars:
                    bars_by_time[bar.timestamp] = bar
            current += timedelta(days=1)

        self._memory_cache[cache_key] = bars_by_time
        print(f"Loaded {len(bars_by_time)} {symbol} bars into memory")
        return bars_by_time

    def get_price_at_time(
        self,
        symbol: str,
        target_time: datetime,
        bars_cache: dict[datetime, CachedBar] | None = None,
    ) -> float | None:
        """Get price at a specific time.

        Uses the close price of the bar containing target_time.
        Falls back to nearest bar if exact match not found.

        Args:
            symbol: Stock symbol
            target_time: Time to look up
            bars_cache: Pre-loaded bars (from load_range_to_memory)

        Returns:
            Price or None if not found
        """
        if bars_cache is None:
            # Load just this day
            day_bars = self._load_cached_date(symbol, target_time.date())
            bars_cache = {bar.timestamp: bar for bar in day_bars}

        if not bars_cache:
            return None

        # Round target_time down to minute
        target_minute = target_time.replace(second=0, microsecond=0)

        # Try exact match first
        if target_minute in bars_cache:
            return bars_cache[target_minute].close

        # Find nearest bar within 5 minutes
        best_bar = None
        best_diff = timedelta(minutes=5)

        for ts, bar in bars_cache.items():
            diff = abs(ts - target_time)
            if diff < best_diff:
                best_diff = diff
                best_bar = bar

        return best_bar.close if best_bar else None


# Convenience function for one-off fetches
async def ensure_bars_cached(
    symbol: str,
    start_date: date,
    end_date: date,
) -> BarCache:
    """Ensure bars are cached for a date range, fetching if needed.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date

    Returns:
        BarCache instance with data loaded
    """
    cache = BarCache()
    await cache.fetch_and_cache_range(symbol, start_date, end_date)
    return cache


class SecondBarCache:
    """Cache for 1-second bars built from trade data.

    Alpaca doesn't provide sub-minute bars, so we fetch trades and aggregate
    them into 1-second OHLCV bars ourselves.

    Cache structure: cache/second_bars/{symbol}/{YYYY-MM-DD}.json
    """

    ALPACA_DATA_URL = "https://data.alpaca.markets"

    def __init__(self, cache_dir: str = "cache/second_bars"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[datetime, CachedBar]] = {}
        self._config = _get_alpaca_config()

    def _get_cache_path(self, symbol: str, target_date: date) -> Path:
        """Get cache file path for a symbol/date."""
        symbol_dir = self.cache_dir / symbol.upper()
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / f"{target_date.isoformat()}.json"

    def _has_cached_date(self, symbol: str, target_date: date) -> bool:
        """Check if we have cached data for a date."""
        return self._get_cache_path(symbol, target_date).exists()

    def _load_cached_date(self, symbol: str, target_date: date) -> list[CachedBar]:
        """Load cached 1-second bars for a date."""
        cache_path = self._get_cache_path(symbol, target_date)
        if not cache_path.exists():
            return []

        with open(cache_path) as f:
            data = json.load(f)

        bars = []
        for bar in data.get("bars", []):
            bars.append(CachedBar(
                timestamp=datetime.fromisoformat(bar["timestamp"]),
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                vwap=bar.get("vwap"),
            ))
        return bars

    def _save_cached_date(
        self, symbol: str, target_date: date, bars: list[CachedBar], trade_count: int
    ) -> None:
        """Save 1-second bars to cache for a date."""
        cache_path = self._get_cache_path(symbol, target_date)

        data = {
            "symbol": symbol,
            "date": target_date.isoformat(),
            "fetched_at": datetime.now().isoformat(),
            "bar_count": len(bars),
            "trade_count": trade_count,
            "timeframe": "1Sec",
            "bars": [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                }
                for bar in bars
            ]
        }

        with open(cache_path, "w") as f:
            json.dump(data, f)

    async def _fetch_trades_for_day(
        self, symbol: str, target_date: date
    ) -> list[dict[str, Any]]:
        """Fetch all trades for a single day from Alpaca.

        Uses pagination to get all trades (TSLA can have 7+ million trades/day).
        """
        trades = []
        page_token = None

        # Market hours in UTC: 14:30 - 21:00
        start_time = datetime.combine(target_date, datetime.min.time().replace(hour=14, minute=30))
        end_time = datetime.combine(target_date, datetime.min.time().replace(hour=21, minute=0))

        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        headers = {
            "APCA-API-KEY-ID": self._config.api_key,
            "APCA-API-SECRET-KEY": self._config.secret_key,
        }

        async with aiohttp.ClientSession() as session:
            page_count = 0
            while True:
                url = f"{self.ALPACA_DATA_URL}/v2/stocks/{symbol}/trades"
                params = {
                    "start": start_str,
                    "end": end_str,
                    "limit": 10000,
                }
                if page_token:
                    params["page_token"] = page_token

                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Alpaca API error: {resp.status} - {text}")

                    data = await resp.json()

                page_trades = data.get("trades", [])
                trades.extend(page_trades)
                page_count += 1

                if page_count % 50 == 0:
                    print(f"    Fetched {len(trades):,} trades ({page_count} pages)...")

                page_token = data.get("next_page_token")
                if not page_token:
                    break

                # Small delay to avoid rate limits
                await asyncio.sleep(0.02)

        return trades

    def _aggregate_trades_to_second_bars(
        self, trades: list[dict[str, Any]]
    ) -> list[CachedBar]:
        """Aggregate trades into 1-second OHLCV bars.

        Args:
            trades: List of Alpaca trade objects with 't' (timestamp), 'p' (price), 's' (size)

        Returns:
            List of CachedBar objects, one per second with trading activity
        """
        if not trades:
            return []

        # Group trades by second
        trades_by_second: dict[datetime, list[dict]] = defaultdict(list)

        for trade in trades:
            ts_str = trade["t"]
            # Parse timestamp and truncate to second
            if "." in ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            else:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

            second_ts = ts.replace(microsecond=0)
            trades_by_second[second_ts].append(trade)

        # Build bars from grouped trades
        bars = []
        for second_ts in sorted(trades_by_second.keys()):
            second_trades = trades_by_second[second_ts]

            prices = [t["p"] for t in second_trades]
            sizes = [t["s"] for t in second_trades]
            total_volume = sum(sizes)

            # Calculate VWAP for this second
            vwap = sum(p * s for p, s in zip(prices, sizes)) / total_volume if total_volume > 0 else prices[0]

            bars.append(CachedBar(
                timestamp=second_ts,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=total_volume,
                vwap=vwap,
            ))

        return bars

    async def fetch_and_cache_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        force_refresh: bool = False,
    ) -> int:
        """Fetch trades and cache as 1-second bars for a date range.

        Args:
            symbol: Stock symbol (e.g., "TSLA")
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            force_refresh: If True, re-fetch even if cached

        Returns:
            Number of dates fetched (not cached)
        """
        dates_fetched = 0

        # Find dates that need fetching
        current = start_date
        dates_to_fetch: list[date] = []

        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:  # Mon-Fri
                if force_refresh or not self._has_cached_date(symbol, current):
                    dates_to_fetch.append(current)
            current += timedelta(days=1)

        if not dates_to_fetch:
            print(f"All {symbol} 1-second bars already cached for {start_date} to {end_date}")
            return 0

        print(f"Fetching 1-second bars for {len(dates_to_fetch)} days of {symbol}...")

        # Fetch each day individually (trades are date-specific)
        for i, target_date in enumerate(dates_to_fetch):
            print(f"  [{i+1}/{len(dates_to_fetch)}] {target_date}...", end=" ", flush=True)

            try:
                trades = await self._fetch_trades_for_day(symbol, target_date)

                if not trades:
                    print("no trades (holiday?)")
                    continue

                bars = self._aggregate_trades_to_second_bars(trades)
                self._save_cached_date(symbol, target_date, bars, len(trades))
                dates_fetched += 1

                print(f"{len(trades):,} trades -> {len(bars):,} bars")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

            # Delay between days to be nice to API
            if i < len(dates_to_fetch) - 1:
                await asyncio.sleep(0.5)

        return dates_fetched

    def load_range_to_memory(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> dict[datetime, CachedBar]:
        """Load cached 1-second bars into memory for fast lookup.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping timestamp to bar
        """
        cache_key = f"1s:{symbol}:{start_date}:{end_date}"

        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        bars_by_time: dict[datetime, CachedBar] = {}
        current = start_date

        while current <= end_date:
            if current.weekday() < 5:  # Mon-Fri
                day_bars = self._load_cached_date(symbol, current)
                for bar in day_bars:
                    bars_by_time[bar.timestamp] = bar
            current += timedelta(days=1)

        self._memory_cache[cache_key] = bars_by_time
        print(f"Loaded {len(bars_by_time):,} {symbol} 1-second bars into memory")
        return bars_by_time
