"""Test script for fetching 1-second TSLA bars."""
import asyncio
from datetime import date
from backend.scalping.bar_cache import SecondBarCache

async def main():
    cache = SecondBarCache()

    # Test with a single day first
    print("Testing single day fetch...")
    await cache.fetch_and_cache_range('TSLA', date(2024, 1, 2), date(2024, 1, 2))

    # Verify the cached file
    import json
    from pathlib import Path

    cache_file = Path("cache/second_bars/TSLA/2024-01-02.json")
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        print(f"\nCached file info:")
        print(f"  Trades: {data.get('trade_count', 'N/A'):,}")
        print(f"  1-second bars: {data.get('bar_count', 'N/A'):,}")

        # Show first few bars
        bars = data.get('bars', [])
        if bars:
            print(f"\nFirst 5 bars:")
            for bar in bars[:5]:
                print(f"  {bar['timestamp']}: O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f} V={bar['volume']}")
    else:
        print(f"Cache file not found: {cache_file}")

if __name__ == "__main__":
    asyncio.run(main())
