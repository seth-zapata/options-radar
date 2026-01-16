"""Fetch 1-second TSLA bars for volatile test periods."""
import asyncio
import sys
from datetime import date
from backend.scalping.bar_cache import SecondBarCache

async def main():
    cache = SecondBarCache()

    if len(sys.argv) > 1 and sys.argv[1] == "2022":
        # 2022 volatile period (Feb 23 - Mar 8)
        print("=" * 60)
        print("Fetching 2022 volatile period (Feb 23 - Mar 8)")
        print("=" * 60)
        await cache.fetch_and_cache_range("TSLA", date(2022, 2, 23), date(2022, 3, 8))
    else:
        # 2024 volatile period (Dec 13-27)
        print("=" * 60)
        print("Fetching 2024 volatile period (Dec 13-27)")
        print("=" * 60)
        await cache.fetch_and_cache_range("TSLA", date(2024, 12, 13), date(2024, 12, 27))

    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
