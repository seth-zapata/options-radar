"""Fetch 1-second TSLA bars for random test weeks."""
import asyncio
import sys
from datetime import date
from backend.scalping.bar_cache import SecondBarCache

async def main():
    cache = SecondBarCache()

    if len(sys.argv) > 1 and sys.argv[1] == "2022":
        # 2022 random weeks (not hand-picked for volatility)
        weeks = [
            ("April 4-8", date(2022, 4, 4), date(2022, 4, 8)),
            ("July 11-15", date(2022, 7, 11), date(2022, 7, 15)),
            ("October 17-21", date(2022, 10, 17), date(2022, 10, 21)),
        ]
        year = "2022"
    else:
        # 2024 random weeks (as specified)
        weeks = [
            ("March 11-15", date(2024, 3, 11), date(2024, 3, 15)),
            ("June 17-21", date(2024, 6, 17), date(2024, 6, 21)),
            ("September 9-13", date(2024, 9, 9), date(2024, 9, 13)),
        ]
        year = "2024"

    print(f"Fetching {year} random weeks...")
    for name, start, end in weeks:
        print(f"\n{'='*60}")
        print(f"{year} {name}")
        print(f"{'='*60}")
        await cache.fetch_and_cache_range("TSLA", start, end)

    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
