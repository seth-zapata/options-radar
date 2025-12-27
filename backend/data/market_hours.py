"""Market hours checking using Alpaca Clock API.

Avoids unnecessary connections when market is closed.
Handles weekends, holidays, and early closes correctly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from backend.config import AlpacaConfig

logger = logging.getLogger(__name__)

# Timezone constants
ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")  # CST/CDT
UTC = timezone.utc


@dataclass
class MarketStatus:
    """Current market status from Alpaca."""
    is_open: bool
    timestamp: datetime
    next_open: datetime | None
    next_close: datetime | None

    def time_until_open(self) -> str:
        """Human-readable time until market opens."""
        if self.is_open:
            return "Market is open"
        if not self.next_open:
            return "Unknown"

        delta = self.next_open - self.timestamp
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)

        if hours > 24:
            days = hours // 24
            return f"{days}d {hours % 24}h until open"
        elif hours > 0:
            return f"{hours}h {minutes}m until open"
        else:
            return f"{minutes}m until open"

    def format_for_timezone(self, tz: ZoneInfo = CT) -> dict[str, str]:
        """Format times for display in specified timezone."""
        def fmt(dt: datetime | None) -> str:
            if not dt:
                return "N/A"
            return dt.astimezone(tz).strftime("%a %b %d, %I:%M %p %Z")

        return {
            "current_time": fmt(self.timestamp),
            "next_open": fmt(self.next_open),
            "next_close": fmt(self.next_close),
            "is_open": self.is_open,
            "time_until_open": self.time_until_open(),
        }


async def check_market_hours(config: AlpacaConfig) -> MarketStatus:
    """Check current market status using Alpaca Clock API.

    Args:
        config: Alpaca configuration with API credentials

    Returns:
        MarketStatus with current market state
    """
    url = "https://api.alpaca.markets/v2/clock"
    if config.paper:
        url = "https://paper-api.alpaca.markets/v2/clock"

    headers = {
        "APCA-API-KEY-ID": config.api_key,
        "APCA-API-SECRET-KEY": config.secret_key,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

    # Parse timestamps
    timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
    next_open = datetime.fromisoformat(data["next_open"].replace("Z", "+00:00")) if data.get("next_open") else None
    next_close = datetime.fromisoformat(data["next_close"].replace("Z", "+00:00")) if data.get("next_close") else None

    return MarketStatus(
        is_open=data["is_open"],
        timestamp=timestamp,
        next_open=next_open,
        next_close=next_close,
    )


async def wait_for_market_open(config: AlpacaConfig, check_interval: int = 60) -> None:
    """Wait until market opens, checking periodically.

    Args:
        config: Alpaca configuration
        check_interval: Seconds between checks (default 60)
    """
    import asyncio

    while True:
        status = await check_market_hours(config)

        if status.is_open:
            logger.info("Market is now open!")
            return

        info = status.format_for_timezone(CT)
        logger.info(f"Market closed. {info['time_until_open']}. Next open: {info['next_open']}")

        await asyncio.sleep(check_interval)
