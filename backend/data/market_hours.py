"""Market hours checking using Alpaca Clock API.

Avoids unnecessary connections when market is closed.
Handles weekends, holidays, and early closes correctly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from backend.config import AlpacaConfig

logger = logging.getLogger(__name__)

# Timezone constants
ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")  # CST/CDT
UTC = timezone.utc


def format_duration(total_seconds: float) -> str:
    """Format seconds into d h m s string."""
    if total_seconds < 0:
        return "0s"

    days = int(total_seconds // 86400)
    hours = int((total_seconds % 86400) // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")

    return " ".join(parts)


@dataclass
class MarketStatus:
    """Current market status from Alpaca."""
    is_open: bool
    timestamp: datetime
    next_open: datetime | None
    next_close: datetime | None

    def seconds_until_open(self) -> float | None:
        """Seconds until market opens, or None if open."""
        if self.is_open or not self.next_open:
            return None
        return (self.next_open - self.timestamp).total_seconds()

    def seconds_until_close(self) -> float | None:
        """Seconds until market closes, or None if closed."""
        if not self.is_open or not self.next_close:
            return None
        return (self.next_close - self.timestamp).total_seconds()

    def time_until_open(self) -> str:
        """Human-readable time until market opens."""
        if self.is_open:
            return "Market is open"
        seconds = self.seconds_until_open()
        if seconds is None:
            return "Unknown"
        return format_duration(seconds)

    def time_until_close(self) -> str:
        """Human-readable time until market closes."""
        if not self.is_open:
            return "Market is closed"
        seconds = self.seconds_until_close()
        if seconds is None:
            return "Unknown"
        return format_duration(seconds)

    def format_for_timezone(self, tz: ZoneInfo = CT) -> dict[str, Any]:
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
            "time_until_close": self.time_until_close(),
            "seconds_until_open": self.seconds_until_open(),
            "seconds_until_close": self.seconds_until_close(),
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
