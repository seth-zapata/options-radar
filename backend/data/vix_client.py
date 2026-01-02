"""VIX (CBOE Volatility Index) client for market regime detection.

Uses yfinance to fetch VIX data. Provides current VIX level
and regime classification for position sizing adjustments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Cache file for historical VIX data (for backtesting)
CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
VIX_CACHE_FILE = CACHE_DIR / "vix_historical.csv"


class VIXRegime(str, Enum):
    """VIX regime classification."""

    LOW = "low"  # VIX < 15
    NORMAL = "normal"  # VIX 15-25
    ELEVATED = "elevated"  # VIX 25-35
    PANIC = "panic"  # VIX > 35


@dataclass
class VIXStatus:
    """Current VIX status and regime."""

    vix_level: float
    regime: VIXRegime
    position_size_modifier: float
    message: str


class VIXClient:
    """Client for fetching VIX data and determining market regime.

    VIX Regime Definitions:
    - LOW (< 15): Calm markets, normal trading
    - NORMAL (15-25): Typical volatility, normal trading
    - ELEVATED (25-35): High volatility, reduce position size to 50%
    - PANIC (> 35): Extreme volatility, block all entries
    """

    def __init__(
        self,
        low_threshold: float = 15.0,
        elevated_threshold: float = 25.0,
        panic_threshold: float = 35.0,
        cache_ttl_seconds: int = 60,
    ):
        """Initialize VIX client.

        Args:
            low_threshold: VIX below this is LOW regime
            elevated_threshold: VIX above this is ELEVATED regime
            panic_threshold: VIX above this is PANIC regime
            cache_ttl_seconds: How long to cache current VIX (default 60s)
        """
        self.low_threshold = low_threshold
        self.elevated_threshold = elevated_threshold
        self.panic_threshold = panic_threshold
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)

        # Cache for current VIX
        self._current_vix: Optional[float] = None
        self._cache_time: Optional[datetime] = None

        # Historical data cache (for backtesting)
        self._historical_data: Optional[pd.DataFrame] = None

    def get_current_vix(self) -> float:
        """Fetch current VIX level with caching.

        Returns:
            Current VIX level (defaults to 20.0 on error)
        """
        # Check cache
        now = datetime.now()
        if (
            self._current_vix is not None
            and self._cache_time is not None
            and (now - self._cache_time) < self.cache_ttl
        ):
            return self._current_vix

        try:
            import yfinance as yf

            vix = yf.Ticker("^VIX")
            # Try to get current price
            current = vix.info.get("regularMarketPrice")
            if current is None:
                current = vix.info.get("previousClose")
            if current is None:
                # Fallback: get latest from history
                hist = vix.history(period="1d")
                if not hist.empty:
                    current = hist["Close"].iloc[-1]

            if current is not None:
                self._current_vix = float(current)
                self._cache_time = now
                logger.debug(f"VIX fetched: {self._current_vix:.2f}")
                return self._current_vix

        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}, using default 20.0")

        # Default to normal VIX
        return 20.0

    def get_regime(self, vix_level: Optional[float] = None) -> VIXStatus:
        """Get current VIX regime and position size modifier.

        Args:
            vix_level: Optional VIX level (fetches current if not provided)

        Returns:
            VIXStatus with regime and modifiers
        """
        if vix_level is None:
            vix_level = self.get_current_vix()

        if vix_level > self.panic_threshold:
            return VIXStatus(
                vix_level=vix_level,
                regime=VIXRegime.PANIC,
                position_size_modifier=0.0,
                message=f"VIX at {vix_level:.1f} - PANIC mode, blocking all entries",
            )
        elif vix_level > self.elevated_threshold:
            return VIXStatus(
                vix_level=vix_level,
                regime=VIXRegime.ELEVATED,
                position_size_modifier=0.5,
                message=f"VIX at {vix_level:.1f} - ELEVATED mode, reducing position size to 50%",
            )
        elif vix_level > self.low_threshold:
            return VIXStatus(
                vix_level=vix_level,
                regime=VIXRegime.NORMAL,
                position_size_modifier=1.0,
                message=f"VIX at {vix_level:.1f} - NORMAL mode, full position size",
            )
        else:
            return VIXStatus(
                vix_level=vix_level,
                regime=VIXRegime.LOW,
                position_size_modifier=1.0,
                message=f"VIX at {vix_level:.1f} - LOW volatility mode, full position size",
            )

    def load_historical_data(self, start_date: str = "2024-01-01") -> None:
        """Load or fetch historical VIX data for backtesting.

        Args:
            start_date: Start date for historical data (YYYY-MM-DD)
        """
        # Try to load from cache first (simplified format)
        if VIX_CACHE_FILE.exists():
            try:
                self._historical_data = pd.read_csv(
                    VIX_CACHE_FILE,
                    index_col=0,  # First column is the date index
                    parse_dates=True,
                )
                # Check if this is our simplified format (has 'Close' column directly)
                if 'Close' in self._historical_data.columns:
                    logger.info(f"Loaded {len(self._historical_data)} days of VIX data from cache")
                    return
                # Otherwise it might be the old multi-level format, re-fetch
                logger.debug("Cache format outdated, re-fetching")
            except Exception as e:
                logger.warning(f"Failed to load VIX cache: {e}")

        # Fetch from yfinance
        try:
            import yfinance as yf

            logger.info(f"Fetching VIX historical data from {start_date}")
            vix = yf.download("^VIX", start=start_date, progress=False)

            if vix.empty:
                logger.warning("No VIX historical data fetched")
                return

            # Flatten multi-level columns if present
            if isinstance(vix.columns, pd.MultiIndex):
                # Extract just the price type (Close, Open, etc.) from multi-level
                vix.columns = [col[0] for col in vix.columns]

            # Save to cache in simplified format
            CACHE_DIR.mkdir(exist_ok=True)
            vix.to_csv(VIX_CACHE_FILE)
            self._historical_data = vix
            logger.info(f"Fetched and cached {len(vix)} days of VIX data")

        except Exception as e:
            logger.error(f"Failed to fetch VIX historical data: {e}")

    def get_vix_for_date(self, check_date: date) -> Optional[float]:
        """Get VIX level for a specific date (for backtesting).

        Args:
            check_date: The date to look up

        Returns:
            VIX close price on that date, or None if not available
        """
        if self._historical_data is None:
            self.load_historical_data()

        if self._historical_data is None or self._historical_data.empty:
            return None

        try:
            # Convert date to datetime for lookup
            dt = datetime.combine(check_date, datetime.min.time())

            # Handle multi-level column index from yfinance
            if isinstance(self._historical_data.columns, pd.MultiIndex):
                close_col = ("Close", "^VIX")
            else:
                close_col = "Close"

            # Try exact date match first
            if dt in self._historical_data.index:
                return float(self._historical_data.loc[dt, close_col])

            # Find closest previous trading day
            mask = self._historical_data.index <= dt
            if mask.any():
                closest = self._historical_data.index[mask][-1]
                return float(self._historical_data.loc[closest, close_col])

        except Exception as e:
            logger.debug(f"VIX lookup failed for {check_date}: {e}")

        return None

    def get_regime_for_date(self, check_date: date) -> VIXStatus:
        """Get VIX regime for a specific date (for backtesting).

        Args:
            check_date: The date to check

        Returns:
            VIXStatus with regime and modifiers
        """
        vix_level = self.get_vix_for_date(check_date)

        if vix_level is None:
            # Default to normal if no data
            return VIXStatus(
                vix_level=20.0,
                regime=VIXRegime.NORMAL,
                position_size_modifier=1.0,
                message="VIX data unavailable, assuming NORMAL regime",
            )

        return self.get_regime(vix_level)
