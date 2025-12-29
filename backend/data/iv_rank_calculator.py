"""IV Rank calculator using EODHD historical options data.

Calculates IV Rank for backtesting without requiring ORATS subscription.

IV Rank = (Current IV - 52wk Low) / (52wk High - 52wk Low) × 100

Usage:
    calculator = IVRankCalculator(eodhd_client)
    iv_rank = await calculator.get_iv_rank("TSLA", "2024-06-15", underlying_price=180.0)
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from backend.data.eodhd_client import EODHDClient, OptionsChain

logger = logging.getLogger(__name__)

# Cache database path
CACHE_DB_PATH = Path(__file__).parent.parent.parent / "cache" / "iv_data.db"


@dataclass
class IVDataPoint:
    """IV data for a single date."""
    symbol: str
    date: str
    atm_iv: float  # ATM implied volatility
    underlying_price: float


@dataclass
class IVRankResult:
    """IV Rank calculation result."""
    symbol: str
    date: str
    current_iv: float
    iv_high_52w: float
    iv_low_52w: float
    iv_rank: float  # 0-100
    data_points: int  # Number of historical data points used


class IVRankCalculator:
    """Calculates IV Rank from EODHD historical options data.

    Uses SQLite caching to avoid repeated API calls.
    """

    def __init__(self, eodhd_client: EODHDClient):
        self.eodhd_client = eodhd_client
        self._init_cache_db()
        # Cache statistics
        self.cache_hits = 0
        self.api_fetches = 0

    def _init_cache_db(self) -> None:
        """Initialize SQLite cache database."""
        CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS iv_cache (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                atm_iv REAL NOT NULL,
                underlying_price REAL NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (symbol, date)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_iv_cache_symbol_date
            ON iv_cache (symbol, date)
        """)

        conn.commit()
        conn.close()

        logger.info(f"IV cache database initialized at {CACHE_DB_PATH}")

    def _get_cached_iv(self, symbol: str, date: str) -> float | None:
        """Get cached IV for a date, or None if not cached."""
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT atm_iv FROM iv_cache WHERE symbol = ? AND date = ?",
            (symbol, date)
        )
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def _cache_iv(self, symbol: str, date: str, atm_iv: float, underlying_price: float) -> None:
        """Cache IV data for a date."""
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT OR REPLACE INTO iv_cache
               (symbol, date, atm_iv, underlying_price, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (symbol, date, atm_iv, underlying_price, datetime.now().isoformat())
        )

        conn.commit()
        conn.close()

    def _get_cached_iv_range(self, symbol: str, start_date: str, end_date: str) -> list[tuple[str, float]]:
        """Get all cached IV data for a date range."""
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """SELECT date, atm_iv FROM iv_cache
               WHERE symbol = ? AND date >= ? AND date <= ?
               ORDER BY date""",
            (symbol, start_date, end_date)
        )
        rows = cursor.fetchall()
        conn.close()

        return rows

    def _extract_atm_iv(
        self,
        chain: OptionsChain,
        underlying_price: float,
        target_dte: int = 30,
    ) -> float | None:
        """Extract ATM implied volatility from options chain.

        Strategy:
        1. Find expiry closest to target_dte (default 30 days)
        2. Find strike closest to underlying price
        3. Average call and put IV at that strike

        Args:
            chain: Options chain from EODHD
            underlying_price: Current underlying stock price
            target_dte: Target days to expiration (default 30)

        Returns:
            ATM IV or None if insufficient data
        """
        if not chain.contracts:
            return None

        # Parse chain date
        try:
            chain_date = datetime.strptime(chain.date, "%Y-%m-%d")
        except ValueError:
            return None

        # Group contracts by expiry
        expiry_contracts: dict[str, list] = {}
        for contract in chain.contracts:
            if contract.expiry not in expiry_contracts:
                expiry_contracts[contract.expiry] = []
            expiry_contracts[contract.expiry].append(contract)

        if not expiry_contracts:
            return None

        # Find expiry closest to target DTE
        target_expiry = None
        min_dte_diff = float('inf')

        for expiry in expiry_contracts.keys():
            try:
                expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
                dte = (expiry_date - chain_date).days
                if dte > 0:  # Only future expiries
                    dte_diff = abs(dte - target_dte)
                    if dte_diff < min_dte_diff:
                        min_dte_diff = dte_diff
                        target_expiry = expiry
            except ValueError:
                continue

        if not target_expiry:
            return None

        # Get contracts for target expiry
        contracts = expiry_contracts[target_expiry]

        # Find strike closest to underlying price
        strikes = set(c.strike for c in contracts)
        if not strikes:
            return None

        atm_strike = min(strikes, key=lambda s: abs(s - underlying_price))

        # Get call and put IV at ATM strike
        call_iv = None
        put_iv = None

        for contract in contracts:
            if contract.strike == atm_strike:
                if contract.is_call and contract.iv > 0:
                    call_iv = contract.iv
                elif contract.is_put and contract.iv > 0:
                    put_iv = contract.iv

        # Average call and put IV (or use whichever is available)
        if call_iv and put_iv:
            return (call_iv + put_iv) / 2
        elif call_iv:
            return call_iv
        elif put_iv:
            return put_iv

        return None

    async def get_iv_for_date(
        self,
        symbol: str,
        date: str,
        underlying_price: float,
        use_cache: bool = True,
    ) -> float | None:
        """Get ATM IV for a specific date.

        Args:
            symbol: Stock symbol
            date: Date (YYYY-MM-DD)
            underlying_price: Stock price on that date
            use_cache: Whether to use cached data

        Returns:
            ATM IV or None if unavailable
        """
        # Check cache first
        if use_cache:
            cached = self._get_cached_iv(symbol, date)
            if cached is not None:
                self.cache_hits += 1
                logger.info(f"[CACHE HIT] {symbol} {date} IV={cached:.4f}")
                return cached

        # Fetch from EODHD
        self.api_fetches += 1
        logger.info(f"[API FETCH] {symbol} {date}")
        chain = await self.eodhd_client.fetch_options_chain(symbol, date, use_cache=False)

        if chain is None:
            return None

        atm_iv = self._extract_atm_iv(chain, underlying_price)

        if atm_iv is not None:
            self._cache_iv(symbol, date, atm_iv, underlying_price)
            logger.info(f"[CACHED] {symbol} {date} IV={atm_iv:.4f}")

        return atm_iv

    async def get_iv_rank(
        self,
        symbol: str,
        date: str,
        underlying_price: float,
        lookback_days: int = 60,
        min_data_points: int = 5,
    ) -> IVRankResult | None:
        """Calculate IV Rank for a symbol on a specific date.

        IV Rank = (Current IV - 52wk Low) / (52wk High - 52wk Low) × 100

        Args:
            symbol: Stock symbol
            date: Date to calculate IV Rank for (YYYY-MM-DD)
            underlying_price: Stock price on that date
            lookback_days: Number of trading days to look back (default 252 = 1 year)
            min_data_points: Minimum data points required for valid calculation

        Returns:
            IVRankResult or None if insufficient data
        """
        # Get current IV
        current_iv = await self.get_iv_for_date(symbol, date, underlying_price)

        if current_iv is None:
            logger.warning(f"No IV data for {symbol} on {date}")
            return None

        # Calculate date range for lookback
        try:
            end_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return None

        start_date = end_date - timedelta(days=int(lookback_days * 1.5))  # Extra days for weekends/holidays

        # Check cache for historical data
        cached_data = self._get_cached_iv_range(
            symbol,
            start_date.strftime("%Y-%m-%d"),
            date
        )

        if len(cached_data) >= min_data_points:
            # Use cached data
            ivs = [iv for _, iv in cached_data if iv > 0]

            if len(ivs) >= min_data_points:
                iv_high = max(ivs)
                iv_low = min(ivs)

                if iv_high > iv_low:
                    iv_rank = ((current_iv - iv_low) / (iv_high - iv_low)) * 100
                    iv_rank = max(0, min(100, iv_rank))  # Clamp to 0-100

                    return IVRankResult(
                        symbol=symbol,
                        date=date,
                        current_iv=current_iv,
                        iv_high_52w=iv_high,
                        iv_low_52w=iv_low,
                        iv_rank=iv_rank,
                        data_points=len(ivs),
                    )

        # Not enough cached data - need to fetch more
        # This is expensive, so we sample dates (every 7 calendar days to skip weekends)
        logger.info(f"Fetching historical IV data for {symbol} ({lookback_days} day lookback)...")

        ivs = [current_iv]
        dates_to_fetch = []

        # Generate sample dates (every 7 calendar days to skip weekends)
        current = end_date - timedelta(days=7)
        while current >= start_date and len(dates_to_fetch) < 15:  # Max 15 samples (reduce API calls)
            date_str = current.strftime("%Y-%m-%d")
            if self._get_cached_iv(symbol, date_str) is None:
                dates_to_fetch.append(date_str)
            current -= timedelta(days=7)

        # Fetch missing data (with rate limiting) - limit to 10 API calls for speed
        for fetch_date in dates_to_fetch[:10]:
            try:
                # Use a rough estimate of underlying price (not perfect but good enough for IV calculation)
                iv = await self.get_iv_for_date(symbol, fetch_date, underlying_price * 0.95)  # Rough estimate
                if iv and iv > 0:
                    ivs.append(iv)
                await asyncio.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.debug(f"Failed to fetch IV for {symbol} on {fetch_date}: {e}")
                continue

        # Recalculate with fetched data
        if len(ivs) < min_data_points:
            logger.warning(f"Insufficient IV data for {symbol}: {len(ivs)} points")
            return None

        iv_high = max(ivs)
        iv_low = min(ivs)

        if iv_high <= iv_low:
            return None

        iv_rank = ((current_iv - iv_low) / (iv_high - iv_low)) * 100
        iv_rank = max(0, min(100, iv_rank))

        return IVRankResult(
            symbol=symbol,
            date=date,
            current_iv=current_iv,
            iv_high_52w=iv_high,
            iv_low_52w=iv_low,
            iv_rank=iv_rank,
            data_points=len(ivs),
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM iv_cache")
        total_rows = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT symbol FROM iv_cache")
        symbols = [row[0] for row in cursor.fetchall()]

        conn.close()

        return {
            "total_cached_points": total_rows,
            "symbols": symbols,
            "cache_path": str(CACHE_DB_PATH),
        }
