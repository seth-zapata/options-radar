"""Comprehensive options data cache using SQLite.

Caches full options chain data from EODHD including:
- Bid/Ask/Last prices (GROUND TRUTH)
- Open/High/Low
- Volume and Open Interest
- Greeks (Delta, Gamma, Theta, Vega)
- Implied Volatility

This data is worth thousands from other providers.
EODHD at $40/mo is drastically underpriced.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from backend.data.eodhd_client import EODHDClient, OptionContract, OptionsChain

logger = logging.getLogger(__name__)

# Cache database path
OPTIONS_CACHE_DB = Path(__file__).parent.parent.parent / "cache" / "options_data.db"


@dataclass
class CachedContract:
    """Option contract with all price data."""

    symbol: str  # Underlying symbol
    contract_id: str  # Full contract ID (e.g., TSLA261218C00075000)
    trade_date: str  # YYYY-MM-DD
    expiry: str  # YYYY-MM-DD
    option_type: str  # 'call' or 'put'
    strike: float

    # Prices (GROUND TRUTH)
    bid: float
    ask: float
    last: float
    open: float
    high: float
    low: float

    # Volume and OI
    volume: int
    open_interest: int

    # Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float

    @property
    def mid(self) -> float:
        """Mid price (average of bid/ask)."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last or 0

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as percentage of mid."""
        if self.mid > 0:
            return (self.ask - self.bid) / self.mid * 100
        return 0

    def is_atm(self, underlying_price: float, tolerance: float = 0.05) -> bool:
        """Check if strike is at-the-money (within tolerance % of underlying)."""
        return abs(self.strike - underlying_price) / underlying_price < tolerance


class OptionsCache:
    """SQLite cache for comprehensive options data.

    Schema stores ALL contract data from EODHD for:
    - True P&L backtesting
    - Historical analysis
    - IV rank calculations

    Usage:
        cache = OptionsCache()
        await cache.cache_chain("TSLA", "2024-06-14", eodhd_client)
        contracts = cache.get_atm_contracts("TSLA", "2024-06-14", 180.0)
    """

    def __init__(self, db_path: Path = OPTIONS_CACHE_DB):
        self.db_path = db_path
        self._init_db()

        # Statistics
        self.cache_hits = 0
        self.api_fetches = 0

    def _init_db(self) -> None:
        """Initialize SQLite database with schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main contracts table
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
                open REAL,
                high REAL,
                low REAL,
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

        logger.info(f"Options cache initialized at {self.db_path}")

    def is_date_cached(self, symbol: str, trade_date: str) -> bool:
        """Check if we have cached data for a symbol/date."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT 1 FROM cache_metadata WHERE symbol = ? AND trade_date = ?",
            (symbol, trade_date)
        )
        result = cursor.fetchone() is not None
        conn.close()

        return result

    async def cache_chain(
        self,
        symbol: str,
        trade_date: str,
        client: EODHDClient,
        force: bool = False,
    ) -> int:
        """Fetch and cache options chain for a symbol/date.

        Args:
            symbol: Underlying symbol
            trade_date: Date to fetch (YYYY-MM-DD)
            client: EODHD client
            force: Force refetch even if cached

        Returns:
            Number of contracts cached
        """
        # Check if already cached
        if not force and self.is_date_cached(symbol, trade_date):
            self.cache_hits += 1
            logger.debug(f"[CACHE HIT] {symbol} {trade_date}")
            return 0

        # Fetch from EODHD
        self.api_fetches += 1
        logger.info(f"[API FETCH] {symbol} {trade_date}")

        chain = await client.fetch_options_chain(symbol, trade_date, use_cache=False)
        if chain is None or not chain.contracts:
            logger.warning(f"No options data for {symbol} on {trade_date}")
            return 0

        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        for c in chain.contracts:
            cursor.execute("""
                INSERT OR REPLACE INTO options_contracts (
                    symbol, contract_id, trade_date, expiry, option_type, strike,
                    bid, ask, last, open, high, low,
                    volume, open_interest,
                    delta, gamma, theta, vega, iv,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, c.symbol, trade_date, c.expiry, c.option_type, c.strike,
                c.bid, c.ask, c.last_price, 0, 0, 0,  # open/high/low not in current schema
                c.volume, c.open_interest,
                c.delta, c.gamma, c.theta, c.vega, c.iv,
                now
            ))

        # Update metadata
        cursor.execute("""
            INSERT OR REPLACE INTO cache_metadata (symbol, trade_date, num_contracts, cached_at)
            VALUES (?, ?, ?, ?)
        """, (symbol, trade_date, len(chain.contracts), now))

        conn.commit()
        conn.close()

        logger.info(f"[CACHED] {symbol} {trade_date}: {len(chain.contracts)} contracts")
        return len(chain.contracts)

    def get_contracts(
        self,
        symbol: str,
        trade_date: str,
        option_type: str | None = None,
        min_strike: float | None = None,
        max_strike: float | None = None,
        expiry: str | None = None,
    ) -> list[CachedContract]:
        """Get cached contracts with optional filters.

        Args:
            symbol: Underlying symbol
            trade_date: Trade date
            option_type: 'call' or 'put' (optional)
            min_strike: Minimum strike price (optional)
            max_strike: Maximum strike price (optional)
            expiry: Specific expiry date (optional)

        Returns:
            List of CachedContract objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT symbol, contract_id, trade_date, expiry, option_type, strike,
                   bid, ask, last, open, high, low,
                   volume, open_interest,
                   delta, gamma, theta, vega, iv
            FROM options_contracts
            WHERE symbol = ? AND trade_date = ?
        """
        params: list[Any] = [symbol, trade_date]

        if option_type:
            query += " AND option_type = ?"
            params.append(option_type)

        if min_strike is not None:
            query += " AND strike >= ?"
            params.append(min_strike)

        if max_strike is not None:
            query += " AND strike <= ?"
            params.append(max_strike)

        if expiry:
            query += " AND expiry = ?"
            params.append(expiry)

        query += " ORDER BY strike, expiry"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        contracts = []
        for row in rows:
            contracts.append(CachedContract(
                symbol=row[0],
                contract_id=row[1],
                trade_date=row[2],
                expiry=row[3],
                option_type=row[4],
                strike=row[5],
                bid=row[6] or 0,
                ask=row[7] or 0,
                last=row[8] or 0,
                open=row[9] or 0,
                high=row[10] or 0,
                low=row[11] or 0,
                volume=row[12] or 0,
                open_interest=row[13] or 0,
                delta=row[14] or 0,
                gamma=row[15] or 0,
                theta=row[16] or 0,
                vega=row[17] or 0,
                iv=row[18] or 0,
            ))

        return contracts

    def get_atm_contract(
        self,
        symbol: str,
        trade_date: str,
        underlying_price: float,
        option_type: str = "call",
        target_dte: int = 30,
    ) -> CachedContract | None:
        """Get the ATM contract closest to target DTE.

        Args:
            symbol: Underlying symbol
            trade_date: Trade date
            underlying_price: Current underlying price
            option_type: 'call' or 'put'
            target_dte: Target days to expiration (default 30)

        Returns:
            Best ATM contract or None
        """
        contracts = self.get_contracts(
            symbol, trade_date,
            option_type=option_type,
            min_strike=underlying_price * 0.95,
            max_strike=underlying_price * 1.05,
        )

        if not contracts:
            return None

        # Parse trade date
        try:
            td = datetime.strptime(trade_date, "%Y-%m-%d")
        except ValueError:
            return None

        # Find best contract: closest to ATM strike + closest to target DTE
        best = None
        best_score = float('inf')

        for c in contracts:
            try:
                expiry_date = datetime.strptime(c.expiry, "%Y-%m-%d")
                dte = (expiry_date - td).days

                if dte <= 0:
                    continue

                # Score: distance from ATM + distance from target DTE
                strike_dist = abs(c.strike - underlying_price) / underlying_price
                dte_dist = abs(dte - target_dte) / target_dte
                score = strike_dist + dte_dist * 0.5  # Weight DTE less

                if score < best_score and c.bid > 0:  # Must have valid bid
                    best_score = score
                    best = c

            except ValueError:
                continue

        return best

    def get_contract_on_date(
        self,
        contract_id: str,
        trade_date: str,
    ) -> CachedContract | None:
        """Get a specific contract on a specific date.

        Used for tracking positions over time.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT symbol, contract_id, trade_date, expiry, option_type, strike,
                   bid, ask, last, open, high, low,
                   volume, open_interest,
                   delta, gamma, theta, vega, iv
            FROM options_contracts
            WHERE contract_id = ? AND trade_date = ?
        """, (contract_id, trade_date))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return CachedContract(
            symbol=row[0],
            contract_id=row[1],
            trade_date=row[2],
            expiry=row[3],
            option_type=row[4],
            strike=row[5],
            bid=row[6] or 0,
            ask=row[7] or 0,
            last=row[8] or 0,
            open=row[9] or 0,
            high=row[10] or 0,
            low=row[11] or 0,
            volume=row[12] or 0,
            open_interest=row[13] or 0,
            delta=row[14] or 0,
            gamma=row[15] or 0,
            theta=row[16] or 0,
            vega=row[17] or 0,
            iv=row[18] or 0,
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM options_contracts")
        total_contracts = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cache_metadata")
        total_dates = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT symbol FROM cache_metadata")
        symbols = [row[0] for row in cursor.fetchall()]

        cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM cache_metadata")
        min_date, max_date = cursor.fetchone()

        # Size on disk
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        conn.close()

        return {
            "total_contracts": total_contracts,
            "total_dates_cached": total_dates,
            "symbols": symbols,
            "date_range": (min_date, max_date),
            "db_size_mb": db_size / 1024 / 1024,
            "cache_hits": self.cache_hits,
            "api_fetches": self.api_fetches,
        }
