"""Paper position tracking for confirmed trades.

Tracks positions the user confirms they took, with actual fill prices.
Generates exit signals based on P/L, time decay, and delta changes.

Positions are persisted to SQLite so they survive server restarts.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "cache" / "positions.db"


@dataclass
class TrackedPosition:
    """A confirmed paper position.

    Attributes:
        id: Unique position ID
        recommendation_id: Original recommendation ID
        opened_at: When position was confirmed
        underlying: Ticker symbol
        expiry: Option expiration date
        strike: Strike price
        right: C or P
        action: Original action (BUY_CALL, etc.)
        contracts: Number of contracts
        fill_price: Actual fill price per contract
        entry_cost: Total entry cost (contracts * fill_price * 100)
        current_price: Current market price (updated)
        current_value: Current value (contracts * current_price * 100)
        pnl: Current P/L in dollars
        pnl_percent: Current P/L as percentage
        dte: Days to expiration
        delta: Current delta (updated)
        status: open, closed, or exit_signal
        exit_reason: Reason for exit signal if any
        closed_at: When position was closed
        close_price: Price at close
    """
    id: str
    recommendation_id: str
    opened_at: str
    underlying: str
    expiry: str
    strike: float
    right: str
    action: str
    contracts: int
    fill_price: float
    entry_cost: float
    current_price: float | None = None
    current_value: float | None = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    dte: int | None = None
    delta: float | None = None
    status: Literal["open", "closed", "exit_signal"] = "open"
    exit_reason: str | None = None
    closed_at: str | None = None
    close_price: float | None = None



@dataclass
class ExitSignal:
    """An exit signal for a tracked position."""
    position_id: str
    reason: str
    current_price: float
    pnl: float
    pnl_percent: float
    urgency: Literal["low", "medium", "high"]
    trigger: Literal["take_profit", "stop_loss", "time_exit"]


@dataclass
class SymbolStats:
    """Per-symbol performance statistics for paper trading evaluation.

    Used to track each ticker's performance and decide whether to
    keep or drop symbols based on live results (e.g., drop PLTR if
    avg P&L < +5% after 4 weeks while TSLA/NVDA are > +15%).
    """
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_dollars: float = 0.0
    total_pnl_percent: float = 0.0  # Sum of all trade P&L %

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0

    @property
    def avg_pnl_percent(self) -> float:
        """Average P&L per trade as percentage."""
        return (self.total_pnl_percent / self.trades) if self.trades > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 1),
            "total_pnl_dollars": round(self.total_pnl_dollars, 2),
            "total_pnl_percent": round(self.total_pnl_percent, 1),
            "avg_pnl_percent": round(self.avg_pnl_percent, 1),
        }


@dataclass
class PositionTrackerConfig:
    """Configuration for position tracking.

    Fixed Exit Rules (validated from TSLA backtesting 2024-01 to 2025-01):
    - Take profit: Exit at +40% gain
    - Stop loss: Exit at -20% loss
    - Time exit: Exit when DTE < 1 (expiring)

    These fixed thresholds were found optimal for the regime-filtered
    intraday strategy with 7-day windows and 1.5% pullback/bounce entries.

    Attributes:
        take_profit_percent: Take profit threshold (default +40%)
        stop_loss_percent: Stop loss threshold (default -20%)
        min_dte_exit: Exit when DTE falls to this (default 1)
        max_positions: Maximum open positions (default 3)
        position_size_pct: Percent of portfolio per trade (default 10%)
    """
    take_profit_percent: float = 40.0
    stop_loss_percent: float = -20.0
    min_dte_exit: int = 1
    max_positions: int = 3
    position_size_pct: float = 10.0


class PositionTracker:
    """Tracks confirmed paper positions and generates exit signals.

    Positions are persisted to SQLite so they survive server restarts.

    Usage:
        tracker = PositionTracker()

        # User confirms a trade
        position = tracker.open_position(
            recommendation_id="abc123",
            underlying="NVDA",
            expiry="2025-01-10",
            strike=140.0,
            right="C",
            action="BUY_CALL",
            contracts=1,
            fill_price=4.25,  # Actual fill, not displayed price
        )

        # Update positions with current prices
        exit_signals = tracker.update_positions(current_prices)

        # User closes a position
        tracker.close_position(position_id, close_price=5.50)
    """

    def __init__(self, config: PositionTrackerConfig | None = None, db_path: Path | None = None):
        self.config = config or PositionTrackerConfig()
        self._positions: dict[str, TrackedPosition] = {}
        self._symbol_stats: dict[str, SymbolStats] = {}
        self._db_path = db_path or DEFAULT_DB_PATH

        # Initialize database and load existing positions
        self._init_db()
        self._load_from_db()

        logger.info(f"Position tracker initialized (db: {self._db_path}, loaded {len(self._positions)} positions)")

    def _init_db(self) -> None:
        """Initialize SQLite database with positions table."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    recommendation_id TEXT,
                    opened_at TEXT,
                    underlying TEXT,
                    expiry TEXT,
                    strike REAL,
                    right TEXT,
                    action TEXT,
                    contracts INTEGER,
                    fill_price REAL,
                    entry_cost REAL,
                    current_price REAL,
                    current_value REAL,
                    pnl REAL,
                    pnl_percent REAL,
                    dte INTEGER,
                    delta REAL,
                    status TEXT,
                    exit_reason TEXT,
                    closed_at TEXT,
                    close_price REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbol_stats (
                    symbol TEXT PRIMARY KEY,
                    trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    total_pnl_dollars REAL,
                    total_pnl_percent REAL
                )
            """)
            conn.commit()

    def _load_from_db(self) -> None:
        """Load positions and symbol stats from database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Load positions
            cursor = conn.execute("SELECT * FROM positions")
            for row in cursor:
                position = TrackedPosition(
                    id=row["id"],
                    recommendation_id=row["recommendation_id"],
                    opened_at=row["opened_at"],
                    underlying=row["underlying"],
                    expiry=row["expiry"],
                    strike=row["strike"],
                    right=row["right"],
                    action=row["action"],
                    contracts=row["contracts"],
                    fill_price=row["fill_price"],
                    entry_cost=row["entry_cost"],
                    current_price=row["current_price"],
                    current_value=row["current_value"],
                    pnl=row["pnl"] or 0.0,
                    pnl_percent=row["pnl_percent"] or 0.0,
                    dte=row["dte"],
                    delta=row["delta"],
                    status=row["status"] or "open",
                    exit_reason=row["exit_reason"],
                    closed_at=row["closed_at"],
                    close_price=row["close_price"],
                )
                self._positions[position.id] = position

            # Load symbol stats
            cursor = conn.execute("SELECT * FROM symbol_stats")
            for row in cursor:
                stats = SymbolStats(
                    symbol=row["symbol"],
                    trades=row["trades"],
                    wins=row["wins"],
                    losses=row["losses"],
                    total_pnl_dollars=row["total_pnl_dollars"],
                    total_pnl_percent=row["total_pnl_percent"],
                )
                self._symbol_stats[stats.symbol] = stats

    def _save_position(self, position: TrackedPosition) -> None:
        """Save or update a position in the database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions (
                    id, recommendation_id, opened_at, underlying, expiry, strike,
                    right, action, contracts, fill_price, entry_cost, current_price,
                    current_value, pnl, pnl_percent, dte, delta, status,
                    exit_reason, closed_at, close_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.id, position.recommendation_id, position.opened_at,
                position.underlying, position.expiry, position.strike,
                position.right, position.action, position.contracts,
                position.fill_price, position.entry_cost, position.current_price,
                position.current_value, position.pnl, position.pnl_percent,
                position.dte, position.delta, position.status,
                position.exit_reason, position.closed_at, position.close_price,
            ))
            conn.commit()

    def _save_symbol_stats(self, stats: SymbolStats) -> None:
        """Save or update symbol stats in the database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO symbol_stats (
                    symbol, trades, wins, losses, total_pnl_dollars, total_pnl_percent
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                stats.symbol, stats.trades, stats.wins, stats.losses,
                stats.total_pnl_dollars, stats.total_pnl_percent,
            ))
            conn.commit()

    def open_position(
        self,
        recommendation_id: str,
        underlying: str,
        expiry: str,
        strike: float,
        right: str,
        action: str,
        contracts: int,
        fill_price: float,
    ) -> TrackedPosition:
        """Open a new tracked position.

        Args:
            recommendation_id: ID of the original recommendation
            underlying: Ticker symbol
            expiry: Option expiration (ISO date)
            strike: Strike price
            right: C or P
            action: BUY_CALL, BUY_PUT, etc.
            contracts: Number of contracts
            fill_price: Actual fill price per contract

        Returns:
            The created TrackedPosition
        """
        if len(self.get_open_positions()) >= self.config.max_positions:
            raise ValueError(f"Maximum {self.config.max_positions} positions reached")

        position_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        entry_cost = contracts * fill_price * 100

        position = TrackedPosition(
            id=position_id,
            recommendation_id=recommendation_id,
            opened_at=now.isoformat(),
            underlying=underlying,
            expiry=expiry,
            strike=strike,
            right=right,
            action=action,
            contracts=contracts,
            fill_price=fill_price,
            entry_cost=entry_cost,
            current_price=fill_price,
            current_value=entry_cost,
        )

        self._positions[position_id] = position

        # Persist to database
        self._save_position(position)

        logger.info(
            f"Position opened: {action} {underlying} ${strike}{right} "
            f"x{contracts} @ ${fill_price:.2f} (${entry_cost:.0f})"
        )

        return position

    def close_position(
        self,
        position_id: str,
        close_price: float,
    ) -> TrackedPosition | None:
        """Close a tracked position.

        Args:
            position_id: ID of position to close
            close_price: Price at which position was closed

        Returns:
            The closed position, or None if not found
        """
        position = self._positions.get(position_id)
        if not position:
            logger.warning(f"Position not found: {position_id}")
            return None

        if position.status == "closed":
            logger.warning(f"Position already closed: {position_id}")
            return position

        now = datetime.now(timezone.utc)

        # Calculate final P/L
        close_value = position.contracts * close_price * 100
        pnl = close_value - position.entry_cost
        pnl_percent = (pnl / position.entry_cost) * 100 if position.entry_cost > 0 else 0

        # Update position
        position.status = "closed"
        position.closed_at = now.isoformat()
        position.close_price = close_price
        position.current_price = close_price
        position.current_value = close_value
        position.pnl = pnl
        position.pnl_percent = pnl_percent

        logger.info(
            f"Position closed: {position.underlying} ${position.strike}{position.right} "
            f"@ ${close_price:.2f} (P/L: ${pnl:.0f}, {pnl_percent:.1f}%)"
        )

        # Update per-symbol stats
        self._update_symbol_stats(position)

        # Persist to database
        self._save_position(position)

        return position

    def _update_symbol_stats(self, position: TrackedPosition) -> None:
        """Update per-symbol statistics when a position is closed."""
        symbol = position.underlying

        if symbol not in self._symbol_stats:
            self._symbol_stats[symbol] = SymbolStats(symbol=symbol)

        stats = self._symbol_stats[symbol]
        stats.trades += 1
        stats.total_pnl_dollars += position.pnl
        stats.total_pnl_percent += position.pnl_percent

        if position.pnl > 0:
            stats.wins += 1
        else:
            stats.losses += 1

        # Persist symbol stats to database
        self._save_symbol_stats(stats)

        logger.info(
            f"[SYMBOL_STATS] {symbol}: {stats.trades} trades, "
            f"{stats.wins} wins ({stats.win_rate:.0f}%), "
            f"avg P&L: {stats.avg_pnl_percent:+.1f}%"
        )

    def update_position(
        self,
        position_id: str,
        current_price: float | None = None,
        delta: float | None = None,
        dte: int | None = None,
        sentiment_score: float | None = None,
    ) -> ExitSignal | None:
        """Update a position with current market data and check for exit signals.

        Args:
            position_id: Position to update
            current_price: Current option price
            delta: Current delta
            dte: Days to expiration
            sentiment_score: Combined sentiment score (-100 to +100)

        Returns:
            ExitSignal if exit criteria met, None otherwise
        """
        position = self._positions.get(position_id)
        if not position or position.status == "closed":
            return None

        # Update market data
        if current_price is not None:
            position.current_price = current_price
            position.current_value = position.contracts * current_price * 100
            position.pnl = position.current_value - position.entry_cost
            position.pnl_percent = (
                (position.pnl / position.entry_cost) * 100
                if position.entry_cost > 0 else 0
            )

        if delta is not None:
            position.delta = delta

        if dte is not None:
            position.dte = dte

        # Check exit signals
        exit_signal = self._check_exit_signals(position, sentiment_score)

        if exit_signal:
            # Update status if not already set (first time triggering)
            if position.status != "exit_signal":
                position.status = "exit_signal"
                position.exit_reason = exit_signal.reason
            # Persist exit signal status to database
            self._save_position(position)
            return exit_signal

        # Persist updated position data (price, pnl, etc.)
        self._save_position(position)
        return None

    def _check_exit_signals(
        self,
        position: TrackedPosition,
        sentiment_score: float | None = None,
    ) -> ExitSignal | None:
        """Check if position meets exit criteria.

        Fixed Exit Rules (validated from TSLA backtesting):
        1. Take profit: +40% gain
        2. Stop loss: -20% loss
        3. Time exit: DTE < 1 (close before expiration)

        Args:
            position: Position to check
            sentiment_score: Combined sentiment score (unused, kept for API compat)

        Returns:
            ExitSignal if exit criteria met, None otherwise
        """

        # 1. Take profit check (+40%)
        if position.pnl_percent >= self.config.take_profit_percent:
            return ExitSignal(
                position_id=position.id,
                reason=f"Take profit triggered (+{position.pnl_percent:.0f}%)",
                current_price=position.current_price or 0,
                pnl=position.pnl,
                pnl_percent=position.pnl_percent,
                urgency="medium",
                trigger="take_profit",
            )

        # 2. Stop loss check (-20%)
        if position.pnl_percent <= self.config.stop_loss_percent:
            return ExitSignal(
                position_id=position.id,
                reason=f"Stop loss triggered ({position.pnl_percent:.0f}%)",
                current_price=position.current_price or 0,
                pnl=position.pnl,
                pnl_percent=position.pnl_percent,
                urgency="high",
                trigger="stop_loss",
            )

        # 3. Time exit (close before expiration)
        if position.dte is not None and position.dte <= self.config.min_dte_exit:
            return ExitSignal(
                position_id=position.id,
                reason=f"Time exit ({position.dte} DTE remaining)",
                current_price=position.current_price or 0,
                pnl=position.pnl,
                pnl_percent=position.pnl_percent,
                urgency="high",
                trigger="time_exit",
            )

        return None

    def get_position(self, position_id: str) -> TrackedPosition | None:
        """Get a position by ID."""
        return self._positions.get(position_id)

    def get_open_positions(self) -> list[TrackedPosition]:
        """Get all open positions."""
        return [p for p in self._positions.values() if p.status != "closed"]

    def get_all_positions(self) -> list[TrackedPosition]:
        """Get all positions (open and closed)."""
        return list(self._positions.values())

    def get_total_exposure(self) -> float:
        """Get total exposure from open positions only."""
        return sum(p.entry_cost for p in self.get_open_positions())

    def get_total_pnl(self) -> float:
        """Get total P/L from all positions."""
        return sum(p.pnl for p in self._positions.values())

    def clear_exit_signal(self, position_id: str) -> bool:
        """Clear exit signal status for a position (user dismissed it).

        Returns True if position was updated, False if not found.
        """
        position = self._positions.get(position_id)
        if not position:
            return False

        if position.status == "exit_signal":
            position.status = "open"
            position.exit_reason = None
            # Persist the update to database
            self._save_position(position)
            logger.info(f"Exit signal dismissed for {position.underlying} ${position.strike}")
            return True

        return False

    def _delete_position(self, position_id: str) -> None:
        """Delete a position from the database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM positions WHERE id = ?", (position_id,))
            conn.commit()

    def clear_closed_positions(self) -> int:
        """Remove closed positions from tracking. Returns count removed."""
        closed_ids = [p.id for p in self._positions.values() if p.status == "closed"]
        for pid in closed_ids:
            self._delete_position(pid)
            del self._positions[pid]
        return len(closed_ids)

    def get_symbol_stats(self, symbol: str) -> SymbolStats | None:
        """Get statistics for a specific symbol."""
        return self._symbol_stats.get(symbol)

    def get_all_symbol_stats(self) -> dict[str, SymbolStats]:
        """Get statistics for all symbols."""
        return dict(self._symbol_stats)

    def get_performance_summary(self) -> dict:
        """Get a summary of performance across all symbols.

        Returns dict with:
            - per_symbol: Dict of symbol -> stats
            - total: Aggregate stats across all symbols
            - underperformers: List of symbols with avg P&L < 5% while others > 15%
        """
        if not self._symbol_stats:
            return {
                "per_symbol": {},
                "total": {
                    "trades": 0,
                    "wins": 0,
                    "win_rate": 0,
                    "total_pnl_percent": 0,
                    "avg_pnl_percent": 0,
                },
                "underperformers": [],
            }

        per_symbol = {s: stats.to_dict() for s, stats in self._symbol_stats.items()}

        # Calculate totals
        total_trades = sum(s.trades for s in self._symbol_stats.values())
        total_wins = sum(s.wins for s in self._symbol_stats.values())
        total_pnl_pct = sum(s.total_pnl_percent for s in self._symbol_stats.values())

        total = {
            "trades": total_trades,
            "wins": total_wins,
            "win_rate": round((total_wins / total_trades * 100) if total_trades > 0 else 0, 1),
            "total_pnl_percent": round(total_pnl_pct, 1),
            "avg_pnl_percent": round((total_pnl_pct / total_trades) if total_trades > 0 else 0, 1),
        }

        # Identify underperformers: symbols with avg P&L < 5% while others > 15%
        # Only flag if we have enough data (at least 4 trades per symbol)
        underperformers = []
        for symbol, stats in self._symbol_stats.items():
            if stats.trades < 4:
                continue  # Not enough data yet

            other_stats = [s for s in self._symbol_stats.values() if s.symbol != symbol and s.trades >= 4]
            if not other_stats:
                continue

            other_avg = sum(s.avg_pnl_percent for s in other_stats) / len(other_stats)

            # Flag if this symbol is < 5% avg AND others are > 15% avg
            if stats.avg_pnl_percent < 5 and other_avg > 15:
                underperformers.append({
                    "symbol": symbol,
                    "avg_pnl": round(stats.avg_pnl_percent, 1),
                    "trades": stats.trades,
                    "other_avg": round(other_avg, 1),
                    "recommendation": f"Consider dropping {symbol} (avg {stats.avg_pnl_percent:+.1f}% vs others {other_avg:+.1f}%)",
                })

        return {
            "per_symbol": per_symbol,
            "total": total,
            "underperformers": underperformers,
        }

    def print_symbol_stats(self) -> None:
        """Print formatted symbol statistics to log."""
        summary = self.get_performance_summary()

        logger.info("=" * 60)
        logger.info("PAPER TRADING PERFORMANCE BY SYMBOL")
        logger.info("=" * 60)

        for symbol, stats in summary["per_symbol"].items():
            logger.info(
                f"  {symbol}: {stats['trades']} trades, "
                f"{stats['wins']} wins ({stats['win_rate']}%), "
                f"avg P&L: {stats['avg_pnl_percent']:+.1f}%"
            )

        logger.info("-" * 60)
        total = summary["total"]
        logger.info(
            f"  TOTAL: {total['trades']} trades, "
            f"{total['wins']} wins ({total['win_rate']}%), "
            f"avg P&L: {total['avg_pnl_percent']:+.1f}%"
        )

        if summary["underperformers"]:
            logger.info("-" * 60)
            logger.warning("UNDERPERFORMERS DETECTED:")
            for up in summary["underperformers"]:
                logger.warning(f"  {up['recommendation']}")
