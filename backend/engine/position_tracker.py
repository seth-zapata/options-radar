"""Paper position tracking for confirmed trades.

Tracks positions the user confirms they took, with actual fill prices.
Generates exit signals based on P/L, time decay, and delta changes.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)


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

    def __init__(self, config: PositionTrackerConfig | None = None):
        self.config = config or PositionTrackerConfig()
        self._positions: dict[str, TrackedPosition] = {}
        self._symbol_stats: dict[str, SymbolStats] = {}

        logger.info("Position tracker initialized")

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

        # Check exit signals (only if position doesn't already have an active exit signal)
        if position.status != "exit_signal":
            exit_signal = self._check_exit_signals(position, sentiment_score)

            if exit_signal:
                position.status = "exit_signal"
                position.exit_reason = exit_signal.reason
                return exit_signal

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
            logger.info(f"Exit signal dismissed for {position.underlying} ${position.strike}")
            return True

        return False

    def clear_closed_positions(self) -> int:
        """Remove closed positions from tracking. Returns count removed."""
        closed_ids = [p.id for p in self._positions.values() if p.status == "closed"]
        for pid in closed_ids:
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
