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
    trigger: Literal["profit_target", "stop_loss", "dte_warning"]


@dataclass
class PositionTrackerConfig:
    """Configuration for position tracking.

    Attributes:
        profit_target_percent: Take profit at this gain (default 50%)
        stop_loss_percent: Stop loss at this loss (default -30%)
        min_dte_warning: Warn when DTE below this (default 7)
        max_positions: Maximum open positions (default 10)
    """
    profit_target_percent: float = 50.0
    stop_loss_percent: float = -30.0
    min_dte_warning: int = 7
    max_positions: int = 10


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

        return position

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

        Args:
            position: Position to check
            sentiment_score: Combined sentiment score (-100 to +100)
                Positive = bullish, Negative = bearish

        Returns:
            ExitSignal if exit criteria met, None otherwise
        """

        # Profit target (TastyTrade 50% rule)
        if position.pnl_percent >= self.config.profit_target_percent:
            return ExitSignal(
                position_id=position.id,
                reason=f"Profit target reached (+{position.pnl_percent:.0f}%)",
                current_price=position.current_price or 0,
                pnl=position.pnl,
                pnl_percent=position.pnl_percent,
                urgency="medium",
                trigger="profit_target",
            )

        # Stop loss
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

        # Time decay warning (avoid gamma acceleration)
        if position.dte is not None and position.dte <= self.config.min_dte_warning:
            return ExitSignal(
                position_id=position.id,
                reason=f"DTE warning ({position.dte} days remaining)",
                current_price=position.current_price or 0,
                pnl=position.pnl,
                pnl_percent=position.pnl_percent,
                urgency="medium" if position.dte > 3 else "high",
                trigger="dte_warning",
            )

        # NOTE: Sentiment reversal exit was REMOVED after P&L backtest showed
        # +381% improvement without it. Profit target gains outweigh the
        # stop loss protection that sentiment reversals provided.
        # See: backend/scripts/compare_exits.py for the analysis.

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
