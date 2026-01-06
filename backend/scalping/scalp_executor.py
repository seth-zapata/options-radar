"""Scalp trade execution and exit monitoring.

Handles the complete scalp trade lifecycle:
1. Execute scalp signals (place orders)
2. Track open scalp positions
3. Monitor for exit conditions (TP, SL, time)
4. Close positions and track P&L

Designed for fast intraday scalping with:
- Sub-second exit monitoring
- Time-based exits (max hold 15 min)
- Per-signal TP/SL thresholds
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

if TYPE_CHECKING:
    from backend.data.alpaca_trader import AlpacaTrader, OrderResult
    from backend.data.mock_trader import MockAlpacaTrader

from backend.scalping.signal_generator import ScalpSignal
from backend.scalping.scalp_backtester import ScalpTrade

logger = logging.getLogger(__name__)


@dataclass
class ScalpPosition:
    """Active scalp position being monitored.

    Extends ScalpTrade with live monitoring state.
    """

    # Core trade info (from ScalpTrade)
    signal_id: str
    symbol: str
    signal_type: Literal["SCALP_CALL", "SCALP_PUT"]
    trigger: str
    confidence: int
    option_symbol: str
    strike: float
    expiry: str
    delta: float
    dte: int
    entry_time: datetime
    entry_price: float
    underlying_at_entry: float
    contracts: int

    # Exit targets (from signal)
    take_profit_pct: float
    stop_loss_pct: float
    max_hold_seconds: int

    # Live state
    current_price: float | None = None
    underlying_price: float | None = None
    current_pnl_pct: float = 0.0
    max_gain_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    last_update: datetime | None = None

    # Order tracking
    entry_order_id: str | None = None
    exit_order_id: str | None = None

    @property
    def hold_seconds(self) -> int:
        """Seconds since entry."""
        if not self.entry_time:
            return 0
        now = datetime.now(timezone.utc)
        return int((now - self.entry_time).total_seconds())

    @property
    def should_exit_time(self) -> bool:
        """Check if max hold time exceeded."""
        return self.hold_seconds >= self.max_hold_seconds

    @property
    def should_exit_profit(self) -> bool:
        """Check if take profit hit."""
        return self.current_pnl_pct >= self.take_profit_pct

    @property
    def should_exit_loss(self) -> bool:
        """Check if stop loss hit."""
        return self.current_pnl_pct <= -self.stop_loss_pct

    def update_price(self, price: float, underlying: float | None = None) -> str | None:
        """Update current price and check exit conditions.

        Args:
            price: Current option price
            underlying: Current underlying price (optional)

        Returns:
            Exit reason if exit triggered, None otherwise
        """
        self.current_price = price
        self.underlying_price = underlying
        self.last_update = datetime.now(timezone.utc)

        # Calculate P&L
        if self.entry_price > 0:
            self.current_pnl_pct = ((price - self.entry_price) / self.entry_price) * 100

        # Track max gain/drawdown
        if self.current_pnl_pct > self.max_gain_pct:
            self.max_gain_pct = self.current_pnl_pct
        if self.current_pnl_pct < self.max_drawdown_pct:
            self.max_drawdown_pct = self.current_pnl_pct

        # Check exit conditions in order of priority
        if self.should_exit_profit:
            return "take_profit"
        if self.should_exit_loss:
            return "stop_loss"
        if self.should_exit_time:
            return "time_exit"

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "trigger": self.trigger,
            "confidence": self.confidence,
            "option_symbol": self.option_symbol,
            "strike": self.strike,
            "expiry": self.expiry,
            "delta": round(self.delta, 3),
            "dte": self.dte,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": round(self.entry_price, 2),
            "underlying_at_entry": round(self.underlying_at_entry, 2),
            "contracts": self.contracts,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_hold_seconds": self.max_hold_seconds,
            "current_price": round(self.current_price, 2) if self.current_price else None,
            "current_pnl_pct": round(self.current_pnl_pct, 2),
            "hold_seconds": self.hold_seconds,
            "max_gain_pct": round(self.max_gain_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
        }


@dataclass
class ScalpExecutionResult:
    """Result of a scalp execution attempt."""

    success: bool
    signal: ScalpSignal
    position: ScalpPosition | None = None
    order_id: str | None = None
    error: str | None = None
    fill_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "signal": self.signal.to_dict(),
            "position": self.position.to_dict() if self.position else None,
            "order_id": self.order_id,
            "error": self.error,
            "fill_price": round(self.fill_price, 2) if self.fill_price else None,
        }


@dataclass
class ScalpExitResult:
    """Result of a scalp exit."""

    position: ScalpPosition
    exit_reason: str
    exit_price: float
    pnl_dollars: float
    pnl_pct: float
    hold_seconds: int
    order_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "signal_id": self.position.signal_id,
            "symbol": self.position.symbol,
            "signal_type": self.position.signal_type,
            "option_symbol": self.position.option_symbol,
            "exit_reason": self.exit_reason,
            "entry_price": round(self.position.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "pnl_dollars": round(self.pnl_dollars, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "hold_seconds": self.hold_seconds,
            "max_gain_pct": round(self.position.max_gain_pct, 2),
            "max_drawdown_pct": round(self.position.max_drawdown_pct, 2),
            "order_id": self.order_id,
        }


@dataclass
class ScalpExecutorConfig:
    """Configuration for scalp execution."""

    enabled: bool = False
    max_concurrent_scalps: int = 1  # Only one scalp at a time by default
    exit_check_interval: float = 0.5  # Check every 500ms for fast exits
    use_limit_orders: bool = True
    limit_offset_pct: float = 0.5  # Offset from mid for limit orders
    slippage_pct: float = 0.5  # Expected slippage for P&L calculations


class ScalpExecutor:
    """Executes and monitors scalp trades.

    Usage:
        executor = ScalpExecutor(
            trader=alpaca_trader,
            config=ScalpExecutorConfig(enabled=True),
        )

        # Execute a scalp signal
        result = await executor.execute_signal(scalp_signal)

        # Start exit monitoring (runs in background)
        await executor.start_exit_monitor()

        # Get open positions
        positions = executor.get_open_positions()
    """

    def __init__(
        self,
        trader: "AlpacaTrader | MockAlpacaTrader",
        config: ScalpExecutorConfig | None = None,
        on_execution: Callable[[ScalpExecutionResult], Awaitable[None]] | None = None,
        on_exit: Callable[[ScalpExitResult], Awaitable[None]] | None = None,
        on_update: Callable[[ScalpPosition], Awaitable[None]] | None = None,
    ):
        """Initialize the scalp executor.

        Args:
            trader: Alpaca or mock trader for order execution
            config: Configuration options
            on_execution: Callback when scalp is opened
            on_exit: Callback when scalp is closed
            on_update: Callback on position price updates
        """
        self.trader = trader
        self.config = config or ScalpExecutorConfig()
        self.on_execution = on_execution
        self.on_exit = on_exit
        self.on_update = on_update

        # Track open scalp positions
        self._positions: dict[str, ScalpPosition] = {}  # signal_id -> position

        # Exit monitor state
        self._exit_monitor_task: asyncio.Task | None = None
        self._running = False

        # Statistics
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0

    async def execute_signal(self, signal: ScalpSignal) -> ScalpExecutionResult:
        """Execute a scalp signal by opening a position.

        Args:
            signal: The scalp signal to execute

        Returns:
            ScalpExecutionResult with success/failure info
        """
        if not self.config.enabled:
            return ScalpExecutionResult(
                success=False,
                signal=signal,
                error="Scalp execution disabled",
            )

        # Check position limit
        if len(self._positions) >= self.config.max_concurrent_scalps:
            return ScalpExecutionResult(
                success=False,
                signal=signal,
                error=f"Max concurrent scalps ({self.config.max_concurrent_scalps}) reached",
            )

        # Calculate order details
        occ_symbol = signal.option_symbol
        contracts = signal.suggested_contracts

        # Use ask price for buys (with optional limit offset)
        if self.config.use_limit_orders:
            mid = (signal.bid_price + signal.ask_price) / 2
            limit_price = mid * (1 + self.config.limit_offset_pct / 100)
        else:
            limit_price = None

        logger.info(
            f"[SCALP-EXEC] Executing {signal.signal_type}: "
            f"{contracts}x {occ_symbol} @ ${signal.ask_price:.2f}"
        )

        # Submit order
        try:
            if self.config.use_limit_orders and limit_price:
                order_result = self.trader.submit_option_order(
                    occ_symbol=occ_symbol,
                    qty=contracts,
                    side="buy",
                    order_type="limit",
                    limit_price=limit_price,
                )
            else:
                order_result = self.trader.submit_option_order(
                    occ_symbol=occ_symbol,
                    qty=contracts,
                    side="buy",
                    order_type="market",
                )
        except Exception as e:
            logger.error(f"[SCALP-EXEC] Order submission failed: {e}")
            return ScalpExecutionResult(
                success=False,
                signal=signal,
                error=str(e),
            )

        if not order_result.success:
            return ScalpExecutionResult(
                success=False,
                signal=signal,
                order_id=order_result.order_id,
                error=order_result.error or "Order failed",
            )

        # Get fill price
        fill_price = order_result.filled_avg_price
        if not fill_price:
            # Wait briefly for fill
            await asyncio.sleep(0.5)
            order_status = self.trader.get_order(order_result.order_id)
            if order_status and order_status.get("filledAvgPrice"):
                fill_price = order_status["filledAvgPrice"]

        # Default to ask if no fill price
        if not fill_price:
            fill_price = signal.ask_price

        # Apply slippage estimate
        fill_price *= (1 + self.config.slippage_pct / 100)

        # Create position
        position = ScalpPosition(
            signal_id=signal.id,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            trigger=signal.trigger,
            confidence=signal.confidence,
            option_symbol=occ_symbol,
            strike=signal.strike,
            expiry=signal.expiry,
            delta=signal.delta,
            dte=signal.dte,
            entry_time=signal.timestamp,
            entry_price=fill_price,
            underlying_at_entry=signal.underlying_price,
            contracts=contracts,
            take_profit_pct=signal.take_profit_pct,
            stop_loss_pct=signal.stop_loss_pct,
            max_hold_seconds=signal.max_hold_minutes * 60,
            current_price=fill_price,
            entry_order_id=order_result.order_id,
        )

        # Track position
        self._positions[signal.id] = position
        self._total_trades += 1

        logger.info(
            f"[SCALP-EXEC] Position opened: {signal.signal_type} "
            f"{contracts}x {occ_symbol} @ ${fill_price:.2f} "
            f"(TP: +{signal.take_profit_pct}%, SL: -{signal.stop_loss_pct}%)"
        )

        result = ScalpExecutionResult(
            success=True,
            signal=signal,
            position=position,
            order_id=order_result.order_id,
            fill_price=fill_price,
        )

        # Callback
        if self.on_execution:
            try:
                await self.on_execution(result)
            except Exception as e:
                logger.error(f"on_execution callback error: {e}")

        return result

    async def start_exit_monitor(self) -> None:
        """Start the background exit monitoring loop."""
        if self._running:
            return

        self._running = True
        self._exit_monitor_task = asyncio.create_task(self._exit_monitor_loop())
        logger.info(
            f"[SCALP-EXEC] Exit monitor started "
            f"(interval: {self.config.exit_check_interval}s)"
        )

    async def stop_exit_monitor(self) -> None:
        """Stop the exit monitoring loop."""
        self._running = False
        if self._exit_monitor_task:
            self._exit_monitor_task.cancel()
            try:
                await self._exit_monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[SCALP-EXEC] Exit monitor stopped")

    async def _exit_monitor_loop(self) -> None:
        """Background loop that monitors positions for exit conditions."""
        while self._running:
            try:
                await asyncio.sleep(self.config.exit_check_interval)
                await self._check_exits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SCALP-EXIT] Monitor error: {e}")

    async def _check_exits(self) -> None:
        """Check all open positions for exit conditions."""
        if not self._positions:
            return

        # Get current prices from trader
        try:
            alpaca_positions = {
                p.symbol: p for p in self.trader.get_option_positions()
            }
        except Exception as e:
            logger.debug(f"[SCALP-EXIT] Failed to get positions: {e}")
            return

        # Check each position
        positions_to_close = []
        for signal_id, position in list(self._positions.items()):
            # Get current price
            alpaca_pos = alpaca_positions.get(position.option_symbol)
            if alpaca_pos:
                current_price = alpaca_pos.current_price
            else:
                # Position not found - may have been closed externally
                current_price = position.current_price

            if not current_price:
                continue

            # Update and check exit
            exit_reason = position.update_price(current_price)

            # Callback for position updates
            if self.on_update:
                try:
                    await self.on_update(position)
                except Exception as e:
                    logger.debug(f"on_update callback error: {e}")

            if exit_reason:
                positions_to_close.append((signal_id, position, exit_reason, current_price))

        # Execute exits
        for signal_id, position, exit_reason, exit_price in positions_to_close:
            await self._execute_exit(position, exit_reason, exit_price)

    async def _execute_exit(
        self,
        position: ScalpPosition,
        exit_reason: str,
        exit_price: float,
    ) -> ScalpExitResult | None:
        """Execute an exit for a position.

        Args:
            position: Position to close
            exit_reason: Why we're exiting
            exit_price: Price at exit

        Returns:
            ScalpExitResult or None if failed
        """
        logger.info(
            f"[SCALP-EXIT] Closing {position.option_symbol}: {exit_reason} "
            f"(P&L: {position.current_pnl_pct:+.1f}%)"
        )

        # Close via trader
        try:
            order_result = self.trader.close_position(position.option_symbol)
        except Exception as e:
            logger.error(f"[SCALP-EXIT] Close failed: {e}")
            return None

        if not order_result.success:
            logger.error(f"[SCALP-EXIT] Close order failed: {order_result.error}")
            return None

        # Get fill price
        fill_price = order_result.filled_avg_price
        if not fill_price:
            await asyncio.sleep(0.3)
            order_status = self.trader.get_order(order_result.order_id)
            if order_status and order_status.get("filledAvgPrice"):
                fill_price = order_status["filledAvgPrice"]

        if not fill_price:
            fill_price = exit_price

        # Apply slippage (selling at bid, so negative slippage)
        fill_price *= (1 - self.config.slippage_pct / 100)

        # Calculate P&L
        pnl_pct = ((fill_price - position.entry_price) / position.entry_price) * 100
        pnl_dollars = (fill_price - position.entry_price) * 100 * position.contracts

        # Update statistics
        if pnl_dollars > 0:
            self._winning_trades += 1
        self._total_pnl += pnl_dollars

        # Remove from tracking
        del self._positions[position.signal_id]

        result = ScalpExitResult(
            position=position,
            exit_reason=exit_reason,
            exit_price=fill_price,
            pnl_dollars=pnl_dollars,
            pnl_pct=pnl_pct,
            hold_seconds=position.hold_seconds,
            order_id=order_result.order_id,
        )

        logger.info(
            f"[SCALP-EXIT] Closed {position.symbol}: "
            f"${pnl_dollars:+.2f} ({pnl_pct:+.1f}%), "
            f"held {position.hold_seconds}s, reason: {exit_reason}"
        )

        # Callback
        if self.on_exit:
            try:
                await self.on_exit(result)
            except Exception as e:
                logger.error(f"on_exit callback error: {e}")

        return result

    async def force_close_all(self, reason: str = "manual") -> list[ScalpExitResult]:
        """Force close all open positions.

        Args:
            reason: Exit reason to record

        Returns:
            List of exit results
        """
        results = []
        for signal_id, position in list(self._positions.items()):
            current_price = position.current_price or position.entry_price
            result = await self._execute_exit(position, reason, current_price)
            if result:
                results.append(result)
        return results

    def get_open_positions(self) -> list[ScalpPosition]:
        """Get all open scalp positions."""
        return list(self._positions.values())

    def get_position(self, signal_id: str) -> ScalpPosition | None:
        """Get a specific position by signal ID."""
        return self._positions.get(signal_id)

    @property
    def has_open_position(self) -> bool:
        """Check if there are any open positions."""
        return len(self._positions) > 0

    @property
    def stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        win_rate = (
            self._winning_trades / self._total_trades
            if self._total_trades > 0
            else 0.0
        )
        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._total_trades - self._winning_trades,
            "win_rate": round(win_rate, 3),
            "total_pnl": round(self._total_pnl, 2),
            "open_positions": len(self._positions),
        }
