"""Automated signal-to-order execution engine.

Ties together:
- RegimeSignalGenerator: Produces BUY_CALL/BUY_PUT signals
- AlpacaTrader: Executes orders via Alpaca API
- PositionTracker: Tracks positions for exit monitoring

Flow:
    Signal → Validate → Size → Order → Track → Monitor → Exit

Exit monitoring runs in a background loop checking:
- Take profit: +40%
- Stop loss: -20%
- Time exit: DTE < 1
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from backend.data.alpaca_trader import (
    AlpacaTrader,
    OrderResult,
    PositionInfo,
    build_occ_symbol,
    parse_occ_symbol,
)
from backend.engine.position_tracker import (
    PositionTracker,
    TrackedPosition,
    ExitSignal,
)
from backend.engine.regime_signals import (
    RegimeSignalGenerator,
    TradeSignal,
    SignalType,
    OptionSelection,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of an automated execution attempt."""
    success: bool
    signal: TradeSignal
    order_result: OrderResult | None = None
    position_id: str | None = None
    error: str | None = None
    occ_symbol: str | None = None
    contracts: int = 0
    fill_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "signal": self.signal.to_dict(),
            "orderId": self.order_result.order_id if self.order_result else None,
            "positionId": self.position_id,
            "error": self.error,
            "occSymbol": self.occ_symbol,
            "contracts": self.contracts,
            "fillPrice": self.fill_price,
        }


@dataclass
class AutoExecutorConfig:
    """Configuration for automated execution.

    Attributes:
        enabled: Whether auto-execution is enabled
        position_size_pct: Percent of portfolio per trade (default 10%)
        max_positions: Maximum concurrent positions (default 3)
        max_contract_price: Don't buy options over this price (default $20)
        min_contract_price: Don't buy options under this price (default $0.50)
        exit_check_interval: Seconds between exit checks (default 30)
        use_limit_orders: Use limit orders at mid price (default True)
        limit_offset_pct: Offset from mid for limit orders (default 0.5%)
    """
    enabled: bool = False
    position_size_pct: float = 10.0
    max_positions: int = 3
    max_contract_price: float = 20.0
    min_contract_price: float = 0.50
    exit_check_interval: float = 30.0
    use_limit_orders: bool = True
    limit_offset_pct: float = 0.5


class AutoExecutor:
    """Automated signal-to-order execution engine.

    Handles the complete trade lifecycle:
    1. Receives signals from RegimeSignalGenerator
    2. Validates trading conditions (market hours, limits, etc.)
    3. Calculates position size based on portfolio value
    4. Submits orders via AlpacaTrader
    5. Tracks positions in PositionTracker
    6. Monitors positions for exit conditions
    7. Automatically closes on take profit/stop loss/time exit

    Usage:
        executor = AutoExecutor(
            trader=alpaca_trader,
            position_tracker=position_tracker,
            signal_generator=regime_signal_generator,
            config=AutoExecutorConfig(enabled=True),
        )

        # Execute on a signal
        result = await executor.execute_signal(signal, option_selection)

        # Start exit monitoring loop
        await executor.start_exit_monitor()
    """

    def __init__(
        self,
        trader: AlpacaTrader,
        position_tracker: PositionTracker,
        signal_generator: RegimeSignalGenerator,
        config: AutoExecutorConfig | None = None,
        on_execution: Callable[[ExecutionResult], Awaitable[None]] | None = None,
        on_exit: Callable[[TrackedPosition, OrderResult], Awaitable[None]] | None = None,
    ):
        """Initialize the auto executor.

        Args:
            trader: AlpacaTrader instance for order execution
            position_tracker: PositionTracker for position management
            signal_generator: RegimeSignalGenerator for cooldown tracking
            config: AutoExecutorConfig or defaults
            on_execution: Callback when trade is executed
            on_exit: Callback when position is closed
        """
        self.trader = trader
        self.position_tracker = position_tracker
        self.signal_generator = signal_generator
        self.config = config or AutoExecutorConfig()
        self.on_execution = on_execution
        self.on_exit = on_exit

        self._exit_monitor_task: asyncio.Task | None = None
        self._running = False

        # Map position_id -> occ_symbol for Alpaca position lookups
        self._position_symbols: dict[str, str] = {}

        logger.info(
            f"AutoExecutor initialized (enabled={self.config.enabled}, "
            f"position_size={self.config.position_size_pct}%)"
        )

    @property
    def is_enabled(self) -> bool:
        """Check if auto-execution is enabled."""
        return self.config.enabled and self.trader.auto_execute

    async def execute_signal(
        self,
        signal: TradeSignal,
        option: OptionSelection,
    ) -> ExecutionResult:
        """Execute a trade signal.

        Args:
            signal: The trade signal to execute
            option: The selected option contract

        Returns:
            ExecutionResult with order details
        """
        if not self.is_enabled:
            return ExecutionResult(
                success=False,
                signal=signal,
                error="Auto-execution is disabled",
            )

        if signal.signal_type == SignalType.NO_SIGNAL:
            return ExecutionResult(
                success=False,
                signal=signal,
                error="No signal to execute",
            )

        # Build OCC symbol
        right = "C" if signal.signal_type == SignalType.BUY_CALL else "P"
        occ_symbol = build_occ_symbol(
            underlying=signal.symbol,
            expiry=option.expiry,
            strike=option.strike,
            right=right,
        )

        # Validate option price
        mid_price = option.mid
        if mid_price > self.config.max_contract_price:
            return ExecutionResult(
                success=False,
                signal=signal,
                occ_symbol=occ_symbol,
                error=f"Option price ${mid_price:.2f} exceeds max ${self.config.max_contract_price}",
            )

        if mid_price < self.config.min_contract_price:
            return ExecutionResult(
                success=False,
                signal=signal,
                occ_symbol=occ_symbol,
                error=f"Option price ${mid_price:.2f} below min ${self.config.min_contract_price}",
            )

        # Calculate position size
        try:
            account = self.trader.get_account()
            portfolio_value = account.portfolio_value
            target_position = portfolio_value * (self.config.position_size_pct / 100)
            contract_cost = mid_price * 100  # Options are 100 shares per contract
            contracts = max(1, int(target_position / contract_cost))

            logger.info(
                f"[AUTO-EXEC] Sizing: ${portfolio_value:.0f} portfolio, "
                f"{self.config.position_size_pct}% = ${target_position:.0f}, "
                f"${mid_price:.2f}/contract = {contracts} contracts"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                signal=signal,
                occ_symbol=occ_symbol,
                error=f"Failed to get account info: {e}",
            )

        # Submit order
        if self.config.use_limit_orders:
            # Use limit order at mid price with small offset
            limit_price = round(mid_price * (1 + self.config.limit_offset_pct / 100), 2)
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

        if not order_result.success:
            return ExecutionResult(
                success=False,
                signal=signal,
                order_result=order_result,
                occ_symbol=occ_symbol,
                contracts=contracts,
                error=order_result.error or "Order failed",
            )

        # Wait briefly for fill (for market orders)
        fill_price = order_result.filled_avg_price
        if not fill_price:
            # Poll for fill status
            await asyncio.sleep(1)
            order_status = self.trader.get_order(order_result.order_id)
            if order_status and order_status.get("filledAvgPrice"):
                fill_price = order_status["filledAvgPrice"]

        # Default to mid if no fill price yet
        if not fill_price:
            fill_price = mid_price

        # Track in PositionTracker
        action = "BUY_CALL" if signal.signal_type == SignalType.BUY_CALL else "BUY_PUT"
        position = self.position_tracker.open_position(
            recommendation_id=f"auto_{order_result.order_id}",
            underlying=signal.symbol,
            expiry=option.expiry,
            strike=option.strike,
            right=right,
            action=action,
            contracts=contracts,
            fill_price=fill_price,
        )

        # Store OCC symbol mapping for exit
        self._position_symbols[position.id] = occ_symbol

        # Record entry for cooldown
        direction = "call" if signal.signal_type == SignalType.BUY_CALL else "put"
        self.signal_generator.record_entry(signal.symbol, direction)

        result = ExecutionResult(
            success=True,
            signal=signal,
            order_result=order_result,
            position_id=position.id,
            occ_symbol=occ_symbol,
            contracts=contracts,
            fill_price=fill_price,
        )

        logger.info(
            f"[AUTO-EXEC] Order filled: {action} {contracts}x {occ_symbol} "
            f"@ ${fill_price:.2f} (order {order_result.order_id})"
        )

        # Callback
        if self.on_execution:
            try:
                await self.on_execution(result)
            except Exception as e:
                logger.error(f"on_execution callback error: {e}")

        return result

    async def check_and_execute_exits(self) -> list[tuple[TrackedPosition, OrderResult]]:
        """Check all open positions for exit conditions and execute closes.

        Returns:
            List of (position, order_result) tuples for executed exits
        """
        exits = []

        # Get Alpaca positions for current prices
        try:
            alpaca_positions = {p.symbol: p for p in self.trader.get_option_positions()}
        except Exception as e:
            logger.error(f"Failed to get Alpaca positions: {e}")
            return exits

        # Check each tracked position
        for position in self.position_tracker.get_open_positions():
            occ_symbol = self._position_symbols.get(position.id)
            if not occ_symbol:
                # Try to reconstruct from position data
                right = position.right
                occ_symbol = build_occ_symbol(
                    underlying=position.underlying,
                    expiry=position.expiry,
                    strike=position.strike,
                    right=right,
                )
                self._position_symbols[position.id] = occ_symbol
                logger.debug(f"[EXIT-MONITOR] Reconstructed OCC symbol: {occ_symbol}")

            # Get current price from Alpaca
            alpaca_pos = alpaca_positions.get(occ_symbol)
            current_price = None
            if alpaca_pos:
                current_price = alpaca_pos.current_price
                logger.debug(
                    f"[EXIT-MONITOR] {position.underlying}: price=${current_price:.2f}, "
                    f"entry=${position.fill_price:.2f}"
                )
            else:
                # Position not found in Alpaca - log for debugging
                logger.warning(
                    f"[EXIT-MONITOR] Position {occ_symbol} not found in trader. "
                    f"Known symbols: {list(alpaca_positions.keys())}"
                )

            # Calculate DTE
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).date()
            try:
                expiry_date = datetime.strptime(position.expiry, "%Y-%m-%d").date()
                dte = (expiry_date - today).days
            except:
                dte = None

            # Update position and check for exit signal
            exit_signal = self.position_tracker.update_position(
                position_id=position.id,
                current_price=current_price,
                dte=dte,
            )

            if exit_signal:
                logger.info(
                    f"[EXIT-MONITOR] Exit signal triggered for {position.underlying}: "
                    f"{exit_signal.reason} (P&L: {exit_signal.pnl_percent:.1f}%)"
                )
                # Execute the exit
                exit_result = await self._execute_exit(position, exit_signal, occ_symbol)
                if exit_result:
                    exits.append(exit_result)

        return exits

    async def _execute_exit(
        self,
        position: TrackedPosition,
        exit_signal: ExitSignal,
        occ_symbol: str,
    ) -> tuple[TrackedPosition, OrderResult] | None:
        """Execute an exit order for a position.

        Args:
            position: The position to close
            exit_signal: The exit signal that triggered
            occ_symbol: OCC symbol for the position

        Returns:
            (position, order_result) tuple or None if failed
        """
        logger.info(
            f"[AUTO-EXIT] {position.underlying}: {exit_signal.reason} "
            f"(P&L: ${exit_signal.pnl:.0f}, {exit_signal.pnl_percent:.1f}%)"
        )

        # Close via Alpaca
        order_result = self.trader.close_position(occ_symbol)

        if not order_result.success:
            logger.error(f"[AUTO-EXIT] Failed to close {occ_symbol}: {order_result.error}")
            return None

        # Wait briefly for fill
        close_price = order_result.filled_avg_price
        if not close_price:
            await asyncio.sleep(1)
            order_status = self.trader.get_order(order_result.order_id)
            if order_status and order_status.get("filledAvgPrice"):
                close_price = order_status["filledAvgPrice"]

        # Default to current price if no fill price
        if not close_price and position.current_price:
            close_price = position.current_price
        elif not close_price:
            close_price = position.fill_price  # Fallback

        # Close in tracker
        closed_position = self.position_tracker.close_position(
            position_id=position.id,
            close_price=close_price,
        )

        # Record exit for cooldown
        direction = "call" if position.right == "C" else "put"
        self.signal_generator.record_exit(position.underlying, direction)

        # Clean up mapping
        if position.id in self._position_symbols:
            del self._position_symbols[position.id]

        logger.info(
            f"[AUTO-EXIT] Closed {position.underlying} ${position.strike}{position.right} "
            f"@ ${close_price:.2f} (order {order_result.order_id})"
        )

        # Callback
        if self.on_exit and closed_position:
            try:
                await self.on_exit(closed_position, order_result)
            except Exception as e:
                logger.error(f"on_exit callback error: {e}")

        return (closed_position or position, order_result)

    async def start_exit_monitor(self) -> None:
        """Start the background exit monitoring loop."""
        if self._running:
            logger.warning("Exit monitor already running")
            return

        self._running = True
        self._exit_monitor_task = asyncio.create_task(self._exit_monitor_loop())
        logger.info(
            f"Exit monitor started (interval: {self.config.exit_check_interval}s)"
        )

    async def stop_exit_monitor(self) -> None:
        """Stop the background exit monitoring loop."""
        self._running = False
        if self._exit_monitor_task:
            self._exit_monitor_task.cancel()
            try:
                await self._exit_monitor_task
            except asyncio.CancelledError:
                pass
            self._exit_monitor_task = None
        logger.info("Exit monitor stopped")

    async def _exit_monitor_loop(self) -> None:
        """Background loop that checks positions for exit conditions."""
        while self._running:
            try:
                # Only check during market hours
                if self.trader.is_market_open():
                    exits = await self.check_and_execute_exits()
                    if exits:
                        logger.info(f"[EXIT-MONITOR] Executed {len(exits)} exit(s)")
                else:
                    logger.debug("[EXIT-MONITOR] Market closed, skipping check")

            except Exception as e:
                logger.error(f"[EXIT-MONITOR] Error: {e}")

            await asyncio.sleep(self.config.exit_check_interval)

    def sync_from_alpaca(self) -> int:
        """Sync position tracker with Alpaca positions.

        Useful on startup to reconcile any positions that were opened
        outside of this session.

        Returns:
            Number of positions synced
        """
        synced = 0
        try:
            alpaca_positions = self.trader.get_option_positions()

            for ap in alpaca_positions:
                # Check if we're already tracking this position
                existing = None
                for tp in self.position_tracker.get_open_positions():
                    occ = self._position_symbols.get(tp.id)
                    if occ == ap.symbol:
                        existing = tp
                        break

                if existing:
                    # Update with current data
                    self.position_tracker.update_position(
                        position_id=existing.id,
                        current_price=ap.current_price,
                    )
                else:
                    # Parse OCC symbol and create new tracking entry
                    parsed = parse_occ_symbol(ap.symbol)
                    direction = "call" if parsed["right"] == "C" else "put"
                    action = "BUY_CALL" if parsed["right"] == "C" else "BUY_PUT"

                    position = self.position_tracker.open_position(
                        recommendation_id=f"synced_{ap.symbol}",
                        underlying=parsed["underlying"],
                        expiry=parsed["expiry"],
                        strike=parsed["strike"],
                        right=parsed["right"],
                        action=action,
                        contracts=int(ap.qty),
                        fill_price=ap.avg_entry_price,
                    )

                    self._position_symbols[position.id] = ap.symbol
                    self.signal_generator.record_entry(parsed["underlying"], direction)

                    logger.info(f"[SYNC] Imported position: {ap.symbol}")
                    synced += 1

        except Exception as e:
            logger.error(f"Failed to sync from Alpaca: {e}")

        return synced

    def get_status(self) -> dict[str, Any]:
        """Get current executor status.

        Returns snake_case fields to match frontend TradingStatus type.
        """
        return {
            "enabled": self.is_enabled,
            "auto_execution": self.trader.auto_execute if self.trader else False,
            "open_positions": len(self._position_symbols),
            "max_positions": self.config.max_positions,
            "position_size_pct": self.config.position_size_pct,
            "exit_monitor_running": self._running,
        }
