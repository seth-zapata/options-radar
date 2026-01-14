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
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

if TYPE_CHECKING:
    from backend.data.alpaca_trader import AlpacaTrader, OrderResult
    from backend.data.mock_trader import MockAlpacaTrader
    from backend.scalping.velocity_tracker import PriceVelocityTracker

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

    # Trailing stop config
    trailing_stop_activation_pct: float = 10.0  # Activate after +10% gain
    trailing_stop_distance_pct: float = 8.0     # Trail 8% below peak

    # Momentum reversal config
    entry_velocity_direction: Literal["up", "down"] = "up"  # Direction at entry

    # Live state
    current_price: float | None = None
    underlying_price: float | None = None
    current_pnl_pct: float = 0.0
    max_gain_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    last_update: datetime | None = None
    trailing_stop_active: bool = False  # Whether trailing stop has activated
    trailing_stop_level_pct: float = 0.0  # Current trail level (% P&L)

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

    @property
    def should_exit_trailing_stop(self) -> bool:
        """Check if trailing stop hit."""
        if not self.trailing_stop_active:
            return False
        return self.current_pnl_pct <= self.trailing_stop_level_pct

    def update_price(
        self,
        price: float,
        underlying: float | None = None,
        current_velocity_direction: Literal["up", "down", "flat"] | None = None,
    ) -> str | None:
        """Update current price and check exit conditions.

        Args:
            price: Current option price
            underlying: Current underlying price (optional)
            current_velocity_direction: Current momentum direction for reversal check

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

        # Update trailing stop
        if not self.trailing_stop_active:
            # Activate trailing stop once we hit activation threshold
            if self.current_pnl_pct >= self.trailing_stop_activation_pct:
                self.trailing_stop_active = True
                self.trailing_stop_level_pct = self.max_gain_pct - self.trailing_stop_distance_pct
                logger.info(
                    f"[TRAIL] Trailing stop activated at {self.max_gain_pct:.1f}%, "
                    f"trail level: {self.trailing_stop_level_pct:.1f}%"
                )
        else:
            # Update trail level as price makes new highs
            new_trail_level = self.max_gain_pct - self.trailing_stop_distance_pct
            if new_trail_level > self.trailing_stop_level_pct:
                self.trailing_stop_level_pct = new_trail_level
                logger.debug(f"[TRAIL] Trail level raised to {self.trailing_stop_level_pct:.1f}%")

        # Check exit conditions in order of priority
        # 1. Momentum reversal (thesis dead - exit fast)
        if current_velocity_direction and current_velocity_direction != "flat":
            if self.entry_velocity_direction == "up" and current_velocity_direction == "down":
                return "momentum_reversal"
            if self.entry_velocity_direction == "down" and current_velocity_direction == "up":
                return "momentum_reversal"

        # 2. Take profit (quick win)
        if self.should_exit_profit:
            return "take_profit"

        # 3. Trailing stop (lock in gains)
        if self.should_exit_trailing_stop:
            return "trailing_stop"

        # 4. Stop loss (hard floor)
        if self.should_exit_loss:
            return "stop_loss"

        # 5. Time stop (don't hold stale positions)
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
            # Trailing stop
            "trailing_stop_active": self.trailing_stop_active,
            "trailing_stop_level_pct": round(self.trailing_stop_level_pct, 2),
            # Momentum reversal
            "entry_velocity_direction": self.entry_velocity_direction,
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
        velocity_trackers: dict[str, "PriceVelocityTracker"] | None = None,
        momentum_window_seconds: int = 15,
    ):
        """Initialize the scalp executor.

        Args:
            trader: Alpaca or mock trader for order execution
            config: Configuration options
            on_execution: Callback when scalp is opened
            on_exit: Callback when scalp is closed
            on_update: Callback on position price updates
            velocity_trackers: Per-symbol velocity trackers for momentum reversal exits
            momentum_window_seconds: Window for velocity calculations
        """
        self.trader = trader
        self.config = config or ScalpExecutorConfig()
        self.on_execution = on_execution
        self.on_exit = on_exit
        self.on_update = on_update
        self.velocity_trackers = velocity_trackers or {}
        self.momentum_window_seconds = momentum_window_seconds

        # Track open scalp positions
        self._positions: dict[str, ScalpPosition] = {}  # signal_id -> position

        # Exit monitor state
        self._exit_monitor_task: asyncio.Task | None = None
        self._running = False

        # Statistics
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0

        # Velocity-scaled position sizing state
        self._daily_pnl = 0.0  # Track P&L for the day
        self._daily_starting_equity: float | None = None
        self._consecutive_losses = 0  # Count of consecutive losses
        self._trading_halted = False  # If daily loss limit hit
        self._last_reset_date: date | None = None  # For resetting daily stats

        # Position sizing config
        self._max_risk_pct = 5.0  # Max risk per trade
        self._daily_loss_limit_pct = 10.0  # Stop trading if hit
        self._loss_streak_threshold = 3  # Consecutive losses to reduce sizing

        # Trade log directory
        self._trade_log_dir = Path("scalp_trades")
        self._trade_log_dir.mkdir(exist_ok=True)

    def _get_trade_log_path(self) -> Path:
        """Get path to today's trade log file."""
        today = date.today().isoformat()
        return self._trade_log_dir / f"scalp_trades_{today}.json"

    def _load_trade_log(self) -> dict[str, Any]:
        """Load today's trade log or create empty structure."""
        log_path = self._get_trade_log_path()
        if log_path.exists():
            try:
                with open(log_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "date": date.today().isoformat(),
            "trades": [],
            "summary": {
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
            },
        }

    def _save_trade_log(self, log: dict[str, Any]) -> None:
        """Save trade log to file."""
        log_path = self._get_trade_log_path()
        try:
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)
            logger.debug(f"[SCALP] Trade log saved to {log_path}")
        except IOError as e:
            logger.error(f"[SCALP] Failed to save trade log: {e}")

    def _log_trade_entry(self, position: ScalpPosition, signal: "ScalpSignal") -> None:
        """Log a trade entry to the daily trade file."""
        log = self._load_trade_log()

        # Determine market time category (useful for analysis)
        entry_time = position.entry_time
        hour = entry_time.hour
        minute = entry_time.minute
        minutes_from_open = (hour - 9) * 60 + (minute - 30)  # Market opens 9:30 ET
        if minutes_from_open < 30:
            market_period = "open"  # First 30 min
        elif minutes_from_open > 360:  # 6 hours = 15:30
            market_period = "close"  # Last 30 min
        else:
            market_period = "mid"  # Middle of day

        entry = {
            # Identification
            "signal_id": position.signal_id,
            "entry_time": entry_time.isoformat(),
            "market_period": market_period,  # open/mid/close - valuable for analysis
            # Position details
            "symbol": position.symbol,
            "option_symbol": position.option_symbol,
            "signal_type": position.signal_type,
            "trigger": position.trigger,
            "strike": position.strike,
            "expiry": position.expiry,
            "dte": position.dte,
            "delta": round(position.delta, 3),
            "contracts": position.contracts,
            # Pricing
            "entry_price": round(position.entry_price, 2),
            "bid_price": round(signal.bid_price, 2),
            "ask_price": round(signal.ask_price, 2),
            "spread_pct": round(signal.spread_pct, 2),
            # Underlying state
            "underlying_at_entry": round(position.underlying_at_entry, 2),
            # Momentum metrics (key for threshold tuning)
            "velocity_pct": round(signal.velocity_pct, 3),
            "threshold_used": round(signal.threshold_used, 3),
            "velocity_margin": round(signal.velocity_margin, 3),
            "volume_ratio": round(signal.volume_ratio, 2),
            # Risk params
            "confidence": position.confidence,
            "take_profit_pct": position.take_profit_pct,
            "stop_loss_pct": position.stop_loss_pct,
            # Exit fields will be filled in later
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "pnl_dollars": None,
            "pnl_pct": None,
            "hold_seconds": None,
            "max_gain_pct": None,
            "max_drawdown_pct": None,
        }
        log["trades"].append(entry)
        log["summary"]["total_trades"] = len(log["trades"])
        self._save_trade_log(log)

    def _log_trade_exit(self, result: "ScalpExitResult") -> None:
        """Update trade entry with exit information."""
        log = self._load_trade_log()

        # Find the trade entry by signal_id
        for trade in log["trades"]:
            if trade["signal_id"] == result.position.signal_id:
                trade["exit_time"] = datetime.now(timezone.utc).isoformat()
                trade["exit_price"] = round(result.exit_price, 2)
                trade["exit_reason"] = result.exit_reason
                trade["pnl_dollars"] = round(result.pnl_dollars, 2)
                trade["pnl_pct"] = round(result.pnl_pct, 2)
                trade["hold_seconds"] = result.hold_seconds
                trade["max_gain_pct"] = round(result.position.max_gain_pct, 2)
                trade["max_drawdown_pct"] = round(result.position.max_drawdown_pct, 2)
                break

        # Update summary with comprehensive metrics for analysis
        completed_trades = [t for t in log["trades"] if t["exit_time"] is not None]
        winning_trades = [t for t in completed_trades if t["pnl_dollars"] and t["pnl_dollars"] > 0]
        total_pnl = sum(t["pnl_dollars"] or 0 for t in completed_trades)

        # Basic stats
        log["summary"]["total_trades"] = len(log["trades"])
        log["summary"]["completed_trades"] = len(completed_trades)
        log["summary"]["winning_trades"] = len(winning_trades)
        log["summary"]["total_pnl"] = round(total_pnl, 2)
        log["summary"]["win_rate"] = round(len(winning_trades) / len(completed_trades) * 100, 1) if completed_trades else 0

        if completed_trades:
            # Averages for threshold tuning
            velocity_margins = [t.get("velocity_margin", 0) for t in completed_trades]
            hold_times = [t.get("hold_seconds", 0) for t in completed_trades if t.get("hold_seconds")]
            pnls = [t["pnl_dollars"] for t in completed_trades if t["pnl_dollars"] is not None]

            log["summary"]["avg_velocity_margin"] = round(sum(velocity_margins) / len(velocity_margins), 3) if velocity_margins else 0
            log["summary"]["avg_hold_seconds"] = round(sum(hold_times) / len(hold_times), 1) if hold_times else 0
            log["summary"]["avg_pnl_per_trade"] = round(sum(pnls) / len(pnls), 2) if pnls else 0
            log["summary"]["best_trade"] = round(max(pnls), 2) if pnls else 0
            log["summary"]["worst_trade"] = round(min(pnls), 2) if pnls else 0

            # Win rate by signal type (CALL vs PUT)
            calls = [t for t in completed_trades if t.get("signal_type") == "SCALP_CALL"]
            puts = [t for t in completed_trades if t.get("signal_type") == "SCALP_PUT"]
            call_wins = len([t for t in calls if t["pnl_dollars"] and t["pnl_dollars"] > 0])
            put_wins = len([t for t in puts if t["pnl_dollars"] and t["pnl_dollars"] > 0])

            log["summary"]["call_trades"] = len(calls)
            log["summary"]["call_win_rate"] = round(call_wins / len(calls) * 100, 1) if calls else 0
            log["summary"]["put_trades"] = len(puts)
            log["summary"]["put_win_rate"] = round(put_wins / len(puts) * 100, 1) if puts else 0

            # Win rate by market period
            for period in ["open", "mid", "close"]:
                period_trades = [t for t in completed_trades if t.get("market_period") == period]
                period_wins = len([t for t in period_trades if t["pnl_dollars"] and t["pnl_dollars"] > 0])
                log["summary"][f"{period}_trades"] = len(period_trades)
                log["summary"][f"{period}_win_rate"] = round(period_wins / len(period_trades) * 100, 1) if period_trades else 0

            # Exit reason breakdown
            exit_reasons = {}
            for t in completed_trades:
                reason = t.get("exit_reason", "unknown")
                if reason not in exit_reasons:
                    exit_reasons[reason] = {"count": 0, "wins": 0, "total_pnl": 0}
                exit_reasons[reason]["count"] += 1
                if t["pnl_dollars"] and t["pnl_dollars"] > 0:
                    exit_reasons[reason]["wins"] += 1
                exit_reasons[reason]["total_pnl"] += t["pnl_dollars"] or 0

            log["summary"]["by_exit_reason"] = {
                reason: {
                    "count": data["count"],
                    "win_rate": round(data["wins"] / data["count"] * 100, 1) if data["count"] > 0 else 0,
                    "total_pnl": round(data["total_pnl"], 2),
                }
                for reason, data in exit_reasons.items()
            }

        self._save_trade_log(log)
        logger.info(f"[SCALP] Trade logged: {result.position.option_symbol} P&L=${result.pnl_dollars:+.2f}")

    def _reset_daily_stats_if_needed(self) -> None:
        """Reset daily stats at start of new trading day."""
        today = date.today()
        if self._last_reset_date != today:
            self._daily_pnl = 0.0
            self._daily_starting_equity = None
            self._trading_halted = False
            self._last_reset_date = today
            logger.info("[SCALP] Daily stats reset for new trading day")

    def _get_risk_pct_for_velocity(self, velocity_pct: float) -> float:
        """Get risk percentage based on velocity magnitude.

        Higher velocity = higher conviction = larger position.

        Args:
            velocity_pct: Absolute velocity percentage

        Returns:
            Risk percentage (2-5%)
        """
        abs_vel = abs(velocity_pct)

        # Velocity-scaled risk tiers
        if abs_vel >= 0.80:
            base_risk = 5.0
        elif abs_vel >= 0.60:
            base_risk = 4.0
        elif abs_vel >= 0.40:
            base_risk = 3.0
        else:  # 0.30-0.40%
            base_risk = 2.0

        # Apply consecutive loss reduction
        if self._consecutive_losses >= self._loss_streak_threshold:
            base_risk = 2.0  # Revert to minimum until a win
            logger.info(
                f"[SCALP] Reduced to {base_risk}% risk due to {self._consecutive_losses} consecutive losses"
            )

        # Cap at max risk
        return min(base_risk, self._max_risk_pct)

    def _calculate_position_size(
        self,
        velocity_pct: float,
        entry_price: float,
        stop_loss_pct: float,
    ) -> tuple[int, float]:
        """Calculate position size based on velocity and risk management.

        Args:
            velocity_pct: Signal velocity percentage
            entry_price: Option entry price
            stop_loss_pct: Stop loss percentage for the trade

        Returns:
            Tuple of (contracts, risk_pct_used)
        """
        # Get account equity
        try:
            account = self.trader.get_account()
            equity = account.equity

            # Track starting equity for daily P&L
            if self._daily_starting_equity is None:
                self._daily_starting_equity = equity
        except Exception as e:
            logger.warning(f"[SCALP] Failed to get account equity: {e}, using default 1 contract")
            return 1, 2.0

        # Get risk percentage based on velocity
        risk_pct = self._get_risk_pct_for_velocity(velocity_pct)

        # Calculate risk amount in dollars
        risk_amount = equity * (risk_pct / 100)

        # Calculate risk per contract: entry * stop_loss_pct * 100 shares
        risk_per_contract = entry_price * (stop_loss_pct / 100) * 100

        if risk_per_contract <= 0:
            logger.warning("[SCALP] Invalid risk per contract, using 1 contract")
            return 1, risk_pct

        # Calculate contracts
        contracts = int(risk_amount / risk_per_contract)

        # Enforce minimum and maximum
        contracts = max(1, min(contracts, 100))  # Min 1, max 100

        return contracts, risk_pct

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been hit.

        Returns:
            True if trading should continue, False if halted
        """
        if self._trading_halted:
            return False

        if self._daily_starting_equity is None:
            return True

        try:
            account = self.trader.get_account()
            current_equity = account.equity
            daily_pnl_pct = ((current_equity - self._daily_starting_equity) / self._daily_starting_equity) * 100

            if daily_pnl_pct <= -self._daily_loss_limit_pct:
                self._trading_halted = True
                logger.warning(
                    f"[SCALP] DAILY LOSS LIMIT HIT: {daily_pnl_pct:.1f}% "
                    f"(limit: -{self._daily_loss_limit_pct}%). Trading halted for today."
                )
                return False
        except Exception as e:
            logger.warning(f"[SCALP] Failed to check daily P&L: {e}")

        return True

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

        # Reset daily stats if new day
        self._reset_daily_stats_if_needed()

        # Check daily loss limit
        if not self._check_daily_loss_limit():
            return ScalpExecutionResult(
                success=False,
                signal=signal,
                error="Daily loss limit reached - trading halted",
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

        # Use velocity-scaled position sizing
        contracts, risk_pct = self._calculate_position_size(
            velocity_pct=signal.velocity_pct,
            entry_price=signal.ask_price,
            stop_loss_pct=signal.stop_loss_pct,
        )

        # Log position sizing decision
        logger.info(
            f"[SCALP] Signal velocity={abs(signal.velocity_pct):.2f}%, "
            f"using {risk_pct:.0f}% risk ({contracts} contracts)"
        )

        # Use ask price for buys (with optional limit offset)
        if self.config.use_limit_orders:
            mid = (signal.bid_price + signal.ask_price) / 2
            # Round to 2 decimal places - Alpaca requires this for options
            limit_price = round(mid * (1 + self.config.limit_offset_pct / 100), 2)
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

        # Determine entry velocity direction from signal type
        # SCALP_CALL = entered on upward momentum, SCALP_PUT = entered on downward momentum
        entry_direction: Literal["up", "down"] = (
            "up" if signal.signal_type == "SCALP_CALL" else "down"
        )

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
            # Handle None max_hold_minutes (means no time limit - use 24 hours as "unlimited")
            max_hold_seconds=(signal.max_hold_minutes * 60) if signal.max_hold_minutes else 86400,
            entry_velocity_direction=entry_direction,
            current_price=fill_price,
            entry_order_id=order_result.order_id,
        )

        # Track position
        self._positions[signal.id] = position
        self._total_trades += 1

        # Log trade entry to daily file
        self._log_trade_entry(position, signal)

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

            # Get current velocity direction for momentum reversal check
            current_velocity_direction: Literal["up", "down", "flat"] | None = None
            tracker = self.velocity_trackers.get(position.symbol)
            if tracker:
                velocity = tracker.get_velocity(self.momentum_window_seconds)
                if velocity:
                    current_velocity_direction = velocity.direction

            # Update and check exit
            exit_reason = position.update_price(
                current_price,
                current_velocity_direction=current_velocity_direction,
            )

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
            # Reset consecutive losses on a win
            if self._consecutive_losses > 0:
                logger.info(f"[SCALP] Win breaks {self._consecutive_losses}-loss streak, resetting to full sizing")
            self._consecutive_losses = 0
        else:
            # Track consecutive losses
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._loss_streak_threshold:
                logger.warning(
                    f"[SCALP] {self._consecutive_losses} consecutive losses - "
                    f"reverting to 2% sizing until a win"
                )

        self._total_pnl += pnl_dollars
        self._daily_pnl += pnl_dollars

        # Log daily P&L status
        if self._daily_starting_equity:
            daily_pnl_pct = (self._daily_pnl / self._daily_starting_equity) * 100
            logger.info(f"[SCALP] Daily P&L: ${self._daily_pnl:+.2f} ({daily_pnl_pct:+.2f}%)")

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

        # Log trade exit to daily file
        self._log_trade_exit(result)

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
        daily_pnl_pct = 0.0
        if self._daily_starting_equity and self._daily_starting_equity > 0:
            daily_pnl_pct = (self._daily_pnl / self._daily_starting_equity) * 100

        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._total_trades - self._winning_trades,
            "win_rate": round(win_rate, 3),
            "total_pnl": round(self._total_pnl, 2),
            "open_positions": len(self._positions),
            # Daily tracking
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "consecutive_losses": self._consecutive_losses,
            "trading_halted": self._trading_halted,
        }
