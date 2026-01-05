"""Alpaca Trading Client for automated order execution.

Handles order submission, position management, and trade lifecycle for
paper trading. Uses Alpaca's official SDK for reliability.

SAFETY: Only paper trading is supported. Live trading requires explicit
confirmation and additional safeguards.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, time
from typing import Any, Literal
from enum import Enum

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    AssetClass,
    QueryOrderStatus,
)
from alpaca.common.exceptions import APIError

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    """Trading mode enumeration."""
    BACKTEST = "backtest"  # No Alpaca connection, use SQLite
    PAPER = "paper"        # Alpaca paper trading
    LIVE = "live"          # Alpaca live trading (requires extra confirmation)


@dataclass
class OrderResult:
    """Result of an order submission."""
    success: bool
    order_id: str | None
    symbol: str
    qty: int
    side: str
    status: str
    filled_qty: int = 0
    filled_avg_price: float | None = None
    error: str | None = None
    submitted_at: str | None = None
    filled_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "orderId": self.order_id,
            "symbol": self.symbol,
            "qty": self.qty,
            "side": self.side,
            "status": self.status,
            "filledQty": self.filled_qty,
            "filledAvgPrice": self.filled_avg_price,
            "error": self.error,
            "submittedAt": self.submitted_at,
            "filledAt": self.filled_at,
        }


@dataclass
class PositionInfo:
    """Information about an open position."""
    symbol: str
    qty: float
    side: str  # "long" or "short"
    avg_entry_price: float
    market_value: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_percent: float
    asset_class: str

    @property
    def is_option(self) -> bool:
        return self.asset_class == "us_option"

    def to_dict(self) -> dict[str, Any]:
        """Return snake_case dict to match frontend AlpacaPosition type."""
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "side": self.side,
            "avg_entry_price": self.avg_entry_price,
            "market_value": self.market_value,
            "cost_basis": self.avg_entry_price * self.qty * 100,  # Options: price * qty * 100
            "current_price": self.current_price,
            "unrealized_pl": self.unrealized_pl,
            "unrealized_plpc": self.unrealized_pl_percent / 100,  # Convert % to decimal
            "is_option": self.is_option,
        }


@dataclass
class AccountInfo:
    """Alpaca account information."""
    buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    day_trade_count: int
    pattern_day_trader: bool
    trading_blocked: bool

    def to_dict(self) -> dict[str, Any]:
        """Return snake_case dict to match frontend AlpacaAccount type."""
        return {
            "buying_power": self.buying_power,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "equity": self.equity,
            "positions_count": 0,  # Positions tracked separately
            "day_trade_count": self.day_trade_count,
            "pattern_day_trader": self.pattern_day_trader,
        }


class AlpacaTrader:
    """Alpaca trading client for automated order execution.

    Features:
    - Paper trading only by default (live requires explicit enable)
    - Order submission (market/limit)
    - Position monitoring and closing
    - Account info and buying power checks
    - Safety checks (market hours, position limits, etc.)

    Usage:
        trader = AlpacaTrader(api_key, secret_key, paper=True)

        # Check account
        account = trader.get_account()
        print(f"Buying power: ${account.buying_power}")

        # Submit order
        result = trader.submit_option_order(
            occ_symbol="TSLA240119C00250000",
            qty=1,
            side="buy"
        )

        # Get positions
        positions = trader.get_positions()

        # Close position
        result = trader.close_position("TSLA240119C00250000")
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        auto_execute: bool = False,
    ):
        """Initialize the Alpaca trading client.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: If True, use paper trading (REQUIRED for safety)
            auto_execute: If True, enable automatic order execution
        """
        if not paper:
            raise ValueError(
                "Live trading is disabled for safety. "
                "Only paper=True is supported."
            )

        self.paper = paper
        self.auto_execute = auto_execute
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

        # Safety limits
        self.max_positions = 3
        self.max_position_pct = 0.15  # 15% of portfolio per position
        self.daily_loss_limit_pct = 0.15  # Stop trading if down 15%
        self._daily_starting_equity: float | None = None

        logger.info(
            f"AlpacaTrader initialized (paper={paper}, auto_execute={auto_execute})"
        )

    def get_account(self) -> AccountInfo:
        """Get account information.

        Returns:
            AccountInfo with buying power, equity, etc.
        """
        try:
            account = self._client.get_account()

            # Track daily starting equity for loss limit
            if self._daily_starting_equity is None:
                self._daily_starting_equity = float(account.equity)

            return AccountInfo(
                buying_power=float(account.buying_power),
                cash=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                equity=float(account.equity),
                day_trade_count=int(account.daytrade_count),
                pattern_day_trader=account.pattern_day_trader,
                trading_blocked=account.trading_blocked,
            )
        except APIError as e:
            logger.error(f"Failed to get account: {e}")
            raise

    def get_positions(self) -> list[PositionInfo]:
        """Get all open positions.

        Returns:
            List of PositionInfo objects
        """
        try:
            positions = self._client.get_all_positions()
            return [
                PositionInfo(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    side=p.side.value if hasattr(p.side, 'value') else str(p.side),
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    current_price=float(p.current_price),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_pl_percent=float(p.unrealized_plpc) * 100,
                    asset_class=p.asset_class.value if hasattr(p.asset_class, 'value') else str(p.asset_class),
                )
                for p in positions
            ]
        except APIError as e:
            logger.error(f"Failed to get positions: {e}")
            raise

    def get_option_positions(self) -> list[PositionInfo]:
        """Get only option positions.

        Returns:
            List of option PositionInfo objects
        """
        return [p for p in self.get_positions() if p.is_option]

    def get_position(self, symbol: str) -> PositionInfo | None:
        """Get a specific position by symbol.

        Args:
            symbol: The symbol (OCC format for options)

        Returns:
            PositionInfo or None if not found
        """
        try:
            p = self._client.get_open_position(symbol)
            return PositionInfo(
                symbol=p.symbol,
                qty=float(p.qty),
                side=p.side.value if hasattr(p.side, 'value') else str(p.side),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                current_price=float(p.current_price),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_pl_percent=float(p.unrealized_plpc) * 100,
                asset_class=p.asset_class.value if hasattr(p.asset_class, 'value') else str(p.asset_class),
            )
        except APIError as e:
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Failed to get position {symbol}: {e}")
            raise

    def submit_option_order(
        self,
        occ_symbol: str,
        qty: int,
        side: Literal["buy", "sell"],
        order_type: Literal["market", "limit"] = "market",
        limit_price: float | None = None,
    ) -> OrderResult:
        """Submit an option order.

        Args:
            occ_symbol: OCC-format symbol (e.g., "TSLA240119C00250000")
            qty: Number of contracts
            side: "buy" or "sell"
            order_type: "market" or "limit"
            limit_price: Required if order_type is "limit"

        Returns:
            OrderResult with order details
        """
        # Safety checks
        if not self._can_trade():
            return OrderResult(
                success=False,
                order_id=None,
                symbol=occ_symbol,
                qty=qty,
                side=side,
                status="rejected",
                error="Trading disabled (safety check failed)",
            )

        # Check position limits
        if side == "buy":
            current_positions = len(self.get_option_positions())
            if current_positions >= self.max_positions:
                return OrderResult(
                    success=False,
                    order_id=None,
                    symbol=occ_symbol,
                    qty=qty,
                    side=side,
                    status="rejected",
                    error=f"Max positions reached ({current_positions}/{self.max_positions})",
                )

        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            if order_type == "market":
                request = MarketOrderRequest(
                    symbol=occ_symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                if limit_price is None:
                    return OrderResult(
                        success=False,
                        order_id=None,
                        symbol=occ_symbol,
                        qty=qty,
                        side=side,
                        status="rejected",
                        error="Limit price required for limit orders",
                    )
                request = LimitOrderRequest(
                    symbol=occ_symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )

            order = self._client.submit_order(request)

            logger.info(
                f"[ORDER] Submitted {side.upper()} {qty}x {occ_symbol} "
                f"@ {order_type} - Order ID: {order.id}"
            )

            return OrderResult(
                success=True,
                order_id=str(order.id),
                symbol=occ_symbol,
                qty=qty,
                side=side,
                status=order.status.value if hasattr(order.status, 'value') else str(order.status),
                filled_qty=int(order.filled_qty) if order.filled_qty else 0,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                submitted_at=order.submitted_at.isoformat() if order.submitted_at else None,
                filled_at=order.filled_at.isoformat() if order.filled_at else None,
            )

        except APIError as e:
            logger.error(f"[ORDER] Failed to submit {side} {qty}x {occ_symbol}: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                symbol=occ_symbol,
                qty=qty,
                side=side,
                status="failed",
                error=str(e),
            )

    def close_position(self, symbol: str) -> OrderResult:
        """Close an open position.

        Args:
            symbol: The symbol to close (OCC format for options)

        Returns:
            OrderResult with close order details
        """
        try:
            # Get current position to determine qty
            position = self.get_position(symbol)
            if not position:
                return OrderResult(
                    success=False,
                    order_id=None,
                    symbol=symbol,
                    qty=0,
                    side="sell",
                    status="rejected",
                    error="Position not found",
                )

            # Close the position
            order = self._client.close_position(symbol)

            logger.info(
                f"[CLOSE] Closing position {symbol} - Order ID: {order.id}"
            )

            return OrderResult(
                success=True,
                order_id=str(order.id),
                symbol=symbol,
                qty=int(position.qty),
                side="sell",
                status=order.status.value if hasattr(order.status, 'value') else str(order.status),
                filled_qty=int(order.filled_qty) if order.filled_qty else 0,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                submitted_at=order.submitted_at.isoformat() if order.submitted_at else None,
            )

        except APIError as e:
            logger.error(f"[CLOSE] Failed to close {symbol}: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                symbol=symbol,
                qty=0,
                side="sell",
                status="failed",
                error=str(e),
            )

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Get order status by ID.

        Args:
            order_id: The order ID

        Returns:
            Order details dict or None if not found
        """
        try:
            order = self._client.get_order_by_id(order_id)
            return {
                "id": str(order.id),
                "symbol": order.symbol,
                "qty": int(order.qty),
                "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "filledQty": int(order.filled_qty) if order.filled_qty else 0,
                "filledAvgPrice": float(order.filled_avg_price) if order.filled_avg_price else None,
                "submittedAt": order.submitted_at.isoformat() if order.submitted_at else None,
                "filledAt": order.filled_at.isoformat() if order.filled_at else None,
            }
        except APIError as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def get_recent_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent orders.

        Args:
            limit: Maximum number of orders to return

        Returns:
            List of order dicts
        """
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=limit,
            )
            orders = self._client.get_orders(request)
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "qty": int(o.qty),
                    "side": o.side.value if hasattr(o.side, 'value') else str(o.side),
                    "status": o.status.value if hasattr(o.status, 'value') else str(o.status),
                    "filledQty": int(o.filled_qty) if o.filled_qty else 0,
                    "filledAvgPrice": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "submittedAt": o.submitted_at.isoformat() if o.submitted_at else None,
                    "filledAt": o.filled_at.isoformat() if o.filled_at else None,
                }
                for o in orders
            ]
        except APIError as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def _can_trade(self) -> bool:
        """Check if trading is allowed based on safety rules.

        Returns:
            True if trading is allowed
        """
        # Check if auto-execute is enabled
        if not self.auto_execute:
            logger.warning("Auto-execute is disabled")
            return False

        # Check market hours (simplified - options trade 9:30 AM - 4:00 PM ET)
        now = datetime.now(timezone.utc)
        # Convert to ET (UTC-5 or UTC-4 depending on DST)
        # Simplified: just check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            logger.warning("Market closed (weekend)")
            return False

        # Check daily loss limit
        if self._daily_starting_equity:
            try:
                account = self.get_account()
                daily_pnl_pct = (account.equity - self._daily_starting_equity) / self._daily_starting_equity
                if daily_pnl_pct <= -self.daily_loss_limit_pct:
                    logger.warning(
                        f"Daily loss limit hit: {daily_pnl_pct*100:.1f}% "
                        f"(limit: -{self.daily_loss_limit_pct*100:.0f}%)"
                    )
                    return False
            except Exception as e:
                logger.warning(f"Failed to check daily P&L: {e}")

        # Check if account is blocked
        try:
            account = self.get_account()
            if account.trading_blocked:
                logger.warning("Trading is blocked on this account")
                return False
        except Exception as e:
            logger.warning(f"Failed to check account status: {e}")
            return False

        return True

    def is_market_open(self) -> bool:
        """Check if the market is currently open.

        Returns:
            True if market is open
        """
        try:
            clock = self._client.get_clock()
            return clock.is_open
        except APIError as e:
            logger.error(f"Failed to get market clock: {e}")
            return False

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        self._daily_starting_equity = None
        logger.info("Daily tracking reset")


def build_occ_symbol(
    underlying: str,
    expiry: str,
    strike: float,
    right: Literal["C", "P"],
) -> str:
    """Build OCC-format option symbol.

    Args:
        underlying: Underlying symbol (e.g., "TSLA")
        expiry: Expiration date in ISO format (e.g., "2024-01-19")
        strike: Strike price
        right: "C" for call, "P" for put

    Returns:
        OCC-format symbol (e.g., "TSLA240119C00250000")
    """
    # Parse expiry date
    from datetime import datetime
    exp_date = datetime.strptime(expiry, "%Y-%m-%d")

    # Format: SYMBOL + YYMMDD + C/P + 8-digit strike (price * 1000, zero-padded)
    date_str = exp_date.strftime("%y%m%d")
    strike_str = f"{int(strike * 1000):08d}"

    return f"{underlying}{date_str}{right}{strike_str}"


def parse_occ_symbol(occ_symbol: str) -> dict[str, Any]:
    """Parse OCC-format option symbol.

    Args:
        occ_symbol: OCC-format symbol (e.g., "TSLA240119C00250000")

    Returns:
        Dict with underlying, expiry, strike, right
    """
    # Find where the date portion starts (first digit)
    date_start = 0
    for i, c in enumerate(occ_symbol):
        if c.isdigit():
            date_start = i
            break

    underlying = occ_symbol[:date_start]
    date_str = occ_symbol[date_start:date_start + 6]
    right = occ_symbol[date_start + 6]
    strike_str = occ_symbol[date_start + 7:]

    # Parse date
    from datetime import datetime
    exp_date = datetime.strptime(date_str, "%y%m%d")

    # Parse strike (stored as price * 1000)
    strike = int(strike_str) / 1000

    return {
        "underlying": underlying,
        "expiry": exp_date.strftime("%Y-%m-%d"),
        "strike": strike,
        "right": right,
    }
