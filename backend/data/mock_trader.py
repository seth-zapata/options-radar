"""Mock Alpaca Trader for simulation mode testing.

Simulates the full trading lifecycle without connecting to Alpaca.
Useful for testing auto-execution pipeline before market hours.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from backend.data.alpaca_trader import (
    OrderResult,
    PositionInfo,
    AccountInfo,
    build_occ_symbol,
    parse_occ_symbol,
)

logger = logging.getLogger(__name__)


@dataclass
class MockPosition:
    """Simulated position for testing."""
    symbol: str
    qty: int
    entry_price: float
    current_price: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def market_value(self) -> float:
        return self.current_price * self.qty * 100

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.qty * 100

    @property
    def unrealized_pl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pl_percent(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pl / self.cost_basis) * 100


@dataclass
class MockOrder:
    """Simulated order for testing."""
    id: str
    symbol: str
    qty: int
    side: str
    status: str
    filled_price: float | None = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: datetime | None = None


class MockAlpacaTrader:
    """Mock Alpaca trader for simulation testing.

    Simulates:
    - Account with configurable starting balance
    - Order submission with instant fills
    - Position tracking with simulated price movements
    - Full lifecycle matching real AlpacaTrader interface

    Usage:
        trader = MockAlpacaTrader(starting_balance=100000.0)

        # Submit order
        result = trader.submit_option_order("TSLA250117C00250000", 2, "buy")

        # Simulate price movement
        trader.simulate_price_movement()

        # Get positions
        positions = trader.get_positions()
    """

    def __init__(
        self,
        starting_balance: float = 100000.0,
        auto_execute: bool = True,
        simulation_speed: float = 1.0,
    ):
        """Initialize mock trader.

        Args:
            starting_balance: Starting account balance
            auto_execute: Always True for simulation
            simulation_speed: Speed multiplier for price movements
        """
        self.starting_balance = starting_balance
        self.cash = starting_balance
        self.auto_execute = auto_execute
        self.simulation_speed = simulation_speed
        self.paper = True  # Always True for simulation mode

        self._positions: dict[str, MockPosition] = {}
        self._orders: list[MockOrder] = []
        self._closed_pnl = 0.0

        # Safety limits (same as real trader)
        self.max_positions = 3
        self.max_position_pct = 0.15
        self.daily_loss_limit_pct = 0.15

        # Price simulation settings
        self._price_drift = 0.001  # Slight upward bias
        self._price_volatility = 0.03  # 3% max move per tick

        logger.info(
            f"MockAlpacaTrader initialized (balance=${starting_balance:,.0f}, "
            f"speed={simulation_speed}x)"
        )

    def get_account(self) -> AccountInfo:
        """Get simulated account information."""
        portfolio_value = self.cash + sum(
            p.market_value for p in self._positions.values()
        )
        equity = portfolio_value + self._closed_pnl

        return AccountInfo(
            buying_power=self.cash,
            cash=self.cash,
            portfolio_value=portfolio_value,
            equity=equity,
            day_trade_count=0,
            pattern_day_trader=False,
            trading_blocked=False,
        )

    def get_positions(self) -> list[PositionInfo]:
        """Get all open positions."""
        return [
            PositionInfo(
                symbol=p.symbol,
                qty=float(p.qty),
                side="long",
                avg_entry_price=p.entry_price,
                market_value=p.market_value,
                current_price=p.current_price,
                unrealized_pl=p.unrealized_pl,
                unrealized_pl_percent=p.unrealized_pl_percent,
                asset_class="us_option",
            )
            for p in self._positions.values()
        ]

    def get_option_positions(self) -> list[PositionInfo]:
        """Get only option positions (all positions in simulation)."""
        return self.get_positions()

    def get_position(self, symbol: str) -> PositionInfo | None:
        """Get a specific position by symbol."""
        pos = self._positions.get(symbol)
        if not pos:
            return None

        return PositionInfo(
            symbol=pos.symbol,
            qty=float(pos.qty),
            side="long",
            avg_entry_price=pos.entry_price,
            market_value=pos.market_value,
            current_price=pos.current_price,
            unrealized_pl=pos.unrealized_pl,
            unrealized_pl_percent=pos.unrealized_pl_percent,
            asset_class="us_option",
        )

    def submit_option_order(
        self,
        occ_symbol: str,
        qty: int,
        side: Literal["buy", "sell"],
        order_type: Literal["market", "limit"] = "market",
        limit_price: float | None = None,
    ) -> OrderResult:
        """Submit a simulated option order.

        Orders are filled instantly at a simulated price.
        """
        # Check position limits
        if side == "buy":
            if len(self._positions) >= self.max_positions:
                return OrderResult(
                    success=False,
                    order_id=None,
                    symbol=occ_symbol,
                    qty=qty,
                    side=side,
                    status="rejected",
                    error=f"Max positions reached ({len(self._positions)}/{self.max_positions})",
                )

        # Generate fill price
        fill_price = self._get_simulated_fill_price(occ_symbol, limit_price)
        cost = fill_price * qty * 100

        # Check buying power
        if side == "buy" and cost > self.cash:
            return OrderResult(
                success=False,
                order_id=None,
                symbol=occ_symbol,
                qty=qty,
                side=side,
                status="rejected",
                error=f"Insufficient buying power (need ${cost:,.0f}, have ${self.cash:,.0f})",
            )

        # Create order
        order_id = str(uuid4())[:8]
        now = datetime.now(timezone.utc)

        order = MockOrder(
            id=order_id,
            symbol=occ_symbol,
            qty=qty,
            side=side,
            status="filled",
            filled_price=fill_price,
            submitted_at=now,
            filled_at=now,
        )
        self._orders.append(order)

        # Update state
        if side == "buy":
            self.cash -= cost
            self._positions[occ_symbol] = MockPosition(
                symbol=occ_symbol,
                qty=qty,
                entry_price=fill_price,
                current_price=fill_price,
                opened_at=now,
            )
            logger.info(
                f"[MOCK ORDER] BUY {qty}x {occ_symbol} @ ${fill_price:.2f} "
                f"(cost: ${cost:,.0f})"
            )
        else:
            # Selling to close
            if occ_symbol in self._positions:
                pos = self._positions.pop(occ_symbol)
                proceeds = fill_price * qty * 100
                self.cash += proceeds
                pnl = proceeds - pos.cost_basis
                self._closed_pnl += pnl
                logger.info(
                    f"[MOCK ORDER] SELL {qty}x {occ_symbol} @ ${fill_price:.2f} "
                    f"(P&L: ${pnl:,.0f})"
                )

        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=occ_symbol,
            qty=qty,
            side=side,
            status="filled",
            filled_qty=qty,
            filled_avg_price=fill_price,
            submitted_at=now.isoformat(),
            filled_at=now.isoformat(),
        )

    def close_position(self, symbol: str) -> OrderResult:
        """Close an open position."""
        pos = self._positions.get(symbol)
        if not pos:
            return OrderResult(
                success=False,
                order_id=None,
                symbol=symbol,
                qty=0,
                side="sell",
                status="rejected",
                error="Position not found",
            )

        return self.submit_option_order(symbol, pos.qty, "sell")

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Get order by ID."""
        for order in self._orders:
            if order.id == order_id:
                return {
                    "id": order.id,
                    "symbol": order.symbol,
                    "qty": order.qty,
                    "side": order.side,
                    "status": order.status,
                    "filledQty": order.qty if order.status == "filled" else 0,
                    "filledAvgPrice": order.filled_price,
                    "submittedAt": order.submitted_at.isoformat(),
                    "filledAt": order.filled_at.isoformat() if order.filled_at else None,
                }
        return None

    def get_recent_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent orders."""
        return [
            {
                "id": o.id,
                "symbol": o.symbol,
                "qty": o.qty,
                "side": o.side,
                "status": o.status,
                "filledQty": o.qty if o.status == "filled" else 0,
                "filledAvgPrice": o.filled_price,
                "submittedAt": o.submitted_at.isoformat(),
                "filledAt": o.filled_at.isoformat() if o.filled_at else None,
            }
            for o in reversed(self._orders[-limit:])
        ]

    def is_market_open(self) -> bool:
        """Always return True for simulation."""
        return True

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking."""
        self._closed_pnl = 0.0
        logger.info("Daily tracking reset (simulation)")

    def simulate_price_movement(self) -> None:
        """Simulate price movements for all positions.

        Called periodically to update position prices with random walk.
        """
        for symbol, pos in self._positions.items():
            # Random walk with slight drift
            change = random.gauss(self._price_drift, self._price_volatility)
            new_price = pos.current_price * (1 + change)

            # Ensure price stays positive and reasonable
            new_price = max(0.05, new_price)  # Min $0.05
            pos.current_price = new_price

    def _get_simulated_fill_price(
        self,
        occ_symbol: str,
        limit_price: float | None = None,
    ) -> float:
        """Generate a simulated fill price.

        If position exists, use current price.
        Otherwise, generate based on the option specs.
        """
        # If closing existing position, use current price
        if occ_symbol in self._positions:
            return self._positions[occ_symbol].current_price

        # If limit price specified, use it
        if limit_price is not None:
            return limit_price

        # Parse OCC symbol to determine reasonable price
        try:
            parsed = parse_occ_symbol(occ_symbol)
            underlying = parsed["underlying"]
            strike = parsed["strike"]

            # Simulate based on underlying price (rough approximation)
            # Assume underlying is around $250 for TSLA
            underlying_price = 250.0 if underlying == "TSLA" else 100.0

            # ATM options are roughly 3-5% of underlying
            atm_premium = underlying_price * 0.04

            # Adjust for moneyness
            moneyness = strike / underlying_price
            if moneyness < 0.95:  # ITM
                premium = atm_premium * 1.5
            elif moneyness > 1.05:  # OTM
                premium = atm_premium * 0.5
            else:
                premium = atm_premium

            # Add some randomness
            premium *= random.uniform(0.9, 1.1)

            return round(premium, 2)
        except Exception:
            # Default to a reasonable price
            return random.uniform(3.0, 8.0)

    def update_position_price(self, symbol: str, new_price: float) -> None:
        """Manually update a position's price (for testing)."""
        if symbol in self._positions:
            self._positions[symbol].current_price = new_price

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get summary of current portfolio state."""
        account = self.get_account()
        positions = self.get_positions()

        total_unrealized = sum(p.unrealized_pl for p in positions)

        return {
            "cash": self.cash,
            "equity": account.equity,
            "buyingPower": account.buying_power,
            "openPositions": len(positions),
            "unrealizedPl": total_unrealized,
            "closedPl": self._closed_pnl,
            "totalPl": total_unrealized + self._closed_pnl,
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "entryPrice": p.avg_entry_price,
                    "currentPrice": p.current_price,
                    "pl": p.unrealized_pl,
                    "plPercent": p.unrealized_pl_percent,
                }
                for p in positions
            ],
        }
