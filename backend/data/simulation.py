"""Simulation mode for testing auto-execution.

Provides accelerated price movements and regime cycling for testing
the full trading pipeline without market hours.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from itertools import cycle
from typing import TYPE_CHECKING, Callable, Awaitable

if TYPE_CHECKING:
    from backend.data.mock_trader import MockAlpacaTrader
    from backend.engine.regime_detector import RegimeDetector
    from backend.engine.regime_signals import RegimeSignalGenerator

logger = logging.getLogger(__name__)


class SimulationController:
    """Controls the simulation loop for testing auto-execution.

    Features:
    - Cycles through different regime states
    - Simulates price movements that trigger entries/exits
    - Updates mock trader positions with P&L changes
    - Runs accelerated (faster than real-time)

    Usage:
        controller = SimulationController(
            mock_trader=trader,
            regime_detector=detector,
            signal_generator=signal_gen,
            on_regime_change=handle_regime,
            on_signal=handle_signal,
            speed=10.0,  # 10x faster
        )

        await controller.start()
    """

    def __init__(
        self,
        mock_trader: MockAlpacaTrader,
        regime_detector: RegimeDetector | None = None,
        signal_generator: RegimeSignalGenerator | None = None,
        on_regime_change: Callable[[str, str, float], Awaitable[None]] | None = None,
        on_price_update: Callable[[str, float, float, float], Awaitable[None]] | None = None,
        speed: float = 5.0,
    ):
        """Initialize simulation controller.

        Args:
            mock_trader: MockAlpacaTrader instance
            regime_detector: Optional RegimeDetector to update
            signal_generator: Optional signal generator to trigger
            on_regime_change: Callback for regime changes (symbol, regime_type, sentiment)
            on_price_update: Callback for price updates (symbol, price, high, low)
            speed: Simulation speed multiplier (1.0 = real-time, 10.0 = 10x faster)
        """
        self.mock_trader = mock_trader
        self.regime_detector = regime_detector
        self.signal_generator = signal_generator
        self.on_regime_change = on_regime_change
        self.on_price_update = on_price_update
        self.speed = speed

        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Simulation state
        self._prices: dict[str, dict] = {}  # symbol -> {price, high, low}
        self._current_regime: dict = {"type": "neutral", "sentiment": 0.0, "duration": 0}

        # Regime cycle configuration
        self._regime_cycle = [
            {"type": "strong_bullish", "sentiment": 0.15, "duration": 60},  # 60 ticks
            {"type": "moderate_bullish", "sentiment": 0.08, "duration": 40},
            {"type": "neutral", "sentiment": 0.00, "duration": 20},
            {"type": "moderate_bearish", "sentiment": -0.10, "duration": 40},
            {"type": "strong_bearish", "sentiment": -0.18, "duration": 60},
            {"type": "moderate_bearish", "sentiment": -0.08, "duration": 30},
            {"type": "neutral", "sentiment": 0.02, "duration": 20},
            {"type": "moderate_bullish", "sentiment": 0.09, "duration": 30},
        ]

        logger.info(f"SimulationController initialized (speed={speed}x)")

    def _init_price(self, symbol: str, base_price: float) -> None:
        """Initialize price tracking for a symbol."""
        self._prices[symbol] = {
            "price": base_price,
            "high": base_price,
            "low": base_price,
            "open": base_price,
        }

    async def start(self, symbols: list[str] | None = None) -> None:
        """Start the simulation loops.

        Args:
            symbols: Symbols to simulate (defaults to TSLA)
        """
        if self._running:
            logger.warning("Simulation already running")
            return

        symbols = symbols or ["TSLA"]

        # Initialize prices
        base_prices = {
            "TSLA": 250.0,
            "NVDA": 140.0,
            "PLTR": 75.0,
            "COIN": 280.0,
        }
        for symbol in symbols:
            self._init_price(symbol, base_prices.get(symbol, 100.0))

        self._running = True

        # Start simulation tasks
        self._tasks = [
            asyncio.create_task(self._price_loop(symbols)),
            asyncio.create_task(self._regime_loop(symbols)),
            asyncio.create_task(self._position_update_loop()),
        ]

        logger.info(
            f"Simulation started: {len(symbols)} symbols, "
            f"{self.speed}x speed"
        )

    async def stop(self) -> None:
        """Stop the simulation loops."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks = []
        logger.info("Simulation stopped")

    async def _price_loop(self, symbols: list[str]) -> None:
        """Generate price movements.

        Simulates:
        - Random walk with trend following current regime
        - Pullbacks during bullish regimes (entry triggers)
        - Bounces during bearish regimes (entry triggers)
        """
        tick_interval = 2.0 / self.speed  # Base 2-second ticks

        while self._running:
            try:
                await asyncio.sleep(tick_interval)

                for symbol in symbols:
                    if symbol not in self._prices:
                        continue

                    state = self._prices[symbol]
                    current_price = state["price"]

                    # Get current regime for trend direction
                    regime = self._get_current_regime()
                    trend = 0.0
                    if regime["type"].startswith("bullish") or regime["type"] == "strong_bullish":
                        trend = 0.001  # Slight upward bias
                    elif regime["type"].startswith("bearish") or regime["type"] == "strong_bearish":
                        trend = -0.001  # Slight downward bias

                    # Random walk with trend
                    volatility = 0.008  # 0.8% per tick
                    change = random.gauss(trend, volatility)

                    # Occasionally create larger moves for entries
                    if random.random() < 0.1:  # 10% chance of larger move
                        if regime["type"].startswith("bullish") or regime["type"] == "strong_bullish":
                            # Pullback during bullish (entry trigger)
                            change = -random.uniform(0.015, 0.025)
                        else:
                            # Bounce during bearish (entry trigger)
                            change = random.uniform(0.015, 0.025)

                    new_price = current_price * (1 + change)
                    new_price = max(1.0, new_price)  # Floor at $1

                    # Update state
                    state["price"] = new_price
                    state["high"] = max(state["high"], new_price)
                    state["low"] = min(state["low"], new_price)

                    # Callback
                    if self.on_price_update:
                        try:
                            await self.on_price_update(
                                symbol,
                                new_price,
                                state["high"],
                                state["low"],
                            )
                        except Exception as e:
                            logger.error(f"Price callback error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Price loop error: {e}")

    async def _regime_loop(self, symbols: list[str]) -> None:
        """Cycle through regime states.

        Each regime has a duration (in ticks), then transitions to the next.
        """
        tick_interval = 1.0 / self.speed
        tick_count = 0
        regime_start_tick = 0

        regime_iter = cycle(self._regime_cycle)
        current_regime = next(regime_iter)
        self._current_regime = current_regime  # Store for status endpoint

        # Announce initial regime
        for symbol in symbols:
            if self.on_regime_change:
                try:
                    await self.on_regime_change(
                        symbol,
                        current_regime["type"],
                        current_regime["sentiment"],
                    )
                except Exception as e:
                    logger.error(f"Regime callback error: {e}")

        logger.info(
            f"[SIMULATION] Initial regime: {current_regime['type']} "
            f"(sentiment: {current_regime['sentiment']})"
        )

        while self._running:
            try:
                await asyncio.sleep(tick_interval)
                tick_count += 1

                # Check if regime should change
                ticks_in_regime = tick_count - regime_start_tick
                if ticks_in_regime >= current_regime["duration"]:
                    # Move to next regime
                    current_regime = next(regime_iter)
                    self._current_regime = current_regime  # Update for status endpoint
                    regime_start_tick = tick_count

                    logger.info(
                        f"[SIMULATION] Regime change: {current_regime['type']} "
                        f"(sentiment: {current_regime['sentiment']}, "
                        f"duration: {current_regime['duration']} ticks)"
                    )

                    # Notify
                    for symbol in symbols:
                        if self.on_regime_change:
                            try:
                                await self.on_regime_change(
                                    symbol,
                                    current_regime["type"],
                                    current_regime["sentiment"],
                                )
                            except Exception as e:
                                logger.error(f"Regime callback error: {e}")

                        # Reset daily high/low for fresh pullback/bounce detection
                        if symbol in self._prices:
                            price = self._prices[symbol]["price"]
                            self._prices[symbol]["high"] = price
                            self._prices[symbol]["low"] = price

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Regime loop error: {e}")

    async def _position_update_loop(self) -> None:
        """Update mock trader position prices based on simulation prices."""
        tick_interval = 1.0 / self.speed

        while self._running:
            try:
                await asyncio.sleep(tick_interval)

                # Update all mock positions based on simulated prices
                for pos in self.mock_trader.get_positions():
                    # Try to extract underlying from OCC symbol
                    symbol = pos.symbol
                    underlying = self._extract_underlying(symbol)

                    if underlying in self._prices:
                        # Simulate option price based on underlying move
                        underlying_price = self._prices[underlying]["price"]
                        option_price = self._simulate_option_price(
                            pos.avg_entry_price,
                            pos.current_price,
                            underlying_price,
                        )
                        self.mock_trader.update_position_price(symbol, option_price)

                # Simulate general price movement for all positions
                self.mock_trader.simulate_price_movement()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position update loop error: {e}")

    def _get_current_regime(self) -> dict:
        """Get the current regime."""
        return self._current_regime

    def _extract_underlying(self, occ_symbol: str) -> str:
        """Extract underlying symbol from OCC format."""
        # Find first digit
        for i, c in enumerate(occ_symbol):
            if c.isdigit():
                return occ_symbol[:i]
        return occ_symbol

    def _simulate_option_price(
        self,
        entry_price: float,
        current_price: float,
        underlying_price: float,
    ) -> float:
        """Simulate option price movement based on underlying.

        Uses simplified delta assumption to move option prices.
        """
        # Assume delta of ~0.5 for ATM options
        delta = 0.5

        # Calculate underlying % change from last tick
        # (This is approximate - real options have complex pricing)
        underlying_change_pct = random.uniform(-0.02, 0.025)

        # Option price change = delta * underlying change
        option_change = current_price * delta * underlying_change_pct

        new_price = current_price + option_change
        new_price = max(0.05, new_price)  # Floor at $0.05

        return round(new_price, 2)

    def get_simulation_status(self) -> dict:
        """Get current simulation status."""
        current_regime = self._get_current_regime()
        return {
            "running": self._running,
            "speed": self.speed,
            "currentRegime": current_regime["type"],
            "sentiment": current_regime["sentiment"],
            "prices": {
                symbol: {
                    "price": round(state["price"], 2),
                    "high": round(state["high"], 2),
                    "low": round(state["low"], 2),
                }
                for symbol, state in self._prices.items()
            },
            "mockPortfolio": self.mock_trader.get_portfolio_summary(),
        }


async def run_simulation_test():
    """Quick test of simulation mode."""
    from backend.data.mock_trader import MockAlpacaTrader

    trader = MockAlpacaTrader(starting_balance=100000.0)

    async def on_regime_change(symbol: str, regime_type: str, sentiment: float):
        print(f"[REGIME] {symbol}: {regime_type} (sentiment: {sentiment})")

    async def on_price_update(symbol: str, price: float, high: float, low: float):
        pullback = ((high - price) / high) * 100
        bounce = ((price - low) / low) * 100
        print(f"[PRICE] {symbol}: ${price:.2f} (pullback: {pullback:.1f}%, bounce: {bounce:.1f}%)")

    controller = SimulationController(
        mock_trader=trader,
        on_regime_change=on_regime_change,
        on_price_update=on_price_update,
        speed=10.0,
    )

    print("Starting simulation test (10x speed)...")
    await controller.start(["TSLA"])

    # Run for 30 seconds
    await asyncio.sleep(30)

    print("\nSimulation status:")
    print(controller.get_simulation_status())

    await controller.stop()
    print("Simulation test complete")


if __name__ == "__main__":
    asyncio.run(run_simulation_test())
