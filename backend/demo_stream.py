#!/usr/bin/env python3
"""Demo script for Alpaca options WebSocket streaming.

Usage:
    1. Copy .env.example to .env and fill in your Alpaca credentials
    2. Run: python -m backend.demo_stream

This will connect to Alpaca's options feed and stream NVDA option quotes,
dynamically subscribing to ±10 strikes around ATM.
"""

import asyncio
import logging
import sys

from backend.config import load_config
from backend.data.alpaca_client import AlpacaOptionsClient, ConnectionState
from backend.data.subscription_manager import SubscriptionManager
from backend.models.market_data import QuoteData, TradeData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Track underlying price from option trades for ATM updates
_last_underlying_update: float = 0
_update_interval: float = 30.0  # Update ATM every 30 seconds at most


def on_quote(quote: QuoteData) -> None:
    """Handle incoming quote."""
    spread_pct = quote.spread_percent
    logger.info(
        f"QUOTE | {quote.canonical_id} | "
        f"Bid: {quote.bid:.2f} x {quote.bid_size} | "
        f"Ask: {quote.ask:.2f} x {quote.ask_size} | "
        f"Spread: {spread_pct:.1f}%"
    )


def on_trade(trade: TradeData) -> None:
    """Handle incoming trade."""
    logger.info(
        f"TRADE | {trade.canonical_id} | "
        f"Price: {trade.price:.2f} | Size: {trade.size}"
    )


def on_state_change(state: ConnectionState) -> None:
    """Handle connection state changes."""
    logger.warning(f"Connection state: {state.value}")


def on_error(error: str) -> None:
    """Handle errors."""
    logger.error(f"Error: {error}")


def on_kill_switch(msg: str) -> None:
    """Handle kill switch trigger."""
    logger.critical(f"KILL SWITCH: {msg}")


def on_subscriptions_changed(symbols: list[str]) -> None:
    """Handle subscription changes."""
    logger.info(f"Now subscribed to {len(symbols)} option contracts")


async def main() -> None:
    """Run the demo streaming client with dynamic ATM subscriptions."""
    try:
        config = load_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please copy .env.example to .env and fill in your credentials")
        sys.exit(1)

    logger.info("Starting Alpaca options stream demo")
    logger.info(f"Using {'paper' if config.alpaca.paper else 'live'} environment")

    # Create WebSocket client
    client = AlpacaOptionsClient(
        config=config.alpaca,
        on_quote=on_quote,
        on_trade=on_trade,
        on_state_change=on_state_change,
        on_error=on_error,
        on_kill_switch=on_kill_switch,
    )

    # Create subscription manager for dynamic ATM-based subscriptions
    sub_manager = SubscriptionManager(
        config=config.alpaca,
        client=client,
        symbol="NVDA",
        strikes_around_atm=10,
        on_subscriptions_changed=on_subscriptions_changed,
    )

    try:
        # Connect WebSocket
        await client.connect()

        # Start subscription manager (fetches price and subscribes)
        await sub_manager.start()

        logger.info(
            f"Subscribed to ±{sub_manager.strikes_around_atm} strikes around "
            f"ATM ${sub_manager.current_atm:.2f}"
        )

        # Process messages
        logger.info("Streaming... Press Ctrl+C to stop")
        async for _ in client.messages():
            pass

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"Error: {e}")
    finally:
        await sub_manager.stop()
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
