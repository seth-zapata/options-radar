"""Alpaca WebSocket client for real-time options quotes.

Handles:
- WebSocket connection to Alpaca Options (OPRA) feed
- Authentication via API keys
- MsgPack message decoding
- Subscription management for option symbols
- Automatic reconnection with exponential backoff
- Kill switch on repeated failures

See spec section 3.1 and Appendix for details.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import msgpack
import websockets
from websockets.asyncio.client import ClientConnection

from backend.config import AlpacaConfig
from backend.models.canonical import CanonicalOptionId, parse_alpaca
from backend.models.market_data import QuoteData, TradeData

logger = logging.getLogger(__name__)

# Kill switch configuration
KILL_SWITCH_FAILURE_THRESHOLD = 5  # failures
KILL_SWITCH_WINDOW_SECONDS = 300  # 5 minutes


class ConnectionState(str, Enum):
    """WebSocket connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class MessageType(str, Enum):
    """Alpaca message types."""

    SUCCESS = "success"
    ERROR = "error"
    SUBSCRIPTION = "subscription"
    QUOTE = "q"
    TRADE = "t"


@dataclass
class AlpacaOptionsClient:
    """WebSocket client for Alpaca options data feed.

    Usage:
        config = AlpacaConfig(api_key="...", secret_key="...", paper=True)
        client = AlpacaOptionsClient(config)

        async def handle_quote(quote: QuoteData):
            print(f"Quote: {quote}")

        client.on_quote = handle_quote
        await client.connect()
        await client.subscribe(["NVDA250117C00500000"])

        # Run the message loop
        async for _ in client.messages():
            pass  # Messages are handled via callbacks

    Attributes:
        config: Alpaca API configuration
        on_quote: Callback for quote updates
        on_trade: Callback for trade updates
        on_error: Callback for error messages
        on_state_change: Callback for connection state changes
    """

    config: AlpacaConfig

    # Callbacks
    on_quote: Callable[[QuoteData], None] | None = None
    on_trade: Callable[[TradeData], None] | None = None
    on_error: Callable[[str], None] | None = None
    on_state_change: Callable[[ConnectionState], None] | None = None
    on_kill_switch: Callable[[str], None] | None = None  # Called when kill switch triggers

    # Reconnection settings
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 60.0

    # Internal state
    _ws: ClientConnection | None = field(default=None, init=False)
    _state: ConnectionState = field(default=ConnectionState.DISCONNECTED, init=False)
    _subscribed_symbols: set[str] = field(default_factory=set, init=False)
    _reconnect_attempts: int = field(default=0, init=False)
    _should_reconnect: bool = field(default=True, init=False)
    _failure_timestamps: deque[float] = field(default_factory=deque, init=False)
    _kill_switch_active: bool = field(default=False, init=False)

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """True if connected and authenticated."""
        return self._state == ConnectionState.CONNECTED

    @property
    def kill_switch_active(self) -> bool:
        """True if kill switch has been triggered."""
        return self._kill_switch_active

    def _set_state(self, state: ConnectionState) -> None:
        """Update state and notify callback."""
        old_state = self._state
        self._state = state
        if old_state != state and self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception as e:
                logger.exception(f"Error in state change callback: {e}")

    async def connect(self) -> None:
        """Establish WebSocket connection and authenticate.

        Raises:
            ConnectionError: If connection or authentication fails
        """
        self._should_reconnect = True
        await self._connect()

    async def _connect(self) -> None:
        """Internal connection logic."""
        self._set_state(ConnectionState.CONNECTING)

        try:
            # Connect to WebSocket
            self._ws = await websockets.connect(
                self.config.options_stream_url,
                additional_headers={
                    "APCA-API-KEY-ID": self.config.api_key,
                    "APCA-API-SECRET-KEY": self.config.secret_key,
                },
                ping_interval=20,
                ping_timeout=20,
            )
            logger.info(f"Connected to {self.config.options_stream_url}")

            # Wait for welcome message
            self._set_state(ConnectionState.AUTHENTICATING)
            welcome = await self._receive()

            if not welcome:
                raise ConnectionError("No welcome message received")

            # Check for successful connection
            # Alpaca sends [{"T":"success","msg":"connected"}]
            if isinstance(welcome, list) and len(welcome) > 0:
                msg = welcome[0]
                if msg.get("T") == "success" and msg.get("msg") == "connected":
                    logger.info("Authentication successful")
                    self._set_state(ConnectionState.CONNECTED)
                    self._reconnect_attempts = 0

                    # Re-subscribe to previous symbols if reconnecting
                    if self._subscribed_symbols:
                        await self._send_subscribe(list(self._subscribed_symbols))
                    return

            raise ConnectionError(f"Unexpected welcome message: {welcome}")

        except Exception as e:
            self._set_state(ConnectionState.DISCONNECTED)
            logger.error(f"Connection failed: {e}")
            raise ConnectionError(f"Failed to connect: {e}") from e

    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully."""
        self._should_reconnect = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self._ws = None
                self._set_state(ConnectionState.DISCONNECTED)

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to option symbols.

        Args:
            symbols: List of Alpaca-format option symbols
                     (e.g., ["NVDA250117C00500000"])

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected:
            raise ConnectionError("Not connected")

        await self._send_subscribe(symbols)
        self._subscribed_symbols.update(symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from option symbols.

        Args:
            symbols: List of Alpaca-format option symbols

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected:
            raise ConnectionError("Not connected")

        await self._send_unsubscribe(symbols)
        self._subscribed_symbols.difference_update(symbols)

    async def _send_subscribe(self, symbols: list[str]) -> None:
        """Send subscription message."""
        msg = {"action": "subscribe", "quotes": symbols, "trades": symbols}
        await self._send(msg)
        logger.info(f"Subscribed to {len(symbols)} symbols")

    async def _send_unsubscribe(self, symbols: list[str]) -> None:
        """Send unsubscription message."""
        msg = {"action": "unsubscribe", "quotes": symbols, "trades": symbols}
        await self._send(msg)
        logger.info(f"Unsubscribed from {len(symbols)} symbols")

    async def _send(self, data: dict) -> None:
        """Send JSON message via WebSocket."""
        if not self._ws:
            raise ConnectionError("WebSocket not connected")

        import json
        await self._ws.send(json.dumps(data))

    async def _receive(self) -> list[dict] | None:
        """Receive and decode a message.

        Returns:
            Decoded message (list of dicts) or None if connection closed
        """
        if not self._ws:
            return None

        try:
            raw = await self._ws.recv()

            # Alpaca sends MsgPack-encoded messages
            if isinstance(raw, bytes):
                return msgpack.unpackb(raw, raw=False)
            else:
                # Fallback to JSON for text messages
                import json
                return json.loads(raw)

        except websockets.exceptions.ConnectionClosed:
            return None

    async def messages(self) -> AsyncIterator[None]:
        """Async generator that processes incoming messages.

        Messages are handled via callbacks (on_quote, on_trade, etc.).
        Yields None after each message batch is processed.

        Handles automatic reconnection on disconnect.

        Usage:
            async for _ in client.messages():
                pass  # Messages handled via callbacks
        """
        while self._should_reconnect:
            try:
                while True:
                    messages = await self._receive()

                    if messages is None:
                        # Connection closed
                        break

                    self._process_messages(messages)
                    yield

            except Exception as e:
                logger.error(f"Error in message loop: {e}")

            # Handle reconnection
            if self._should_reconnect:
                await self._handle_reconnect()

    def _record_failure(self) -> None:
        """Record a connection failure and check kill switch."""
        now = datetime.now(timezone.utc).timestamp()
        self._failure_timestamps.append(now)

        # Remove failures outside the window
        cutoff = now - KILL_SWITCH_WINDOW_SECONDS
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()

        # Check if kill switch should trigger
        if len(self._failure_timestamps) >= KILL_SWITCH_FAILURE_THRESHOLD:
            self._trigger_kill_switch()

    def _trigger_kill_switch(self) -> None:
        """Trigger the kill switch due to repeated failures."""
        self._kill_switch_active = True
        self._should_reconnect = False

        msg = (
            f"CRITICAL: Kill switch triggered - "
            f"{len(self._failure_timestamps)} failures in "
            f"{KILL_SWITCH_WINDOW_SECONDS}s window"
        )
        logger.critical(msg)

        if self.on_kill_switch:
            try:
                self.on_kill_switch(msg)
            except Exception as e:
                logger.exception(f"Error in kill switch callback: {e}")

    def reset_kill_switch(self) -> None:
        """Reset the kill switch to allow reconnection.

        Call this after investigating and resolving the underlying issue.
        """
        self._kill_switch_active = False
        self._failure_timestamps.clear()
        self._reconnect_attempts = 0
        logger.info("Kill switch reset - reconnection enabled")

    async def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff.

        Tracks failures and triggers kill switch if too many failures
        occur within the configured window (5 failures in 5 minutes).
        """
        self._set_state(ConnectionState.RECONNECTING)
        self._reconnect_attempts += 1
        self._record_failure()

        # Check if kill switch was triggered
        if self._kill_switch_active:
            self._set_state(ConnectionState.DISCONNECTED)
            return

        # Calculate delay with exponential backoff: 1s -> 2s -> 4s -> 8s -> 16s -> 32s -> 60s cap
        delay = min(
            self.reconnect_base_delay * (2 ** (self._reconnect_attempts - 1)),
            self.reconnect_max_delay,
        )

        logger.warning(
            f"Reconnecting in {delay:.1f}s "
            f"(attempt {self._reconnect_attempts}, "
            f"{len(self._failure_timestamps)} failures in window)"
        )
        await asyncio.sleep(delay)

        try:
            await self._connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    def _process_messages(self, messages: list[dict]) -> None:
        """Process a batch of messages from Alpaca.

        Args:
            messages: List of message dicts from Alpaca
        """
        receive_ts = datetime.now(timezone.utc).isoformat()

        for msg in messages:
            msg_type = msg.get("T")

            if msg_type == MessageType.QUOTE.value:
                self._handle_quote(msg, receive_ts)
            elif msg_type == MessageType.TRADE.value:
                self._handle_trade(msg, receive_ts)
            elif msg_type == MessageType.ERROR.value:
                self._handle_error(msg)
            elif msg_type == MessageType.SUBSCRIPTION.value:
                self._handle_subscription(msg)
            elif msg_type == MessageType.SUCCESS.value:
                logger.debug(f"Success message: {msg.get('msg')}")
            else:
                logger.debug(f"Unknown message type: {msg_type}")

    def _handle_quote(self, msg: dict, receive_ts: str) -> None:
        """Handle quote message.

        Alpaca quote format:
        {
            "T": "q",
            "S": "NVDA250117C00500000",
            "bx": "C",
            "bp": 12.50,
            "bs": 10,
            "ax": "P",
            "ap": 12.75,
            "as": 5,
            "c": "A",
            "t": "2025-01-15T14:30:00.123Z"
        }
        """
        try:
            symbol = msg.get("S", "")
            canonical_id = parse_alpaca(symbol)

            quote = QuoteData(
                canonical_id=canonical_id,
                bid=float(msg.get("bp", 0)),
                ask=float(msg.get("ap", 0)),
                bid_size=int(msg.get("bs", 0)),
                ask_size=int(msg.get("as", 0)),
                last=None,  # Quotes don't include last price
                timestamp=msg.get("t", receive_ts),
                receive_timestamp=receive_ts,
                source="alpaca",
            )

            if self.on_quote:
                self.on_quote(quote)

        except Exception as e:
            logger.error(f"Error processing quote: {e}, msg: {msg}")

    def _handle_trade(self, msg: dict, receive_ts: str) -> None:
        """Handle trade message.

        Alpaca trade format:
        {
            "T": "t",
            "S": "NVDA250117C00500000",
            "x": "C",
            "p": 12.60,
            "s": 1,
            "c": ["A"],
            "t": "2025-01-15T14:30:00.456Z"
        }
        """
        try:
            symbol = msg.get("S", "")
            canonical_id = parse_alpaca(symbol)

            trade = TradeData(
                canonical_id=canonical_id,
                price=float(msg.get("p", 0)),
                size=int(msg.get("s", 0)),
                timestamp=msg.get("t", receive_ts),
                exchange=msg.get("x", ""),
                conditions=msg.get("c", []),
            )

            if self.on_trade:
                self.on_trade(trade)

        except Exception as e:
            logger.error(f"Error processing trade: {e}, msg: {msg}")

    def _handle_error(self, msg: dict) -> None:
        """Handle error message."""
        error_msg = msg.get("msg", "Unknown error")
        error_code = msg.get("code")
        full_msg = f"Alpaca error: {error_msg} (code: {error_code})"

        logger.error(full_msg)

        if self.on_error:
            try:
                self.on_error(full_msg)
            except Exception as e:
                logger.exception(f"Error in error callback: {e}")

    def _handle_subscription(self, msg: dict) -> None:
        """Handle subscription confirmation."""
        quotes = msg.get("quotes", [])
        trades = msg.get("trades", [])
        logger.info(f"Subscription confirmed: {len(quotes)} quotes, {len(trades)} trades")
