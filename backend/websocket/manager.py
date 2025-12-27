"""WebSocket connection manager for frontend clients.

Handles:
- Client connection lifecycle
- Broadcasting option updates to all clients
- Message serialization (JSON)

See spec section 8.3 for architecture.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""
    # Server -> Client
    OPTION_UPDATE = "option_update"
    UNDERLYING_UPDATE = "underlying_update"
    GATE_STATUS = "gate_status"
    ABSTAIN = "abstain"
    CONNECTION_STATUS = "connection_status"
    ERROR = "error"

    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"


@dataclass
class ConnectionManager:
    """Manages WebSocket connections to frontend clients.

    Usage:
        manager = ConnectionManager()

        # In WebSocket endpoint
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                # Handle client messages
        except WebSocketDisconnect:
            manager.disconnect(websocket)

        # Broadcasting updates
        await manager.broadcast_option_update(option_data)
    """

    _connections: set[WebSocket] = field(default_factory=set, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new client connection.

        Args:
            websocket: The WebSocket connection to register
        """
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info(f"Client connected. Total connections: {self.connection_count}")

        # Send initial connection status
        await self._send_to_client(websocket, {
            "type": MessageType.CONNECTION_STATUS.value,
            "data": {
                "connected": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        })

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a client connection.

        Args:
            websocket: The WebSocket connection to remove
        """
        self._connections.discard(websocket)
        logger.info(f"Client disconnected. Total connections: {self.connection_count}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients.

        Args:
            message: Message dict to send (will be JSON encoded)
        """
        if not self._connections:
            return

        # Encode once for all clients
        data = json.dumps(message)

        # Send to all clients, removing any that fail
        disconnected: list[WebSocket] = []

        async with self._lock:
            for websocket in self._connections:
                try:
                    await websocket.send_text(data)
                except Exception as e:
                    logger.warning(f"Failed to send to client: {e}")
                    disconnected.append(websocket)

            # Clean up disconnected clients
            for ws in disconnected:
                self._connections.discard(ws)

    async def broadcast_option_update(self, option_data: dict[str, Any]) -> None:
        """Broadcast an option update to all clients.

        Args:
            option_data: Aggregated option data dict
        """
        await self.broadcast({
            "type": MessageType.OPTION_UPDATE.value,
            "data": option_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def broadcast_underlying_update(self, underlying_data: dict[str, Any]) -> None:
        """Broadcast an underlying update to all clients.

        Args:
            underlying_data: Underlying data dict
        """
        await self.broadcast({
            "type": MessageType.UNDERLYING_UPDATE.value,
            "data": underlying_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def broadcast_gate_status(self, gate_results: list[dict[str, Any]]) -> None:
        """Broadcast gate evaluation results to all clients.

        Args:
            gate_results: List of gate result dicts
        """
        await self.broadcast({
            "type": MessageType.GATE_STATUS.value,
            "data": {"gates": gate_results},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def broadcast_abstain(self, abstain_data: dict[str, Any]) -> None:
        """Broadcast an abstain decision to all clients.

        Args:
            abstain_data: Abstain object as dict
        """
        await self.broadcast({
            "type": MessageType.ABSTAIN.value,
            "data": abstain_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def _send_to_client(self, websocket: WebSocket, message: dict[str, Any]) -> bool:
        """Send a message to a specific client.

        Args:
            websocket: Target client
            message: Message to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            return False
