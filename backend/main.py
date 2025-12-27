"""FastAPI application entry point for OptionsRadar.

Provides:
- WebSocket endpoint for frontend clients
- REST endpoints for health checks
- Orchestrates data flow from Alpaca/ORATS to frontend

Usage:
    uvicorn backend.main:app --reload --port 8000

See spec section 8.3 for architecture.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.config import load_config
from backend.data import AlpacaOptionsClient, DataAggregator, MockDataGenerator, ORATSClient, SubscriptionManager
from backend.data.aggregator import AggregatedOptionData
from backend.engine import GatingPipeline, PortfolioState, evaluate_option_for_signal
from backend.models.market_data import UnderlyingData
from backend.websocket import ConnectionManager

# Check for mock mode
MOCK_MODE = os.environ.get("MOCK_DATA", "").lower() in ("true", "1", "yes")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global state
connection_manager = ConnectionManager()
aggregator: DataAggregator | None = None
alpaca_client: AlpacaOptionsClient | None = None
orats_client: ORATSClient | None = None
subscription_manager: SubscriptionManager | None = None
mock_generator: MockDataGenerator | None = None

# Background tasks
_background_tasks: set[asyncio.Task] = set()


def option_to_dict(option: AggregatedOptionData) -> dict[str, Any]:
    """Convert AggregatedOptionData to JSON-serializable dict."""
    now = datetime.now(timezone.utc)

    # Serialize timestamps to ISO strings
    quote_ts = option.quote_timestamp
    if isinstance(quote_ts, datetime):
        quote_ts = quote_ts.isoformat()

    greeks_ts = option.greeks_timestamp
    if isinstance(greeks_ts, datetime):
        greeks_ts = greeks_ts.isoformat()

    return {
        "canonicalId": {
            "underlying": option.canonical_id.underlying,
            "expiry": option.canonical_id.expiry,
            "right": option.canonical_id.right,
            "strike": option.canonical_id.strike,
        },
        "bid": option.bid,
        "ask": option.ask,
        "bidSize": option.bid_size,
        "askSize": option.ask_size,
        "last": option.last,
        "mid": option.mid,
        "spread": option.spread,
        "spreadPercent": option.spread_percent,
        "delta": option.delta,
        "gamma": option.gamma,
        "theta": option.theta,
        "vega": option.vega,
        "iv": option.iv,
        "theoreticalValue": option.theoretical_value,
        "quoteTimestamp": quote_ts,
        "greeksTimestamp": greeks_ts,
        "quoteAge": option.quote_age_seconds(now),
        "greeksAge": option.greeks_age_seconds(now),
    }


def underlying_to_dict(underlying: UnderlyingData) -> dict[str, Any]:
    """Convert UnderlyingData to JSON-serializable dict."""
    now = datetime.now(timezone.utc)
    return {
        "symbol": underlying.symbol,
        "price": underlying.price,
        "ivRank": underlying.iv_rank,
        "ivPercentile": underlying.iv_percentile,
        "timestamp": underlying.timestamp,
        "age": underlying.age_seconds(now),
    }


async def on_option_update(option: AggregatedOptionData) -> None:
    """Callback when option data updates - broadcast to clients."""
    try:
        if connection_manager.connection_count > 0:
            await connection_manager.broadcast_option_update(option_to_dict(option))
    except Exception as e:
        logger.error(f"Error broadcasting option update: {e}")


async def on_underlying_update(underlying: UnderlyingData) -> None:
    """Callback when underlying data updates - broadcast to clients."""
    await connection_manager.broadcast_underlying_update(underlying_to_dict(underlying))


async def gate_evaluation_loop() -> None:
    """Background task to periodically evaluate gates and broadcast results."""
    global mock_generator, aggregator

    logger.info("Starting gate evaluation loop (5s interval)")

    while True:
        try:
            await asyncio.sleep(5)  # Evaluate every 5 seconds

            # Get options and underlying
            if mock_generator:
                options = mock_generator.get_all_options()
                underlying = mock_generator.get_underlying()
            elif aggregator:
                options = aggregator.get_all_options()
                underlying = aggregator.get_underlying("NVDA")
            else:
                continue

            if not options or not underlying:
                continue

            # Find ATM option (closest to underlying price)
            atm_option = min(
                options,
                key=lambda o: abs(o.canonical_id.strike - underlying.price)
            )

            # Evaluate gates
            result = evaluate_option_for_signal(
                option=atm_option,
                underlying=underlying,
                action="BUY_CALL" if atm_option.canonical_id.right == "C" else "BUY_PUT",
            )

            # Format gate results for frontend
            gate_results = [
                {
                    "name": g.gate_name,
                    "passed": g.passed,
                    "value": g.value,
                    "threshold": g.threshold,
                    "message": g.message,
                }
                for g in result.all_results
            ]

            # Broadcast gate status
            await connection_manager.broadcast_gate_status(gate_results)

            # Broadcast abstain if applicable
            if result.abstain:
                await connection_manager.broadcast_abstain({
                    "reason": result.abstain.reason.value,
                    "resumeCondition": result.abstain.resume_condition,
                    "failedGates": [
                        {"name": g.gate_name, "message": g.message}
                        for g in result.all_results if not g.passed
                    ],
                })

        except asyncio.CancelledError:
            logger.info("Gate evaluation loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in gate evaluation: {e}")
            await asyncio.sleep(5)


async def greeks_polling_loop() -> None:
    """Background task to poll ORATS for Greeks updates."""
    global orats_client, aggregator, subscription_manager

    if not orats_client or not aggregator:
        logger.warning("ORATS client not configured, skipping Greeks polling")
        return

    logger.info("Starting Greeks polling loop (60s interval)")

    while True:
        try:
            await asyncio.sleep(60)  # Poll every 60 seconds

            if not subscription_manager:
                continue

            symbol = subscription_manager.symbol
            logger.debug(f"Fetching Greeks for {symbol}")

            # Fetch Greeks for subscribed options
            greeks_list = await orats_client.fetch_greeks(symbol, "", dte_max=60)

            if greeks_list:
                aggregator.update_greeks_batch(greeks_list)
                logger.debug(f"Updated {len(greeks_list)} Greeks for {symbol}")

            # Fetch IV rank
            iv_data = await orats_client.fetch_iv_rank(symbol)
            if iv_data:
                aggregator.update_underlying(iv_data)

        except asyncio.CancelledError:
            logger.info("Greeks polling cancelled")
            break
        except Exception as e:
            logger.error(f"Error in Greeks polling: {e}")
            await asyncio.sleep(10)  # Brief pause on error


async def alpaca_message_loop() -> None:
    """Background task to process Alpaca WebSocket messages."""
    global alpaca_client

    if not alpaca_client:
        return

    logger.info("Starting Alpaca message processing loop")

    try:
        async for message in alpaca_client.messages():
            # Message processing is handled via callbacks
            pass
    except asyncio.CancelledError:
        logger.info("Alpaca message loop cancelled")
    except Exception as e:
        logger.error(f"Error in Alpaca message loop: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    global aggregator, alpaca_client, orats_client, subscription_manager, mock_generator

    logger.info("Starting OptionsRadar server...")

    # Check for mock mode
    if MOCK_MODE:
        logger.info("=" * 60)
        logger.info("MOCK DATA MODE ENABLED")
        logger.info("  Using simulated NVDA options data")
        logger.info("  Set MOCK_DATA=false to use real market data")
        logger.info("=" * 60)

        # Set up mock data callbacks
        def mock_option_callback(option: AggregatedOptionData) -> None:
            asyncio.create_task(on_option_update(option))

        def mock_underlying_callback(underlying: UnderlyingData) -> None:
            asyncio.create_task(on_underlying_update(underlying))

        mock_generator = MockDataGenerator(
            on_option_update=mock_option_callback,
            on_underlying_update=mock_underlying_callback,
            update_interval=1.0,  # Update every second
        )

        # Generate initial data
        underlying, options = mock_generator.generate_initial_chain()
        logger.info(f"Generated {len(options)} mock options")

        # Send initial data to any connected clients
        mock_underlying_callback(underlying)
        for option in options:
            mock_option_callback(option)

        # Start streaming updates
        task = asyncio.create_task(mock_generator.start_streaming())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        # Start gate evaluation loop
        task = asyncio.create_task(gate_evaluation_loop())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        logger.info("OptionsRadar server ready (MOCK MODE)")

        yield

        # Shutdown mock mode
        logger.info("Shutting down OptionsRadar server...")
        mock_generator.stop_streaming()

        for task in _background_tasks:
            task.cancel()

        if _background_tasks:
            await asyncio.gather(*_background_tasks, return_exceptions=True)

        logger.info("Shutdown complete")
        return

    # Real mode - connect to Alpaca/ORATS
    try:
        config = load_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.warning("Running in demo mode without data sources")
        yield
        return

    # Check market hours before connecting
    from backend.data.market_hours import check_market_hours, CT

    try:
        market_status = await check_market_hours(config.alpaca)
        info = market_status.format_for_timezone(CT)

        logger.info(f"Current time (CT): {info['current_time']}")

        if not market_status.is_open:
            logger.warning("=" * 60)
            logger.warning("MARKET IS CLOSED")
            logger.warning(f"  {info['time_until_open']}")
            logger.warning(f"  Next open: {info['next_open']}")
            logger.warning("  No quotes will be received until market opens.")
            logger.warning("  Server will stay running for UI testing.")
            logger.warning("=" * 60)
    except Exception as e:
        logger.warning(f"Could not check market hours: {e}")

    # Initialize aggregator
    aggregator = DataAggregator(config=config)

    # Initialize Alpaca client
    def sync_option_callback(option: AggregatedOptionData) -> None:
        """Sync wrapper for async callback."""
        asyncio.create_task(on_option_update(option))

    def sync_underlying_callback(underlying: UnderlyingData) -> None:
        """Sync wrapper for async callback."""
        asyncio.create_task(on_underlying_update(underlying))

    aggregator.on_option_update = sync_option_callback
    aggregator.on_underlying_update = sync_underlying_callback

    # Quote counter for logging
    quote_count = [0]

    def on_quote_received(quote):
        """Log and forward quotes to aggregator."""
        quote_count[0] += 1
        if quote_count[0] <= 5 or quote_count[0] % 100 == 0:
            logger.info(
                f"Quote #{quote_count[0]}: {quote.canonical_id.underlying} "
                f"{quote.canonical_id.strike}{quote.canonical_id.right} "
                f"Bid: ${quote.bid:.2f} Ask: ${quote.ask:.2f}"
            )
        aggregator.update_quote(quote)

    alpaca_client = AlpacaOptionsClient(
        config=config.alpaca,
        on_quote=on_quote_received,
        on_state_change=lambda state: logger.info(f"Alpaca state: {state.value}"),
    )

    # Initialize ORATS client if configured
    if config.orats.api_token:
        orats_client = ORATSClient(config=config.orats)
        await orats_client.__aenter__()

    # Connect to Alpaca
    await alpaca_client.connect()

    # Set up subscription manager
    subscription_manager = SubscriptionManager(
        config=config.alpaca,
        client=alpaca_client,
        symbol="NVDA",  # MVP: Single symbol
        strikes_around_atm=10,
    )
    await subscription_manager.start()

    logger.info(f"Subscribed to {subscription_manager.subscribed_count} contracts")

    # Start background tasks
    task = asyncio.create_task(alpaca_message_loop())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    if orats_client:
        task = asyncio.create_task(greeks_polling_loop())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    # Start gate evaluation loop
    task = asyncio.create_task(gate_evaluation_loop())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    logger.info("OptionsRadar server ready")

    yield

    # Shutdown
    logger.info("Shutting down OptionsRadar server...")

    # Cancel background tasks
    for task in _background_tasks:
        task.cancel()

    if _background_tasks:
        await asyncio.gather(*_background_tasks, return_exceptions=True)

    # Cleanup
    if subscription_manager:
        await subscription_manager.stop()

    if alpaca_client:
        await alpaca_client.disconnect()

    if orats_client:
        await orats_client.__aexit__(None, None, None)

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="OptionsRadar API",
    description="Real-time options recommendation system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    if mock_generator:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "connections": connection_manager.connection_count,
            "optionsCount": mock_generator.option_count,
            "mockMode": True,
            "alpaca_connected": False,
            "orats_configured": False,
        }

    options_count = aggregator.quote_count if aggregator else 0
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "connections": connection_manager.connection_count,
        "optionsCount": options_count,
        "mockMode": False,
        "alpaca_connected": alpaca_client is not None,
        "orats_configured": orats_client is not None,
    }


@app.get("/api/market")
async def get_market_status() -> dict[str, Any]:
    """Get current market status from Alpaca."""
    from backend.data.market_hours import check_market_hours, CT

    try:
        config = load_config()
        status = await check_market_hours(config.alpaca)
        info = status.format_for_timezone(CT)
        return {
            "isOpen": status.is_open,
            "currentTime": info["current_time"],
            "nextOpen": info["next_open"],
            "nextClose": info["next_close"],
            "timeUntilOpen": info["time_until_open"],
            "timeUntilClose": info["time_until_close"],
            "secondsUntilOpen": info["seconds_until_open"],
            "secondsUntilClose": info["seconds_until_close"],
            "timezone": "Central Time",
        }
    except Exception as e:
        return {
            "error": str(e),
            "isOpen": None,
        }


@app.get("/api/options")
async def get_options() -> dict[str, Any]:
    """Get current snapshot of all options data."""
    # Check mock mode first
    if mock_generator:
        options = mock_generator.get_all_options()
        underlying = mock_generator.get_underlying()
        return {
            "options": [option_to_dict(opt) for opt in options],
            "underlying": underlying_to_dict(underlying),
        }

    if not aggregator:
        return {"options": [], "underlying": None}

    options = aggregator.get_all_options()
    underlying = aggregator.get_underlying("NVDA")

    return {
        "options": [option_to_dict(opt) for opt in options],
        "underlying": underlying_to_dict(underlying) if underlying else None,
    }


@app.get("/api/gates/{symbol}")
async def evaluate_gates(symbol: str) -> dict[str, Any]:
    """Evaluate gates for a symbol's options."""
    if not aggregator:
        return {"error": "Aggregator not initialized"}

    options = aggregator.get_options_for_underlying(symbol)
    underlying = aggregator.get_underlying(symbol)

    results = []
    for option in options[:5]:  # Limit for demo
        result = evaluate_option_for_signal(
            option=option,
            underlying=underlying,
            action="BUY_CALL" if option.canonical_id.right == "C" else "BUY_PUT",
        )

        gate_results = [
            {
                "name": g.gate_name,
                "passed": g.passed,
                "value": g.value,
                "threshold": g.threshold,
                "message": g.message,
            }
            for g in result.all_results
        ]

        results.append({
            "option": f"{option.canonical_id.strike}{option.canonical_id.right}",
            "passed": result.passed,
            "stageReached": result.stage_reached.value,
            "confidenceCap": result.confidence_cap,
            "gates": gate_results,
            "abstain": {
                "reason": result.abstain.reason.value,
                "resumeCondition": result.abstain.resume_condition,
            } if result.abstain else None,
        })

    return {"evaluations": results}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    await connection_manager.connect(websocket)

    try:
        # Send initial data snapshot
        if mock_generator:
            # Mock mode - send mock data
            options = mock_generator.get_all_options()
            for option in options:
                await connection_manager._send_to_client(
                    websocket,
                    {
                        "type": "option_update",
                        "data": option_to_dict(option),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            underlying = mock_generator.get_underlying()
            await connection_manager._send_to_client(
                websocket,
                {
                    "type": "underlying_update",
                    "data": underlying_to_dict(underlying),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        elif aggregator:
            # Real mode - send aggregated data
            options = aggregator.get_all_options()
            for option in options:
                await connection_manager._send_to_client(
                    websocket,
                    {
                        "type": "option_update",
                        "data": option_to_dict(option),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            underlying = aggregator.get_underlying("NVDA")
            if underlying:
                await connection_manager._send_to_client(
                    websocket,
                    {
                        "type": "underlying_update",
                        "data": underlying_to_dict(underlying),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        # Keep connection alive with ping/pong and handle client messages
        while True:
            try:
                # Wait for message with timeout, send ping if timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle client messages
                if data == "ping":
                    await websocket.send_text("pong")
                else:
                    logger.debug(f"Received from client: {data}")
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_text('{"type":"ping"}')
                except Exception:
                    break  # Connection dead

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)
