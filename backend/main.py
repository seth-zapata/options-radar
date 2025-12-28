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
from pydantic import BaseModel

from backend.config import load_config
from backend.data import AlpacaOptionsClient, DataAggregator, MockDataGenerator, ORATSClient, SubscriptionManager
from backend.data.aggregator import AggregatedOptionData
from backend.engine import (
    GatingPipeline,
    PortfolioState,
    PositionTracker,
    Recommendation,
    Recommender,
    SessionTracker,
    TrackedPosition,
    evaluate_option_for_signal,
)
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

# Recommender and session tracking
recommender = Recommender()
session_tracker = SessionTracker()
position_tracker = PositionTracker()

# Rate limiting: track last recommendation time per contract
_last_recommendation_time: dict[str, float] = {}
RECOMMENDATION_COOLDOWN = 60.0  # Don't recommend same contract within 60 seconds

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


def recommendation_to_dict(rec: Recommendation) -> dict[str, Any]:
    """Convert Recommendation to JSON-serializable dict."""
    return {
        "id": rec.id,
        "generatedAt": rec.generated_at,
        "underlying": rec.underlying,
        "action": rec.action,
        "strike": rec.strike,
        "expiry": rec.expiry,
        "right": rec.right,
        "contracts": rec.contracts,
        "premium": rec.premium,
        "totalCost": rec.total_cost,
        "confidence": rec.confidence,
        "rationale": rec.rationale,
        "gateResults": list(rec.gate_results),
        "quoteAge": rec.quote_age,
        "greeksAge": rec.greeks_age,
        "underlyingAge": rec.underlying_age,
        "validUntil": rec.valid_until,
    }


def session_stats_to_dict() -> dict[str, Any]:
    """Get current session stats as JSON-serializable dict.

    Now uses position tracker for exposure (confirmed positions only).
    """
    stats = session_tracker.get_stats()

    # Override exposure with position tracker (confirmed positions only)
    confirmed_exposure = position_tracker.get_total_exposure()
    open_positions = position_tracker.get_open_positions()

    return {
        "sessionId": stats.session_id,
        "startedAt": stats.started_at,
        "recommendationCount": stats.recommendation_count,
        "totalExposure": confirmed_exposure,  # From confirmed positions
        "exposureRemaining": max(0, 5000 - confirmed_exposure),  # $5k limit
        "exposurePercent": round((confirmed_exposure / 5000) * 100, 1) if confirmed_exposure > 0 else 0,
        "isAtLimit": confirmed_exposure >= 5000,
        "isWarning": confirmed_exposure >= 4000 and confirmed_exposure < 5000,
        "recommendationsBySymbol": stats.recommendations_by_symbol,
        "lastRecommendationAt": stats.last_recommendation_at,
        "openPositionCount": len(open_positions),
        "totalPnl": position_tracker.get_total_pnl(),
    }


def position_to_dict(pos: TrackedPosition) -> dict[str, Any]:
    """Convert TrackedPosition to JSON-serializable dict."""
    return {
        "id": pos.id,
        "recommendationId": pos.recommendation_id,
        "openedAt": pos.opened_at,
        "underlying": pos.underlying,
        "expiry": pos.expiry,
        "strike": pos.strike,
        "right": pos.right,
        "action": pos.action,
        "contracts": pos.contracts,
        "fillPrice": pos.fill_price,
        "entryCost": pos.entry_cost,
        "currentPrice": pos.current_price,
        "currentValue": pos.current_value,
        "pnl": pos.pnl,
        "pnlPercent": pos.pnl_percent,
        "dte": pos.dte,
        "delta": pos.delta,
        "status": pos.status,
        "exitReason": pos.exit_reason,
        "closedAt": pos.closed_at,
        "closePrice": pos.close_price,
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


def score_option_candidate(
    option: AggregatedOptionData,
    underlying: UnderlyingData,
) -> tuple[float, int]:
    """Score an option for recommendation quality.

    Returns (score, confidence) where:
    - score: 0-100, higher is better candidate
    - confidence: 0-100, how confident we are in this recommendation

    Scoring factors:
    - Delta: 0.25-0.45 is ideal for directional plays (not too aggressive, not too conservative)
    - Spread: Tighter spreads are better
    - IV rank alignment: Low IV for buying premium
    - Time to expiry: 14-45 DTE is ideal
    """
    score = 50.0  # Base score
    confidence = 70  # Base confidence

    # Delta scoring (ideal: 0.30-0.40 for calls, -0.40 to -0.30 for puts)
    if option.delta is not None:
        abs_delta = abs(option.delta)
        if 0.30 <= abs_delta <= 0.40:
            score += 20  # Ideal range
            confidence += 15
        elif 0.25 <= abs_delta <= 0.45:
            score += 10  # Good range
            confidence += 8
        elif 0.20 <= abs_delta <= 0.50:
            score += 5  # Acceptable
            confidence += 3
        else:
            score -= 10  # Too aggressive or too conservative
            confidence -= 10

    # Spread scoring (tighter is better)
    if option.spread_percent is not None:
        if option.spread_percent < 2:
            score += 15
            confidence += 10
        elif option.spread_percent < 5:
            score += 10
            confidence += 5
        elif option.spread_percent < 8:
            score += 5
        else:
            score -= 5
            confidence -= 5

    # IV rank scoring (for buying premium, lower IV is better)
    if underlying.iv_rank is not None:
        if underlying.iv_rank < 30:
            score += 15
            confidence += 10
        elif underlying.iv_rank < 50:
            score += 8
            confidence += 5
        else:
            score -= 5
            confidence -= 5

    # Premium scoring (not too cheap, not too expensive)
    if option.mid is not None:
        if 2.0 <= option.mid <= 8.0:
            score += 10
            confidence += 5
        elif 1.0 <= option.mid <= 15.0:
            score += 5
        else:
            score -= 5

    # Clamp values
    score = max(0, min(100, score))
    confidence = max(40, min(95, confidence))  # Never 100%, never below 40%

    return (score, confidence)


def get_option_key(option: AggregatedOptionData) -> str:
    """Get unique key for rate-limiting."""
    cid = option.canonical_id
    return f"{cid.underlying}-{cid.expiry}-{cid.strike}-{cid.right}"


async def gate_evaluation_loop() -> None:
    """Background task to periodically evaluate gates and broadcast results."""
    global mock_generator, aggregator, _last_recommendation_time

    logger.info("Starting gate evaluation loop (5s interval)")

    while True:
        try:
            await asyncio.sleep(5)  # Evaluate every 5 seconds

            # Get options and underlying
            if mock_generator:
                all_options = mock_generator.get_all_options()
                underlying = mock_generator.get_underlying()
            elif aggregator:
                all_options = aggregator.get_all_options()
                underlying = aggregator.get_underlying("NVDA")
            else:
                continue

            if not all_options or not underlying:
                continue

            now = datetime.now(timezone.utc)
            now_ts = now.timestamp()

            # Filter to calls only for BUY_CALL evaluation
            # Get options within reasonable strike range (ATM +/- 5 strikes)
            atm_strike = round(underlying.price / 2.5) * 2.5
            candidates = [
                o for o in all_options
                if o.canonical_id.right == "C"
                and abs(o.canonical_id.strike - atm_strike) <= 12.5  # Within 5 strikes
            ]

            if not candidates:
                continue

            # Score all candidates and find the best one that passes gates
            best_option = None
            best_result = None
            best_score = -1
            best_confidence = 0

            for option in candidates:
                # Check rate limiting
                option_key = get_option_key(option)
                last_rec_time = _last_recommendation_time.get(option_key, 0)
                if now_ts - last_rec_time < RECOMMENDATION_COOLDOWN:
                    continue  # Skip, recently recommended

                # Score the option
                score, confidence = score_option_candidate(option, underlying)

                # Evaluate gates
                result = evaluate_option_for_signal(
                    option=option,
                    underlying=underlying,
                    action="BUY_CALL",
                )

                if result.passed and score > best_score:
                    best_option = option
                    best_result = result
                    best_score = score
                    best_confidence = confidence

            # If no passing candidate, just evaluate ATM for display
            if best_option is None:
                atm_option = min(
                    candidates,
                    key=lambda o: abs(o.canonical_id.strike - underlying.price)
                )
                result = evaluate_option_for_signal(
                    option=atm_option,
                    underlying=underlying,
                    action="BUY_CALL",
                )
                display_option = atm_option
            else:
                result = best_result
                display_option = best_option

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

            # Broadcast gate status with the option being evaluated
            await connection_manager.broadcast({
                "type": "gate_status",
                "data": {
                    "gates": gate_results,
                    "evaluatedOption": {
                        "strike": display_option.canonical_id.strike,
                        "right": display_option.canonical_id.right,
                        "expiry": display_option.canonical_id.expiry,
                        "premium": display_option.mid,
                    },
                },
                "timestamp": now.isoformat(),
            })

            # Broadcast abstain status (null clears previous abstain)
            if result.abstain or best_option is None:
                if result.abstain:
                    await connection_manager.broadcast_abstain({
                        "reason": result.abstain.reason.value,
                        "resumeCondition": result.abstain.resume_condition,
                        "failedGates": [
                            {"name": g.gate_name, "message": g.message}
                            for g in result.all_results if not g.passed
                        ],
                    })
                # If best_option is None due to rate limiting, don't send abstain
            else:
                # Best option found - generate recommendation with calculated confidence
                await connection_manager.broadcast_abstain(None)

                # Override confidence in the result for the recommender
                # We do this by generating with the result then adjusting
                recommendation = recommender.generate(
                    result=best_result,
                    option=best_option,
                    underlying=underlying,
                    action="BUY_CALL",
                )

                if recommendation:
                    # Create new recommendation with adjusted confidence
                    from backend.engine.recommender import Recommendation
                    adjusted_rec = Recommendation(
                        id=recommendation.id,
                        generated_at=recommendation.generated_at,
                        underlying=recommendation.underlying,
                        action=recommendation.action,
                        strike=recommendation.strike,
                        expiry=recommendation.expiry,
                        right=recommendation.right,
                        contracts=recommendation.contracts,
                        premium=recommendation.premium,
                        total_cost=recommendation.total_cost,
                        confidence=best_confidence,  # Use our calculated confidence
                        rationale=recommendation.rationale,
                        gate_results=recommendation.gate_results,
                        quote_age=recommendation.quote_age,
                        greeks_age=recommendation.greeks_age,
                        underlying_age=recommendation.underlying_age,
                        valid_until=recommendation.valid_until,
                    )

                    # Check session limits
                    allowed, reason = session_tracker.can_add_recommendation(adjusted_rec)

                    if allowed:
                        # Record recommendation time for rate limiting
                        option_key = get_option_key(best_option)
                        _last_recommendation_time[option_key] = now_ts

                        # Add to session and broadcast
                        session_tracker.add_recommendation(adjusted_rec)

                        await connection_manager.broadcast({
                            "type": "recommendation",
                            "data": recommendation_to_dict(adjusted_rec),
                            "timestamp": now.isoformat(),
                        })

                        logger.info(
                            f"Recommendation: {adjusted_rec.action} "
                            f"{adjusted_rec.underlying} ${adjusted_rec.strike} "
                            f"@ ${adjusted_rec.premium:.2f} "
                            f"(score: {best_score:.0f}, conf: {best_confidence}%)"
                        )
                    else:
                        logger.warning(f"Recommendation blocked: {reason}")

                # Always broadcast session stats after evaluation
                await connection_manager.broadcast({
                    "type": "session_status",
                    "data": session_stats_to_dict(),
                    "timestamp": now.isoformat(),
                })

        except asyncio.CancelledError:
            logger.info("Gate evaluation loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in gate evaluation: {e}")
            import traceback
            traceback.print_exc()
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


@app.get("/api/session")
async def get_session_status() -> dict[str, Any]:
    """Get current session tracking status."""
    return session_stats_to_dict()


@app.get("/api/recommendations")
async def get_recommendations(limit: int = 10) -> dict[str, Any]:
    """Get recent recommendations from the session."""
    recs = session_tracker.get_recent_recommendations(limit)
    return {
        "recommendations": [recommendation_to_dict(r) for r in recs],
        "session": session_stats_to_dict(),
    }


@app.get("/api/positions")
async def get_positions() -> dict[str, Any]:
    """Get all tracked positions."""
    open_positions = position_tracker.get_open_positions()
    all_positions = position_tracker.get_all_positions()
    return {
        "openPositions": [position_to_dict(p) for p in open_positions],
        "allPositions": [position_to_dict(p) for p in all_positions],
        "totalExposure": position_tracker.get_total_exposure(),
        "totalPnl": position_tracker.get_total_pnl(),
    }


class OpenPositionRequest(BaseModel):
    """Request body for opening a position."""
    recommendation_id: str
    fill_price: float
    contracts: int = 1


@app.post("/api/positions/open")
async def open_position(request: OpenPositionRequest) -> dict[str, Any]:
    """Confirm a trade and open a tracked position.

    Args:
        request: OpenPositionRequest with recommendation_id, fill_price, contracts

    Returns:
        The created position
    """
    # Find the recommendation
    recs = session_tracker.get_recent_recommendations(50)
    rec = next((r for r in recs if r.id == request.recommendation_id), None)

    if not rec:
        return {"error": f"Recommendation not found: {request.recommendation_id}"}

    # Check exposure limit before opening
    proposed_cost = request.fill_price * request.contracts * 100
    current_exposure = position_tracker.get_total_exposure()
    exposure_limit = 5000.0  # $5k session limit

    if current_exposure + proposed_cost > exposure_limit:
        remaining = exposure_limit - current_exposure
        return {
            "error": f"Trade would exceed session limit. Current: ${current_exposure:.0f}, "
                     f"Trade: ${proposed_cost:.0f}, Limit: ${exposure_limit:.0f}, "
                     f"Remaining: ${remaining:.0f}"
        }

    try:
        position = position_tracker.open_position(
            recommendation_id=request.recommendation_id,
            underlying=rec.underlying,
            expiry=rec.expiry,
            strike=rec.strike,
            right=rec.right,
            action=rec.action,
            contracts=request.contracts,
            fill_price=request.fill_price,
        )

        # Broadcast position update to all clients
        await connection_manager.broadcast({
            "type": "position_opened",
            "data": position_to_dict(position),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Also broadcast updated session stats
        await connection_manager.broadcast({
            "type": "session_status",
            "data": session_stats_to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return {"position": position_to_dict(position)}

    except ValueError as e:
        return {"error": str(e)}


class ClosePositionRequest(BaseModel):
    """Request body for closing a position."""
    close_price: float


@app.post("/api/positions/{position_id}/close")
async def close_position(position_id: str, request: ClosePositionRequest) -> dict[str, Any]:
    """Close a tracked position.

    Args:
        position_id: ID of position to close (path parameter)
        request: ClosePositionRequest with close_price

    Returns:
        The closed position
    """
    position = position_tracker.close_position(position_id, request.close_price)

    if not position:
        return {"error": f"Position not found: {position_id}"}

    # Broadcast position update
    await connection_manager.broadcast({
        "type": "position_closed",
        "data": position_to_dict(position),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # Broadcast updated session stats
    await connection_manager.broadcast({
        "type": "session_status",
        "data": session_stats_to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {"position": position_to_dict(position)}


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

        # Send initial session status
        await connection_manager._send_to_client(
            websocket,
            {
                "type": "session_status",
                "data": session_stats_to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Send recent recommendations
        recent_recs = session_tracker.get_recent_recommendations(10)
        for rec in recent_recs:
            await connection_manager._send_to_client(
                websocket,
                {
                    "type": "recommendation",
                    "data": recommendation_to_dict(rec),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Send open positions
        open_positions = position_tracker.get_open_positions()
        for pos in open_positions:
            await connection_manager._send_to_client(
                websocket,
                {
                    "type": "position_opened",
                    "data": position_to_dict(pos),
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
