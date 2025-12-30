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
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import load_config
from backend.data import AlpacaAccountClient, AlpacaOptionsClient, AlpacaRestClient, DataAggregator, EODHDClient, MockDataGenerator, OptionsIndicators, ORATSClient, SubscriptionManager
from backend.data.aggregator import AggregatedOptionData
from backend.engine import (
    ExitSignal,
    GatingPipeline,
    PortfolioState,
    PositionTracker,
    Recommendation,
    Recommender,
    SessionTracker,
    TrackedPosition,
    evaluate_option_for_signal,
)
from backend.engine.regime_detector import RegimeDetector, RegimeConfig, RegimeType
from backend.engine.regime_signals import (
    RegimeSignalGenerator,
    SignalGeneratorConfig,
    PriceData,
    SignalType,
    TechnicalIndicators,
    select_atm_option,
)
from backend.logging import (
    EvaluationLogger,
    EvaluationMetrics,
    MetricsCalculator,
    Outcome,
    SessionRecorder,
    SessionReplayer,
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
alpaca_rest_client: AlpacaRestClient | None = None  # For options chain OI (live trading)
orats_client: ORATSClient | None = None
eodhd_client: EODHDClient | None = None  # For historical backtesting only
subscription_manager: SubscriptionManager | None = None

# Options indicators cache (symbol -> (indicators, timestamp))
# Cache for 1 hour since options OI data is from EOD
_options_indicators_cache: dict[str, tuple[OptionsIndicators | None, float]] = {}
OPTIONS_CACHE_TTL = 3600.0  # 1 hour

# Multi-symbol mock mode support
# Dictionary of symbol -> MockDataGenerator for all watchlist symbols
mock_generators: dict[str, MockDataGenerator] = {}
# Track which symbols are in the user's watchlist
watchlist_symbols: set[str] = set()
# The currently selected symbol (for UI focus)
current_symbol: str = "QQQ"

# Recommender and session tracking
recommender = Recommender()
session_tracker = SessionTracker()
position_tracker = PositionTracker()
evaluation_logger = EvaluationLogger(persist_path="./logs/evaluations")
metrics_calculator = MetricsCalculator()
session_recorder = SessionRecorder(persist_path="./logs/replays")
session_replayer = SessionReplayer(persist_path="./logs/replays")

# Regime-filtered strategy components (validated: 71 trades, 43.7% win, +17.4% avg return)
regime_detector: RegimeDetector | None = None
regime_signal_generator: RegimeSignalGenerator | None = None

# Rate limiting: track last recommendation time per contract AND per symbol
_last_recommendation_time: dict[str, float] = {}  # Per contract (symbol+strike+expiry+right)
_last_symbol_recommendation_time: dict[str, float] = {}  # Per underlying symbol
RECOMMENDATION_COOLDOWN = 300.0  # Don't recommend same contract within 5 minutes (matches TTL)
SYMBOL_COOLDOWN = 300.0  # Don't recommend same underlying within 5 minutes (matches TTL)

# Background tasks
_background_tasks: set[asyncio.Task] = set()


async def get_options_indicators(symbol: str) -> OptionsIndicators | None:
    """Get options indicators (P/C Ratio, Max Pain) for a symbol with caching.

    For live trading: Uses Alpaca Trading API to fetch options chain OI
    For backtesting: Uses EODHD for historical data (handled in run_backtest.py)

    Results are cached for 1 hour since options OI is typically EOD data.

    Args:
        symbol: Stock symbol (e.g., "TSLA")

    Returns:
        OptionsIndicators or None if unavailable
    """
    global alpaca_rest_client

    if alpaca_rest_client is None:
        return None

    import time
    from datetime import datetime, timezone
    now = time.time()

    # Check cache
    if symbol in _options_indicators_cache:
        cached, cached_time = _options_indicators_cache[symbol]
        if now - cached_time < OPTIONS_CACHE_TTL:
            return cached

    # Fetch fresh data from Alpaca
    try:
        chain_data = await alpaca_rest_client.get_options_chain_oi(symbol)

        # Convert to OptionsIndicators format
        put_call_ratio = chain_data.get("put_call_ratio")

        # Determine P/C signal
        pcr_signal = None
        if put_call_ratio is not None:
            if put_call_ratio > 1.2:
                pcr_signal = "bullish"  # Contrarian: excessive bearishness
            elif put_call_ratio < 0.6:
                pcr_signal = "bearish"  # Contrarian: excessive bullishness
            else:
                pcr_signal = "neutral"

        indicators = OptionsIndicators(
            symbol=symbol,
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            put_call_ratio=put_call_ratio,
            max_pain=None,  # Max Pain not calculated from Alpaca (requires full chain analysis)
            total_call_oi=chain_data.get("total_call_oi", 0),
            total_put_oi=chain_data.get("total_put_oi", 0),
            num_contracts=chain_data.get("num_contracts", 0),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        _options_indicators_cache[symbol] = (indicators, now)

        logger.info(
            f"Alpaca {symbol}: P/C={put_call_ratio:.2f if put_call_ratio else 'N/A'} "
            f"({pcr_signal}), {chain_data.get('num_contracts', 0)} contracts"
        )
        return indicators

    except Exception as e:
        logger.warning(f"Failed to fetch options indicators for {symbol}: {e}")
        _options_indicators_cache[symbol] = (None, now)
        return None


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


def exit_signal_to_dict(signal: ExitSignal, position: TrackedPosition) -> dict[str, Any]:
    """Convert ExitSignal to JSON-serializable dict with position context."""
    return {
        "positionId": signal.position_id,
        "reason": signal.reason,
        "currentPrice": signal.current_price,
        "pnl": signal.pnl,
        "pnlPercent": signal.pnl_percent,
        "urgency": signal.urgency,
        "trigger": signal.trigger,
        # Include position context for frontend display
        "position": {
            "underlying": position.underlying,
            "strike": position.strike,
            "right": position.right,
            "expiry": position.expiry,
            "action": position.action,
            "contracts": position.contracts,
            "fillPrice": position.fill_price,
        },
    }


async def on_option_update(option: AggregatedOptionData) -> None:
    """Callback when option data updates - broadcast to clients.

    Only broadcasts if the option is for the currently focused symbol.
    """
    try:
        # Only broadcast options for the current symbol to avoid UI flicker
        if option.canonical_id.underlying != current_symbol:
            return

        if connection_manager.connection_count > 0:
            await connection_manager.broadcast_option_update(option_to_dict(option))
    except Exception as e:
        logger.error(f"Error broadcasting option update: {e}")


async def on_underlying_update(underlying: UnderlyingData) -> None:
    """Callback when underlying data updates - broadcast to clients.

    Only broadcasts if the underlying is for the currently focused symbol.
    """
    # Only broadcast for the current symbol to avoid UI flicker
    if underlying.symbol != current_symbol:
        return

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
    """Background task to periodically evaluate gates for ALL watchlist symbols.

    Also integrates the regime-filtered strategy:
    1. Updates regime detector with WSB sentiment
    2. Checks for pullback/bounce signals during active regimes
    3. Generates regime-based recommendations for enabled symbols (TSLA only validated)
    """
    global mock_generators, watchlist_symbols, current_symbol, aggregator
    global _last_recommendation_time, _last_symbol_recommendation_time
    global regime_detector, regime_signal_generator

    logger.info("Starting gate evaluation loop (5s interval, multi-symbol)")

    while True:
        try:
            await asyncio.sleep(5)  # Evaluate every 5 seconds

            now = datetime.now(timezone.utc)
            now_ts = now.timestamp()

            # Evaluate ALL symbols in the watchlist
            symbols_to_evaluate = list(mock_generators.keys()) if mock_generators else []

            if not symbols_to_evaluate and aggregator:
                # Real mode - just evaluate the subscribed symbol
                symbols_to_evaluate = ["NVDA"]

            # Only evaluate regime-enabled symbols (TSLA only for now)
            # This reduces log noise and CPU waste from evaluating non-validated symbols
            config = load_config()
            enabled_symbols = set(config.regime_strategy.enabled_symbols)

            for symbol in symbols_to_evaluate:
                # Skip non-enabled symbols entirely
                if symbol not in enabled_symbols:
                    continue

                # Check per-symbol rate limit (skip signal generation, but still evaluate for UI)
                last_symbol_rec_time = _last_symbol_recommendation_time.get(symbol, 0)
                symbol_on_cooldown = (now_ts - last_symbol_rec_time) < SYMBOL_COOLDOWN

                # Get options and underlying for this symbol
                if mock_generators and symbol in mock_generators:
                    generator = mock_generators[symbol]
                    all_options = generator.get_all_options()
                    underlying = generator.get_underlying()
                elif aggregator:
                    all_options = aggregator.get_all_options()
                    underlying = aggregator.get_underlying(symbol)
                else:
                    continue

                if not all_options or not underlying:
                    continue

                # Fetch sentiment data for this symbol
                sentiment = None
                try:
                    scanner = get_scanner()
                    if scanner._sentiment_aggregator:
                        sentiment = await scanner._sentiment_aggregator.get_sentiment(symbol)
                except Exception as e:
                    logger.debug(f"Error fetching sentiment for {symbol}: {e}")

                # === REGIME STRATEGY: Update regime and check for signals ===
                # We already filtered to enabled symbols above, so just check components exist
                if regime_detector and regime_signal_generator:

                    # Update regime with WSB sentiment if available
                    if sentiment and sentiment.wsb_score is not None:
                        regime_detector.update_regime(symbol, sentiment.wsb_score)

                    # Check for regime signals using OHLC from underlying
                    # Note: In live trading, we'd get intraday OHLC from minute bars
                    # For now, use current price as approximation
                    active_regime = regime_detector.get_active_regime(symbol)
                    if active_regime and active_regime.is_active:
                        # Create price data for signal check
                        # In real trading, this would use intraday high/low from bars
                        # Using underlying price with simulated volatility for now
                        price_volatility = underlying.price * 0.02  # ~2% daily range
                        price_data = PriceData(
                            symbol=symbol,
                            current=underlying.price,
                            high=underlying.price + price_volatility,  # Simulated
                            low=underlying.price - price_volatility,   # Simulated
                            open=underlying.price,
                            timestamp=now,
                            # Technical indicators would come from real data feed
                            # For now, create placeholder that allows most signals through
                            technicals=TechnicalIndicators(
                                bb_pct=0.4 if active_regime.regime_type.is_bullish else 0.6,
                                macd_hist=0.1 if active_regime.regime_type.is_bullish else -0.1,
                                macd_prev_hist=0.05 if active_regime.regime_type.is_bullish else -0.05,
                                sma_20=underlying.price * 0.98,  # Slightly below = bullish
                                trend_bullish=active_regime.regime_type.is_bullish,
                            ),
                        )

                        # Check for entry signal
                        regime_signal = regime_signal_generator.check_entry_signal(price_data)

                        if regime_signal.signal_type != SignalType.NO_SIGNAL:
                            logger.info(
                                f"[REGIME SIGNAL] {symbol}: {regime_signal.signal_type.value} - "
                                f"{regime_signal.trigger_reason}"
                            )

                            # Select ATM option for this signal
                            # Convert OptionData to dict format expected by select_atm_option
                            option_type = "call" if regime_signal.signal_type == SignalType.BUY_CALL else "put"
                            available_options_list = [
                                {
                                    "type": "call" if o.canonical_id.right == "C" else "put",
                                    "strike": o.canonical_id.strike,
                                    "expiry": o.canonical_id.expiry,
                                    "bid": o.bid or 0,
                                    "ask": o.ask or 0,
                                    "open_interest": o.open_interest or 0,
                                    "volume": o.volume or 0,
                                    "delta": o.delta,
                                }
                                for o in all_options
                                if (o.canonical_id.right == "C") == (option_type == "call")
                            ]

                            # Select the best option
                            selected_option = select_atm_option(
                                regime_signal,
                                available_options_list,
                                regime_signal_generator.config,
                            )

                            # Build option data for broadcast
                            option_data = None
                            if selected_option:
                                # Calculate position size (10% of $10k portfolio = $1000)
                                position_size_pct = regime_signal_generator.config.position_size_pct
                                portfolio_size = 10000  # Default portfolio size
                                max_cost = portfolio_size * (position_size_pct / 100)
                                contract_cost = selected_option.mid * 100
                                suggested_contracts = max(1, int(max_cost / contract_cost)) if contract_cost > 0 else 1

                                option_data = {
                                    "strike": selected_option.strike,
                                    "expiry": selected_option.expiry,
                                    "dte": selected_option.dte,
                                    "bid": selected_option.bid,
                                    "ask": selected_option.ask,
                                    "mid": selected_option.mid,
                                    "delta": selected_option.delta,
                                    "open_interest": selected_option.open_interest,
                                    "volume": selected_option.volume,
                                    "suggested_contracts": suggested_contracts,
                                    "total_cost": round(suggested_contracts * selected_option.mid * 100, 2),
                                }

                            # Generate unique signal ID
                            signal_id = f"{symbol}-{now.strftime('%Y%m%d%H%M%S')}-{regime_signal.signal_type.value}"

                            # Broadcast regime signal to frontend
                            await connection_manager.broadcast({
                                "type": "regime_signal",
                                "data": {
                                    "id": signal_id,
                                    "symbol": symbol,
                                    "signal_type": regime_signal.signal_type.value,
                                    "regime_type": regime_signal.regime_type.value,
                                    "trigger_reason": regime_signal.trigger_reason,
                                    "trigger_pct": regime_signal.trigger_pct,
                                    "entry_price": regime_signal.entry_price,
                                    "generated_at": regime_signal.generated_at.isoformat(),
                                    "option": option_data,
                                },
                                "timestamp": now.isoformat(),
                            })

                # Fetch options indicators (P/C Ratio, Max Pain) for this symbol
                options_ind = await get_options_indicators(symbol)

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

                    # Evaluate gates with sentiment data and options indicators
                    result = evaluate_option_for_signal(
                        option=option,
                        underlying=underlying,
                        action="BUY_CALL",
                        sentiment=sentiment,
                        options_indicators=options_ind,
                    )

                    if result.passed and score > best_score:
                        best_option = option
                        best_result = result
                        best_score = score
                        best_confidence = confidence

                # Only broadcast gate status for the CURRENT symbol (UI focus)
                if symbol == current_symbol:
                    if best_option is None:
                        atm_option = min(
                            candidates,
                            key=lambda o: abs(o.canonical_id.strike - underlying.price)
                        )
                        result = evaluate_option_for_signal(
                            option=atm_option,
                            underlying=underlying,
                            action="BUY_CALL",
                            sentiment=sentiment,
                            options_indicators=options_ind,
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
                            # Options flow indicators (Max Pain, P/C Ratio)
                            "optionsIndicators": options_ind.to_dict() if options_ind else None,
                        },
                        "timestamp": now.isoformat(),
                    })

                    # Broadcast abstain status for current symbol only
                    if result.abstain or best_option is None:
                        if result.abstain:
                            failed_gates = [
                                {"name": g.gate_name, "message": g.message}
                                for g in result.all_results if not g.passed
                            ]
                            await connection_manager.broadcast_abstain({
                                "reason": result.abstain.reason.value,
                                "resumeCondition": result.abstain.resume_condition,
                                "failedGates": failed_gates,
                            })

                            # Log abstain for evaluation
                            portfolio_state = {
                                "total_exposure": position_tracker.get_total_exposure(),
                                "open_position_count": len(position_tracker.get_open_positions()),
                                "cash_available": 5000 - position_tracker.get_total_exposure(),
                            }
                            evaluation_logger.log_abstain(
                                abstain=result.abstain,
                                underlying=underlying,
                                options=all_options,
                                portfolio_state=portfolio_state,
                                session_id=session_tracker.get_stats().session_id,
                                failed_gates=failed_gates,
                            )
                    else:
                        # Clear abstain status for current symbol
                        await connection_manager.broadcast_abstain(None)

                # Generate recommendation for enabled symbols only
                # We already filtered to enabled_symbols at the start of the loop
                if best_option is not None and not symbol_on_cooldown:
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
                            confidence=best_confidence,
                            rationale=recommendation.rationale,
                            gate_results=recommendation.gate_results,
                            quote_age=recommendation.quote_age,
                            greeks_age=recommendation.greeks_age,
                            underlying_age=recommendation.underlying_age,
                            valid_until=recommendation.valid_until,
                        )

                        # Check session limits
                        confirmed_exposure = position_tracker.get_total_exposure()
                        max_exposure = 25000.0
                        max_single_position = 10000.0

                        proposed_cost = adjusted_rec.total_cost
                        if proposed_cost > max_single_position:
                            allowed = False
                            reason = f"Position ${proposed_cost:.0f} exceeds single position limit ${max_single_position:.0f}"
                        elif confirmed_exposure + proposed_cost > max_exposure:
                            allowed = False
                            reason = f"Would exceed session limit: ${confirmed_exposure + proposed_cost:.0f} > ${max_exposure:.0f}"
                        else:
                            allowed = True
                            reason = None

                        if allowed:
                            # Record recommendation time for rate limiting (both contract and symbol)
                            option_key = get_option_key(best_option)
                            _last_recommendation_time[option_key] = now_ts
                            _last_symbol_recommendation_time[symbol] = now_ts

                            # Add to session and broadcast
                            session_tracker.add_recommendation(adjusted_rec)

                            await connection_manager.broadcast({
                                "type": "recommendation",
                                "data": recommendation_to_dict(adjusted_rec),
                                "timestamp": now.isoformat(),
                            })

                            # Log recommendation for evaluation
                            portfolio_state = {
                                "total_exposure": position_tracker.get_total_exposure(),
                                "open_position_count": len(position_tracker.get_open_positions()),
                                "cash_available": 5000 - position_tracker.get_total_exposure(),
                            }
                            evaluation_logger.log_recommendation(
                                recommendation=adjusted_rec,
                                underlying=underlying,
                                options=all_options,
                                portfolio_state=portfolio_state,
                                session_id=session_tracker.get_stats().session_id,
                            )

                            logger.info(
                                f"Recommendation: {adjusted_rec.action} "
                                f"{adjusted_rec.underlying} ${adjusted_rec.strike} "
                                f"@ ${adjusted_rec.premium:.2f} "
                                f"(score: {best_score:.0f}, conf: {best_confidence}%)"
                            )
                        else:
                            logger.warning(f"Recommendation blocked for {symbol}: {reason}")

            # === Exit Signal Checking for Open Positions ===
            # Check all open positions for exit conditions
            open_positions = position_tracker.get_open_positions()
            for pos in open_positions:
                # Find current option data for this position
                current_price = None
                delta = None
                dte = None

                # Get the option data from the appropriate source
                if mock_generators and pos.underlying in mock_generators:
                    gen = mock_generators[pos.underlying]
                    all_opts = gen.get_all_options()
                    # Find the matching option
                    for opt in all_opts:
                        cid = opt.canonical_id
                        if (cid.strike == pos.strike and
                            cid.expiry == pos.expiry and
                            cid.right == pos.right):
                            current_price = opt.mid
                            delta = opt.delta
                            break
                elif aggregator:
                    all_opts = aggregator.get_all_options()
                    for opt in all_opts:
                        cid = opt.canonical_id
                        if (cid.underlying == pos.underlying and
                            cid.strike == pos.strike and
                            cid.expiry == pos.expiry and
                            cid.right == pos.right):
                            current_price = opt.mid
                            delta = opt.delta
                            break

                # Calculate DTE from expiry date
                try:
                    from datetime import date
                    expiry_date = date.fromisoformat(pos.expiry)
                    dte = (expiry_date - now.date()).days
                except (ValueError, AttributeError):
                    dte = None

                # Get sentiment for position's underlying
                sentiment_score = None
                try:
                    scanner = get_scanner()
                    if scanner._sentiment_aggregator:
                        pos_sentiment = await scanner._sentiment_aggregator.get_sentiment(pos.underlying)
                        if pos_sentiment:
                            sentiment_score = pos_sentiment.combined_score
                except Exception:
                    pass  # Sentiment optional, don't block exit signal checking

                # Update position and check for exit signals
                exit_signal = position_tracker.update_position(
                    position_id=pos.id,
                    current_price=current_price,
                    delta=delta,
                    dte=dte,
                    sentiment_score=sentiment_score,
                )

                # Always broadcast position updates so frontend shows live P/L
                await connection_manager.broadcast({
                    "type": "position_updated",
                    "data": position_to_dict(pos),
                    "timestamp": now.isoformat(),
                })

                # Broadcast exit signal if triggered
                if exit_signal:
                    logger.info(
                        f"EXIT SIGNAL: {pos.underlying} ${pos.strike}{pos.right} - {exit_signal.reason}"
                    )
                    await connection_manager.broadcast({
                        "type": "exit_signal",
                        "data": exit_signal_to_dict(exit_signal, pos),
                        "timestamp": now.isoformat(),
                    })

            # Always broadcast session stats after evaluation cycle
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


def create_mock_generator_for_symbol(symbol: str) -> MockDataGenerator:
    """Create a mock data generator for a specific symbol.

    Each generator is independent and generates its own price/options data.
    """
    def mock_option_callback(option: AggregatedOptionData) -> None:
        asyncio.create_task(on_option_update(option))

    def mock_underlying_callback(underlying: UnderlyingData) -> None:
        asyncio.create_task(on_underlying_update(underlying))

    generator = MockDataGenerator(
        on_option_update=mock_option_callback,
        on_underlying_update=mock_underlying_callback,
        update_interval=1.0,
    )
    generator.set_symbol(symbol)
    return generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    global aggregator, alpaca_client, orats_client, subscription_manager
    global mock_generators, watchlist_symbols, current_symbol
    global regime_detector, regime_signal_generator

    logger.info("Starting OptionsRadar server...")

    # Initialize regime strategy components (validated: 71 trades, +17.4% avg return)
    try:
        config = load_config()
        regime_config = RegimeConfig(
            strong_bullish_threshold=config.regime_strategy.strong_bullish_threshold,
            moderate_bullish_threshold=config.regime_strategy.moderate_bullish_threshold,
            moderate_bearish_threshold=config.regime_strategy.moderate_bearish_threshold,
            strong_bearish_threshold=config.regime_strategy.strong_bearish_threshold,
            regime_window_days=config.regime_strategy.regime_window_days,
        )
        regime_detector = RegimeDetector(config=regime_config)

        signal_config = SignalGeneratorConfig(
            pullback_threshold=config.regime_strategy.pullback_threshold,
            bounce_threshold=config.regime_strategy.bounce_threshold,
            target_dte=config.regime_strategy.target_dte,
            min_oi=config.regime_strategy.min_open_interest,
            min_volume=config.regime_strategy.min_volume,
            max_concurrent_positions=config.regime_strategy.max_concurrent_positions,
            min_days_between_entries=config.regime_strategy.min_days_between_entries,
            position_size_pct=config.regime_strategy.position_size_pct,
        )
        regime_signal_generator = RegimeSignalGenerator(
            regime_detector=regime_detector,
            config=signal_config,
        )
        logger.info(
            f"Regime strategy initialized: "
            f"thresholds (+{regime_config.strong_bullish_threshold:.2f}/"
            f"+{regime_config.moderate_bullish_threshold:.2f}/"
            f"{regime_config.moderate_bearish_threshold:.2f}/"
            f"{regime_config.strong_bearish_threshold:.2f}), "
            f"{regime_config.regime_window_days}d window"
        )
    except Exception as e:
        logger.warning(f"Could not initialize regime strategy: {e}")

    # Check for mock mode
    if MOCK_MODE:
        logger.info("=" * 60)
        logger.info("MOCK DATA MODE ENABLED")
        logger.info("  Using simulated multi-symbol options data")
        logger.info("  Set MOCK_DATA=false to use real market data")
        logger.info("=" * 60)

        # Load watchlist from config (use default if config load fails)
        try:
            config = load_config()
            all_symbols = list(config.watchlist)
        except ValueError:
            # No API keys configured, use default watchlist
            all_symbols = [
                "NVDA", "TSLA", "PLTR", "COIN", "MARA", "RKLB", "ASTS",
                "QQQ", "AAPL", "SPY", "AMD", "GOOGL", "AMZN", "META", "MSFT", "GME", "SMCI",
            ]

        watchlist_symbols = set(all_symbols)
        current_symbol = all_symbols[0] if all_symbols else "SPY"

        # Create a mock generator for each watchlist symbol
        for symbol in all_symbols:
            generator = create_mock_generator_for_symbol(symbol)
            mock_generators[symbol] = generator

            # Generate initial data
            underlying, options = generator.generate_initial_chain()
            logger.info(f"Generated {len(options)} mock options for {symbol}")

            # Start streaming updates for this symbol
            task = asyncio.create_task(generator.start_streaming())
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

        logger.info(f"Tracking {len(mock_generators)} symbols: {list(mock_generators.keys())}")

        # Start gate evaluation loop
        task = asyncio.create_task(gate_evaluation_loop())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        logger.info("OptionsRadar server ready (MOCK MODE)")

        yield

        # Shutdown mock mode
        logger.info("Shutting down OptionsRadar server...")
        for symbol, generator in mock_generators.items():
            generator.stop_streaming()

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

    # Initialize Alpaca REST client for options chain OI (P/C Ratio)
    # Uses Alpaca Trading API which has OI data - no EODHD quota burned
    # EODHD is now only used for historical backtesting (in run_backtest.py)
    alpaca_rest_client = AlpacaRestClient(config=config.alpaca)
    logger.info("Alpaca REST client initialized for options chain OI (P/C Ratio)")

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
    if mock_generators:
        total_options = sum(g.option_count for g in mock_generators.values())
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "connections": connection_manager.connection_count,
            "optionsCount": total_options,
            "symbolsTracked": len(mock_generators),
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
    """Get current market status from Alpaca (or simulated in mock mode)."""
    from backend.data.market_hours import CT

    # In mock mode, simulate market as open
    if mock_generators:
        now = datetime.now(CT)
        # Simulate market open 9:30 AM - 4:00 PM CT on weekdays
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute

        is_weekday = weekday < 5
        is_market_hours = (hour > 9 or (hour == 9 and minute >= 30)) and hour < 16
        is_open = is_weekday and is_market_hours

        # Calculate next open/close
        if is_open:
            # Market closes at 4 PM today
            close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
            seconds_until_close = int((close_time - now).total_seconds())
            return {
                "isOpen": True,
                "currentTime": now.strftime("%I:%M:%S %p %Z"),
                "nextOpen": "Tomorrow 9:30 AM CT" if weekday < 4 else "Monday 9:30 AM CT",
                "nextClose": "4:00 PM CT",
                "timeUntilOpen": None,
                "timeUntilClose": f"{seconds_until_close // 3600}h {(seconds_until_close % 3600) // 60}m",
                "secondsUntilOpen": None,
                "secondsUntilClose": seconds_until_close,
                "timezone": "Central Time",
                "mock": True,
            }
        else:
            # Market is closed - calculate next open
            if weekday >= 5:
                # Weekend - next open is Monday
                days_until_monday = (7 - weekday) % 7 or 7
                next_open = (now + timedelta(days=days_until_monday)).replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
            elif hour >= 16:
                # After hours - next open is tomorrow (or Monday)
                if weekday == 4:
                    next_open = (now + timedelta(days=3)).replace(
                        hour=9, minute=30, second=0, microsecond=0
                    )
                else:
                    next_open = (now + timedelta(days=1)).replace(
                        hour=9, minute=30, second=0, microsecond=0
                    )
            else:
                # Before market open today
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

            seconds_until_open = int((next_open - now).total_seconds())
            return {
                "isOpen": False,
                "currentTime": now.strftime("%I:%M:%S %p %Z"),
                "nextOpen": next_open.strftime("%a %I:%M %p CT"),
                "nextClose": "4:00 PM CT",
                "timeUntilOpen": f"{seconds_until_open // 3600}h {(seconds_until_open % 3600) // 60}m",
                "timeUntilClose": None,
                "secondsUntilOpen": seconds_until_open,
                "secondsUntilClose": None,
                "timezone": "Central Time",
                "mock": True,
            }

    # Real mode - use Alpaca API
    from backend.data.market_hours import check_market_hours

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
    """Get current snapshot of all options data for the current symbol."""
    # Check mock mode first
    if mock_generators and current_symbol in mock_generators:
        generator = mock_generators[current_symbol]
        options = generator.get_all_options()
        underlying = generator.get_underlying()
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

    # Fetch sentiment for this symbol
    sentiment = None
    try:
        scanner = get_scanner()
        if scanner._sentiment_aggregator:
            sentiment = await scanner._sentiment_aggregator.get_sentiment(symbol)
    except Exception as e:
        logger.debug(f"Error fetching sentiment for {symbol}: {e}")

    # Fetch options indicators for this symbol
    options_ind = await get_options_indicators(symbol)

    results = []
    for option in options[:5]:  # Limit for demo
        result = evaluate_option_for_signal(
            option=option,
            underlying=underlying,
            action="BUY_CALL" if option.canonical_id.right == "C" else "BUY_PUT",
            sentiment=sentiment,
            options_indicators=options_ind,
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

        # Record entry for regime signal cooldown (only when trade is actually taken)
        # This blocks same-direction entries while position is open
        if rec.underlying == "TSLA":
            # Convert "C"/"P" to "call"/"put" for direction tracking
            direction = "call" if rec.right == "C" else "put"
            regime_signal_generator.record_entry(rec.underlying, direction)

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


class ManualPositionRequest(BaseModel):
    """Request body for opening a manual position (not from recommendation)."""
    underlying: str
    expiry: str
    strike: float
    right: str
    action: str
    fill_price: float
    contracts: int = 1


@app.post("/api/positions/manual")
async def open_manual_position(request: ManualPositionRequest) -> dict[str, Any]:
    """Open a position manually without a recommendation.

    Args:
        request: ManualPositionRequest with option details and fill info

    Returns:
        The created position
    """
    # Check exposure limit before opening
    proposed_cost = request.fill_price * request.contracts * 100
    current_exposure = position_tracker.get_total_exposure()
    exposure_limit = 5000.0  # $5k session limit

    if current_exposure + proposed_cost > exposure_limit:
        remaining = exposure_limit - current_exposure
        raise HTTPException(
            status_code=400,
            detail=f"Trade would exceed session limit. Current: ${current_exposure:.0f}, "
                   f"Trade: ${proposed_cost:.0f}, Limit: ${exposure_limit:.0f}, "
                   f"Remaining: ${remaining:.0f}"
        )

    try:
        import uuid
        manual_rec_id = f"manual-{str(uuid.uuid4())[:8]}"

        position = position_tracker.open_position(
            recommendation_id=manual_rec_id,
            underlying=request.underlying,
            expiry=request.expiry,
            strike=request.strike,
            right=request.right,
            action=request.action,
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
        raise HTTPException(status_code=400, detail=str(e))


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

    # Record exit for regime signal cooldown (allow new same-direction entries)
    if position.underlying == "TSLA":
        # Convert "C"/"P" to "call"/"put" for direction tracking
        direction = "call" if position.right == "C" else "put"
        regime_signal_generator.record_exit(position.underlying, direction)

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


@app.post("/api/positions/{position_id}/dismiss-exit-signal")
async def dismiss_exit_signal(position_id: str) -> dict[str, Any]:
    """Dismiss an exit signal for a position (keep position open).

    This allows the user to acknowledge the exit signal but choose to hold.
    The position returns to 'open' status and exit signals can trigger again.

    Args:
        position_id: ID of position with exit signal to dismiss

    Returns:
        The updated position
    """
    success = position_tracker.clear_exit_signal(position_id)

    if not success:
        return {"error": f"Position not found or no exit signal: {position_id}"}

    position = position_tracker.get_position(position_id)
    if not position:
        return {"error": f"Position not found: {position_id}"}

    # Broadcast position update
    await connection_manager.broadcast({
        "type": "position_updated",
        "data": position_to_dict(position),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {"position": position_to_dict(position)}


# ============================================================================
# Evaluation API Endpoints
# ============================================================================

@app.get("/api/evaluation/logs")
async def get_evaluation_logs(
    session_id: str | None = None,
    decision_type: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Get evaluation log entries.

    Args:
        session_id: Filter by session ID
        decision_type: Filter by "recommendation" or "abstain"
        limit: Maximum entries to return

    Returns:
        List of log entries
    """
    logs = evaluation_logger.get_logs(
        session_id=session_id,
        decision_type=decision_type,
        limit=limit,
    )
    return {
        "logs": [log.to_dict() for log in logs],
        "count": len(logs),
        "totalInMemory": evaluation_logger.log_count,
    }


@app.get("/api/evaluation/metrics")
async def get_evaluation_metrics(session_id: str | None = None) -> dict[str, Any]:
    """Get calculated evaluation metrics.

    Args:
        session_id: Filter by session ID (default: all)

    Returns:
        EvaluationMetrics object
    """
    logs = evaluation_logger.get_logs(session_id=session_id)
    metrics = metrics_calculator.calculate(logs)
    return {
        "metrics": metrics.to_dict(),
        "logCount": len(logs),
    }


@app.get("/api/evaluation/summary")
async def get_evaluation_summary() -> dict[str, Any]:
    """Get a quick summary of evaluation state."""
    return {
        "totalLogs": evaluation_logger.log_count,
        "recommendations": evaluation_logger.recommendation_count,
        "abstains": evaluation_logger.abstain_count,
        "abstentionRate": (
            round(evaluation_logger.abstain_count / evaluation_logger.log_count * 100, 1)
            if evaluation_logger.log_count > 0 else 0
        ),
        "logsNeedingOutcome": len(evaluation_logger.get_logs_needing_outcome()),
    }


@app.get("/api/metrics/dashboard")
async def get_metrics_dashboard() -> dict[str, Any]:
    """Get comprehensive metrics dashboard data.

    Returns:
        - Overall metrics
        - Accuracy by symbol
        - Confidence calibration buckets
        - Aligned vs non-aligned performance
        - Rolling accuracy over time
        - Position P/L summary
    """
    logs = evaluation_logger.get_logs()

    # Overall metrics
    overall = metrics_calculator.calculate(logs)

    # Accuracy by symbol
    symbols_data: dict[str, dict[str, Any]] = {}
    for log in logs:
        if log.decision_type != "recommendation":
            continue
        symbol = log.underlying
        if symbol not in symbols_data:
            symbols_data[symbol] = {
                "symbol": symbol,
                "totalSignals": 0,
                "withOutcomes": 0,
                "wins": 0,
                "losses": 0,
                "totalPnl": 0.0,
            }
        symbols_data[symbol]["totalSignals"] += 1
        if log.outcome is not None:
            symbols_data[symbol]["withOutcomes"] += 1
            if log.outcome.would_have_profited:
                symbols_data[symbol]["wins"] += 1
            elif log.outcome.would_have_profited is False:
                symbols_data[symbol]["losses"] += 1
            if log.outcome.theoretical_pnl:
                symbols_data[symbol]["totalPnl"] += log.outcome.theoretical_pnl

    # Calculate accuracy per symbol
    symbol_stats = []
    for data in symbols_data.values():
        if data["withOutcomes"] > 0:
            data["accuracy"] = round(data["wins"] / data["withOutcomes"] * 100, 1)
        else:
            data["accuracy"] = None
        symbol_stats.append(data)
    symbol_stats.sort(key=lambda x: x["totalSignals"], reverse=True)

    # Confidence calibration buckets
    conf_buckets = {
        "0-50": {"signals": 0, "wins": 0, "expectedRate": 25},
        "51-60": {"signals": 0, "wins": 0, "expectedRate": 55},
        "61-70": {"signals": 0, "wins": 0, "expectedRate": 65},
        "71-80": {"signals": 0, "wins": 0, "expectedRate": 75},
        "81-90": {"signals": 0, "wins": 0, "expectedRate": 85},
        "91-100": {"signals": 0, "wins": 0, "expectedRate": 95},
    }
    for log in logs:
        if log.decision_type != "recommendation" or log.outcome is None:
            continue
        conf = log.recommendation_confidence or 0
        if conf <= 50:
            bucket = "0-50"
        elif conf <= 60:
            bucket = "51-60"
        elif conf <= 70:
            bucket = "61-70"
        elif conf <= 80:
            bucket = "71-80"
        elif conf <= 90:
            bucket = "81-90"
        else:
            bucket = "91-100"
        conf_buckets[bucket]["signals"] += 1
        if log.outcome.would_have_profited:
            conf_buckets[bucket]["wins"] += 1

    # Calculate actual rates
    calibration = []
    for bucket, data in conf_buckets.items():
        if data["signals"] > 0:
            actual = round(data["wins"] / data["signals"] * 100, 1)
        else:
            actual = None
        calibration.append({
            "bucket": bucket,
            "signals": data["signals"],
            "wins": data["wins"],
            "expectedRate": data["expectedRate"],
            "actualRate": actual,
        })

    # Aligned vs non-aligned (parse from rationale which contains [ALIGNED]/[NOT_ALIGNED] tags)
    alignment_data = {
        "aligned": {"signals": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        "notAligned": {"signals": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        "noNews": {"signals": 0, "wins": 0, "losses": 0, "pnl": 0.0},
    }
    for log in logs:
        if log.decision_type != "recommendation":
            continue
        rationale = log.recommendation_rationale or ""
        if "[ALIGNED]" in rationale:
            key = "aligned"
        elif "[NOT_ALIGNED]" in rationale:
            key = "notAligned"
        elif "[NO_NEWS]" in rationale:
            key = "noNews"
        else:
            continue  # Can't determine alignment

        alignment_data[key]["signals"] += 1
        if log.outcome is not None:
            if log.outcome.would_have_profited:
                alignment_data[key]["wins"] += 1
            elif log.outcome.would_have_profited is False:
                alignment_data[key]["losses"] += 1
            if log.outcome.theoretical_pnl:
                alignment_data[key]["pnl"] += log.outcome.theoretical_pnl

    # Calculate alignment accuracy
    alignment_stats = []
    for key, data in alignment_data.items():
        total_outcomes = data["wins"] + data["losses"]
        accuracy = round(data["wins"] / total_outcomes * 100, 1) if total_outcomes > 0 else None
        alignment_stats.append({
            "type": key,
            "signals": data["signals"],
            "withOutcomes": total_outcomes,
            "wins": data["wins"],
            "losses": data["losses"],
            "accuracy": accuracy,
            "pnl": round(data["pnl"], 2),
        })

    # Position P/L summary from position tracker
    all_positions = position_tracker.get_all_positions()
    open_positions = [p for p in all_positions if p.status != "closed"]
    closed_positions = [p for p in all_positions if p.status == "closed"]

    total_open_pnl = sum(p.pnl for p in open_positions)
    total_closed_pnl = sum(p.pnl for p in closed_positions)
    win_trades = [p for p in closed_positions if p.pnl > 0]
    lose_trades = [p for p in closed_positions if p.pnl < 0]

    position_summary = {
        "openPositions": len(open_positions),
        "closedPositions": len(closed_positions),
        "unrealizedPnl": round(total_open_pnl, 2),
        "realizedPnl": round(total_closed_pnl, 2),
        "totalPnl": round(total_open_pnl + total_closed_pnl, 2),
        "winningTrades": len(win_trades),
        "losingTrades": len(lose_trades),
        "winRate": round(len(win_trades) / len(closed_positions) * 100, 1) if closed_positions else None,
    }

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "overall": overall.to_dict(),
        "bySymbol": symbol_stats,
        "confidenceCalibration": calibration,
        "alignmentAnalysis": alignment_stats,
        "positionSummary": position_summary,
        "config": {
            "signalsEnabled": [s for s in load_config().watchlist if s not in load_config().signal_quality.signals_disabled],
            "signalsDisabled": list(load_config().signal_quality.signals_disabled),
        }
    }


# ============================================================================
# Scanner API Endpoints
# ============================================================================

# Lazy-initialized scanner (uses API keys)
_daily_scanner = None


def get_scanner():
    """Get or create the daily scanner instance."""
    global _daily_scanner
    if _daily_scanner is None:
        from backend.engine.scanner import DailyScanner
        config = load_config()
        _daily_scanner = DailyScanner(config)
    return _daily_scanner


@app.get("/api/scanner/hot-picks")
async def get_hot_picks() -> dict[str, Any]:
    """Get curated hot picks from sentiment sources.

    Returns:
        WSB trending and top opportunities
    """
    try:
        scanner = get_scanner()
        picks = await scanner.get_hot_picks()
        return {
            "wsbTrending": picks.get("wsbTrending", []),
            "topOpportunities": picks.get("topOpportunities", []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching hot picks: {e}")
        return {
            "wsbTrending": [],
            "topOpportunities": [],
            "error": str(e),
        }


@app.get("/api/scanner/sentiment/{symbol}")
async def get_symbol_sentiment(symbol: str) -> dict[str, Any]:
    """Get combined sentiment for a specific symbol.

    Args:
        symbol: Stock symbol (e.g., "NVDA")

    Returns:
        Combined sentiment data
    """
    try:
        scanner = get_scanner()
        if scanner._sentiment_aggregator:
            sentiment = await scanner._sentiment_aggregator.get_sentiment(symbol)
            return {
                "symbol": symbol,
                "sentiment": sentiment.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        return {"error": "Sentiment aggregator not configured"}
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        return {"error": str(e)}


@app.get("/api/scanner/options-indicators/{symbol}")
async def get_symbol_options_indicators(symbol: str) -> dict[str, Any]:
    """Get options flow indicators (P/C Ratio, Max Pain) for a symbol.

    Args:
        symbol: Stock symbol (e.g., "TSLA")

    Returns:
        Options indicators including P/C Ratio, Max Pain, and open interest
    """
    try:
        indicators = await get_options_indicators(symbol)
        if indicators:
            return {
                "symbol": symbol,
                "indicators": indicators.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        return {"error": f"No options data available for {symbol}"}
    except Exception as e:
        logger.error(f"Error fetching options indicators for {symbol}: {e}")
        return {"error": str(e)}


@app.get("/api/scanner/scan")
async def run_scanner(symbols: str | None = None) -> dict[str, Any]:
    """Run the daily scanner on specified symbols.

    Args:
        symbols: Comma-separated list of symbols (default: watchlist)

    Returns:
        Scan results sorted by score
    """
    try:
        scanner = get_scanner()
        symbol_list = symbols.split(",") if symbols else None
        results = await scanner.scan(symbols=symbol_list)
        return {
            "results": [r.to_dict() for r in results],
            "opportunityCount": len([r for r in results if r.is_opportunity]),
            "strongOpportunityCount": len([r for r in results if r.is_strong_opportunity]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error running scanner: {e}")
        return {"error": str(e)}


@app.get("/api/config/watchlist")
async def get_watchlist() -> dict[str, Any]:
    """Get the configured watchlist symbols."""
    config = load_config()
    return {
        "symbols": list(config.watchlist),
        "count": len(config.watchlist),
    }


# ============================================================================
# Regime Strategy API Endpoints
# ============================================================================

@app.get("/api/regime/status")
async def get_regime_status(symbol: str = "TSLA") -> dict[str, Any]:
    """Get current regime status for a symbol.

    This endpoint returns the current market regime based on WSB sentiment
    and whether the regime strategy is ready to generate signals.

    Args:
        symbol: Stock symbol (default: TSLA, only validated ticker)

    Returns:
        - active_regime: Current regime type (strong_bullish, moderate_bullish, etc.)
        - regime_triggered_at: When the regime was triggered
        - regime_expires_at: When the regime window expires
        - sentiment_value: The WSB sentiment that triggered the regime
        - signal_generator_status: Current signal generator state
        - config: Strategy configuration parameters
    """
    if not regime_detector or not regime_signal_generator:
        return {
            "error": "Regime strategy not initialized",
            "hint": "Check startup logs for initialization errors",
        }

    try:
        # Get active regime for symbol
        active_regime = regime_detector.get_active_regime(symbol)

        # Try to update regime from latest sentiment if available
        try:
            scanner = get_scanner()
            if scanner._sentiment_aggregator:
                sentiment = await scanner._sentiment_aggregator.get_sentiment(symbol)
                if sentiment and sentiment.wsb_score is not None:
                    # Update regime with fresh WSB sentiment
                    regime_detector.update_regime(symbol, sentiment.wsb_score)
                    active_regime = regime_detector.get_active_regime(symbol)
        except Exception as e:
            logger.debug(f"Could not fetch WSB sentiment for {symbol}: {e}")

        # Build response
        regime_info = None
        if active_regime and active_regime.is_active:
            regime_info = {
                "type": active_regime.regime_type.value,
                "is_bullish": active_regime.regime_type.is_bullish,
                "is_bearish": active_regime.regime_type.is_bearish,
                "triggered_at": active_regime.triggered_date.isoformat(),
                "expires_at": active_regime.window_expires.isoformat(),
                "sentiment_value": active_regime.trigger_sentiment,
                "days_remaining": active_regime.days_remaining,
                "is_active": active_regime.is_active,
            }

        # Get signal generator status
        generator_status = regime_signal_generator.get_status()

        # Get config
        config = load_config()

        return {
            "symbol": symbol,
            "active_regime": regime_info,
            "signal_generator": generator_status,
            "config": {
                "strong_bullish_threshold": config.regime_strategy.strong_bullish_threshold,
                "moderate_bullish_threshold": config.regime_strategy.moderate_bullish_threshold,
                "moderate_bearish_threshold": config.regime_strategy.moderate_bearish_threshold,
                "strong_bearish_threshold": config.regime_strategy.strong_bearish_threshold,
                "regime_window_days": config.regime_strategy.regime_window_days,
                "pullback_threshold": config.regime_strategy.pullback_threshold,
                "bounce_threshold": config.regime_strategy.bounce_threshold,
                "target_dte": config.regime_strategy.target_dte,
                "enabled_symbols": list(config.regime_strategy.enabled_symbols),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting regime status for {symbol}: {e}")
        return {"error": str(e)}


@app.post("/api/regime/update")
async def update_regime_sentiment(symbol: str, wsb_sentiment: float) -> dict[str, Any]:
    """Manually update regime with a WSB sentiment value (for testing).

    Args:
        symbol: Stock symbol
        wsb_sentiment: WSB sentiment value (-1 to +1 scale)

    Returns:
        Updated regime status
    """
    if not regime_detector:
        return {"error": "Regime detector not initialized"}

    try:
        regime_detector.update_regime(symbol, wsb_sentiment)
        active_regime = regime_detector.get_active_regime(symbol)

        regime_info = None
        if active_regime and active_regime.is_active:
            regime_info = {
                "type": active_regime.regime_type.value,
                "is_bullish": active_regime.regime_type.is_bullish,
                "is_bearish": active_regime.regime_type.is_bearish,
                "triggered_at": active_regime.triggered_date.isoformat(),
                "expires_at": active_regime.window_expires.isoformat(),
                "sentiment_value": active_regime.trigger_sentiment,
                "days_remaining": active_regime.days_remaining,
            }

        return {
            "symbol": symbol,
            "wsb_sentiment": wsb_sentiment,
            "active_regime": regime_info,
            "message": f"Regime updated for {symbol}",
        }

    except Exception as e:
        logger.error(f"Error updating regime for {symbol}: {e}")
        return {"error": str(e)}


@app.get("/api/symbols/search")
async def search_symbols(q: str, limit: int = 20) -> dict[str, Any]:
    """Search for symbols matching a query.

    Args:
        q: Search query (partial symbol or company name)
        limit: Maximum number of results (default 20)

    Returns:
        List of matching symbols with company names
    """
    if not q or len(q) < 1:
        return {"results": [], "count": 0}

    try:
        # In mock mode, return some common symbols
        if MOCK_MODE:
            mock_symbols = [
                # Popular tech stocks
                {"symbol": "AAPL", "name": "Apple Inc."},
                {"symbol": "AMD", "name": "Advanced Micro Devices Inc."},
                {"symbol": "AMZN", "name": "Amazon.com Inc."},
                {"symbol": "ARM", "name": "Arm Holdings plc"},
                {"symbol": "AVGO", "name": "Broadcom Inc."},
                {"symbol": "CRWD", "name": "CrowdStrike Holdings Inc."},
                {"symbol": "CRWV", "name": "CoreWeave Inc."},
                {"symbol": "CRM", "name": "Salesforce Inc."},
                {"symbol": "CSCO", "name": "Cisco Systems Inc."},
                {"symbol": "GOOG", "name": "Alphabet Inc."},
                {"symbol": "GOOGL", "name": "Alphabet Inc. Class A"},
                {"symbol": "IBM", "name": "International Business Machines"},
                {"symbol": "INTC", "name": "Intel Corporation"},
                {"symbol": "META", "name": "Meta Platforms Inc."},
                {"symbol": "MSFT", "name": "Microsoft Corporation"},
                {"symbol": "MU", "name": "Micron Technology Inc."},
                {"symbol": "NFLX", "name": "Netflix Inc."},
                {"symbol": "NVDA", "name": "NVIDIA Corporation"},
                {"symbol": "ORCL", "name": "Oracle Corporation"},
                {"symbol": "PLTR", "name": "Palantir Technologies Inc."},
                {"symbol": "QCOM", "name": "Qualcomm Inc."},
                {"symbol": "SNOW", "name": "Snowflake Inc."},
                {"symbol": "TSM", "name": "Taiwan Semiconductor"},
                # AI/Tech growth
                {"symbol": "AI", "name": "C3.ai Inc."},
                {"symbol": "IONQ", "name": "IonQ Inc."},
                {"symbol": "PATH", "name": "UiPath Inc."},
                {"symbol": "RKLB", "name": "Rocket Lab USA Inc."},
                {"symbol": "SMCI", "name": "Super Micro Computer Inc."},
                {"symbol": "U", "name": "Unity Software Inc."},
                # Consumer / E-commerce
                {"symbol": "BABA", "name": "Alibaba Group Holding Ltd."},
                {"symbol": "BYND", "name": "Beyond Meat Inc."},
                {"symbol": "CHWY", "name": "Chewy Inc."},
                {"symbol": "DIS", "name": "Walt Disney Company"},
                {"symbol": "DKNG", "name": "DraftKings Inc."},
                {"symbol": "EBAY", "name": "eBay Inc."},
                {"symbol": "ETSY", "name": "Etsy Inc."},
                {"symbol": "HD", "name": "Home Depot Inc."},
                {"symbol": "KO", "name": "Coca-Cola Company"},
                {"symbol": "MCD", "name": "McDonald's Corporation"},
                {"symbol": "NKE", "name": "Nike Inc."},
                {"symbol": "PEP", "name": "PepsiCo Inc."},
                {"symbol": "SBUX", "name": "Starbucks Corporation"},
                {"symbol": "SHOP", "name": "Shopify Inc."},
                {"symbol": "TGT", "name": "Target Corporation"},
                {"symbol": "WMT", "name": "Walmart Inc."},
                # EV / Automotive
                {"symbol": "F", "name": "Ford Motor Company"},
                {"symbol": "GM", "name": "General Motors Company"},
                {"symbol": "LCID", "name": "Lucid Group Inc."},
                {"symbol": "RIVN", "name": "Rivian Automotive Inc."},
                {"symbol": "TSLA", "name": "Tesla Inc."},
                # Finance / Fintech
                {"symbol": "BAC", "name": "Bank of America Corporation"},
                {"symbol": "C", "name": "Citigroup Inc."},
                {"symbol": "COIN", "name": "Coinbase Global Inc."},
                {"symbol": "GS", "name": "Goldman Sachs Group Inc."},
                {"symbol": "HOOD", "name": "Robinhood Markets Inc."},
                {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
                {"symbol": "MA", "name": "Mastercard Inc."},
                {"symbol": "MSTR", "name": "MicroStrategy Inc."},
                {"symbol": "PYPL", "name": "PayPal Holdings Inc."},
                {"symbol": "SOFI", "name": "SoFi Technologies Inc."},
                {"symbol": "SQ", "name": "Block Inc."},
                {"symbol": "V", "name": "Visa Inc."},
                # Healthcare / Pharma
                {"symbol": "ABBV", "name": "AbbVie Inc."},
                {"symbol": "BMY", "name": "Bristol-Myers Squibb"},
                {"symbol": "JNJ", "name": "Johnson & Johnson"},
                {"symbol": "LLY", "name": "Eli Lilly and Company"},
                {"symbol": "MRNA", "name": "Moderna Inc."},
                {"symbol": "NVO", "name": "Novo Nordisk A/S"},
                {"symbol": "PFE", "name": "Pfizer Inc."},
                {"symbol": "UNH", "name": "UnitedHealth Group Inc."},
                # Aerospace / Defense
                {"symbol": "BA", "name": "Boeing Company"},
                {"symbol": "LMT", "name": "Lockheed Martin Corporation"},
                {"symbol": "RTX", "name": "RTX Corporation"},
                # Energy
                {"symbol": "CVX", "name": "Chevron Corporation"},
                {"symbol": "XOM", "name": "Exxon Mobil Corporation"},
                # Social / Media / Entertainment
                {"symbol": "ROKU", "name": "Roku Inc."},
                {"symbol": "SNAP", "name": "Snap Inc."},
                {"symbol": "SPOT", "name": "Spotify Technology S.A."},
                {"symbol": "UBER", "name": "Uber Technologies Inc."},
                {"symbol": "ZM", "name": "Zoom Video Communications Inc."},
                # ETFs
                {"symbol": "ARKK", "name": "ARK Innovation ETF"},
                {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Average ETF"},
                {"symbol": "GLD", "name": "SPDR Gold Shares"},
                {"symbol": "IWM", "name": "iShares Russell 2000 ETF"},
                {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
                {"symbol": "SOXL", "name": "Direxion Daily Semiconductor Bull 3X"},
                {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust"},
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ"},
                {"symbol": "VIX", "name": "CBOE Volatility Index"},
                {"symbol": "XLF", "name": "Financial Select Sector SPDR Fund"},
            ]
            query_upper = q.upper()
            # Prioritize symbols that start with query, then those that contain it
            starts_with = [s for s in mock_symbols if s["symbol"].startswith(query_upper)]
            contains = [s for s in mock_symbols if query_upper in s["name"].upper() and s not in starts_with]
            results = (starts_with + contains)[:limit]
            return {"results": results, "count": len(results)}

        # Real mode: use Alpaca API
        config = load_config()
        account_client = AlpacaAccountClient(config=config.alpaca)
        results = await account_client.search_symbols(q, limit=limit)
        return {"results": results, "count": len(results)}

    except Exception as e:
        logger.error(f"Error searching symbols: {e}")
        return {"results": [], "count": 0, "error": str(e)}


# ============================================================================
# Replay API Endpoints
# ============================================================================

@app.get("/api/replay/sessions")
async def list_replay_sessions(date: str | None = None) -> dict[str, Any]:
    """List available replay sessions.

    Args:
        date: Filter by date (YYYY-MM-DD)

    Returns:
        List of session metadata
    """
    sessions = session_replayer.list_sessions(date=date)
    return {
        "sessions": sessions,
        "count": len(sessions),
    }


@app.get("/api/replay/sessions/{date}/{session_id}")
async def get_replay_session(date: str, session_id: str) -> dict[str, Any]:
    """Get a specific replay session.

    Args:
        date: Date string (YYYY-MM-DD)
        session_id: Session ID

    Returns:
        Full session data
    """
    session = session_replayer.load_session(date, session_id)
    if not session:
        return {"error": f"Session not found: {date}/{session_id}"}

    return {
        "session": session.to_dict(),
    }


@app.post("/api/replay/record/start")
async def start_recording(
    description: str = "",
) -> dict[str, Any]:
    """Start recording the current session for replay.

    Args:
        description: Optional description for this recording

    Returns:
        Recording session info
    """
    current_session_id = session_tracker.get_stats().session_id
    symbols = ["SPY"]  # Currently only tracking SPY

    session = session_recorder.start_session(
        session_id=current_session_id,
        symbols=symbols,
        description=description,
    )

    return {
        "recording": True,
        "sessionId": session.session_id,
        "startTime": session.start_time,
    }


@app.post("/api/replay/record/stop")
async def stop_recording() -> dict[str, Any]:
    """Stop recording and save the session.

    Returns:
        Saved session info
    """
    session = session_recorder.end_session()
    if not session:
        return {"error": "No active recording session"}

    return {
        "recording": False,
        "sessionId": session.session_id,
        "tickCount": len(session.market_ticks),
        "decisionCount": len(session.expected_decisions),
        "savedTo": f"./logs/replays/{session.start_time[:10]}/{session.session_id}.json",
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    global current_symbol, watchlist_symbols  # Needed for modifying these in mock mode
    await connection_manager.connect(websocket)

    try:
        # Send initial data snapshot
        if mock_generators and current_symbol in mock_generators:
            # Mock mode - send mock data for current symbol
            generator = mock_generators[current_symbol]
            options = generator.get_all_options()
            for option in options:
                await connection_manager._send_to_client(
                    websocket,
                    {
                        "type": "option_update",
                        "data": option_to_dict(option),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            underlying = generator.get_underlying()
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
                elif data.startswith("{"):
                    # Parse JSON message
                    try:
                        msg = json.loads(data)
                        msg_type = msg.get("type")

                        if msg_type == "subscribe":
                            new_symbol = msg.get("symbol", "").upper()
                            if new_symbol:
                                logger.info(f"Client requesting symbol switch to {new_symbol}")

                                if MOCK_MODE:
                                    # Mock mode: switch to symbol (add to watchlist if new)
                                    # Add new symbol to watchlist if not already tracked
                                    if new_symbol not in mock_generators:
                                        logger.info(f"Adding {new_symbol} to watchlist")
                                        generator = create_mock_generator_for_symbol(new_symbol)
                                        mock_generators[new_symbol] = generator
                                        watchlist_symbols.add(new_symbol)

                                        # Generate initial data for new symbol
                                        generator.generate_initial_chain()

                                        # Start streaming for new symbol
                                        task = asyncio.create_task(generator.start_streaming())
                                        _background_tasks.add(task)
                                        task.add_done_callback(_background_tasks.discard)

                                    # Update current symbol (for UI focus)
                                    current_symbol = new_symbol

                                    # Send updated underlying data immediately
                                    generator = mock_generators[new_symbol]
                                    underlying = generator.get_underlying()
                                    await connection_manager._send_to_client(
                                        websocket,
                                        {
                                            "type": "underlying_update",
                                            "data": {
                                                "symbol": underlying.symbol,
                                                "price": underlying.price,
                                                "ivRank": underlying.iv_rank,
                                                "ivPercentile": underlying.iv_percentile,
                                                "timestamp": underlying.timestamp,
                                            },
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                        }
                                    )
                                    # Send all options for the new symbol
                                    for option in generator.get_all_options():
                                        await on_option_update(option)
                                elif subscription_manager:
                                    # Real mode: switch subscription
                                    await subscription_manager.switch_symbol(new_symbol)
                                    # Clear old data from aggregator
                                    if aggregator:
                                        aggregator.clear()

                                # Send confirmation
                                await connection_manager._send_to_client(
                                    websocket,
                                    {
                                        "type": "symbol_changed",
                                        "data": {"symbol": new_symbol},
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                    }
                                )
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from client: {data}")
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
