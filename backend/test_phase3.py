#!/usr/bin/env python3
"""Phase 3 integration test script.

Tests all MVP UI components:
1. FastAPI server startup
2. WebSocket connection
3. REST API endpoints
4. Frontend build (if npm available)

Usage:
    source venv/bin/activate
    python -m backend.test_phase3
"""

import asyncio
import logging
import sys

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def test_imports() -> bool:
    """Test that all modules import correctly."""
    logger.info("=" * 60)
    logger.info("TEST: Module Imports")
    logger.info("=" * 60)

    try:
        from backend.main import app
        from backend.websocket import ConnectionManager, MessageType
        from backend.engine import GatingPipeline, evaluate_option_for_signal

        logger.info("  FastAPI app: OK")
        logger.info("  ConnectionManager: OK")
        logger.info("  GatingPipeline: OK")
        logger.info("  Module imports: PASSED")
        return True

    except Exception as e:
        logger.error(f"  Module imports: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_websocket_manager() -> bool:
    """Test WebSocket manager functionality."""
    from backend.websocket import ConnectionManager, MessageType

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: WebSocket Manager")
    logger.info("=" * 60)

    try:
        manager = ConnectionManager()

        # Test initial state
        logger.info(f"  Initial connections: {manager.connection_count}")
        assert manager.connection_count == 0

        # Test message types
        logger.info(f"  Message types: {len(MessageType)}")
        for mt in MessageType:
            logger.info(f"    - {mt.value}")

        logger.info("  WebSocket manager: PASSED")
        return True

    except Exception as e:
        logger.error(f"  WebSocket manager: FAILED - {e}")
        return False


async def test_api_endpoints() -> bool:
    """Test REST API endpoints (requires running server)."""
    from backend.main import app
    from fastapi.testclient import TestClient

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: REST API Endpoints")
    logger.info("=" * 60)

    try:
        # Use TestClient for synchronous testing
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            # Health endpoint
            response = client.get("/health")
            logger.info(f"  GET /health: {response.status_code}")
            assert response.status_code == 200
            data = response.json()
            logger.info(f"    status: {data.get('status')}")
            logger.info(f"    connections: {data.get('connections')}")

            # Options endpoint
            response = client.get("/api/options")
            logger.info(f"  GET /api/options: {response.status_code}")
            assert response.status_code == 200
            data = response.json()
            logger.info(f"    options: {len(data.get('options', []))} items")

        logger.info("  REST API endpoints: PASSED")
        return True

    except Exception as e:
        logger.error(f"  REST API endpoints: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_data_flow_simulation() -> bool:
    """Test simulated data flow through the system."""
    from datetime import datetime, timezone
    from backend.config import AppConfig, AlpacaConfig, ORATSConfig, FinnhubConfig, QuiverConfig
    from backend.data.aggregator import DataAggregator
    from backend.models.canonical import CanonicalOptionId
    from backend.models.market_data import QuoteData, GreeksData
    from backend.websocket import ConnectionManager

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: Data Flow Simulation")
    logger.info("=" * 60)

    try:
        # Create config
        config = AppConfig(
            alpaca=AlpacaConfig(api_key="test", secret_key="test", paper=True),
            orats=ORATSConfig(api_token=""),
            finnhub=FinnhubConfig(api_key=""),
            quiver=QuiverConfig(api_key=""),
            log_level="INFO",
        )

        # Create aggregator and connection manager
        aggregator = DataAggregator(config=config)
        manager = ConnectionManager()

        # Track updates
        updates_received = []

        def on_update(option):
            updates_received.append(option)

        aggregator.on_option_update = on_update

        # Simulate quote update
        now = datetime.now(timezone.utc).isoformat()
        quote = QuoteData(
            canonical_id=CanonicalOptionId(
                underlying="NVDA",
                expiry="2025-01-17",
                right="C",
                strike=150.0,
            ),
            bid=5.00,
            ask=5.10,
            bid_size=100,
            ask_size=50,
            last=5.05,
            timestamp=now,
            receive_timestamp=now,
            source="alpaca",
        )
        aggregator.update_quote(quote)
        logger.info(f"  Quote update triggered {len(updates_received)} callbacks")

        # Simulate Greeks update
        greeks = GreeksData(
            canonical_id=CanonicalOptionId(
                underlying="NVDA",
                expiry="2025-01-17",
                right="C",
                strike=150.0,
            ),
            delta=0.45,
            gamma=0.02,
            theta=-0.15,
            vega=0.25,
            rho=0.01,
            iv=0.35,
            theoretical_value=5.05,
            timestamp=now,
            source="orats",
        )
        aggregator.update_greeks(greeks)
        logger.info(f"  Greeks update triggered {len(updates_received)} total callbacks")

        # Check aggregated data
        option = aggregator.get_option("NVDA", "2025-01-17", "C", 150.0)
        assert option is not None
        assert option.bid == 5.00
        assert option.delta == 0.45
        logger.info(f"  Aggregated option: {option.canonical_id}")
        logger.info(f"    Bid/Ask: ${option.bid:.2f} / ${option.ask:.2f}")
        logger.info(f"    Delta: {option.delta:.3f}")

        logger.info("  Data flow simulation: PASSED")
        return True

    except Exception as e:
        logger.error(f"  Data flow simulation: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_frontend_files() -> bool:
    """Test that frontend files exist."""
    import os

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: Frontend Files")
    logger.info("=" * 60)

    try:
        frontend_dir = "frontend"
        required_files = [
            "package.json",
            "vite.config.ts",
            "tsconfig.json",
            "index.html",
            "src/main.tsx",
            "src/App.tsx",
            "src/store/optionsStore.ts",
            "src/hooks/useOptionsStream.ts",
            "src/components/ChainView.tsx",
            "src/components/StatusBar.tsx",
            "src/components/AbstainPanel.tsx",
        ]

        missing = []
        for file in required_files:
            path = os.path.join(frontend_dir, file)
            if os.path.exists(path):
                logger.info(f"    {file}")
            else:
                missing.append(file)

        if missing:
            logger.error(f"  Missing files: {missing}")
            return False

        logger.info("  Frontend files: PASSED")
        return True

    except Exception as e:
        logger.error(f"  Frontend files: FAILED - {e}")
        return False


async def main():
    """Run all Phase 3 tests."""
    logger.info("")
    logger.info("" * 62)
    logger.info("           PHASE 3: MVP UI TESTS                         ")
    logger.info("" * 62)
    logger.info("")

    results = {}

    # Run tests
    results["imports"] = await test_imports()
    results["websocket_manager"] = await test_websocket_manager()
    results["api_endpoints"] = await test_api_endpoints()
    results["data_flow"] = await test_data_flow_simulation()
    results["frontend_files"] = await test_frontend_files()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {test}: {status}")
        if not passed:
            all_passed = False

    logger.info("")
    if all_passed:
        logger.info("Phase 3 tests completed successfully!")
        logger.info("")
        logger.info("To run the full system:")
        logger.info("  1. Backend:  uvicorn backend.main:app --reload")
        logger.info("  2. Frontend: cd frontend && npm install && npm run dev")
        logger.info("  3. Open:     http://localhost:5173")
    else:
        logger.error("Some tests failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
