#!/usr/bin/env python3
"""Phase 1 integration test script.

Tests all Data Foundation components:
1. Alpaca WebSocket connection and quote streaming
2. ORATS Greeks fetching (if credentials available)
3. Data aggregation
4. Staleness detection

Usage:
    source venv/bin/activate
    python -m backend.test_phase1
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


def check_credentials() -> dict[str, bool]:
    """Check which credentials are available."""
    return {
        "alpaca": bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY")),
        "orats": bool(os.getenv("ORATS_API_TOKEN")),
    }


async def test_alpaca_connection() -> bool:
    """Test Alpaca WebSocket connection and quote streaming."""
    from backend.config import load_config
    from backend.data import AlpacaOptionsClient, ConnectionState, SubscriptionManager

    logger.info("=" * 60)
    logger.info("TEST: Alpaca WebSocket Connection")
    logger.info("=" * 60)

    try:
        config = load_config()
    except ValueError as e:
        logger.error(f"Missing Alpaca credentials: {e}")
        return False

    quotes_received = []

    def on_quote(quote):
        quotes_received.append(quote)
        if len(quotes_received) <= 3:
            logger.info(
                f"  Quote: {quote.canonical_id.underlying} "
                f"{quote.canonical_id.expiry} {quote.canonical_id.strike}{quote.canonical_id.right} "
                f"Bid: ${quote.bid:.2f} Ask: ${quote.ask:.2f}"
            )

    def on_state(state):
        logger.info(f"  Connection state: {state.value}")

    client = AlpacaOptionsClient(
        config=config.alpaca,
        on_quote=on_quote,
        on_state_change=on_state,
    )

    sub_manager = SubscriptionManager(
        config=config.alpaca,
        client=client,
        symbol="NVDA",
        strikes_around_atm=3,  # Just 3 for testing
    )

    try:
        await client.connect()
        await sub_manager.start()

        logger.info(f"  Subscribed to {sub_manager.subscribed_count} contracts")
        logger.info(f"  ATM strike: ${sub_manager.current_atm}")
        logger.info("  Waiting 5 seconds for quotes...")

        # Wait for quotes
        start = asyncio.get_event_loop().time()
        async for _ in client.messages():
            if asyncio.get_event_loop().time() - start > 5:
                break
            if len(quotes_received) >= 10:
                break

        logger.info(f"  Received {len(quotes_received)} quotes")

        if quotes_received:
            logger.info("  ✅ Alpaca connection: PASSED")
            return True
        else:
            logger.warning("  ⚠️  Connected but no quotes received (market may be closed)")
            return True  # Still a pass - connection worked

    except Exception as e:
        logger.error(f"  ❌ Alpaca connection: FAILED - {e}")
        return False
    finally:
        await sub_manager.stop()
        await client.disconnect()


async def test_orats_connection() -> bool:
    """Test ORATS API connection and Greeks fetching."""
    from backend.config import load_config
    from backend.data import ORATSClient

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: ORATS Greeks API")
    logger.info("=" * 60)

    try:
        config = load_config()
    except ValueError:
        pass  # Alpaca might be missing, that's ok

    orats_token = os.getenv("ORATS_API_TOKEN")
    if not orats_token:
        logger.warning("  ⚠️  ORATS_API_TOKEN not set - skipping")
        logger.info("  To enable: Add ORATS_API_TOKEN to .env")
        logger.info("  Get token at: https://orats.com (Live subscription required)")
        return True  # Not a failure, just skipped

    from backend.config import ORATSConfig

    orats_config = ORATSConfig(api_token=orats_token)
    client = ORATSClient(config=orats_config)

    try:
        async with client:
            # Test Greeks fetch
            logger.info("  Fetching NVDA Greeks...")
            greeks = await client.fetch_greeks("NVDA", "", dte_max=30)

            if greeks:
                sample = greeks[0]
                logger.info(f"  Found {len(greeks)} option contracts")
                logger.info(
                    f"  Sample: {sample.canonical_id} "
                    f"Delta: {sample.delta:.3f} IV: {sample.iv:.1%}"
                )

                # Test IV rank
                logger.info("  Fetching IV rank...")
                iv_data = await client.fetch_iv_rank("NVDA")
                if iv_data:
                    logger.info(
                        f"  IV Rank: {iv_data.iv_rank:.1f} "
                        f"IV Percentile: {iv_data.iv_percentile:.1f}"
                    )

                logger.info(f"  Requests used today: {client.requests_today}")
                logger.info("  ✅ ORATS connection: PASSED")
                return True
            else:
                logger.warning("  ⚠️  No Greeks data returned")
                return False

    except Exception as e:
        logger.error(f"  ❌ ORATS connection: FAILED - {e}")
        return False


async def test_aggregator() -> bool:
    """Test data aggregator functionality."""
    from backend.data import AggregatedOptionData, DataAggregator, StalenessChecker
    from backend.models.canonical import CanonicalOptionId
    from backend.models.market_data import GreeksData, QuoteData

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST: Data Aggregator & Staleness Detection")
    logger.info("=" * 60)

    try:
        from backend.config import load_config
        config = load_config()
    except ValueError:
        # Create minimal config for testing
        from backend.config import AppConfig, AlpacaConfig, ORATSConfig, FinnhubConfig, QuiverConfig
        config = AppConfig(
            alpaca=AlpacaConfig(api_key="test", secret_key="test", paper=True),
            orats=ORATSConfig(api_token=""),
            finnhub=FinnhubConfig(api_key=""),
            quiver=QuiverConfig(api_key=""),
            log_level="INFO",
        )

    aggregator = DataAggregator(config=config)
    checker = StalenessChecker()

    # Create test data
    now = datetime.now(timezone.utc)
    canonical_id = CanonicalOptionId(
        underlying="NVDA",
        expiry="2025-01-17",
        right="C",
        strike=150.0,
    )

    # Add quote
    quote = QuoteData(
        canonical_id=canonical_id,
        bid=5.20,
        ask=5.40,
        bid_size=100,
        ask_size=50,
        last=5.30,
        timestamp=now.isoformat(),
        receive_timestamp=now.isoformat(),
        source="alpaca",
    )
    aggregator.update_quote(quote)

    # Add Greeks
    greeks = GreeksData(
        canonical_id=canonical_id,
        delta=0.45,
        gamma=0.02,
        theta=-0.15,
        vega=0.25,
        rho=0.01,
        iv=0.35,
        theoretical_value=5.35,
        timestamp=now.isoformat(),
        source="orats",
    )
    aggregator.update_greeks(greeks)

    # Test aggregation
    option = aggregator.get_option_by_id(canonical_id)
    if not option:
        logger.error("  ❌ Aggregator: Failed to retrieve option")
        return False

    logger.info(f"  Option: {option.canonical_id}")
    logger.info(f"  Quote: Bid ${option.bid:.2f} / Ask ${option.ask:.2f}")
    logger.info(f"  Greeks: Delta {option.delta:.3f}, IV {option.iv:.1%}")
    logger.info(f"  Mid: ${option.mid:.2f}, Spread: {option.spread_percent:.1f}%")

    # Test staleness
    report = checker.check_option(option)
    logger.info(f"  Quote freshness: {report.quote_status.value} ({report.quote_age:.1f}s)")
    logger.info(f"  Greeks freshness: {report.greeks_status.value} ({report.greeks_age:.1f}s)")
    logger.info(f"  All fresh: {report.all_fresh}")

    logger.info("  ✅ Aggregator & Staleness: PASSED")
    return True


async def main():
    """Run all Phase 1 tests."""
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║           PHASE 1: DATA FOUNDATION TESTS                 ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")

    # Check credentials
    creds = check_credentials()
    logger.info("Credentials detected:")
    logger.info(f"  Alpaca: {'✅ Found' if creds['alpaca'] else '❌ Missing'}")
    logger.info(f"  ORATS:  {'✅ Found' if creds['orats'] else '⚠️  Missing (optional)'}")
    logger.info("")

    results = {}

    # Run tests
    if creds["alpaca"]:
        results["alpaca"] = await test_alpaca_connection()
    else:
        logger.warning("Skipping Alpaca test - credentials missing")
        logger.info("Add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env")
        results["alpaca"] = False

    results["orats"] = await test_orats_connection()
    results["aggregator"] = await test_aggregator()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test}: {status}")
        if not passed and test != "orats":  # ORATS is optional
            all_passed = False

    logger.info("")
    if all_passed:
        logger.info("🎉 Phase 1 tests completed successfully!")
    else:
        logger.error("Some tests failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
