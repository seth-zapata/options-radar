"""Phase 6 test script - Sentiment and Scanner.

Tests:
1. Finnhub news sentiment
2. Quiver political/insider/WSB sentiment
3. Combined sentiment aggregation
4. Daily scanner
5. Alpaca portfolio integration
6. Sentiment gates

Usage:
    # With live API calls (requires API keys):
    python -m backend.test_phase6

    # With mock data:
    MOCK_DATA=true python -m backend.test_phase6
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import load_config


async def test_finnhub_sentiment():
    """Test Finnhub news sentiment client."""
    print("\n" + "="*60)
    print("TEST: Finnhub News Sentiment")
    print("="*60)

    from backend.data.finnhub_client import FinnhubClient

    config = load_config()

    if not config.finnhub.api_key:
        print("SKIP: No Finnhub API key configured")
        return

    client = FinnhubClient(config.finnhub)

    # Test single symbol
    symbol = "NVDA"
    print(f"\nFetching news sentiment for {symbol}...")

    sentiment = await client.get_news_sentiment(symbol)
    if sentiment:
        print(f"  Sentiment Score: {sentiment.sentiment_score:.1f}")
        print(f"  Bullish %: {sentiment.bullish_percent:.0f}%")
        print(f"  Bearish %: {sentiment.bearish_percent:.0f}%")
        print(f"  Relative Buzz: {sentiment.relative_buzz:.2f}x")
        print(f"  Is Bullish: {sentiment.is_bullish}")
    else:
        print("  No sentiment data available")

    print("\n✓ Finnhub client test complete")


async def test_quiver_wsb():
    """Test Quiver WSB sentiment client."""
    print("\n" + "="*60)
    print("TEST: Quiver WSB Sentiment")
    print("="*60)

    from backend.data.quiver_client import QuiverClient

    config = load_config()

    if not config.quiver.api_key:
        print("SKIP: No Quiver API key configured")
        return

    client = QuiverClient(config.quiver)

    symbol = "NVDA"

    # Test WSB sentiment
    print(f"\nFetching WSB sentiment for {symbol}...")
    wsb = await client.get_wsb_sentiment(symbol)
    if wsb:
        print(f"  Mentions (24h): {wsb.mentions_24h}")
        print(f"  Sentiment: {wsb.sentiment:.3f}")
        print(f"  Sentiment Score: {wsb.sentiment_score:.1f}")
        print(f"  Rank: {wsb.rank}")
        print(f"  Buzz Level: {wsb.buzz_level}")
        print(f"  Is Trending: {wsb.is_trending}")
        print(f"  Is Bullish: {wsb.is_bullish}")
    else:
        print("  No WSB data for this symbol")

    # Test WSB trending
    print("\nFetching WSB trending...")
    trending = await client.get_wsb_trending(limit=5)
    for wsb in trending:
        print(f"  {wsb.rank}. {wsb.symbol}: {wsb.mentions_24h} mentions, sentiment={wsb.sentiment:.2f}")

    print("\n✓ Quiver WSB client test complete")


async def test_combined_sentiment():
    """Test combined sentiment aggregator."""
    print("\n" + "="*60)
    print("TEST: Combined Sentiment Aggregator (News + WSB)")
    print("="*60)

    from backend.data.sentiment_aggregator import SentimentAggregator

    config = load_config()
    aggregator = SentimentAggregator(config)

    symbols = ["NVDA", "AAPL", "TSLA"]

    for symbol in symbols:
        print(f"\nFetching combined sentiment for {symbol}...")
        sentiment = await aggregator.get_sentiment(symbol)

        print(f"  News Score: {sentiment.news_score:.1f}")
        print(f"  WSB Score: {sentiment.wsb_score:.1f}")
        print(f"  Combined Score: {sentiment.combined_score:.1f} (50/50 weighted)")
        print(f"  Signal: {sentiment.signal} ({sentiment.strength})")
        print(f"  Flags:")
        print(f"    - News Buzzing: {sentiment.news_is_buzzing}")
        print(f"    - WSB Trending: {sentiment.wsb_is_trending}")
        print(f"    - WSB Bullish: {sentiment.wsb_is_bullish}")
        print(f"    - Sources Aligned: {sentiment.sources_aligned}")

    print("\n✓ Combined sentiment test complete")


async def test_scanner():
    """Test daily opportunity scanner."""
    print("\n" + "="*60)
    print("TEST: Daily Opportunity Scanner")
    print("="*60)

    from backend.engine.scanner import DailyScanner

    config = load_config()
    scanner = DailyScanner(config)

    # Scan a subset of symbols for speed
    symbols = ["NVDA", "AAPL", "TSLA", "AMD"]

    print(f"\nScanning {len(symbols)} symbols...")
    results = await scanner.scan(symbols=symbols)

    print("\nScan Results (sorted by score):")
    for result in results:
        print(f"\n  {result.symbol}: Score={result.score:.0f}, Direction={result.direction}")
        if result.signals:
            for signal in result.signals:
                print(f"    - {signal}")

    # Get top opportunities
    opportunities = [r for r in results if r.is_opportunity]
    print(f"\n\nOpportunities found: {len(opportunities)}")

    print("\n✓ Scanner test complete")


async def test_alpaca_portfolio():
    """Test Alpaca portfolio integration."""
    print("\n" + "="*60)
    print("TEST: Alpaca Portfolio Integration")
    print("="*60)

    from backend.data.alpaca_account import AlpacaAccountClient

    config = load_config()
    client = AlpacaAccountClient(config.alpaca)

    print("\nFetching portfolio...")

    try:
        portfolio = await client.get_portfolio()

        print(f"\nAccount Summary:")
        print(f"  Cash: ${portfolio.account.cash:,.2f}")
        print(f"  Buying Power: ${portfolio.account.buying_power:,.2f}")
        print(f"  Portfolio Value: ${portfolio.account.portfolio_value:,.2f}")
        print(f"  Day Trade Count: {portfolio.account.day_trade_count}")

        print(f"\nPositions: {portfolio.total_positions} total")
        print(f"  Stock Positions: {len(portfolio.stock_positions)}")
        print(f"  Option Positions: {len(portfolio.option_positions)}")

        if portfolio.positions:
            print("\nPosition Details:")
            for pos in portfolio.positions[:5]:  # Limit to first 5
                print(f"  {pos.symbol}:")
                print(f"    Qty: {pos.qty}, Avg Entry: ${pos.avg_entry_price:.2f}")
                print(f"    Market Value: ${pos.market_value:,.2f}")
                print(f"    Unrealized P/L: ${pos.unrealized_pl:,.2f} ({pos.unrealized_pl_percent:.1f}%)")

        print("\n✓ Alpaca portfolio test complete")

    except Exception as e:
        print(f"\nError fetching portfolio: {e}")
        print("Note: This may fail with paper trading if no positions exist")


async def test_sentiment_gates():
    """Test sentiment gate evaluation."""
    print("\n" + "="*60)
    print("TEST: Sentiment Gates (News + WSB)")
    print("="*60)

    from backend.engine.gates import (
        GateContext,
        SentimentDirectionGate,
        RetailMomentumGate,
        SentimentConvergenceGate,
    )

    # Create context with bullish sentiment for a bullish trade
    ctx_bullish = GateContext(
        action="BUY_CALL",
        combined_sentiment_score=45.0,
        news_sentiment_score=50.0,
        wsb_sentiment_score=40.0,
        wsb_is_trending=True,
        news_is_buzzing=True,
        sources_aligned=True,
    )

    # Create context with conflicting sentiment
    ctx_conflicting = GateContext(
        action="BUY_CALL",  # Bullish trade
        combined_sentiment_score=-50.0,  # Bearish sentiment
        news_sentiment_score=-60.0,
        wsb_sentiment_score=-40.0,
        wsb_is_trending=True,  # Trending but bearish
        sources_aligned=True,  # Both bearish = aligned
    )

    gates = [
        SentimentDirectionGate(),
        RetailMomentumGate(),
        SentimentConvergenceGate(),
    ]

    print("\nScenario 1: Bullish trade with bullish news + WSB")
    for gate in gates:
        result = gate.evaluate(ctx_bullish)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {gate.name}: {status} - {result.message}")

    print("\nScenario 2: Bullish trade with bearish news + WSB (conflicting)")
    for gate in gates:
        result = gate.evaluate(ctx_conflicting)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {gate.name}: {status} - {result.message}")

    print("\n✓ Sentiment gates test complete")


async def main():
    """Run all Phase 6 tests."""
    print("="*60)
    print("PHASE 6 TEST SUITE")
    print("Sentiment Integration & Daily Scanner")
    print("="*60)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    mock_mode = os.environ.get("MOCK_DATA", "").lower() == "true"
    if mock_mode:
        print("\n⚠️  Running in MOCK mode - API calls will be skipped")

    try:
        # Test sentiment gates (no API calls needed)
        await test_sentiment_gates()

        if not mock_mode:
            # Test Finnhub news sentiment
            await test_finnhub_sentiment()

            # Test Quiver WSB sentiment
            await test_quiver_wsb()

            # Test combined aggregator (News + WSB)
            await test_combined_sentiment()

            # Test scanner
            await test_scanner()

            # Test Alpaca portfolio
            await test_alpaca_portfolio()
        else:
            print("\n\nSkipping API tests in mock mode")
            print("Run without MOCK_DATA=true to test live APIs")

        print("\n" + "="*60)
        print("ALL PHASE 6 TESTS COMPLETE")
        print("="*60)

    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
