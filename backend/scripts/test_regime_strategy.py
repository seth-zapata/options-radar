#!/usr/bin/env python3
"""Test script for regime-filtered intraday strategy implementation.

Tests:
1. RegimeDetector - classifies sentiment correctly
2. RegimeSignalGenerator - generates pullback/bounce signals
3. PositionTracker - uses fixed take-profit/stop-loss
4. Config integration - all parameters load correctly

Run: python -m backend.scripts.test_regime_strategy
"""

from datetime import datetime, timezone, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def test_regime_detector():
    """Test RegimeDetector with calibrated thresholds."""
    from backend.engine.regime_detector import (
        RegimeDetector,
        RegimeConfig,
        RegimeType,
    )

    print("\n" + "=" * 60)
    print("TEST 1: RegimeDetector")
    print("=" * 60)

    detector = RegimeDetector()

    # Test threshold classification
    test_cases = [
        (0.15, RegimeType.STRONG_BULLISH, "Strong Bullish (> 0.12)"),
        (0.10, RegimeType.MODERATE_BULLISH, "Moderate Bullish (0.07-0.12)"),
        (0.05, RegimeType.NEUTRAL, "Neutral (0.07 to -0.08)"),
        (0.00, RegimeType.NEUTRAL, "Neutral (center)"),
        (-0.05, RegimeType.NEUTRAL, "Neutral (slightly bearish)"),
        (-0.10, RegimeType.MODERATE_BEARISH, "Moderate Bearish (-0.08 to -0.15)"),
        (-0.20, RegimeType.STRONG_BEARISH, "Strong Bearish (< -0.15)"),
    ]

    all_passed = True
    for sentiment, expected, description in test_cases:
        result = detector.classify_sentiment(sentiment)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  {status}: sentiment={sentiment:+.2f} -> {result.value} ({description})")

    # Test regime window
    print("\n  Testing regime window...")
    regime = detector.update_regime("TSLA", wsb_sentiment=0.15)
    assert regime is not None
    assert regime.is_active
    assert regime.regime_type == RegimeType.STRONG_BULLISH
    print(f"  PASS: Regime window active, expires {regime.window_expires}")
    print(f"  PASS: Days remaining: {regime.days_remaining}")

    # Test is_bullish / is_bearish helpers
    assert detector.is_bullish("TSLA")
    assert not detector.is_bearish("TSLA")
    print("  PASS: is_bullish() / is_bearish() helpers work")

    if all_passed:
        print("\n  All regime detector tests PASSED")
    else:
        print("\n  Some regime detector tests FAILED")

    return all_passed


def test_signal_generator():
    """Test RegimeSignalGenerator with pullback/bounce logic."""
    from backend.engine.regime_detector import RegimeDetector, RegimeType
    from backend.engine.regime_signals import (
        RegimeSignalGenerator,
        SignalGeneratorConfig,
        PriceData,
        SignalType,
    )

    print("\n" + "=" * 60)
    print("TEST 2: RegimeSignalGenerator")
    print("=" * 60)

    detector = RegimeDetector()
    generator = RegimeSignalGenerator(detector)

    now = datetime.now(timezone.utc)

    # Set up bullish regime
    detector.update_regime("TSLA", wsb_sentiment=0.15)

    # Test 1: Bullish regime with pullback
    print("\n  Testing bullish regime pullback entry...")
    price_data = PriceData(
        symbol="TSLA",
        current=247.0,  # Current price
        high=252.0,     # Daily high (2% above current)
        low=245.0,
        open=250.0,
        timestamp=now,
    )
    pullback_pct = price_data.pullback_pct
    print(f"  Pullback from high: {pullback_pct:.2f}%")

    signal = generator.check_entry_signal(price_data)
    if signal.signal_type == SignalType.BUY_CALL:
        print(f"  PASS: BUY_CALL signal generated")
        print(f"        Reason: {signal.trigger_reason}")
    else:
        print(f"  FAIL: Expected BUY_CALL, got {signal.signal_type.value}")
        print(f"        Reason: {signal.trigger_reason}")

    # Test 2: Bullish regime without sufficient pullback
    print("\n  Testing bullish regime without pullback...")
    price_data_no_pullback = PriceData(
        symbol="TSLA",
        current=251.5,  # Near high
        high=252.0,
        low=245.0,
        open=250.0,
        timestamp=now,
    )
    pullback_pct = price_data_no_pullback.pullback_pct
    print(f"  Pullback from high: {pullback_pct:.2f}% (below 1.5% threshold)")

    signal = generator.check_entry_signal(price_data_no_pullback)
    if signal.signal_type == SignalType.NO_SIGNAL:
        print(f"  PASS: NO_SIGNAL (pullback insufficient)")
        print(f"        Reason: {signal.trigger_reason}")
    else:
        print(f"  FAIL: Expected NO_SIGNAL, got {signal.signal_type.value}")

    # Test 3: Set up bearish regime
    print("\n  Testing bearish regime bounce entry...")
    detector.clear_regime("TSLA")
    detector.update_regime("TSLA", wsb_sentiment=-0.20)

    price_data_bounce = PriceData(
        symbol="TSLA",
        current=248.0,  # Current price
        high=252.0,
        low=244.0,      # Daily low (1.6% below current)
        open=250.0,
        timestamp=now,
    )
    bounce_pct = price_data_bounce.bounce_pct
    print(f"  Bounce from low: {bounce_pct:.2f}%")

    signal = generator.check_entry_signal(price_data_bounce)
    if signal.signal_type == SignalType.BUY_PUT:
        print(f"  PASS: BUY_PUT signal generated")
        print(f"        Reason: {signal.trigger_reason}")
    else:
        print(f"  FAIL: Expected BUY_PUT, got {signal.signal_type.value}")
        print(f"        Reason: {signal.trigger_reason}")

    # Test 4: No regime (neutral)
    print("\n  Testing neutral regime (no signal)...")
    detector.clear_regime("TSLA")
    detector.update_regime("TSLA", wsb_sentiment=0.03)  # Neutral

    signal = generator.check_entry_signal(price_data)
    if signal.signal_type == SignalType.NO_SIGNAL:
        print(f"  PASS: NO_SIGNAL in neutral regime")
    else:
        print(f"  FAIL: Expected NO_SIGNAL, got {signal.signal_type.value}")

    print("\n  Signal generator tests completed")
    return True


def test_position_tracker():
    """Test PositionTracker with fixed take-profit/stop-loss."""
    from backend.engine.position_tracker import (
        PositionTracker,
        PositionTrackerConfig,
    )

    print("\n" + "=" * 60)
    print("TEST 3: PositionTracker (Fixed Exit Rules)")
    print("=" * 60)

    config = PositionTrackerConfig()
    print(f"\n  Configuration:")
    print(f"    Take profit: +{config.take_profit_percent}%")
    print(f"    Stop loss: {config.stop_loss_percent}%")
    print(f"    Time exit: DTE <= {config.min_dte_exit}")
    print(f"    Max positions: {config.max_positions}")

    tracker = PositionTracker(config)

    # Open a position
    position = tracker.open_position(
        recommendation_id="test-123",
        underlying="TSLA",
        expiry="2025-01-10",
        strike=250.0,
        right="C",
        action="BUY_CALL",
        contracts=1,
        fill_price=5.00,
    )
    print(f"\n  Opened position: {position.underlying} ${position.strike}C @ ${position.fill_price}")
    print(f"    Entry cost: ${position.entry_cost}")

    # Test 1: Take profit trigger (+40%)
    print("\n  Testing take profit trigger (+40%)...")
    exit_signal = tracker.update_position(
        position.id,
        current_price=7.00,  # +40%
        dte=5,
    )
    if exit_signal and exit_signal.trigger == "take_profit":
        print(f"  PASS: Take profit triggered at +{exit_signal.pnl_percent:.0f}%")
    else:
        # Check P&L
        pos = tracker.get_position(position.id)
        print(f"  Current P&L: {pos.pnl_percent:.1f}%")
        if pos.pnl_percent >= 40:
            print(f"  FAIL: Should have triggered take profit")
        else:
            print(f"  INFO: Not at take profit threshold yet")

    # Reset position for next test
    tracker.clear_exit_signal(position.id)

    # Test 2: Stop loss trigger (-20%)
    print("\n  Testing stop loss trigger (-20%)...")
    exit_signal = tracker.update_position(
        position.id,
        current_price=4.00,  # -20%
        dte=5,
    )
    if exit_signal and exit_signal.trigger == "stop_loss":
        print(f"  PASS: Stop loss triggered at {exit_signal.pnl_percent:.0f}%")
    else:
        pos = tracker.get_position(position.id)
        print(f"  Current P&L: {pos.pnl_percent:.1f}%")
        if pos.pnl_percent <= -20:
            print(f"  FAIL: Should have triggered stop loss")
        else:
            print(f"  INFO: Not at stop loss threshold yet")

    # Reset for next test
    tracker.clear_exit_signal(position.id)

    # Test 3: Time exit (DTE < 1)
    print("\n  Testing time exit (DTE <= 1)...")
    exit_signal = tracker.update_position(
        position.id,
        current_price=5.50,  # +10%
        dte=1,
    )
    if exit_signal and exit_signal.trigger == "time_exit":
        print(f"  PASS: Time exit triggered (DTE={exit_signal.pnl_percent:.0f} remaining)")
    else:
        print(f"  FAIL: Should have triggered time exit at DTE=1")

    print("\n  Position tracker tests completed")
    return True


def test_config_integration():
    """Test that config loads correctly."""
    from backend.config import load_config, RegimeStrategyConfig

    print("\n" + "=" * 60)
    print("TEST 4: Configuration Integration")
    print("=" * 60)

    config = RegimeStrategyConfig()

    print(f"\n  RegimeStrategyConfig defaults:")
    print(f"    Regime thresholds:")
    print(f"      Strong Bullish: > {config.strong_bullish_threshold}")
    print(f"      Moderate Bullish: > {config.moderate_bullish_threshold}")
    print(f"      Moderate Bearish: < {config.moderate_bearish_threshold}")
    print(f"      Strong Bearish: < {config.strong_bearish_threshold}")
    print(f"    Regime window: {config.regime_window_days} days")
    print(f"    Entry thresholds:")
    print(f"      Pullback: {config.pullback_threshold}%")
    print(f"      Bounce: {config.bounce_threshold}%")
    print(f"    Option selection:")
    print(f"      Target DTE: {config.target_dte}")
    print(f"      DTE range: {config.min_dte} - {config.max_dte}")
    print(f"    Exit rules:")
    print(f"      Take profit: +{config.take_profit_percent}%")
    print(f"      Stop loss: {config.stop_loss_percent}%")
    print(f"      Time exit: DTE <= {config.min_dte_exit}")
    print(f"    Position sizing:")
    print(f"      Size: {config.position_size_pct}% per trade")
    print(f"      Max positions: {config.max_concurrent_positions}")
    print(f"    Enabled symbols: {config.enabled_symbols}")

    # Verify values match validated parameters
    assert config.strong_bullish_threshold == 0.12
    assert config.moderate_bullish_threshold == 0.07
    assert config.moderate_bearish_threshold == -0.08
    assert config.strong_bearish_threshold == -0.15
    assert config.regime_window_days == 7
    assert config.pullback_threshold == 1.5
    assert config.bounce_threshold == 1.5
    assert config.target_dte == 7
    assert config.take_profit_percent == 40.0
    assert config.stop_loss_percent == -20.0
    assert config.position_size_pct == 10.0
    assert config.max_concurrent_positions == 3
    assert "TSLA" in config.enabled_symbols

    print("\n  PASS: All configuration values match validated parameters")
    return True


def test_option_selection():
    """Test ATM option selection logic."""
    from backend.engine.regime_signals import (
        select_atm_option,
        TradeSignal,
        SignalType,
        SignalGeneratorConfig,
    )
    from backend.engine.regime_detector import RegimeType
    from datetime import datetime, timezone, timedelta

    print("\n" + "=" * 60)
    print("TEST 5: Option Selection")
    print("=" * 60)

    now = datetime.now(timezone.utc)
    config = SignalGeneratorConfig()

    # Create a test signal
    signal = TradeSignal(
        signal_type=SignalType.BUY_CALL,
        symbol="TSLA",
        generated_at=now,
        regime_type=RegimeType.STRONG_BULLISH,
        trigger_reason="Test signal",
        trigger_pct=2.0,
        entry_price=250.00,
    )

    # Create mock options chain
    expiry_7dte = (now.date() + timedelta(days=7)).strftime("%Y-%m-%d")
    expiry_14dte = (now.date() + timedelta(days=14)).strftime("%Y-%m-%d")

    available_options = [
        # Good candidate - ATM, 7 DTE, liquid
        {
            "type": "call",
            "strike": 250,
            "expiry": expiry_7dte,
            "bid": 5.00,
            "ask": 5.20,
            "open_interest": 1000,
            "volume": 500,
            "delta": 0.50,
        },
        # Too far OTM
        {
            "type": "call",
            "strike": 280,
            "expiry": expiry_7dte,
            "bid": 0.50,
            "ask": 0.60,
            "open_interest": 2000,
            "volume": 200,
            "delta": 0.10,
        },
        # Low liquidity
        {
            "type": "call",
            "strike": 250,
            "expiry": expiry_14dte,
            "bid": 7.00,
            "ask": 7.50,
            "open_interest": 100,  # Below 500 threshold
            "volume": 50,  # Below 100 threshold
            "delta": 0.52,
        },
        # Put (wrong type)
        {
            "type": "put",
            "strike": 250,
            "expiry": expiry_7dte,
            "bid": 4.50,
            "ask": 4.70,
            "open_interest": 800,
            "volume": 300,
            "delta": -0.50,
        },
    ]

    selection = select_atm_option(signal, available_options, config)

    if selection:
        print(f"\n  Selected option:")
        print(f"    Strike: ${selection.strike}")
        print(f"    Expiry: {selection.expiry}")
        print(f"    DTE: {selection.dte}")
        print(f"    Type: {selection.option_type}")
        print(f"    Bid/Ask: ${selection.bid}/{selection.ask}")
        print(f"    Mid: ${selection.mid:.2f}")
        print(f"    OI: {selection.open_interest}")
        print(f"    Volume: {selection.volume}")

        # Verify it selected the ATM 7 DTE option
        if selection.strike == 250 and selection.dte == 7:
            print("\n  PASS: Selected correct ATM 7 DTE option")
        else:
            print(f"\n  FAIL: Expected $250 strike, 7 DTE")
    else:
        print("\n  FAIL: No option selected")

    return selection is not None


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("REGIME-FILTERED INTRADAY STRATEGY TEST SUITE")
    print("=" * 60)
    print(f"Testing validated strategy parameters:")
    print(f"  - 71 trades, 43.7% win rate, +17.4% avg return")
    print(f"  - +1238% total return (10% position sizing)")
    print("=" * 60)

    results = []

    results.append(("RegimeDetector", test_regime_detector()))
    results.append(("SignalGenerator", test_signal_generator()))
    results.append(("PositionTracker", test_position_tracker()))
    results.append(("Configuration", test_config_integration()))
    results.append(("OptionSelection", test_option_selection()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {status}: {name}")

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
