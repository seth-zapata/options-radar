"""Tests for subscription manager."""

from datetime import date, timedelta

import pytest

from backend.data.subscription_manager import (
    get_expiration_buckets,
    get_strike_interval,
    round_to_strike,
)


class TestStrikeInterval:
    """Tests for strike interval calculation."""

    def test_low_price_1_dollar_interval(self):
        """Test $1 strikes for prices under $50."""
        assert get_strike_interval(25.0) == 1.0
        assert get_strike_interval(49.99) == 1.0

    def test_mid_price_2_50_interval(self):
        """Test $2.50 strikes for $50-100."""
        assert get_strike_interval(50.0) == 2.5
        assert get_strike_interval(75.0) == 2.5
        assert get_strike_interval(99.99) == 2.5

    def test_medium_price_5_dollar_interval(self):
        """Test $5 strikes for $100-200."""
        assert get_strike_interval(100.0) == 5.0
        assert get_strike_interval(150.0) == 5.0

    def test_high_price_10_dollar_interval(self):
        """Test $10 strikes for $200-500."""
        assert get_strike_interval(200.0) == 10.0
        assert get_strike_interval(350.0) == 10.0

    def test_very_high_price_25_dollar_interval(self):
        """Test $25 strikes for $500-1000."""
        assert get_strike_interval(500.0) == 25.0
        assert get_strike_interval(750.0) == 25.0

    def test_ultra_high_price_50_dollar_interval(self):
        """Test $50 strikes for $1000+."""
        assert get_strike_interval(1000.0) == 50.0
        assert get_strike_interval(2500.0) == 50.0


class TestRoundToStrike:
    """Tests for rounding to nearest strike."""

    def test_round_down(self):
        """Test rounding down to nearest strike."""
        assert round_to_strike(142.3, 5.0) == 140.0

    def test_round_up(self):
        """Test rounding up to nearest strike."""
        assert round_to_strike(143.5, 5.0) == 145.0

    def test_exact_strike(self):
        """Test exact strike value."""
        assert round_to_strike(145.0, 5.0) == 145.0

    def test_small_interval(self):
        """Test with $1 interval."""
        assert round_to_strike(25.4, 1.0) == 25.0
        assert round_to_strike(25.6, 1.0) == 26.0


class TestExpirationBuckets:
    """Tests for expiration bucket calculation."""

    def test_returns_two_expirations(self):
        """Test that we get exactly 2 expiration dates."""
        expirations = get_expiration_buckets()
        assert len(expirations) == 2

    def test_first_is_friday(self):
        """Test that first expiration is a Friday."""
        expirations = get_expiration_buckets()
        assert expirations[0].weekday() == 4  # Friday

    def test_second_is_friday(self):
        """Test that second expiration is a Friday."""
        expirations = get_expiration_buckets()
        assert expirations[1].weekday() == 4  # Friday

    def test_second_is_after_first(self):
        """Test that 45 DTE comes after weekly."""
        expirations = get_expiration_buckets()
        assert expirations[1] > expirations[0]

    def test_from_monday(self):
        """Test expiration calculation from a Monday."""
        # Find a Monday
        today = date.today()
        days_until_monday = (0 - today.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        monday = today + timedelta(days=days_until_monday)

        expirations = get_expiration_buckets(monday)

        # Next Friday should be 4 days away
        assert expirations[0] == monday + timedelta(days=4)

    def test_from_friday(self):
        """Test expiration calculation from a Friday."""
        # Find a Friday
        today = date.today()
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7
        friday = today + timedelta(days=days_until_friday)

        expirations = get_expiration_buckets(friday)

        # Should get next Friday, not today
        assert expirations[0] == friday + timedelta(days=7)
