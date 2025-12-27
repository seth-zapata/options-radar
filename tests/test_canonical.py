"""Tests for canonical option ID module."""

import pytest

from backend.models.canonical import (
    CanonicalOptionId,
    parse_alpaca,
    parse_occ,
    to_alpaca,
    to_occ,
    to_orats,
)


class TestCanonicalOptionId:
    """Tests for CanonicalOptionId dataclass."""

    def test_create_valid_call(self):
        """Test creating a valid call option."""
        opt = CanonicalOptionId(
            underlying="NVDA",
            expiry="2025-01-17",
            right="C",
            strike=500.00,
        )
        assert opt.underlying == "NVDA"
        assert opt.expiry == "2025-01-17"
        assert opt.right == "C"
        assert opt.strike == 500.00
        assert opt.multiplier == 100
        assert opt.is_call
        assert not opt.is_put

    def test_create_valid_put(self):
        """Test creating a valid put option."""
        opt = CanonicalOptionId(
            underlying="AAPL",
            expiry="2025-06-20",
            right="P",
            strike=150.50,
            multiplier=100,
        )
        assert opt.is_put
        assert not opt.is_call

    def test_invalid_underlying_raises(self):
        """Test that empty underlying raises ValueError."""
        with pytest.raises(ValueError, match="Invalid underlying"):
            CanonicalOptionId(
                underlying="",
                expiry="2025-01-17",
                right="C",
                strike=100.00,
            )

    def test_invalid_expiry_format_raises(self):
        """Test that invalid expiry format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid expiry format"):
            CanonicalOptionId(
                underlying="NVDA",
                expiry="01-17-2025",  # Wrong format
                right="C",
                strike=100.00,
            )

    def test_invalid_right_raises(self):
        """Test that invalid right raises ValueError."""
        with pytest.raises(ValueError, match="Invalid right"):
            CanonicalOptionId(
                underlying="NVDA",
                expiry="2025-01-17",
                right="X",  # type: ignore[arg-type]
                strike=100.00,
            )

    def test_negative_strike_raises(self):
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            CanonicalOptionId(
                underlying="NVDA",
                expiry="2025-01-17",
                right="C",
                strike=-100.00,
            )

    def test_expiry_date_property(self):
        """Test expiry_date property returns date object."""
        opt = CanonicalOptionId(
            underlying="NVDA",
            expiry="2025-01-17",
            right="C",
            strike=500.00,
        )
        from datetime import date
        assert opt.expiry_date == date(2025, 1, 17)


class TestToOCC:
    """Tests for to_occ conversion."""

    def test_basic_call(self):
        """Test OCC format for basic call."""
        opt = CanonicalOptionId(
            underlying="NVDA",
            expiry="2025-01-17",
            right="C",
            strike=500.00,
        )
        result = to_occ(opt)
        assert result == "NVDA  250117C00500000"
        assert len(result) == 21

    def test_basic_put(self):
        """Test OCC format for basic put."""
        opt = CanonicalOptionId(
            underlying="AAPL",
            expiry="2025-06-20",
            right="P",
            strike=150.00,
        )
        result = to_occ(opt)
        assert result == "AAPL  250620P00150000"

    def test_fractional_strike(self):
        """Test OCC format with fractional strike."""
        opt = CanonicalOptionId(
            underlying="SPY",
            expiry="2025-03-21",
            right="C",
            strike=450.50,
        )
        result = to_occ(opt)
        assert result == "SPY   250321C00450500"

    def test_long_symbol(self):
        """Test OCC format with 4-letter symbol."""
        opt = CanonicalOptionId(
            underlying="GOOGL",
            expiry="2025-01-17",
            right="C",
            strike=100.00,
        )
        result = to_occ(opt)
        assert result == "GOOGL 250117C00100000"


class TestToAlpaca:
    """Tests for to_alpaca conversion."""

    def test_basic_call(self):
        """Test Alpaca format for basic call."""
        opt = CanonicalOptionId(
            underlying="NVDA",
            expiry="2025-01-17",
            right="C",
            strike=500.00,
        )
        result = to_alpaca(opt)
        assert result == "NVDA250117C00500000"

    def test_no_padding(self):
        """Test that Alpaca format has no symbol padding."""
        opt = CanonicalOptionId(
            underlying="A",
            expiry="2025-01-17",
            right="P",
            strike=100.00,
        )
        result = to_alpaca(opt)
        assert result == "A250117P00100000"
        assert not result.startswith("A ")


class TestToORATS:
    """Tests for to_orats conversion."""

    def test_call_params(self):
        """Test ORATS params for call."""
        opt = CanonicalOptionId(
            underlying="NVDA",
            expiry="2025-01-17",
            right="C",
            strike=500.00,
        )
        result = to_orats(opt)
        assert result == {
            "ticker": "NVDA",
            "expirDate": "2025-01-17",
            "strike": 500.00,
            "callPut": "call",
        }

    def test_put_params(self):
        """Test ORATS params for put."""
        opt = CanonicalOptionId(
            underlying="NVDA",
            expiry="2025-01-17",
            right="P",
            strike=500.00,
        )
        result = to_orats(opt)
        assert result["callPut"] == "put"


class TestParseAlpaca:
    """Tests for parse_alpaca."""

    def test_parse_call(self):
        """Test parsing Alpaca call symbol."""
        opt = parse_alpaca("NVDA250117C00500000")
        assert opt.underlying == "NVDA"
        assert opt.expiry == "2025-01-17"
        assert opt.right == "C"
        assert opt.strike == 500.00

    def test_parse_put(self):
        """Test parsing Alpaca put symbol."""
        opt = parse_alpaca("AAPL250620P00150500")
        assert opt.underlying == "AAPL"
        assert opt.expiry == "2025-06-20"
        assert opt.right == "P"
        assert opt.strike == 150.50

    def test_roundtrip(self):
        """Test that to_alpaca and parse_alpaca are inverses."""
        original = CanonicalOptionId(
            underlying="TSLA",
            expiry="2025-09-19",
            right="C",
            strike=250.00,
        )
        symbol = to_alpaca(original)
        parsed = parse_alpaca(symbol)
        assert parsed == original

    def test_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Alpaca"):
            parse_alpaca("INVALID")


class TestParseOCC:
    """Tests for parse_occ."""

    def test_parse_padded(self):
        """Test parsing padded OCC symbol."""
        opt = parse_occ("NVDA  250117C00500000")
        assert opt.underlying == "NVDA"
        assert opt.expiry == "2025-01-17"
        assert opt.right == "C"
        assert opt.strike == 500.00

    def test_roundtrip(self):
        """Test that to_occ and parse_occ are inverses."""
        original = CanonicalOptionId(
            underlying="SPY",
            expiry="2025-03-21",
            right="P",
            strike=450.00,
        )
        symbol = to_occ(original)
        parsed = parse_occ(symbol)
        assert parsed == original
