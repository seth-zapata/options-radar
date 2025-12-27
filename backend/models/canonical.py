"""Canonical option ID representation and vendor symbol mappings.

All internal logic uses CanonicalOptionId. Vendor-specific symbols are
derived deterministically via the mapping functions.

See spec section 2 for details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Literal


class OptionRight(str, Enum):
    """Option right (call or put)."""

    CALL = "C"
    PUT = "P"


@dataclass(frozen=True, slots=True)
class CanonicalOptionId:
    """Internal canonical representation of an option contract.

    All internal logic uses this representation. Vendor-specific symbols
    are derived using the to_* functions.

    Attributes:
        underlying: Ticker symbol (e.g., "NVDA")
        expiry: Expiration date in ISO 8601 format (YYYY-MM-DD)
        right: Call ("C") or Put ("P")
        strike: Strike price with 2 decimal precision
        multiplier: Contract multiplier (default 100, adjust for minis)
    """

    underlying: str
    expiry: str  # ISO 8601 date: "2025-01-17"
    right: Literal["C", "P"]
    strike: float
    multiplier: int = 100

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        # Validate underlying is non-empty uppercase letters
        if not self.underlying or not self.underlying.isalpha():
            raise ValueError(f"Invalid underlying: {self.underlying}")

        # Validate expiry format (YYYY-MM-DD)
        try:
            date.fromisoformat(self.expiry)
        except ValueError as e:
            raise ValueError(f"Invalid expiry format: {self.expiry}") from e

        # Validate right
        if self.right not in ("C", "P"):
            raise ValueError(f"Invalid right: {self.right}, must be 'C' or 'P'")

        # Validate strike is positive
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive: {self.strike}")

        # Validate multiplier is positive
        if self.multiplier <= 0:
            raise ValueError(f"Multiplier must be positive: {self.multiplier}")

    @property
    def expiry_date(self) -> date:
        """Return expiry as a date object."""
        return date.fromisoformat(self.expiry)

    @property
    def is_call(self) -> bool:
        """Return True if this is a call option."""
        return self.right == "C"

    @property
    def is_put(self) -> bool:
        """Return True if this is a put option."""
        return self.right == "P"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.underlying} {self.expiry} {self.strike:.2f}{self.right}"


def to_occ(opt: CanonicalOptionId) -> str:
    """Convert to OCC format.

    OCC format: SYMBOL (6 chars padded) + YYMMDD + C/P + Strike*1000 (8 digits)
    Example: "NVDA  250117C00500000" for NVDA Jan 17 2025 500 Call

    Args:
        opt: Canonical option ID

    Returns:
        OCC-formatted symbol string (21 characters)
    """
    # Extract YYMMDD from expiry
    exp_date = date.fromisoformat(opt.expiry)
    exp_str = exp_date.strftime("%y%m%d")

    # Strike * 1000, zero-padded to 8 digits
    strike_int = round(opt.strike * 1000)
    strike_str = str(strike_int).zfill(8)

    # Underlying padded to 6 characters
    underlying_padded = opt.underlying.ljust(6)

    return f"{underlying_padded}{exp_str}{opt.right}{strike_str}"


def to_alpaca(opt: CanonicalOptionId) -> str:
    """Convert to Alpaca format.

    Alpaca format: SYMBOL + YYMMDD + C/P + Strike*1000 (8 digits)
    Example: "NVDA250117C00500000" for NVDA Jan 17 2025 500 Call

    Note: Unlike OCC, the symbol is NOT padded.

    Args:
        opt: Canonical option ID

    Returns:
        Alpaca-formatted symbol string
    """
    # Extract YYMMDD from expiry
    exp_date = date.fromisoformat(opt.expiry)
    exp_str = exp_date.strftime("%y%m%d")

    # Strike * 1000, zero-padded to 8 digits
    strike_int = round(opt.strike * 1000)
    strike_str = str(strike_int).zfill(8)

    return f"{opt.underlying}{exp_str}{opt.right}{strike_str}"


def to_orats(opt: CanonicalOptionId) -> dict:
    """Convert to ORATS query parameters.

    ORATS uses separate query parameters for each component.

    Args:
        opt: Canonical option ID

    Returns:
        Dictionary of ORATS API parameters
    """
    return {
        "ticker": opt.underlying,
        "expirDate": opt.expiry,
        "strike": opt.strike,
        "callPut": "call" if opt.right == "C" else "put",
    }


# Regex patterns for parsing vendor symbols
_ALPACA_PATTERN = re.compile(
    r"^([A-Z]+)(\d{6})([CP])(\d{8})$"
)
_OCC_PATTERN = re.compile(
    r"^([A-Z]{1,6})\s*(\d{6})([CP])(\d{8})$"
)


def parse_alpaca(symbol: str) -> CanonicalOptionId:
    """Parse an Alpaca-formatted option symbol.

    Args:
        symbol: Alpaca option symbol (e.g., "NVDA250117C00500000")

    Returns:
        CanonicalOptionId instance

    Raises:
        ValueError: If symbol format is invalid
    """
    match = _ALPACA_PATTERN.match(symbol)
    if not match:
        raise ValueError(f"Invalid Alpaca option symbol: {symbol}")

    underlying, exp_str, right, strike_str = match.groups()

    # Parse expiry: YYMMDD -> YYYY-MM-DD
    year = 2000 + int(exp_str[:2])
    month = int(exp_str[2:4])
    day = int(exp_str[4:6])
    expiry = f"{year:04d}-{month:02d}-{day:02d}"

    # Parse strike: integer / 1000
    strike = int(strike_str) / 1000.0

    return CanonicalOptionId(
        underlying=underlying,
        expiry=expiry,
        right=right,  # type: ignore[arg-type]
        strike=strike,
    )


def parse_occ(symbol: str) -> CanonicalOptionId:
    """Parse an OCC-formatted option symbol.

    Args:
        symbol: OCC option symbol (e.g., "NVDA  250117C00500000")

    Returns:
        CanonicalOptionId instance

    Raises:
        ValueError: If symbol format is invalid
    """
    # Remove spaces for easier matching
    match = _OCC_PATTERN.match(symbol.strip())
    if not match:
        raise ValueError(f"Invalid OCC option symbol: {symbol}")

    underlying, exp_str, right, strike_str = match.groups()

    # Parse expiry: YYMMDD -> YYYY-MM-DD
    year = 2000 + int(exp_str[:2])
    month = int(exp_str[2:4])
    day = int(exp_str[4:6])
    expiry = f"{year:04d}-{month:02d}-{day:02d}"

    # Parse strike: integer / 1000
    strike = int(strike_str) / 1000.0

    return CanonicalOptionId(
        underlying=underlying.strip(),
        expiry=expiry,
        right=right,  # type: ignore[arg-type]
        strike=strike,
    )


@dataclass(frozen=True)
class CorporateActionAdjustment:
    """Tracks corporate action adjustments to option contracts.

    For MVP: Log warning if contract has non-standard multiplier.
    Full implementation via ORATS corporate actions endpoint is deferred.
    """

    original_id: CanonicalOptionId
    adjusted_id: CanonicalOptionId
    adjustment_type: Literal["split", "reverse_split", "special_dividend", "merger"]
    effective_date: str  # ISO 8601 date
    multiplier_change: float | None = None
    strike_adjustment: float | None = None
