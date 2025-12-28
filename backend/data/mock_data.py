"""Mock data generator for development and testing.

Generates realistic NVDA options data without requiring market connection.
Enable with MOCK_DATA=true environment variable.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from backend.data.aggregator import AggregatedOptionData
from backend.models import CanonicalOptionId
from backend.models.market_data import UnderlyingData

logger = logging.getLogger(__name__)

# Base prices by symbol (realistic ranges)
BASE_PRICES = {
    "QQQ": 525.00,
    "SPY": 595.00,
    "NVDA": 137.50,
    "AAPL": 255.00,
    "TSLA": 425.00,
    "MSFT": 435.00,
    "AMZN": 225.00,
    "META": 610.00,
    "GOOGL": 195.00,
    "AMD": 125.00,
}
DEFAULT_PRICE = 200.00  # For unknown symbols
PRICE_VOLATILITY = 0.002  # 0.2% per update

# Option chain configuration
# Weekly options for near-term, then monthly, then quarterly for LEAPS
EXPIRATIONS_WEEKS = [1, 2, 3, 4, 6, 8, 12, 16, 26, 39, 52, 78, 104]  # Weeks out (up to 2 years)

# Strikes configuration - fewer strikes for longer DTEs (realistic)
def get_strikes_around_atm(weeks_out: int) -> int:
    """Get number of strikes around ATM based on DTE.
    Near-term has more strikes, LEAPS have fewer.
    """
    if weeks_out <= 4:
        return 15  # Weekly/near-term: more strikes
    elif weeks_out <= 12:
        return 10  # 1-3 months: moderate strikes
    elif weeks_out <= 26:
        return 7   # 3-6 months: fewer strikes
    elif weeks_out <= 52:
        return 5   # 6-12 months: even fewer
    else:
        return 3   # LEAPS: minimal strikes

# Strike increments vary by price
def get_strike_increment(price: float) -> float:
    """Get strike increment based on underlying price."""
    if price >= 500:
        return 5.0  # $5 increments for high-priced
    elif price >= 200:
        return 2.50  # $2.50 increments for mid-priced
    else:
        return 1.0  # $1 increments for lower-priced


@dataclass
class MockOptionState:
    """Tracks state of a mock option for realistic updates."""
    option: AggregatedOptionData
    base_iv: float
    last_update: datetime


def calculate_black_scholes_delta(
    spot: float,
    strike: float,
    time_to_exp: float,
    iv: float,
    is_call: bool,
) -> float:
    """Simplified Black-Scholes delta calculation."""
    if time_to_exp <= 0:
        # At expiration
        if is_call:
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0

    # Simplified delta approximation
    moneyness = math.log(spot / strike) / (iv * math.sqrt(time_to_exp))
    # Use normal CDF approximation
    delta = 0.5 * (1 + math.erf(moneyness / math.sqrt(2)))

    if is_call:
        return round(delta, 4)
    else:
        return round(delta - 1, 4)


def calculate_option_price(
    spot: float,
    strike: float,
    time_to_exp: float,
    iv: float,
    is_call: bool,
) -> float:
    """Simplified option pricing."""
    if time_to_exp <= 0:
        if is_call:
            return max(0, spot - strike)
        else:
            return max(0, strike - spot)

    # Intrinsic value
    if is_call:
        intrinsic = max(0, spot - strike)
    else:
        intrinsic = max(0, strike - spot)

    # Time value (simplified)
    time_value = spot * iv * math.sqrt(time_to_exp) * 0.4

    # Adjust time value based on moneyness
    moneyness = abs(spot - strike) / spot
    time_value *= math.exp(-moneyness * 2)

    return round(intrinsic + time_value, 2)


def generate_expiration_dates() -> list[str]:
    """Generate realistic Friday expiration dates."""
    today = datetime.now(timezone.utc).date()
    expirations = []

    for weeks in EXPIRATIONS_WEEKS:
        # Find next Friday
        target = today + timedelta(weeks=weeks)
        days_until_friday = (4 - target.weekday()) % 7
        if days_until_friday == 0 and weeks > 0:
            days_until_friday = 7
        expiry = target + timedelta(days=days_until_friday)
        expirations.append(expiry.isoformat())

    return expirations


def generate_strikes(atm_price: float, weeks_out: int = 4) -> list[float]:
    """Generate strikes around ATM price.

    Args:
        atm_price: Current underlying price
        weeks_out: Weeks until expiration (affects number of strikes)
    """
    increment = get_strike_increment(atm_price)
    num_strikes = get_strikes_around_atm(weeks_out)

    # Round ATM to nearest strike increment
    atm_strike = round(atm_price / increment) * increment

    strikes = []
    for i in range(-num_strikes, num_strikes + 1):
        strike = atm_strike + (i * increment)
        if strike > 0:
            strikes.append(strike)

    return strikes


def create_mock_option(
    underlying_symbol: str,
    underlying_price: float,
    strike: float,
    expiry: str,
    right: str,
    now: datetime,
) -> AggregatedOptionData:
    """Create a single mock option with realistic data."""
    # Calculate time to expiration in years
    exp_date = datetime.fromisoformat(expiry).replace(tzinfo=timezone.utc)
    time_to_exp = max(0, (exp_date - now).total_seconds() / (365.25 * 24 * 3600))

    # Base IV varies by moneyness and time
    moneyness = abs(underlying_price - strike) / underlying_price
    base_iv = 0.35 + (moneyness * 0.5) + (0.1 / max(0.1, math.sqrt(time_to_exp * 52)))

    # Add some randomness
    iv = base_iv * random.uniform(0.95, 1.05)
    iv = round(iv, 4)

    is_call = right == "C"

    # Calculate theoretical price
    theo_price = calculate_option_price(underlying_price, strike, time_to_exp, iv, is_call)

    # Create bid/ask spread (wider for OTM, narrower for ATM)
    spread_pct = 0.02 + (moneyness * 0.05)  # 2-7% spread
    spread = max(0.05, theo_price * spread_pct)

    bid = round(max(0.01, theo_price - spread / 2), 2)
    ask = round(theo_price + spread / 2, 2)

    # Calculate Greeks
    delta = calculate_black_scholes_delta(underlying_price, strike, time_to_exp, iv, is_call)

    # Gamma (higher ATM, lower for far OTM/ITM)
    gamma = round(0.05 * math.exp(-moneyness * 10) / max(0.1, math.sqrt(time_to_exp * 52)), 4)

    # Theta (negative, larger magnitude for shorter-dated)
    theta = round(-theo_price * 0.01 / max(0.01, time_to_exp * 365), 4)

    # Vega (higher for longer-dated and ATM)
    vega = round(underlying_price * 0.01 * math.sqrt(time_to_exp) * math.exp(-moneyness * 5), 4)

    canonical_id = CanonicalOptionId(
        underlying=underlying_symbol,
        expiry=expiry,
        right=right,
        strike=strike,
    )

    # OI and volume - higher for ATM, lower for far OTM
    oi_base = 5000 * math.exp(-moneyness * 5)
    open_interest = max(100, int(oi_base * random.uniform(0.5, 1.5)))
    volume = max(50, int(open_interest * random.uniform(0.1, 0.3)))

    return AggregatedOptionData(
        canonical_id=canonical_id,
        bid=bid,
        ask=ask,
        bid_size=random.randint(10, 500),
        ask_size=random.randint(10, 500),
        last=round((bid + ask) / 2, 2),
        quote_timestamp=now.isoformat(),  # Must be ISO string for age_seconds()
        open_interest=open_interest,
        volume=volume,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        iv=iv,
        theoretical_value=theo_price,
        greeks_timestamp=now.isoformat(),  # Must be ISO string for age_seconds()
    )


class MockDataGenerator:
    """Generates and streams mock options data."""

    def __init__(
        self,
        symbol: str = "QQQ",
        on_option_update: Callable[[AggregatedOptionData], None] | None = None,
        on_underlying_update: Callable[[UnderlyingData], None] | None = None,
        update_interval: float = 1.0,
    ):
        self.on_option_update = on_option_update
        self.on_underlying_update = on_underlying_update
        self.update_interval = update_interval

        self._symbol = symbol
        self._underlying_price = BASE_PRICES.get(symbol, DEFAULT_PRICE)
        self._options: dict[str, MockOptionState] = {}
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def symbol(self) -> str:
        return self._symbol

    def set_symbol(self, symbol: str) -> None:
        """Switch to a different symbol, regenerating the chain."""
        if symbol != self._symbol:
            self._symbol = symbol
            self._underlying_price = BASE_PRICES.get(symbol, DEFAULT_PRICE)
            self._options.clear()
            # Generate new chain for this symbol
            self.generate_initial_chain()

    @property
    def underlying_price(self) -> float:
        return self._underlying_price

    @property
    def option_count(self) -> int:
        return len(self._options)

    def generate_initial_chain(self) -> tuple[UnderlyingData, list[AggregatedOptionData]]:
        """Generate initial options chain snapshot."""
        now = datetime.now(timezone.utc)

        # Create underlying data (timestamp must be ISO string for age_seconds())
        # IV rank < 50 is favorable for buying premium (passes iv_rank_appropriate gate)
        underlying = UnderlyingData(
            symbol=self._symbol,
            price=self._underlying_price,
            iv_rank=random.uniform(30, 45),  # Keep in favorable range for BUY actions
            iv_percentile=random.uniform(30, 50),
            timestamp=now.isoformat(),
        )

        # Generate options with different strike counts per expiration
        expirations = generate_expiration_dates()

        options = []
        total_strikes = 0
        for idx, expiry in enumerate(expirations):
            weeks_out = EXPIRATIONS_WEEKS[idx] if idx < len(EXPIRATIONS_WEEKS) else 52
            strikes = generate_strikes(self._underlying_price, weeks_out)
            total_strikes += len(strikes)

            for strike in strikes:
                for right in ["C", "P"]:
                    option = create_mock_option(
                        self._symbol, self._underlying_price, strike, expiry, right, now
                    )
                    options.append(option)

                    # Track for updates
                    key = f"{expiry}-{strike}-{right}"
                    self._options[key] = MockOptionState(
                        option=option,
                        base_iv=option.iv or 0.35,
                        last_update=now,
                    )

        logger.info(
            f"Generated mock chain for {self._symbol}: {len(options)} options, "
            f"{len(expirations)} expirations, avg {total_strikes // len(expirations)} strikes/exp"
        )

        return underlying, options

    def _update_underlying_price(self) -> float:
        """Simulate price movement."""
        change = random.gauss(0, PRICE_VOLATILITY)
        self._underlying_price *= (1 + change)
        self._underlying_price = round(self._underlying_price, 2)
        return self._underlying_price

    def _update_option(self, state: MockOptionState, now: datetime) -> AggregatedOptionData:
        """Update a single option with new prices."""
        option = state.option

        # Recalculate with new underlying price
        exp_date = datetime.fromisoformat(option.canonical_id.expiry).replace(tzinfo=timezone.utc)
        time_to_exp = max(0, (exp_date - now).total_seconds() / (365.25 * 24 * 3600))

        # Slightly vary IV
        iv = state.base_iv * random.uniform(0.98, 1.02)
        iv = round(iv, 4)

        is_call = option.canonical_id.right == "C"
        strike = option.canonical_id.strike

        # Recalculate price and Greeks
        theo_price = calculate_option_price(
            self._underlying_price, strike, time_to_exp, iv, is_call
        )

        moneyness = abs(self._underlying_price - strike) / self._underlying_price
        spread_pct = 0.02 + (moneyness * 0.05)
        spread = max(0.05, theo_price * spread_pct)

        bid = round(max(0.01, theo_price - spread / 2), 2)
        ask = round(theo_price + spread / 2, 2)

        delta = calculate_black_scholes_delta(
            self._underlying_price, strike, time_to_exp, iv, is_call
        )

        # Update the option (keep OI stable, slightly vary volume)
        updated = AggregatedOptionData(
            canonical_id=option.canonical_id,
            bid=bid,
            ask=ask,
            bid_size=random.randint(10, 500),
            ask_size=random.randint(10, 500),
            last=round((bid + ask) / 2, 2),
            quote_timestamp=now.isoformat(),  # Must be ISO string
            open_interest=option.open_interest,  # OI stays stable
            volume=option.volume + random.randint(0, 10),  # Volume accumulates
            delta=delta,
            gamma=option.gamma,  # Keep gamma/theta/vega stable
            theta=option.theta,
            vega=option.vega,
            iv=iv,
            theoretical_value=theo_price,
            greeks_timestamp=now.isoformat(),  # Update greeks timestamp too
        )

        state.option = updated
        state.last_update = now

        return updated

    async def start_streaming(self) -> None:
        """Start streaming mock updates."""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting mock data streaming (interval: {self.update_interval}s)")

        while self._running:
            try:
                await asyncio.sleep(self.update_interval)

                if not self._running:
                    break

                now = datetime.now(timezone.utc)

                # Update underlying price
                self._update_underlying_price()

                if self.on_underlying_update:
                    underlying = UnderlyingData(
                        symbol=self._symbol,
                        price=self._underlying_price,
                        iv_rank=random.uniform(30, 45),  # Keep in favorable range
                        iv_percentile=random.uniform(30, 50),
                        timestamp=now.isoformat(),
                    )
                    self.on_underlying_update(underlying)

                # Update a random subset of options (simulates market activity)
                # Always include ATM options to keep them fresh for gate evaluation
                keys = list(self._options.keys())

                # Find ATM strike using the correct increment for this symbol
                increment = get_strike_increment(self._underlying_price)
                atm_strike = round(self._underlying_price / increment) * increment
                atm_keys = [k for k in keys if f"-{atm_strike}-" in k]

                # Random subset plus ATM options
                num_updates = random.randint(10, 30)
                other_keys = [k for k in keys if k not in atm_keys]
                random_keys = random.sample(other_keys, min(num_updates, len(other_keys)))
                update_keys = list(set(atm_keys + random_keys))

                for key in update_keys:
                    state = self._options[key]
                    updated = self._update_option(state, now)

                    if self.on_option_update:
                        self.on_option_update(updated)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in mock streaming: {e}")
                await asyncio.sleep(1)

        logger.info("Mock data streaming stopped")

    def stop_streaming(self) -> None:
        """Stop streaming mock updates."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    def get_all_options(self) -> list[AggregatedOptionData]:
        """Get all current options."""
        return [state.option for state in self._options.values()]

    def get_underlying(self) -> UnderlyingData:
        """Get current underlying data."""
        return UnderlyingData(
            symbol=self._symbol,
            price=self._underlying_price,
            iv_rank=40.0,  # Favorable for buying premium
            iv_percentile=45.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
