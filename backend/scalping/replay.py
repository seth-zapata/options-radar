"""Quote replay system for backtesting scalping strategies.

Simulates real-time quote arrival from historical DataBento CBBO data,
allowing the scalping components to be tested against historical data
as if it were streaming in real-time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, Iterator

import pandas as pd

from backend.data.databento_loader import DataBentoLoader
from backend.models.canonical import CanonicalOptionId, parse_occ
from backend.scalping.bar_cache import BarCache, CachedBar, SecondBarCache

logger = logging.getLogger(__name__)


@dataclass
class ReplayQuote:
    """A single quote event during replay.

    Mirrors the structure of live QuoteData but optimized for replay.
    """

    timestamp: datetime
    symbol: str  # OCC symbol
    underlying: str
    expiry: str
    right: str  # "C" or "P"
    strike: float
    dte: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid."""
        if self.mid <= 0:
            return 100.0
        return (self.spread / self.mid) * 100


@dataclass
class ReplayTick:
    """A single tick during replay, containing all quotes at that timestamp.

    In CBBO-1m data, each minute may have multiple option quotes.
    This groups them for efficient processing.
    """

    timestamp: datetime
    underlying_price: float | None  # Estimated from options
    quotes: list[ReplayQuote] = field(default_factory=list)

    @property
    def call_quotes(self) -> list[ReplayQuote]:
        """Filter to call options only."""
        return [q for q in self.quotes if q.right == "C"]

    @property
    def put_quotes(self) -> list[ReplayQuote]:
        """Filter to put options only."""
        return [q for q in self.quotes if q.right == "P"]


@dataclass
class BacktestTick:
    """Simple tick format for backtester compatibility.

    The backtester expects individual ticks rather than grouped ReplayTick.
    This provides a flat structure for each price update.
    """

    timestamp: datetime
    symbol: str  # OCC symbol or underlying symbol
    is_underlying: bool
    mid_price: float
    bid_price: float
    ask_price: float
    volume: int | None = None


class QuoteReplaySystem:
    """Replays historical quotes for backtesting.

    Usage:
        loader = DataBentoLoader("/path/to/data")
        replay = QuoteReplaySystem(loader)

        # Iterate through a day's quotes
        for tick in replay.replay_day(date(2024, 1, 3)):
            # Process tick.quotes as if they just arrived
            for quote in tick.quotes:
                velocity_tracker.add_price(tick.underlying_price, tick.timestamp)
                # ... check signals, etc.

        # Or replay a date range
        for tick in replay.replay_range(start, end):
            process(tick)
    """

    # Market hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    def __init__(
        self,
        loader: DataBentoLoader,
        filter_market_hours: bool = True,
        max_dte: int = 1,
        min_bid: float = 0.05,
        max_spread_pct: float = 10.0,
        bar_cache: BarCache | None = None,
        second_bar_cache: SecondBarCache | None = None,
    ):
        """Initialize replay system.

        Args:
            loader: DataBentoLoader instance
            filter_market_hours: Only yield quotes during market hours
            max_dte: Maximum DTE to include (0 = 0DTE only, 1 = include 1DTE)
            min_bid: Minimum bid price filter
            max_spread_pct: Maximum spread as percentage of mid
            bar_cache: Optional BarCache for 1-minute underlying prices (fallback)
            second_bar_cache: Optional SecondBarCache for 1-second underlying prices (preferred)
        """
        self.loader = loader
        self.filter_market_hours = filter_market_hours
        self.max_dte = max_dte
        self.min_bid = min_bid
        self.max_spread_pct = max_spread_pct
        self.bar_cache = bar_cache
        self.second_bar_cache = second_bar_cache
        self._bars_in_memory: dict[datetime, CachedBar] | None = None
        self._second_bars_in_memory: dict[datetime, CachedBar] | None = None

    def replay_day(
        self,
        target_date: date,
        symbol_filter: str | None = "TSLA",
    ) -> Iterator[ReplayTick]:
        """Replay a single day's quotes.

        Args:
            target_date: Date to replay
            symbol_filter: Only include options for this underlying (default TSLA)

        Yields:
            ReplayTick for each unique timestamp
        """
        try:
            df = self.loader.load_day(target_date)
        except FileNotFoundError:
            logger.warning(f"No data for {target_date}")
            return

        # Filter and process
        yield from self._process_dataframe(df, target_date, symbol_filter)

    def load_bars_for_range(self, symbol: str, start_date: date, end_date: date) -> None:
        """Pre-load bar cache into memory for a date range.

        Call this before replay_range for better performance.
        Prefers 1-second bars when available, falls back to 1-minute bars.

        Args:
            symbol: Stock symbol (e.g., "TSLA")
            start_date: Start date
            end_date: End date
        """
        # Try loading 1-second bars first (preferred - more accurate)
        if self.second_bar_cache:
            self._second_bars_in_memory = self.second_bar_cache.load_range_to_memory(
                symbol, start_date, end_date
            )

        # Also load 1-minute bars as fallback
        if self.bar_cache:
            self._bars_in_memory = self.bar_cache.load_range_to_memory(
                symbol, start_date, end_date
            )

    def replay_range(
        self,
        start_date: date,
        end_date: date,
        symbol_filter: str | None = "TSLA",
        on_day_start: Callable[[date], None] | None = None,
        on_day_end: Callable[[date, int], None] | None = None,
    ) -> Iterator[ReplayTick]:
        """Replay quotes across a date range.

        Args:
            start_date: First date (inclusive)
            end_date: Last date (inclusive)
            symbol_filter: Only include options for this underlying
            on_day_start: Callback at start of each day
            on_day_end: Callback at end of each day with tick count

        Yields:
            ReplayTick for each unique timestamp across all days
        """
        # Load bars into memory if we have a cache and haven't loaded yet
        if self.bar_cache and symbol_filter and self._bars_in_memory is None:
            self.load_bars_for_range(symbol_filter, start_date, end_date)

        for current_date, df in self.loader.load_date_range(start_date, end_date):
            if on_day_start:
                on_day_start(current_date)

            tick_count = 0
            for tick in self._process_dataframe(df, current_date, symbol_filter):
                tick_count += 1
                yield tick

            if on_day_end:
                on_day_end(current_date, tick_count)

    def replay(
        self,
        start_date: date,
        end_date: date,
        symbol_filter: str | None = "TSLA",
        on_day_start: Callable[[date], None] | None = None,
        on_day_end: Callable[[date, int], None] | None = None,
    ) -> Iterator[BacktestTick]:
        """Replay quotes as flat BacktestTick objects for backtester.

        This method converts ReplayTick (grouped by timestamp) into
        individual BacktestTick objects that the backtester expects.

        Args:
            start_date: First date (inclusive)
            end_date: Last date (inclusive)
            symbol_filter: Only include options for this underlying
            on_day_start: Callback at start of each day
            on_day_end: Callback at end of each day with tick count

        Yields:
            BacktestTick for each quote (underlying first, then options)
        """
        for tick in self.replay_range(start_date, end_date, symbol_filter, on_day_start, on_day_end):
            # First yield underlying price update if available
            if tick.underlying_price is not None:
                yield BacktestTick(
                    timestamp=tick.timestamp,
                    symbol=symbol_filter or "UND",
                    is_underlying=True,
                    mid_price=tick.underlying_price,
                    bid_price=tick.underlying_price,
                    ask_price=tick.underlying_price,
                    volume=None,
                )

            # Then yield each option quote
            for quote in tick.quotes:
                yield BacktestTick(
                    timestamp=tick.timestamp,
                    symbol=quote.symbol,
                    is_underlying=False,
                    mid_price=quote.mid,
                    bid_price=quote.bid,
                    ask_price=quote.ask,
                    volume=None,
                )

    def _process_dataframe(
        self,
        df: pd.DataFrame,
        current_date: date,
        symbol_filter: str | None,
    ) -> Iterator[ReplayTick]:
        """Process a DataFrame into ReplayTicks.

        Args:
            df: Raw CBBO data
            current_date: Date being processed
            symbol_filter: Underlying to filter for

        Yields:
            ReplayTick for each timestamp
        """
        # Apply scalping filters
        filtered = self.loader.filter_for_scalping(
            df,
            current_date,
            max_dte=self.max_dte,
            min_bid=self.min_bid,
            max_spread_pct=self.max_spread_pct,
        )

        if filtered.empty:
            logger.debug(f"No quotes after filtering for {current_date}")
            return

        # Filter by underlying if specified
        if symbol_filter:
            filtered = filtered[filtered["underlying"] == symbol_filter]
            if filtered.empty:
                logger.debug(f"No {symbol_filter} quotes for {current_date}")
                return

        # Sort by timestamp
        filtered = filtered.sort_values("ts_event")

        # Group by timestamp
        for timestamp, group in filtered.groupby("ts_event"):
            # Convert timestamp
            if isinstance(timestamp, pd.Timestamp):
                ts = timestamp.to_pydatetime()
            elif isinstance(timestamp, str):
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                ts = timestamp

            # Make timezone-aware if not already
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            # Filter market hours if enabled
            if self.filter_market_hours:
                # Convert to Eastern for market hours check
                # Note: This is simplified - production should use pytz
                local_hour = ts.hour - 5  # Rough EST offset from UTC
                local_minute = ts.minute
                local_time = time(local_hour % 24, local_minute)

                if local_time < self.MARKET_OPEN or local_time >= self.MARKET_CLOSE:
                    continue

            # Build quotes
            quotes = []
            for _, row in group.iterrows():
                quote = ReplayQuote(
                    timestamp=ts,
                    symbol=row["symbol"],
                    underlying=row["underlying"],
                    expiry=row["expiry"],
                    right=row["right"],
                    strike=row["strike"],
                    dte=row["dte"],
                    bid=row["bid_px"],
                    ask=row["ask_px"],
                    bid_size=row["bid_sz"],
                    ask_size=row["ask_sz"],
                )
                quotes.append(quote)

            # Get underlying price - prefer real bars over estimation
            underlying_price = self._get_underlying_price(ts, quotes)

            yield ReplayTick(
                timestamp=ts,
                underlying_price=underlying_price,
                quotes=quotes,
            )

    def _get_underlying_price(
        self, timestamp: datetime, quotes: list[ReplayQuote]
    ) -> float | None:
        """Get underlying price from bar cache or estimate from options.

        Prefers real bar data when available, falls back to put-call parity estimation.

        Args:
            timestamp: Time to look up
            quotes: Quotes at this timestamp (for fallback estimation)

        Returns:
            Underlying price or None
        """
        # Try bar cache first (real prices)
        if self._bars_in_memory:
            price = self._lookup_bar_price(timestamp)
            if price is not None:
                return price

        # Fallback to estimation from options
        return self._estimate_underlying_price(quotes)

    def _lookup_bar_price(self, timestamp: datetime) -> float | None:
        """Look up price from loaded bar cache.

        IMPORTANT: To avoid lookahead bias, we use the PREVIOUS bar's close.
        A bar's close price isn't known until the end of that bar, so:
        - For 1-second bars: At 14:33:15, use 14:33:14 bar close (1 second delay)
        - For 1-minute bars: At 14:33:15, use 14:32 bar close (up to 75 second delay)

        Prefers 1-second bars when available for more accurate momentum detection.

        Args:
            timestamp: Time to look up

        Returns:
            Close price of the PREVIOUS completed bar, or None if not found
        """
        # Try 1-second bars first (preferred - only 1 second delay)
        if self._second_bars_in_memory:
            price = self._lookup_second_bar_price(timestamp)
            if price is not None:
                return price

        # Fall back to 1-minute bars
        if self._bars_in_memory:
            return self._lookup_minute_bar_price(timestamp)

        return None

    def _lookup_second_bar_price(self, timestamp: datetime) -> float | None:
        """Look up price from 1-second bars.

        Uses the PREVIOUS second's bar close to avoid lookahead bias.
        At 14:33:15.500, we use the 14:33:14 bar close (last completed second).

        Args:
            timestamp: Time to look up

        Returns:
            Close price or None if not found
        """
        if not self._second_bars_in_memory:
            return None

        # Use the PREVIOUS second's bar to avoid lookahead bias
        prev_second = timestamp.replace(microsecond=0) - timedelta(seconds=1)

        if prev_second in self._second_bars_in_memory:
            return self._second_bars_in_memory[prev_second].close

        # Fallback: find nearest EARLIER bar within 5 seconds
        best_bar = None
        best_diff = timedelta(seconds=5)

        for ts, bar in self._second_bars_in_memory.items():
            if ts >= timestamp:
                continue
            diff = timestamp - ts
            if diff < best_diff:
                best_diff = diff
                best_bar = bar

        return best_bar.close if best_bar else None

    def _lookup_minute_bar_price(self, timestamp: datetime) -> float | None:
        """Look up price from 1-minute bars.

        Uses the PREVIOUS minute's bar close to avoid lookahead bias.
        At 14:33:15, we use the 14:32 bar close (last known price).

        Args:
            timestamp: Time to look up

        Returns:
            Close price or None if not found
        """
        if not self._bars_in_memory:
            return None

        # Use the PREVIOUS minute's bar to avoid lookahead bias
        prev_minute = (timestamp.replace(second=0, microsecond=0) - timedelta(minutes=1))

        if prev_minute in self._bars_in_memory:
            return self._bars_in_memory[prev_minute].close

        # Fallback: find nearest EARLIER bar within 5 minutes
        best_bar = None
        best_diff = timedelta(minutes=5)

        for ts, bar in self._bars_in_memory.items():
            if ts >= timestamp:
                continue
            diff = timestamp - ts
            if diff < best_diff:
                best_diff = diff
                best_bar = bar

        return best_bar.close if best_bar else None

    def _estimate_underlying_price(self, quotes: list[ReplayQuote]) -> float | None:
        """Estimate underlying price from option quotes using put-call parity.

        For ATM options at strike K with call mid C and put mid P:
        underlying ≈ K + (C - P)

        This is more accurate than using strike alone because it captures
        actual market expectations of underlying price.

        Args:
            quotes: List of quotes at this timestamp

        Returns:
            Estimated underlying price or None
        """
        # Filter to valid quotes with tight spreads
        calls = [q for q in quotes if q.right == "C" and q.spread > 0 and q.spread_pct < 20]
        puts = [q for q in quotes if q.right == "P" and q.spread > 0 and q.spread_pct < 20]

        if not calls:
            return None

        # Find tightest-spread call (likely most liquid/ATM)
        best_call = min(calls, key=lambda q: q.spread_pct)

        # Try to find matching put at same strike
        matching_put = next((p for p in puts if p.strike == best_call.strike), None)

        if matching_put:
            # Use put-call parity: underlying = strike + (call_mid - put_mid)
            return best_call.strike + (best_call.mid - matching_put.mid)

        # Fallback: estimate from call alone using delta approximation
        # For ATM call (delta ~0.5), underlying ≈ strike + 2*(call_mid - (strike - underlying)/2)
        # Simplified: just use strike + call_mid as rough estimate
        # Since ATM call mid ~ underlying * delta, and delta ~ 0.5 for ATM
        # This gives underlying ~ strike + 2 * call_mid * 0.05 (for ~5% of strike)
        # But safer to just return strike + call_mid for now as directional proxy
        return best_call.strike + best_call.mid * 0.5

    def get_day_summary(self, target_date: date) -> dict:
        """Get summary statistics for a day's data.

        Args:
            target_date: Date to summarize

        Returns:
            Dict with quote counts, unique symbols, etc.
        """
        try:
            df = self.loader.load_day(target_date)
        except FileNotFoundError:
            return {"date": target_date.isoformat(), "error": "No data"}

        filtered = self.loader.filter_for_scalping(
            df, target_date, max_dte=self.max_dte
        )

        return {
            "date": target_date.isoformat(),
            "total_quotes": len(df),
            "filtered_quotes": len(filtered),
            "unique_symbols": filtered["symbol"].nunique() if not filtered.empty else 0,
            "unique_timestamps": filtered["ts_event"].nunique() if not filtered.empty else 0,
            "calls": len(filtered[filtered["right"] == "C"]) if "right" in filtered.columns else 0,
            "puts": len(filtered[filtered["right"] == "P"]) if "right" in filtered.columns else 0,
        }


@dataclass
class BacktestClock:
    """Simulated clock for backtesting.

    Tracks the current simulation time and provides utilities
    for time-based operations during backtesting.
    """

    current_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    speed_multiplier: float = 1.0  # Not used in tick-by-tick replay

    def advance_to(self, new_time: datetime) -> timedelta:
        """Advance clock to new time.

        Args:
            new_time: New simulation time

        Returns:
            Time elapsed since last update
        """
        elapsed = new_time - self.current_time
        self.current_time = new_time
        return elapsed

    def seconds_since(self, past_time: datetime) -> float:
        """Get seconds elapsed since a past time."""
        return (self.current_time - past_time).total_seconds()

    def minutes_since(self, past_time: datetime) -> float:
        """Get minutes elapsed since a past time."""
        return self.seconds_since(past_time) / 60

    @property
    def is_market_hours(self) -> bool:
        """Check if current time is during market hours (rough)."""
        # Simplified EST check
        hour = self.current_time.hour - 5  # UTC to EST offset
        return 9 <= hour < 16
