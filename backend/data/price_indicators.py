"""Price indicators for bear market detection.

Provides helper functions for calculating:
- 52-week high
- Simple moving averages (50, 200 day)
- Consecutive days below SMA
- Death cross detection

Uses yfinance for historical price data (free).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# Cache for price data to avoid repeated API calls
_price_cache: dict[str, pd.DataFrame] = {}
_cache_timestamps: dict[str, datetime] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_price_data(symbol: str, days_back: int = 300) -> pd.DataFrame:
    """Get historical price data with caching.

    Args:
        symbol: Stock symbol
        days_back: Number of calendar days of history to fetch

    Returns:
        DataFrame with OHLCV data, indexed by date string (YYYY-MM-DD)
    """
    import yfinance as yf

    cache_key = f"{symbol}_{days_back}"
    now = datetime.now()

    # Check cache
    if cache_key in _price_cache:
        cached_time = _cache_timestamps.get(cache_key)
        if cached_time and (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
            return _price_cache[cache_key]

    # Fetch from yfinance
    try:
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            logger.warning(f"No price data for {symbol}")
            return pd.DataFrame()

        # Convert index to string dates
        hist.index = pd.to_datetime(hist.index).strftime("%Y-%m-%d")

        # Cache the result
        _price_cache[cache_key] = hist
        _cache_timestamps[cache_key] = now

        return hist

    except Exception as e:
        logger.error(f"Failed to fetch price data for {symbol}: {e}")
        return pd.DataFrame()


def _get_historical_price_data(
    symbol: str,
    as_of_date: date,
    days_back: int = 300
) -> pd.DataFrame:
    """Get historical price data up to a specific date (for backtesting).

    Args:
        symbol: Stock symbol
        as_of_date: Get data up to this date
        days_back: Number of calendar days of history

    Returns:
        DataFrame with OHLCV data up to as_of_date
    """
    import yfinance as yf

    try:
        end_date = (as_of_date + timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (as_of_date - timedelta(days=days_back)).strftime("%Y-%m-%d")

        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            return pd.DataFrame()

        # Convert index to string dates
        hist.index = pd.to_datetime(hist.index).strftime("%Y-%m-%d")

        # Filter to only dates <= as_of_date
        as_of_str = as_of_date.strftime("%Y-%m-%d")
        hist = hist[hist.index <= as_of_str]

        return hist

    except Exception as e:
        logger.error(f"Failed to fetch historical price data for {symbol}: {e}")
        return pd.DataFrame()


def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Current price or None if unavailable
    """
    df = _get_price_data(symbol, days_back=5)
    if df.empty:
        return None
    return float(df['Close'].iloc[-1])


def get_52_week_high(symbol: str, as_of_date: Optional[date] = None) -> Optional[float]:
    """Get 52-week high price.

    Args:
        symbol: Stock symbol
        as_of_date: Calculate as of this date (None = current)

    Returns:
        52-week high price or None if unavailable
    """
    if as_of_date:
        df = _get_historical_price_data(symbol, as_of_date, days_back=365)
    else:
        df = _get_price_data(symbol, days_back=365)

    if df.empty:
        return None

    return float(df['High'].max())


def get_sma(
    symbol: str,
    period: int,
    as_of_date: Optional[date] = None
) -> Optional[float]:
    """Get simple moving average.

    Args:
        symbol: Stock symbol
        period: SMA period (e.g., 50, 200)
        as_of_date: Calculate as of this date (None = current)

    Returns:
        SMA value or None if insufficient data
    """
    # Convert trading days to calendar days (roughly 252 trading days per 365 calendar days)
    # Use 1.5x multiplier plus buffer to ensure we get enough trading days
    calendar_days_needed = int(period * 1.5) + 30

    if as_of_date:
        df = _get_historical_price_data(symbol, as_of_date, days_back=calendar_days_needed)
    else:
        df = _get_price_data(symbol, days_back=calendar_days_needed)

    if df.empty or len(df) < period:
        return None

    sma = df['Close'].rolling(window=period).mean()
    return float(sma.iloc[-1])


def get_price_for_date(
    symbol: str,
    check_date: date,
    price_data: Optional[pd.DataFrame] = None
) -> Optional[float]:
    """Get closing price for a specific date.

    Args:
        symbol: Stock symbol
        check_date: Date to get price for
        price_data: Optional pre-fetched price DataFrame

    Returns:
        Closing price or None if unavailable
    """
    if price_data is None:
        price_data = _get_historical_price_data(symbol, check_date, days_back=5)

    if price_data.empty:
        return None

    date_str = check_date.strftime("%Y-%m-%d")
    if date_str in price_data.index:
        return float(price_data.loc[date_str, 'Close'])

    return None


def count_consecutive_days_below_sma(
    symbol: str,
    sma_period: int,
    as_of_date: Optional[date] = None,
    threshold_ratio: float = 1.0
) -> int:
    """Count consecutive trading days price has been below SMA.

    Args:
        symbol: Stock symbol
        sma_period: SMA period (e.g., 200)
        as_of_date: Calculate as of this date (None = current)
        threshold_ratio: Price must be below SMA * ratio (default 1.0 = exactly below)

    Returns:
        Number of consecutive days below SMA
    """
    # Convert trading days to calendar days plus extra for lookback
    # Use 1.5x multiplier for SMA period plus buffer for consecutive day counting
    calendar_days_needed = int(sma_period * 1.5) + 100

    if as_of_date:
        df = _get_historical_price_data(symbol, as_of_date, days_back=calendar_days_needed)
    else:
        df = _get_price_data(symbol, days_back=calendar_days_needed)

    if df.empty or len(df) < sma_period:
        return 0

    # Calculate SMA
    df = df.copy()
    df['sma'] = df['Close'].rolling(window=sma_period).mean()
    df['below_sma'] = df['Close'] < (df['sma'] * threshold_ratio)

    # Count consecutive days from the end
    consecutive = 0
    for below in reversed(df['below_sma'].values):
        if pd.isna(below):
            break
        if below:
            consecutive += 1
        else:
            break

    return consecutive


def find_death_cross_date(
    symbol: str,
    as_of_date: Optional[date] = None,
    lookback_days: int = 60
) -> Optional[date]:
    """Find when 50-day SMA crossed below 200-day SMA (death cross).

    Args:
        symbol: Stock symbol
        as_of_date: Look for cross before this date (None = current)
        lookback_days: Only consider crosses within this many days

    Returns:
        Date of death cross or None if not found/not active
    """
    # Convert trading days to calendar days (need 200 trading days for SMA200)
    # Use 1.5x multiplier plus buffer for lookback
    calendar_days_needed = int(200 * 1.5) + lookback_days + 30

    if as_of_date:
        df = _get_historical_price_data(symbol, as_of_date, days_back=calendar_days_needed)
    else:
        df = _get_price_data(symbol, days_back=calendar_days_needed)

    if df.empty or len(df) < 200:
        return None

    # Calculate SMAs
    df = df.copy()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()

    # Find crossover points (50 SMA crosses below 200 SMA)
    df['death_cross'] = (df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))

    # Get dates with death crosses
    cross_dates = df[df['death_cross']].index.tolist()

    if not cross_dates:
        return None

    # Get most recent cross
    most_recent = cross_dates[-1]
    cross_date = datetime.strptime(most_recent, "%Y-%m-%d").date()

    # Check if within lookback period
    if as_of_date:
        reference_date = as_of_date
    else:
        reference_date = date.today()

    if (reference_date - cross_date).days <= lookback_days:
        return cross_date

    return None


def is_death_cross_active(
    symbol: str,
    as_of_date: Optional[date] = None
) -> bool:
    """Check if death cross is currently active (50 SMA < 200 SMA).

    Args:
        symbol: Stock symbol
        as_of_date: Check as of this date (None = current)

    Returns:
        True if 50 SMA is below 200 SMA
    """
    sma_50 = get_sma(symbol, 50, as_of_date)
    sma_200 = get_sma(symbol, 200, as_of_date)

    if sma_50 is None or sma_200 is None:
        return False

    return sma_50 < sma_200


def get_bear_market_indicators(
    symbol: str,
    as_of_date: Optional[date] = None
) -> dict:
    """Get all bear market indicators for a symbol.

    Args:
        symbol: Stock symbol
        as_of_date: Calculate as of this date (None = current)

    Returns:
        Dictionary with all indicator values
    """
    if as_of_date:
        df = _get_historical_price_data(symbol, as_of_date, days_back=300)
        current_price = float(df['Close'].iloc[-1]) if not df.empty else None
    else:
        df = _get_price_data(symbol, days_back=300)
        current_price = get_current_price(symbol)

    high_52w = get_52_week_high(symbol, as_of_date)
    sma_200 = get_sma(symbol, 200, as_of_date)
    sma_50 = get_sma(symbol, 50, as_of_date)
    days_below_sma = count_consecutive_days_below_sma(symbol, 200, as_of_date)
    death_cross_date = find_death_cross_date(symbol, as_of_date)

    # Calculate derived values
    drawdown_pct = None
    if current_price and high_52w:
        drawdown_pct = (current_price - high_52w) / high_52w * 100

    price_vs_sma_pct = None
    if current_price and sma_200:
        price_vs_sma_pct = (current_price - sma_200) / sma_200 * 100

    return {
        "symbol": symbol,
        "as_of_date": as_of_date.strftime("%Y-%m-%d") if as_of_date else "current",
        "current_price": current_price,
        "high_52w": high_52w,
        "drawdown_pct": drawdown_pct,
        "sma_200": sma_200,
        "sma_50": sma_50,
        "price_vs_sma_pct": price_vs_sma_pct,
        "days_below_sma": days_below_sma,
        "death_cross_active": sma_50 < sma_200 if sma_50 and sma_200 else False,
        "death_cross_date": death_cross_date.strftime("%Y-%m-%d") if death_cross_date else None,
    }


def get_recent_low(
    symbol: str,
    as_of_date: Optional[date] = None,
    days: int = 5
) -> Optional[float]:
    """Get the lowest price in the last N trading days.

    Args:
        symbol: Stock symbol
        as_of_date: Calculate as of this date (None = current)
        days: Number of trading days to look back

    Returns:
        Lowest closing price in the period or None if unavailable
    """
    # Convert trading days to calendar days (roughly 1.5x)
    calendar_days = int(days * 1.5) + 5

    if as_of_date:
        df = _get_historical_price_data(symbol, as_of_date, days_back=calendar_days)
    else:
        df = _get_price_data(symbol, days_back=calendar_days)

    if df.empty or len(df) < 1:
        return None

    # Get last N trading days
    recent_data = df.tail(days)
    return float(recent_data['Low'].min())


def get_recent_high(
    symbol: str,
    as_of_date: Optional[date] = None,
    days: int = 5
) -> Optional[float]:
    """Get the highest price in the last N trading days.

    Args:
        symbol: Stock symbol
        as_of_date: Calculate as of this date (None = current)
        days: Number of trading days to look back

    Returns:
        Highest closing price in the period or None if unavailable
    """
    # Convert trading days to calendar days (roughly 1.5x)
    calendar_days = int(days * 1.5) + 5

    if as_of_date:
        df = _get_historical_price_data(symbol, as_of_date, days_back=calendar_days)
    else:
        df = _get_price_data(symbol, days_back=calendar_days)

    if df.empty or len(df) < 1:
        return None

    # Get last N trading days
    recent_data = df.tail(days)
    return float(recent_data['High'].max())


def get_price_n_days_ago(
    symbol: str,
    as_of_date: Optional[date] = None,
    n: int = 1
) -> Optional[float]:
    """Get closing price from N trading days ago.

    Args:
        symbol: Stock symbol
        as_of_date: Calculate as of this date (None = current)
        n: Number of trading days back (1 = yesterday)

    Returns:
        Closing price or None if unavailable
    """
    # Convert trading days to calendar days
    calendar_days = int(n * 1.5) + 5

    if as_of_date:
        df = _get_historical_price_data(symbol, as_of_date, days_back=calendar_days)
    else:
        df = _get_price_data(symbol, days_back=calendar_days)

    if df.empty or len(df) <= n:
        return None

    # Get the price N days ago (from the end)
    return float(df['Close'].iloc[-(n + 1)])


def clear_cache():
    """Clear the price data cache."""
    global _price_cache, _cache_timestamps
    _price_cache = {}
    _cache_timestamps = {}
