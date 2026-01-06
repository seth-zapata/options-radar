"""DataBento CBBO-1m options data loader.

Loads historical TSLA options data from DataBento for backtesting
the scalping module. Supports both compressed (.csv.zst) and
uncompressed (.csv) formats.

Data specs:
- Dataset: OPRA (TSLA.OPT)
- Schema: CBBO-1m (Consolidated Best Bid/Offer, 1-minute intervals)
- Date Range: 2022-01-01 to 2025-12-31
- Size: ~135 GB compressed
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

# Use existing OCC parsing from models
from backend.models.canonical import CanonicalOptionId, parse_occ

logger = logging.getLogger(__name__)


@dataclass
class DataBentoConfig:
    """Configuration for DataBento data loading."""

    data_dir: Path
    # Scalping filters
    max_dte: int = 1  # 0DTE and 1DTE only
    min_bid: float = 0.05  # Filter out penny options
    max_spread_pct: float = 10.0  # Maximum spread as % of mid
    min_open_interest: int = 100  # Minimum OI (if available)
    # Performance tuning
    chunk_size: int = 100_000  # Rows per chunk for large files


class DataBentoLoader:
    """Load and process DataBento CBBO-1m options data.

    Usage:
        loader = DataBentoLoader("/path/to/databento/data")

        # Load single day
        df = loader.load_day(date(2024, 1, 3))

        # Load and filter for scalping
        scalp_df = loader.filter_for_scalping(df, date(2024, 1, 3))

        # Iterate date range
        for day, df in loader.load_date_range(start, end):
            process(df)
    """

    # Expected CBBO-1m columns
    REQUIRED_COLUMNS = {"ts_recv", "ts_event", "symbol", "bid_px", "ask_px", "bid_sz", "ask_sz"}
    OPTIONAL_COLUMNS = {"bid_ct", "ask_ct"}  # Order counts

    def __init__(
        self,
        data_dir: str | Path,
        config: DataBentoConfig | None = None,
    ):
        """Initialize with path to DataBento data directory.

        Args:
            data_dir: Directory containing daily .csv.zst or .csv files
            config: Optional configuration overrides
        """
        self.data_dir = Path(data_dir)
        self.config = config or DataBentoConfig(data_dir=self.data_dir)
        self._zstd_available = self._check_zstd()
        self._validate_data_dir()

    def _check_zstd(self) -> bool:
        """Check if zstandard library is available."""
        try:
            import zstandard  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "zstandard not installed. Install with: pip install zstandard. "
                "Only uncompressed .csv files will be supported."
            )
            return False

    def _validate_data_dir(self) -> None:
        """Verify data directory exists and contains expected files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Check for data files
        zst_files = list(self.data_dir.glob("*.csv.zst"))
        csv_files = list(self.data_dir.glob("*.csv"))

        if not zst_files and not csv_files:
            logger.warning(f"No data files found in {self.data_dir}")
        else:
            total = len(zst_files) + len(csv_files)
            logger.info(
                f"DataBento loader initialized: {total} files "
                f"({len(zst_files)} compressed, {len(csv_files)} uncompressed)"
            )

    def get_available_dates(self) -> list[date]:
        """Get list of available dates in the data directory.

        Returns:
            Sorted list of dates with available data
        """
        dates = set()

        # Check both compressed and uncompressed
        for pattern in ["*.csv.zst", "*.csv"]:
            for path in self.data_dir.glob(pattern):
                # Extract date from filename (e.g., "2024-01-03.csv.zst")
                name = path.name.replace(".csv.zst", "").replace(".csv", "")
                try:
                    d = datetime.strptime(name, "%Y-%m-%d").date()
                    dates.add(d)
                except ValueError:
                    # Try alternate format: YYYYMMDD
                    try:
                        d = datetime.strptime(name, "%Y%m%d").date()
                        dates.add(d)
                    except ValueError:
                        logger.debug(f"Could not parse date from filename: {path.name}")

        return sorted(dates)

    def load_day(self, target_date: date) -> pd.DataFrame:
        """Load a single day's data.

        Args:
            target_date: The date to load

        Returns:
            DataFrame with CBBO data for that day

        Raises:
            FileNotFoundError: If no data file exists for the date
        """
        # Try multiple filename formats
        date_formats = [
            target_date.strftime("%Y-%m-%d"),
            target_date.strftime("%Y%m%d"),
        ]

        for date_str in date_formats:
            # Try compressed first (more common)
            zst_path = self.data_dir / f"{date_str}.csv.zst"
            if zst_path.exists():
                return self._load_zst(zst_path)

            csv_path = self.data_dir / f"{date_str}.csv"
            if csv_path.exists():
                return self._load_csv(csv_path)

        raise FileNotFoundError(
            f"No data file for {target_date}. "
            f"Checked: {date_formats[0]}.csv.zst, {date_formats[0]}.csv"
        )

    def _load_zst(self, path: Path) -> pd.DataFrame:
        """Load a zstd-compressed CSV file."""
        if not self._zstd_available:
            raise ImportError(
                "zstandard library required for .csv.zst files. "
                "Install with: pip install zstandard"
            )

        import zstandard as zstd

        logger.debug(f"Loading compressed file: {path}")

        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            # Stream decompress for memory efficiency
            with dctx.stream_reader(f) as reader:
                # Read in chunks for large files
                df = pd.read_csv(
                    reader,
                    parse_dates=["ts_event", "ts_recv"],
                )

        self._validate_columns(df, path)
        logger.info(f"Loaded {len(df):,} rows from {path.name}")
        return df

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load an uncompressed CSV file."""
        logger.debug(f"Loading CSV file: {path}")

        df = pd.read_csv(
            path,
            parse_dates=["ts_event", "ts_recv"],
        )

        self._validate_columns(df, path)
        logger.info(f"Loaded {len(df):,} rows from {path.name}")
        return df

    def _validate_columns(self, df: pd.DataFrame, path: Path) -> None:
        """Validate that DataFrame has required columns."""
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {path.name}: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        skip_missing: bool = True,
    ) -> Iterator[tuple[date, pd.DataFrame]]:
        """Iterate through days in a date range.

        Args:
            start_date: First date to load
            end_date: Last date to load (inclusive)
            skip_missing: If True, skip days with no data (weekends/holidays)

        Yields:
            Tuple of (date, DataFrame) for each available day
        """
        current = start_date
        loaded_count = 0
        skipped_count = 0

        while current <= end_date:
            try:
                df = self.load_day(current)
                loaded_count += 1
                yield current, df
            except FileNotFoundError:
                skipped_count += 1
                if not skip_missing:
                    raise
                # Skip weekends/holidays silently
                if current.weekday() < 5:  # Weekday
                    logger.debug(f"No data for {current} (holiday?)")

            current += timedelta(days=1)

        logger.info(
            f"Date range complete: {loaded_count} days loaded, {skipped_count} skipped"
        )

    def filter_for_scalping(
        self,
        df: pd.DataFrame,
        current_date: date,
        max_dte: int | None = None,
        min_bid: float | None = None,
        max_spread_pct: float | None = None,
    ) -> pd.DataFrame:
        """Filter options data for scalping candidates.

        Adds parsed option fields and filters by DTE, bid price, and spread.

        Args:
            df: Raw CBBO data from load_day()
            current_date: Current simulation date (for DTE calculation)
            max_dte: Maximum days to expiration (default: config value)
            min_bid: Minimum bid price (default: config value)
            max_spread_pct: Maximum spread as % of mid (default: config value)

        Returns:
            Filtered DataFrame with additional columns:
            - underlying, expiry, right, strike (parsed from symbol)
            - dte (days to expiration)
            - mid_px, spread, spread_pct (calculated)
        """
        # Use config defaults if not specified
        max_dte = max_dte if max_dte is not None else self.config.max_dte
        min_bid = min_bid if min_bid is not None else self.config.min_bid
        max_spread_pct = max_spread_pct if max_spread_pct is not None else self.config.max_spread_pct

        df = df.copy()

        # Parse OCC symbols using existing parser
        logger.debug("Parsing OCC symbols...")
        parsed = df["symbol"].apply(self._safe_parse_occ)

        # Extract fields from parsed results
        df["underlying"] = parsed.apply(lambda x: x.underlying if x else None)
        df["expiry"] = parsed.apply(lambda x: x.expiry if x else None)
        df["right"] = parsed.apply(lambda x: x.right if x else None)
        df["strike"] = parsed.apply(lambda x: x.strike if x else None)

        # Calculate DTE
        df["dte"] = df["expiry"].apply(
            lambda x: (datetime.strptime(x, "%Y-%m-%d").date() - current_date).days
            if x else None
        )

        # Calculate mid price and spread
        df["mid_px"] = (df["bid_px"] + df["ask_px"]) / 2
        df["spread"] = df["ask_px"] - df["bid_px"]
        df["spread_pct"] = (df["spread"] / df["mid_px"] * 100).fillna(100)

        # Apply filters
        initial_count = len(df)
        mask = (
            df["underlying"].notna()  # Successfully parsed
            & (df["dte"] <= max_dte)
            & (df["dte"] >= 0)  # Not expired
            & (df["bid_px"] >= min_bid)
            & (df["spread_pct"] <= max_spread_pct)
        )

        filtered = df[mask].copy()
        logger.info(
            f"Filtered {initial_count:,} → {len(filtered):,} rows "
            f"(DTE≤{max_dte}, bid≥${min_bid}, spread≤{max_spread_pct}%)"
        )

        return filtered

    def _safe_parse_occ(self, symbol: str) -> CanonicalOptionId | None:
        """Parse OCC symbol, returning None on failure."""
        try:
            return parse_occ(symbol)
        except Exception as e:
            logger.debug(f"Failed to parse symbol '{symbol}': {e}")
            return None

    def get_minute_bars(
        self,
        df: pd.DataFrame,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate CBBO data into minute bars for a specific contract.

        Args:
            df: CBBO data (raw or filtered)
            symbol: Specific OCC symbol to filter (optional)

        Returns:
            DataFrame with OHLC-style minute bars
        """
        if symbol:
            df = df[df["symbol"] == symbol].copy()

        if df.empty:
            return pd.DataFrame()

        # Group by minute
        df["minute"] = df["ts_event"].dt.floor("1min")

        bars = df.groupby(["symbol", "minute"]).agg(
            open_bid=("bid_px", "first"),
            high_bid=("bid_px", "max"),
            low_bid=("bid_px", "min"),
            close_bid=("bid_px", "last"),
            open_ask=("ask_px", "first"),
            high_ask=("ask_px", "max"),
            low_ask=("ask_px", "min"),
            close_ask=("ask_px", "last"),
            avg_bid_sz=("bid_sz", "mean"),
            avg_ask_sz=("ask_sz", "mean"),
            records=("bid_px", "count"),
        ).reset_index()

        # Calculate mid prices
        bars["open_mid"] = (bars["open_bid"] + bars["open_ask"]) / 2
        bars["close_mid"] = (bars["close_bid"] + bars["close_ask"]) / 2
        bars["high_mid"] = (bars["high_bid"] + bars["high_ask"]) / 2
        bars["low_mid"] = (bars["low_bid"] + bars["low_ask"]) / 2

        return bars

    def build_quote_timeline(
        self,
        df: pd.DataFrame,
        symbols: list[str] | None = None,
    ) -> Iterator[dict]:
        """Generate quote events in chronological order for replay.

        This is used by the backtester to simulate real-time quote arrival.

        Args:
            df: CBBO data
            symbols: Optional list of symbols to include

        Yields:
            Quote dicts in timestamp order, suitable for replay
        """
        if symbols:
            df = df[df["symbol"].isin(symbols)]

        # Sort by timestamp
        df = df.sort_values("ts_event")

        for _, row in df.iterrows():
            yield {
                "ts_event": row["ts_event"],
                "ts_recv": row["ts_recv"],
                "symbol": row["symbol"],
                "bid": row["bid_px"],
                "ask": row["ask_px"],
                "bid_size": row["bid_sz"],
                "ask_size": row["ask_sz"],
                "mid": (row["bid_px"] + row["ask_px"]) / 2,
            }


def get_dte(expiry: date | str, current_date: date) -> int:
    """Calculate days to expiration.

    Args:
        expiry: Expiration date (date object or ISO string)
        current_date: Current date

    Returns:
        Days until expiration (negative if expired)
    """
    if isinstance(expiry, str):
        expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
    return (expiry - current_date).days
