# Scalping Module Implementation Guide

## Overview

This document provides complete implementation specifications for adding a momentum-based options scalping module to the OptionsRadar system. The scalping module runs **parallel** to the existing regime-based system, targeting rapid intraday moves in TSLA options.

**Key Distinction:**
- **Regime System**: Multi-day sentiment-based signals, 30 DTE options, ~1 trade/week
- **Scalping System**: Intraday momentum signals, 0DTE/1DTE options, multiple trades/day

---

## CRITICAL: Integration with Existing Codebase

**BEFORE implementing any component, Claude Code MUST inspect the existing codebase to avoid duplication and ensure consistency.**

### Existing Structure to Preserve

```
backend/
├── models/          # Data models (canonical IDs, quotes, Greeks)
├── data/            # Data clients (Alpaca, ORATS, Finnhub, Quiver)
│   ├── alpaca_*.py  # Alpaca streaming + account
│   ├── orats_*.py   # ORATS Greeks
│   ├── finnhub_*.py # News sentiment
│   ├── quiver_*.py  # WSB sentiment
│   └── sentiment_aggregator.py
├── engine/          # Core logic
│   ├── gates.py     # 11 trading gates (8 hard, 3 soft)
│   ├── pipeline.py  # Gate orchestration
│   ├── recommender.py
│   ├── scanner.py   # Daily opportunity scanner
│   ├── session_tracker.py
│   └── position_tracker.py
├── websocket/       # WebSocket management
└── logging/         # Shadow mode logging
```

### Pre-Implementation Checklist

Before creating ANY new file, Claude Code must:

1. **Check `backend/models/`** for existing OCC symbol parsing
   - Run: `grep -r "parse" backend/models/` and `grep -r "OCC" backend/models/`
   - If OCC parsing exists, import and use it instead of creating `occ_parser.py`

2. **Check `backend/engine/gates.py`** for reusable gates
   - The scalping module should IMPORT and REUSE these gates:
     - Liquidity gate (OI, volume checks)
     - Spread gate (bid-ask spread checks)
     - Freshness gate (data staleness checks)
   - Run: `cat backend/engine/gates.py` to see available gates

3. **Check `backend/engine/position_tracker.py`** for extension points
   - Do NOT create a separate scalp position tracker
   - EXTEND the existing PositionTracker with a `position_type` field ('regime' | 'scalp')
   - Run: `cat backend/engine/position_tracker.py` to understand the interface

4. **Check `backend/data/` for data client patterns**
   - New data clients (DataBentoLoader) should follow the same patterns
   - Look at `alpaca_*.py` and `orats_*.py` for conventions (logging, error handling, caching)
   - Run: `ls -la backend/data/` and review one existing client

5. **Check `run.py`** for startup patterns
   - Understand how the existing system starts before adding mode flags
   - Run: `cat run.py`

### Integration Points Summary

| New Component | Integrate With | Action |
|---------------|----------------|--------|
| OCC symbol parsing | `backend/models/` | **Check first** — likely exists |
| Liquidity/spread checks | `backend/engine/gates.py` | **Import existing gates** |
| Scalp position tracking | `backend/engine/position_tracker.py` | **Extend, don't duplicate** |
| DataBento loader | `backend/data/` | **Follow existing patterns** |
| Scalp signal generator | `backend/engine/scanner.py` | **Separate but parallel** |
| WebSocket broadcasts | `backend/websocket/` | **Use existing manager** |

### Code Style Requirements

Match the existing codebase style:
- Review existing files for import ordering
- Match logging patterns (check `backend/logging/`)
- Follow existing docstring format
- Use same type hint conventions
- Match error handling patterns

---

## Part 1: DataBento Data Integration

### 1.1 Data Specifications

The system has access to historical TSLA options data from DataBento:

| Field | Value |
|-------|-------|
| Dataset | OPRA (TSLA.OPT) |
| Schema | CBBO-1m (Consolidated Best Bid/Offer, 1-minute) |
| Date Range | 2022-01-01 to 2025-12-31 |
| Format | CSV, zstd compressed |
| Files | ~1007 daily files |
| Total Size | ~135 GB |
| Price Format | Decimal (e.g., 1.25) |
| Timestamp Format | ISO 8601 UTC (e.g., 2023-02-01T01:02:03.123456789Z) |

### 1.2 Expected CBBO-1m Schema

Each row represents the best bid/offer for an option contract at a 1-minute interval:

```
ts_recv,ts_event,symbol,bid_px,ask_px,bid_sz,ask_sz,bid_ct,ask_ct
2024-01-03T14:30:00.000000000Z,2024-01-03T14:30:00.000000000Z,TSLA240105C00250000,1.25,1.35,50,75,3,5
```

| Column | Description |
|--------|-------------|
| ts_recv | Timestamp when exchange received |
| ts_event | Timestamp of the event |
| symbol | OCC option symbol (e.g., TSLA240105C00250000) |
| bid_px | Best bid price |
| ask_px | Best ask price |
| bid_sz | Bid size (contracts) |
| ask_sz | Ask size (contracts) |
| bid_ct | Number of bid orders |
| ask_ct | Number of ask orders |

### 1.3 OCC Symbol Parsing

**FIRST:** Check if OCC parsing already exists:
```bash
grep -r "parse" backend/models/
grep -r "OCC" backend/models/
grep -r "strike" backend/models/
cat backend/models/*.py | head -200
```

**If parsing exists:** Import and use the existing implementation. Skip creating `occ_parser.py`.

**If parsing does NOT exist:** Create the following:

The OCC option symbol format: `TSLA240105C00250000`

```python
# backend/utils/occ_parser.py

from dataclasses import dataclass
from datetime import date

@dataclass
class ParsedOption:
    underlying: str      # "TSLA"
    expiry: date         # 2024-01-05
    option_type: str     # "C" or "P"
    strike: float        # 250.00

def parse_occ_symbol(symbol: str) -> ParsedOption:
    """Parse OCC option symbol into components.
    
    Format: TSLA240105C00250000
            ^^^^---------- underlying (variable length, find where digits start)
                ^^^^^^---- expiry YYMMDD
                      ^--- option type C/P
                       ^^^^^^^^- strike price * 1000
    """
    # Find where the date starts (first digit after letters)
    i = 0
    while i < len(symbol) and not symbol[i].isdigit():
        i += 1
    
    underlying = symbol[:i]
    expiry_str = symbol[i:i+6]  # YYMMDD
    option_type = symbol[i+6]   # C or P
    strike_str = symbol[i+7:]   # Strike * 1000
    
    expiry = date(
        year=2000 + int(expiry_str[:2]),
        month=int(expiry_str[2:4]),
        day=int(expiry_str[4:6])
    )
    strike = int(strike_str) / 1000
    
    return ParsedOption(
        underlying=underlying,
        expiry=expiry,
        option_type=option_type,
        strike=strike
    )

def get_dte(option: ParsedOption, current_date: date) -> int:
    """Calculate days to expiration."""
    return (option.expiry - current_date).days
```

### 1.4 DataBento Data Loader

```python
# backend/data/databento_loader.py

import zstandard as zstd
import pandas as pd
from pathlib import Path
from datetime import date, datetime
from typing import Iterator
import io

class DataBentoLoader:
    """Load and process DataBento CBBO-1m options data."""
    
    def __init__(self, data_dir: str | Path):
        """Initialize with path to DataBento data directory.
        
        Args:
            data_dir: Directory containing daily .csv.zst files
        """
        self.data_dir = Path(data_dir)
        self._validate_data_dir()
    
    def _validate_data_dir(self) -> None:
        """Verify data directory exists and contains expected files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        zst_files = list(self.data_dir.glob("*.csv.zst"))
        if not zst_files:
            # Also check for uncompressed
            csv_files = list(self.data_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No data files found in {self.data_dir}")
    
    def load_day(self, target_date: date) -> pd.DataFrame:
        """Load a single day's data.
        
        Args:
            target_date: The date to load
            
        Returns:
            DataFrame with CBBO data for that day
        """
        # Try compressed first, then uncompressed
        date_str = target_date.strftime("%Y-%m-%d")
        zst_path = self.data_dir / f"{date_str}.csv.zst"
        csv_path = self.data_dir / f"{date_str}.csv"
        
        if zst_path.exists():
            return self._load_zst(zst_path)
        elif csv_path.exists():
            return pd.read_csv(csv_path, parse_dates=['ts_event'])
        else:
            raise FileNotFoundError(f"No data file for {date_str}")
    
    def _load_zst(self, path: Path) -> pd.DataFrame:
        """Load a zstd-compressed CSV file."""
        dctx = zstd.ZstdDecompressor()
        with open(path, 'rb') as f:
            decompressed = dctx.decompress(f.read())
        return pd.read_csv(
            io.BytesIO(decompressed),
            parse_dates=['ts_event']
        )
    
    def load_date_range(
        self, 
        start_date: date, 
        end_date: date
    ) -> Iterator[tuple[date, pd.DataFrame]]:
        """Iterate through days in a date range.
        
        Yields:
            Tuple of (date, DataFrame) for each available day
        """
        current = start_date
        while current <= end_date:
            try:
                df = self.load_day(current)
                yield current, df
            except FileNotFoundError:
                # Skip weekends/holidays with no data
                pass
            current += timedelta(days=1)
    
    def filter_for_scalping(
        self,
        df: pd.DataFrame,
        current_date: date,
        max_dte: int = 1,
        min_bid: float = 0.05,
        max_spread_pct: float = 10.0
    ) -> pd.DataFrame:
        """Filter options data for scalping candidates.
        
        Args:
            df: Raw CBBO data
            current_date: Current simulation date
            max_dte: Maximum days to expiration (0 = 0DTE only, 1 = include 1DTE)
            min_bid: Minimum bid price (filter out pennies)
            max_spread_pct: Maximum bid-ask spread as percentage
            
        Returns:
            Filtered DataFrame with additional columns
        """
        # Parse all symbols
        df = df.copy()
        
        parsed = df['symbol'].apply(parse_occ_symbol)
        df['underlying'] = parsed.apply(lambda x: x.underlying)
        df['expiry'] = parsed.apply(lambda x: x.expiry)
        df['option_type'] = parsed.apply(lambda x: x.option_type)
        df['strike'] = parsed.apply(lambda x: x.strike)
        df['dte'] = parsed.apply(lambda x: get_dte(x, current_date))
        
        # Calculate mid price and spread
        df['mid_px'] = (df['bid_px'] + df['ask_px']) / 2
        df['spread'] = df['ask_px'] - df['bid_px']
        df['spread_pct'] = (df['spread'] / df['mid_px'] * 100).fillna(100)
        
        # Apply filters
        mask = (
            (df['dte'] <= max_dte) &
            (df['dte'] >= 0) &
            (df['bid_px'] >= min_bid) &
            (df['spread_pct'] <= max_spread_pct)
        )
        
        return df[mask].copy()
```

---

## Part 2: Core Scalping Components

### 2.1 Price Velocity Tracker

Tracks underlying stock price velocity to detect rapid moves.

```python
# backend/scalping/velocity_tracker.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from typing import Literal
import statistics

@dataclass
class PricePoint:
    price: float
    timestamp: datetime
    volume: int = 0

@dataclass
class VelocityReading:
    """Result of velocity calculation."""
    change_pct: float           # Percentage change over window
    change_dollars: float       # Dollar change over window
    direction: Literal['up', 'down', 'flat']
    speed: float               # Change per second (absolute)
    window_seconds: int        # Window used for calculation
    data_points: int           # Number of price points in window
    
    @property
    def is_significant(self) -> bool:
        """Check if velocity is significant for trading."""
        return abs(self.change_pct) >= 0.3 and self.data_points >= 3

@dataclass
class SpikeSignal:
    """Detected price spike."""
    direction: Literal['up', 'down']
    change_pct: float
    change_dollars: float
    duration_seconds: float
    start_price: float
    end_price: float
    timestamp: datetime

class PriceVelocityTracker:
    """Tracks price velocity over multiple timeframes."""
    
    def __init__(
        self,
        symbol: str,
        max_history_seconds: int = 300,  # Keep 5 minutes of data
        windows: tuple[int, ...] = (5, 15, 30, 60)  # Windows to calculate
    ):
        self.symbol = symbol
        self.max_history_seconds = max_history_seconds
        self.windows = windows
        self._prices: deque[PricePoint] = deque()
        self._last_spike: SpikeSignal | None = None
    
    def add_price(
        self, 
        price: float, 
        timestamp: datetime,
        volume: int = 0
    ) -> None:
        """Add a new price observation."""
        self._prices.append(PricePoint(price, timestamp, volume))
        self._cleanup_old_prices(timestamp)
    
    def _cleanup_old_prices(self, current_time: datetime) -> None:
        """Remove prices older than max_history_seconds."""
        cutoff = current_time - timedelta(seconds=self.max_history_seconds)
        while self._prices and self._prices[0].timestamp < cutoff:
            self._prices.popleft()
    
    def get_velocity(self, window_seconds: int) -> VelocityReading | None:
        """Calculate price velocity over specified window.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            VelocityReading or None if insufficient data
        """
        if len(self._prices) < 2:
            return None
        
        current_time = self._prices[-1].timestamp
        cutoff = current_time - timedelta(seconds=window_seconds)
        
        # Get prices within window
        window_prices = [p for p in self._prices if p.timestamp >= cutoff]
        
        if len(window_prices) < 2:
            return None
        
        start_price = window_prices[0].price
        end_price = window_prices[-1].price
        
        change_dollars = end_price - start_price
        change_pct = (change_dollars / start_price) * 100 if start_price else 0
        
        # Calculate actual time span
        actual_seconds = (window_prices[-1].timestamp - window_prices[0].timestamp).total_seconds()
        speed = abs(change_pct / actual_seconds) if actual_seconds > 0 else 0
        
        # Determine direction
        if change_pct > 0.05:
            direction = 'up'
        elif change_pct < -0.05:
            direction = 'down'
        else:
            direction = 'flat'
        
        return VelocityReading(
            change_pct=change_pct,
            change_dollars=change_dollars,
            direction=direction,
            speed=speed,
            window_seconds=window_seconds,
            data_points=len(window_prices)
        )
    
    def get_all_velocities(self) -> dict[int, VelocityReading | None]:
        """Get velocity readings for all configured windows."""
        return {w: self.get_velocity(w) for w in self.windows}
    
    def detect_spike(
        self,
        threshold_pct: float = 0.5,
        window_seconds: int = 30,
        cooldown_seconds: float = 60
    ) -> SpikeSignal | None:
        """Detect if price spiked beyond threshold.
        
        Args:
            threshold_pct: Minimum percentage move to trigger
            window_seconds: Window to check for spike
            cooldown_seconds: Minimum time between spike signals
            
        Returns:
            SpikeSignal if spike detected, None otherwise
        """
        velocity = self.get_velocity(window_seconds)
        
        if velocity is None:
            return None
        
        # Check cooldown
        if self._last_spike:
            elapsed = (self._prices[-1].timestamp - self._last_spike.timestamp).total_seconds()
            if elapsed < cooldown_seconds:
                return None
        
        # Check threshold
        if abs(velocity.change_pct) < threshold_pct:
            return None
        
        # Get window prices for spike details
        current_time = self._prices[-1].timestamp
        cutoff = current_time - timedelta(seconds=window_seconds)
        window_prices = [p for p in self._prices if p.timestamp >= cutoff]
        
        spike = SpikeSignal(
            direction='up' if velocity.change_pct > 0 else 'down',
            change_pct=velocity.change_pct,
            change_dollars=velocity.change_dollars,
            duration_seconds=(window_prices[-1].timestamp - window_prices[0].timestamp).total_seconds(),
            start_price=window_prices[0].price,
            end_price=window_prices[-1].price,
            timestamp=current_time
        )
        
        self._last_spike = spike
        return spike
    
    @property
    def current_price(self) -> float | None:
        """Get most recent price."""
        return self._prices[-1].price if self._prices else None
    
    @property
    def price_count(self) -> int:
        """Get number of prices in history."""
        return len(self._prices)
```

### 2.2 Volume Analyzer

```python
# backend/scalping/volume_analyzer.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Literal

@dataclass
class VolumeBar:
    """Volume data for a time period."""
    timestamp: datetime
    volume: int
    trade_count: int

@dataclass
class VolumeSpike:
    """Detected volume spike."""
    contract: str
    current_volume: int
    baseline_volume: float
    ratio: float
    timestamp: datetime

@dataclass  
class SweepSignal:
    """Detected sweep order (large aggressive order)."""
    direction: Literal['call_sweep', 'put_sweep']
    contracts: int
    premium: float  # Total premium in dollars
    strikes: list[float]
    timestamp: datetime

class VolumeAnalyzer:
    """Analyzes option volume for unusual activity."""
    
    def __init__(
        self,
        baseline_window_minutes: int = 30,
        spike_ratio_threshold: float = 2.0
    ):
        self.baseline_window_minutes = baseline_window_minutes
        self.spike_ratio_threshold = spike_ratio_threshold
        
        # Volume history per contract
        self._volume_history: dict[str, deque[VolumeBar]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # Aggregate volume by option type
        self._call_volume: deque[VolumeBar] = deque(maxlen=100)
        self._put_volume: deque[VolumeBar] = deque(maxlen=100)
        
        # Baseline calculations (updated periodically)
        self._baseline_cache: dict[str, float] = {}
        self._last_baseline_update: datetime | None = None
    
    def add_volume(
        self,
        contract: str,
        volume: int,
        timestamp: datetime,
        option_type: Literal['C', 'P'] | None = None
    ) -> None:
        """Record volume observation."""
        bar = VolumeBar(timestamp=timestamp, volume=volume, trade_count=1)
        self._volume_history[contract].append(bar)
        
        # Track aggregate by type
        if option_type == 'C':
            self._call_volume.append(bar)
        elif option_type == 'P':
            self._put_volume.append(bar)
    
    def get_volume_ratio(
        self,
        contract: str,
        window_minutes: int = 5
    ) -> float:
        """Get current volume vs baseline ratio.
        
        Returns:
            Ratio where 1.0 = normal, 2.0 = 2x normal volume
        """
        history = self._volume_history.get(contract)
        if not history or len(history) < 2:
            return 1.0
        
        # Calculate recent volume
        current_time = history[-1].timestamp
        cutoff = current_time - timedelta(minutes=window_minutes)
        recent_volume = sum(
            bar.volume for bar in history 
            if bar.timestamp >= cutoff
        )
        
        # Calculate baseline
        baseline = self._calculate_baseline(contract, current_time)
        
        if baseline <= 0:
            return 1.0
        
        # Scale to same time period
        baseline_scaled = baseline * window_minutes
        
        return recent_volume / baseline_scaled if baseline_scaled > 0 else 1.0
    
    def _calculate_baseline(
        self,
        contract: str,
        current_time: datetime
    ) -> float:
        """Calculate baseline volume per minute for contract."""
        history = self._volume_history.get(contract)
        if not history:
            return 0.0
        
        cutoff = current_time - timedelta(minutes=self.baseline_window_minutes)
        baseline_bars = [bar for bar in history if bar.timestamp >= cutoff]
        
        if len(baseline_bars) < 2:
            return 0.0
        
        total_volume = sum(bar.volume for bar in baseline_bars)
        time_span = (baseline_bars[-1].timestamp - baseline_bars[0].timestamp).total_seconds() / 60
        
        return total_volume / time_span if time_span > 0 else 0.0
    
    def detect_volume_spike(self, contract: str) -> VolumeSpike | None:
        """Detect if contract has unusual volume."""
        ratio = self.get_volume_ratio(contract)
        
        if ratio < self.spike_ratio_threshold:
            return None
        
        history = self._volume_history.get(contract)
        if not history:
            return None
        
        return VolumeSpike(
            contract=contract,
            current_volume=history[-1].volume,
            baseline_volume=self._calculate_baseline(contract, history[-1].timestamp),
            ratio=ratio,
            timestamp=history[-1].timestamp
        )
    
    def get_put_call_ratio(self, window_minutes: int = 5) -> float:
        """Get put/call volume ratio.
        
        Returns:
            Ratio where >1 means more put volume, <1 means more call volume
        """
        if not self._call_volume or not self._put_volume:
            return 1.0
        
        current_time = max(
            self._call_volume[-1].timestamp if self._call_volume else datetime.min,
            self._put_volume[-1].timestamp if self._put_volume else datetime.min
        )
        cutoff = current_time - timedelta(minutes=window_minutes)
        
        call_vol = sum(bar.volume for bar in self._call_volume if bar.timestamp >= cutoff)
        put_vol = sum(bar.volume for bar in self._put_volume if bar.timestamp >= cutoff)
        
        return put_vol / call_vol if call_vol > 0 else 1.0
```

### 2.3 Technical Scalper

```python
# backend/scalping/technical_scalper.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal
import statistics

@dataclass
class VWAPState:
    """VWAP calculation state."""
    cumulative_pv: float = 0.0  # Price * Volume
    cumulative_volume: int = 0
    vwap: float = 0.0
    upper_band: float = 0.0  # +1 std dev
    lower_band: float = 0.0  # -1 std dev
    squared_deviations: list[float] = field(default_factory=list)

@dataclass
class SupportResistance:
    """Support/resistance level."""
    price: float
    strength: int  # Number of touches
    level_type: Literal['support', 'resistance']
    last_touch: datetime

@dataclass
class ScalpTechnicalSignal:
    """Technical signal for scalping."""
    signal_type: Literal['vwap_bounce', 'vwap_rejection', 'breakout', 'momentum_burst']
    direction: Literal['bullish', 'bearish']
    price: float
    confidence: int  # 0-100
    timestamp: datetime
    metadata: dict = field(default_factory=dict)

class TechnicalScalper:
    """Fast technical analysis for scalping decisions."""
    
    def __init__(
        self,
        symbol: str,
        vwap_band_std: float = 1.0,
        sr_threshold_pct: float = 0.3,  # Price must be within 0.3% to "touch" level
        sr_lookback_minutes: int = 60
    ):
        self.symbol = symbol
        self.vwap_band_std = vwap_band_std
        self.sr_threshold_pct = sr_threshold_pct
        self.sr_lookback_minutes = sr_lookback_minutes
        
        # VWAP state
        self._vwap = VWAPState()
        
        # Price history for S/R calculation
        self._price_history: list[tuple[float, datetime]] = []
        
        # Detected levels
        self._support_levels: list[SupportResistance] = []
        self._resistance_levels: list[SupportResistance] = []
        
        # Session tracking
        self._session_high: float = 0.0
        self._session_low: float = float('inf')
        self._session_open: float | None = None
    
    def reset_session(self) -> None:
        """Reset for new trading session."""
        self._vwap = VWAPState()
        self._price_history.clear()
        self._support_levels.clear()
        self._resistance_levels.clear()
        self._session_high = 0.0
        self._session_low = float('inf')
        self._session_open = None
    
    def update(
        self,
        price: float,
        volume: int,
        timestamp: datetime
    ) -> None:
        """Update all technical indicators with new tick."""
        # Track session stats
        if self._session_open is None:
            self._session_open = price
        self._session_high = max(self._session_high, price)
        self._session_low = min(self._session_low, price)
        
        # Update VWAP
        self._update_vwap(price, volume)
        
        # Store price history
        self._price_history.append((price, timestamp))
        self._cleanup_old_prices(timestamp)
        
        # Periodically update S/R levels
        if len(self._price_history) % 10 == 0:  # Every 10 ticks
            self._update_sr_levels(timestamp)
    
    def _update_vwap(self, price: float, volume: int) -> None:
        """Update VWAP calculation."""
        if volume <= 0:
            return
        
        self._vwap.cumulative_pv += price * volume
        self._vwap.cumulative_volume += volume
        
        if self._vwap.cumulative_volume > 0:
            self._vwap.vwap = self._vwap.cumulative_pv / self._vwap.cumulative_volume
            
            # Track squared deviation for std calculation
            deviation = price - self._vwap.vwap
            self._vwap.squared_deviations.append(deviation ** 2)
            
            # Keep last 100 for rolling std
            if len(self._vwap.squared_deviations) > 100:
                self._vwap.squared_deviations = self._vwap.squared_deviations[-100:]
            
            # Calculate bands
            if len(self._vwap.squared_deviations) >= 2:
                std = statistics.stdev(
                    [d ** 0.5 for d in self._vwap.squared_deviations]
                )
                self._vwap.upper_band = self._vwap.vwap + (std * self.vwap_band_std)
                self._vwap.lower_band = self._vwap.vwap - (std * self.vwap_band_std)
    
    def _cleanup_old_prices(self, current_time: datetime) -> None:
        """Remove prices older than lookback period."""
        cutoff = current_time - timedelta(minutes=self.sr_lookback_minutes)
        self._price_history = [
            (p, t) for p, t in self._price_history if t >= cutoff
        ]
    
    def _update_sr_levels(self, current_time: datetime) -> None:
        """Update support/resistance levels from price history."""
        if len(self._price_history) < 20:
            return
        
        prices = [p for p, _ in self._price_history]
        
        # Find local minima (support) and maxima (resistance)
        for i in range(2, len(prices) - 2):
            # Local minimum
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
               prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                self._add_or_update_level(
                    prices[i], 'support', current_time
                )
            
            # Local maximum
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                self._add_or_update_level(
                    prices[i], 'resistance', current_time
                )
    
    def _add_or_update_level(
        self,
        price: float,
        level_type: Literal['support', 'resistance'],
        timestamp: datetime
    ) -> None:
        """Add new S/R level or strengthen existing one."""
        levels = self._support_levels if level_type == 'support' else self._resistance_levels
        threshold = price * (self.sr_threshold_pct / 100)
        
        # Check if near existing level
        for level in levels:
            if abs(level.price - price) <= threshold:
                level.strength += 1
                level.last_touch = timestamp
                return
        
        # Add new level
        levels.append(SupportResistance(
            price=price,
            strength=1,
            level_type=level_type,
            last_touch=timestamp
        ))
        
        # Keep only strongest levels
        if level_type == 'support':
            self._support_levels = sorted(
                self._support_levels, key=lambda x: -x.strength
            )[:5]
        else:
            self._resistance_levels = sorted(
                self._resistance_levels, key=lambda x: -x.strength
            )[:5]
    
    def check_vwap_signal(
        self,
        current_price: float,
        velocity_pct: float
    ) -> ScalpTechnicalSignal | None:
        """Check for VWAP-based signals.
        
        - VWAP Bounce: Price touches VWAP from above/below and reverses
        - VWAP Rejection: Price fails to break through VWAP
        """
        if self._vwap.vwap <= 0:
            return None
        
        vwap = self._vwap.vwap
        distance_to_vwap_pct = ((current_price - vwap) / vwap) * 100
        
        # Near VWAP (within 0.2%)
        if abs(distance_to_vwap_pct) > 0.2:
            return None
        
        timestamp = datetime.now()
        
        # Bullish bounce: Price at VWAP, velocity turning up
        if distance_to_vwap_pct <= 0.1 and velocity_pct > 0.1:
            return ScalpTechnicalSignal(
                signal_type='vwap_bounce',
                direction='bullish',
                price=current_price,
                confidence=70,
                timestamp=timestamp,
                metadata={'vwap': vwap, 'distance_pct': distance_to_vwap_pct}
            )
        
        # Bearish rejection: Price at VWAP, velocity turning down
        if distance_to_vwap_pct >= -0.1 and velocity_pct < -0.1:
            return ScalpTechnicalSignal(
                signal_type='vwap_rejection',
                direction='bearish',
                price=current_price,
                confidence=70,
                timestamp=timestamp,
                metadata={'vwap': vwap, 'distance_pct': distance_to_vwap_pct}
            )
        
        return None
    
    def check_breakout(
        self,
        current_price: float,
        velocity_pct: float,
        volume_ratio: float
    ) -> ScalpTechnicalSignal | None:
        """Check for support/resistance breakout."""
        timestamp = datetime.now()
        
        # Check resistance breakout
        for level in self._resistance_levels:
            if level.strength < 2:
                continue
            
            # Price breaking above resistance with momentum
            if current_price > level.price and \
               velocity_pct > 0.2 and \
               volume_ratio > 1.5:
                return ScalpTechnicalSignal(
                    signal_type='breakout',
                    direction='bullish',
                    price=current_price,
                    confidence=60 + min(level.strength * 5, 20),
                    timestamp=timestamp,
                    metadata={
                        'level': level.price,
                        'level_strength': level.strength,
                        'volume_ratio': volume_ratio
                    }
                )
        
        # Check support breakdown
        for level in self._support_levels:
            if level.strength < 2:
                continue
            
            # Price breaking below support with momentum
            if current_price < level.price and \
               velocity_pct < -0.2 and \
               volume_ratio > 1.5:
                return ScalpTechnicalSignal(
                    signal_type='breakout',
                    direction='bearish',
                    price=current_price,
                    confidence=60 + min(level.strength * 5, 20),
                    timestamp=timestamp,
                    metadata={
                        'level': level.price,
                        'level_strength': level.strength,
                        'volume_ratio': volume_ratio
                    }
                )
        
        return None
    
    @property
    def vwap(self) -> float:
        return self._vwap.vwap
    
    @property
    def vwap_upper(self) -> float:
        return self._vwap.upper_band
    
    @property
    def vwap_lower(self) -> float:
        return self._vwap.lower_band
```

### 2.4 Scalp Signal Generator

**IMPORTANT:** Before implementing, check existing gates in `backend/engine/gates.py`:
```bash
cat backend/engine/gates.py
```

Reuse existing gate classes for:
- **Liquidity checks** (OI, volume) — likely `LiquidityGate` or similar
- **Spread checks** — likely `SpreadGate` or similar
- **Freshness checks** — likely `FreshnessGate` or similar

Import and use them instead of reimplementing:
```python
from backend.engine.gates import LiquidityGate, SpreadGate  # Adjust names based on actual implementation
```

```python
# backend/scalping/signal_generator.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal
import uuid

from .velocity_tracker import PriceVelocityTracker, SpikeSignal
from .volume_analyzer import VolumeAnalyzer
from .technical_scalper import TechnicalScalper, ScalpTechnicalSignal

@dataclass
class ScalpSignal:
    """A complete scalping trade signal."""
    
    # Identification
    id: str
    timestamp: datetime
    symbol: str
    
    # Signal classification
    signal_type: Literal['SCALP_CALL', 'SCALP_PUT']
    trigger: str  # 'momentum_burst', 'vwap_bounce', 'vwap_rejection', 'breakout'
    
    # Underlying state
    underlying_price: float
    velocity_pct: float
    volume_ratio: float
    
    # Option selection
    option_symbol: str
    strike: float
    expiry: str
    delta: float
    dte: int
    
    # Entry
    bid_price: float
    ask_price: float
    entry_price: float  # Expected fill (usually ask for buy)
    spread_pct: float
    
    # Risk management
    take_profit_pct: float = 30.0
    stop_loss_pct: float = 15.0
    max_hold_minutes: int = 15
    
    # Sizing
    confidence: int = 50  # 0-100
    suggested_contracts: int = 1
    max_position_value: float = 500.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'trigger': self.trigger,
            'underlying_price': self.underlying_price,
            'velocity_pct': self.velocity_pct,
            'volume_ratio': self.volume_ratio,
            'option_symbol': self.option_symbol,
            'strike': self.strike,
            'expiry': self.expiry,
            'delta': self.delta,
            'dte': self.dte,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'entry_price': self.entry_price,
            'spread_pct': self.spread_pct,
            'take_profit_pct': self.take_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'max_hold_minutes': self.max_hold_minutes,
            'confidence': self.confidence,
            'suggested_contracts': self.suggested_contracts,
        }

@dataclass
class ScalpConfig:
    """Configuration for scalping module."""
    
    # Enable/disable
    enabled: bool = False
    
    # Evaluation interval (in seconds for backtest, ms for live)
    eval_interval_seconds: float = 1.0  # Check every second in backtest
    
    # Momentum thresholds
    momentum_threshold_pct: float = 0.5  # 0.5% move triggers signal
    momentum_window_seconds: int = 30    # Over 30 second window
    
    # Volume thresholds
    volume_spike_ratio: float = 1.5  # 1.5x normal volume required
    
    # Option selection
    target_delta: float = 0.35
    delta_tolerance: float = 0.10  # Accept 0.25-0.45 delta
    max_spread_pct: float = 8.0    # Max 8% spread
    min_open_interest: int = 100   # Minimum OI
    max_dte: int = 1               # 0DTE or 1DTE only
    prefer_0dte: bool = True
    
    # Risk management  
    take_profit_pct: float = 30.0
    stop_loss_pct: float = 15.0
    max_hold_minutes: int = 15
    
    # Position limits
    max_daily_scalps: int = 10
    max_concurrent_scalps: int = 1
    scalp_position_size_pct: float = 5.0  # % of portfolio
    max_contract_price: float = 5.00      # Don't buy options > $5
    
    # Cooldowns
    min_signal_interval_seconds: float = 60.0   # 1 min between signals
    cooldown_after_loss_seconds: float = 300.0  # 5 min after loss

class ScalpSignalGenerator:
    """Generates scalping signals by combining velocity, volume, and technicals."""
    
    def __init__(
        self,
        symbol: str,
        config: ScalpConfig,
        velocity_tracker: PriceVelocityTracker,
        volume_analyzer: VolumeAnalyzer,
        technical_scalper: TechnicalScalper
    ):
        self.symbol = symbol
        self.config = config
        self.velocity = velocity_tracker
        self.volume = volume_analyzer
        self.technical = technical_scalper
        
        # State tracking
        self._last_signal_time: datetime | None = None
        self._daily_signal_count: int = 0
        self._last_reset_date: datetime | None = None
        self._in_cooldown_until: datetime | None = None
        
        # Available options (updated externally)
        self._available_options: list[dict] = []
    
    def update_available_options(self, options: list[dict]) -> None:
        """Update list of available options for selection.
        
        Args:
            options: List of option dicts with keys:
                - symbol: OCC symbol
                - strike: Strike price
                - expiry: Expiration date string
                - option_type: 'C' or 'P'
                - delta: Option delta
                - bid_px: Bid price
                - ask_px: Ask price
                - dte: Days to expiration
        """
        self._available_options = options
    
    def evaluate(
        self,
        current_time: datetime,
        underlying_price: float
    ) -> ScalpSignal | None:
        """Run scalping evaluation.
        
        Should be called at regular intervals (e.g., every second in backtest).
        
        Args:
            current_time: Current timestamp
            underlying_price: Current underlying stock price
            
        Returns:
            ScalpSignal if conditions met, None otherwise
        """
        if not self.config.enabled:
            return None
        
        # Reset daily counter if new day
        self._check_daily_reset(current_time)
        
        # Check limits
        if self._daily_signal_count >= self.config.max_daily_scalps:
            return None
        
        # Check cooldowns
        if not self._check_cooldowns(current_time):
            return None
        
        # Get current velocity
        velocity = self.velocity.get_velocity(self.config.momentum_window_seconds)
        if velocity is None:
            return None
        
        # Get volume ratio (for any contract, use aggregate)
        volume_ratio = 1.0  # Default if no volume data
        
        # Check for signals
        signal = self._check_momentum_burst(
            current_time, underlying_price, velocity.change_pct, volume_ratio
        )
        
        if signal is None:
            signal = self._check_technical_signals(
                current_time, underlying_price, velocity.change_pct, volume_ratio
            )
        
        if signal:
            self._last_signal_time = current_time
            self._daily_signal_count += 1
        
        return signal
    
    def _check_daily_reset(self, current_time: datetime) -> None:
        """Reset daily counter if new trading day."""
        if self._last_reset_date is None or \
           current_time.date() > self._last_reset_date.date():
            self._daily_signal_count = 0
            self._last_reset_date = current_time
    
    def _check_cooldowns(self, current_time: datetime) -> bool:
        """Check if we're in any cooldown period."""
        # Signal interval cooldown
        if self._last_signal_time:
            elapsed = (current_time - self._last_signal_time).total_seconds()
            if elapsed < self.config.min_signal_interval_seconds:
                return False
        
        # Loss cooldown
        if self._in_cooldown_until and current_time < self._in_cooldown_until:
            return False
        
        return True
    
    def trigger_loss_cooldown(self, current_time: datetime) -> None:
        """Trigger cooldown after a losing trade."""
        self._in_cooldown_until = current_time + timedelta(
            seconds=self.config.cooldown_after_loss_seconds
        )
    
    def _check_momentum_burst(
        self,
        current_time: datetime,
        underlying_price: float,
        velocity_pct: float,
        volume_ratio: float
    ) -> ScalpSignal | None:
        """Check for momentum burst signal."""
        # Require sufficient momentum
        if abs(velocity_pct) < self.config.momentum_threshold_pct:
            return None
        
        # Require volume confirmation (if available)
        if volume_ratio < self.config.volume_spike_ratio:
            return None
        
        # Determine direction
        direction = 'SCALP_CALL' if velocity_pct > 0 else 'SCALP_PUT'
        option_type = 'C' if velocity_pct > 0 else 'P'
        
        # Select option
        option = self._select_option(
            underlying_price, option_type, current_time
        )
        
        if option is None:
            return None
        
        return self._create_signal(
            current_time=current_time,
            underlying_price=underlying_price,
            velocity_pct=velocity_pct,
            volume_ratio=volume_ratio,
            trigger='momentum_burst',
            direction=direction,
            option=option
        )
    
    def _check_technical_signals(
        self,
        current_time: datetime,
        underlying_price: float,
        velocity_pct: float,
        volume_ratio: float
    ) -> ScalpSignal | None:
        """Check for technical pattern signals."""
        # Check VWAP signals
        vwap_signal = self.technical.check_vwap_signal(
            underlying_price, velocity_pct
        )
        
        if vwap_signal:
            direction = 'SCALP_CALL' if vwap_signal.direction == 'bullish' else 'SCALP_PUT'
            option_type = 'C' if vwap_signal.direction == 'bullish' else 'P'
            
            option = self._select_option(
                underlying_price, option_type, current_time
            )
            
            if option:
                return self._create_signal(
                    current_time=current_time,
                    underlying_price=underlying_price,
                    velocity_pct=velocity_pct,
                    volume_ratio=volume_ratio,
                    trigger=vwap_signal.signal_type,
                    direction=direction,
                    option=option,
                    confidence=vwap_signal.confidence
                )
        
        # Check breakout signals
        breakout_signal = self.technical.check_breakout(
            underlying_price, velocity_pct, volume_ratio
        )
        
        if breakout_signal:
            direction = 'SCALP_CALL' if breakout_signal.direction == 'bullish' else 'SCALP_PUT'
            option_type = 'C' if breakout_signal.direction == 'bullish' else 'P'
            
            option = self._select_option(
                underlying_price, option_type, current_time
            )
            
            if option:
                return self._create_signal(
                    current_time=current_time,
                    underlying_price=underlying_price,
                    velocity_pct=velocity_pct,
                    volume_ratio=volume_ratio,
                    trigger='breakout',
                    direction=direction,
                    option=option,
                    confidence=breakout_signal.confidence
                )
        
        return None
    
    def _select_option(
        self,
        underlying_price: float,
        option_type: str,
        current_time: datetime
    ) -> dict | None:
        """Select best option for the scalp trade.
        
        Criteria:
        - Correct type (C/P)
        - 0DTE or 1DTE
        - Delta near target (0.35)
        - Spread under threshold
        - Price under max
        """
        candidates = []
        
        for opt in self._available_options:
            # Filter by type
            if opt.get('option_type') != option_type:
                continue
            
            # Filter by DTE
            dte = opt.get('dte', 99)
            if dte > self.config.max_dte:
                continue
            
            # Filter by delta
            delta = abs(opt.get('delta', 0))
            if abs(delta - self.config.target_delta) > self.config.delta_tolerance:
                continue
            
            # Filter by spread
            bid = opt.get('bid_px', 0)
            ask = opt.get('ask_px', 0)
            if bid <= 0 or ask <= 0:
                continue
            
            mid = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid) * 100
            if spread_pct > self.config.max_spread_pct:
                continue
            
            # Filter by price
            if ask > self.config.max_contract_price:
                continue
            
            # Score candidate
            score = 0
            
            # Prefer 0DTE
            if self.config.prefer_0dte and dte == 0:
                score += 10
            
            # Prefer tighter spreads
            score += (self.config.max_spread_pct - spread_pct)
            
            # Prefer delta closer to target
            score += (self.config.delta_tolerance - abs(delta - self.config.target_delta)) * 10
            
            candidates.append((score, opt))
        
        if not candidates:
            return None
        
        # Return highest scoring candidate
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]
    
    def _create_signal(
        self,
        current_time: datetime,
        underlying_price: float,
        velocity_pct: float,
        volume_ratio: float,
        trigger: str,
        direction: str,
        option: dict,
        confidence: int = 50
    ) -> ScalpSignal:
        """Create a ScalpSignal from components."""
        bid = option['bid_px']
        ask = option['ask_px']
        mid = (bid + ask) / 2
        spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 0
        
        # Calculate suggested contracts based on position size
        max_value = self.config.max_contract_price * 100  # Convert to contract value
        suggested = max(1, int(max_value / (ask * 100)))
        
        return ScalpSignal(
            id=str(uuid.uuid4())[:8],
            timestamp=current_time,
            symbol=self.symbol,
            signal_type=direction,
            trigger=trigger,
            underlying_price=underlying_price,
            velocity_pct=velocity_pct,
            volume_ratio=volume_ratio,
            option_symbol=option['symbol'],
            strike=option['strike'],
            expiry=option['expiry'],
            delta=option.get('delta', 0),
            dte=option.get('dte', 0),
            bid_price=bid,
            ask_price=ask,
            entry_price=ask,  # Assume fill at ask for buys
            spread_pct=spread_pct,
            take_profit_pct=self.config.take_profit_pct,
            stop_loss_pct=self.config.stop_loss_pct,
            max_hold_minutes=self.config.max_hold_minutes,
            confidence=confidence,
            suggested_contracts=suggested
        )
```

---

## Part 3: Backtesting Framework

### 3.1 Scalp Backtester

```python
# backend/scripts/scalp_backtest.py

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Iterator
import pandas as pd
import json

from backend.data.databento_loader import DataBentoLoader
from backend.scalping.velocity_tracker import PriceVelocityTracker
from backend.scalping.volume_analyzer import VolumeAnalyzer
from backend.scalping.technical_scalper import TechnicalScalper
from backend.scalping.signal_generator import ScalpSignalGenerator, ScalpConfig, ScalpSignal

@dataclass
class ScalpTrade:
    """Record of a completed scalp trade."""
    signal: ScalpSignal
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # 'take_profit', 'stop_loss', 'time_exit'
    pnl_pct: float
    pnl_dollars: float
    hold_minutes: float

@dataclass
class BacktestResult:
    """Results from scalp backtesting."""
    start_date: date
    end_date: date
    config: ScalpConfig
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # P&L statistics
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0
    max_winner_pct: float = 0.0
    max_loser_pct: float = 0.0
    
    # Time statistics
    avg_hold_minutes: float = 0.0
    
    # Exit breakdown
    take_profit_exits: int = 0
    stop_loss_exits: int = 0
    time_exits: int = 0
    
    # Detailed trades
    trades: list[ScalpTrade] = field(default_factory=list)
    
    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        gross_loss = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct < 0))
        return gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl_pct': self.total_pnl_pct,
            'avg_pnl_pct': self.avg_pnl_pct,
            'avg_winner_pct': self.avg_winner_pct,
            'avg_loser_pct': self.avg_loser_pct,
            'max_winner_pct': self.max_winner_pct,
            'max_loser_pct': self.max_loser_pct,
            'avg_hold_minutes': self.avg_hold_minutes,
            'profit_factor': self.profit_factor,
            'take_profit_exits': self.take_profit_exits,
            'stop_loss_exits': self.stop_loss_exits,
            'time_exits': self.time_exits,
        }

class ScalpBacktester:
    """Backtester for scalping strategy using DataBento data."""
    
    def __init__(
        self,
        data_dir: str | Path,
        config: ScalpConfig | None = None
    ):
        self.loader = DataBentoLoader(data_dir)
        self.config = config or ScalpConfig(enabled=True)
        
        # Components (re-initialized per day)
        self.velocity: PriceVelocityTracker | None = None
        self.volume: VolumeAnalyzer | None = None
        self.technical: TechnicalScalper | None = None
        self.generator: ScalpSignalGenerator | None = None
    
    def _init_components(self, symbol: str = 'TSLA') -> None:
        """Initialize fresh components for a new day."""
        self.velocity = PriceVelocityTracker(symbol)
        self.volume = VolumeAnalyzer()
        self.technical = TechnicalScalper(symbol)
        self.generator = ScalpSignalGenerator(
            symbol=symbol,
            config=self.config,
            velocity_tracker=self.velocity,
            volume_analyzer=self.volume,
            technical_scalper=self.technical
        )
    
    def run(
        self,
        start_date: date,
        end_date: date,
        symbol: str = 'TSLA'
    ) -> BacktestResult:
        """Run backtest over date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            symbol: Underlying symbol
            
        Returns:
            BacktestResult with all statistics and trades
        """
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            config=self.config
        )
        
        for current_date, day_data in self.loader.load_date_range(start_date, end_date):
            day_trades = self._run_day(current_date, day_data, symbol)
            result.trades.extend(day_trades)
        
        # Calculate statistics
        self._calculate_statistics(result)
        
        return result
    
    def _run_day(
        self,
        current_date: date,
        day_data: pd.DataFrame,
        symbol: str
    ) -> list[ScalpTrade]:
        """Run backtest for a single day."""
        # Re-initialize components for fresh day
        self._init_components(symbol)
        
        # Filter data for scalping
        filtered = self.loader.filter_for_scalping(
            day_data,
            current_date,
            max_dte=self.config.max_dte,
            max_spread_pct=self.config.max_spread_pct
        )
        
        if filtered.empty:
            return []
        
        # Group by timestamp to simulate time progression
        grouped = filtered.groupby('ts_event')
        
        trades = []
        active_position: dict | None = None
        
        for timestamp, minute_data in grouped:
            # Convert timestamp
            if isinstance(timestamp, str):
                ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                ts = timestamp.to_pydatetime()
            
            # Get underlying price (use ATM call as proxy, or need separate stock data)
            # For now, estimate from nearest ATM option
            underlying_price = self._estimate_underlying_price(minute_data)
            
            if underlying_price is None:
                continue
            
            # Update velocity tracker
            self.velocity.add_price(underlying_price, ts)
            
            # Update technical indicators
            self.technical.update(underlying_price, volume=1000, timestamp=ts)
            
            # Update available options for signal generator
            options = self._prepare_options_list(minute_data, current_date)
            self.generator.update_available_options(options)
            
            # Check for exit if we have a position
            if active_position:
                exit_result = self._check_exit(
                    active_position, minute_data, ts
                )
                if exit_result:
                    trades.append(exit_result)
                    active_position = None
            
            # Check for new signal if no position
            if active_position is None:
                signal = self.generator.evaluate(ts, underlying_price)
                if signal:
                    active_position = {
                        'signal': signal,
                        'entry_time': ts,
                        'entry_price': signal.entry_price,
                    }
        
        # Force exit any remaining position at end of day
        if active_position:
            trade = self._force_exit(active_position, filtered, 'end_of_day')
            if trade:
                trades.append(trade)
        
        return trades
    
    def _estimate_underlying_price(self, minute_data: pd.DataFrame) -> float | None:
        """Estimate underlying price from option data.
        
        Uses put-call parity or nearest ATM option mid prices.
        """
        # Simple approach: find options with delta nearest 0.5
        # and back-calculate underlying
        # For now, use strike of nearest ATM call as rough proxy
        
        calls = minute_data[minute_data['option_type'] == 'C']
        if calls.empty:
            return None
        
        # Get call with tightest spread (likely most liquid/ATM)
        calls = calls.copy()
        calls['spread'] = calls['ask_px'] - calls['bid_px']
        calls = calls[calls['spread'] > 0]
        
        if calls.empty:
            return None
        
        best = calls.loc[calls['spread'].idxmin()]
        
        # ATM call strike ≈ underlying price
        return best['strike']
    
    def _prepare_options_list(
        self,
        minute_data: pd.DataFrame,
        current_date: date
    ) -> list[dict]:
        """Convert DataFrame to list of option dicts for signal generator."""
        options = []
        
        for _, row in minute_data.iterrows():
            # Estimate delta from moneyness (rough approximation)
            # In production, would use actual Greeks
            strike = row['strike']
            underlying = strike  # Rough estimate
            moneyness = strike / underlying if underlying else 1.0
            
            # Rough delta estimation
            if row['option_type'] == 'C':
                delta = max(0.1, min(0.9, 1.0 - moneyness + 0.5))
            else:
                delta = -max(0.1, min(0.9, moneyness - 0.5))
            
            options.append({
                'symbol': row['symbol'],
                'strike': row['strike'],
                'expiry': row['expiry'].isoformat() if hasattr(row['expiry'], 'isoformat') else str(row['expiry']),
                'option_type': row['option_type'],
                'delta': delta,
                'bid_px': row['bid_px'],
                'ask_px': row['ask_px'],
                'dte': row['dte'],
            })
        
        return options
    
    def _check_exit(
        self,
        position: dict,
        minute_data: pd.DataFrame,
        current_time: datetime
    ) -> ScalpTrade | None:
        """Check if position should exit."""
        signal: ScalpSignal = position['signal']
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        
        # Find current price of the option
        option_data = minute_data[minute_data['symbol'] == signal.option_symbol]
        
        if option_data.empty:
            return None  # Option not in this minute's data
        
        current_bid = option_data.iloc[0]['bid_px']
        
        # Calculate P&L (sell at bid)
        pnl_pct = ((current_bid - entry_price) / entry_price) * 100
        
        # Check exit conditions
        exit_reason = None
        
        # Stop loss
        if pnl_pct <= -signal.stop_loss_pct:
            exit_reason = 'stop_loss'
        
        # Take profit
        elif pnl_pct >= signal.take_profit_pct:
            exit_reason = 'take_profit'
        
        # Time exit
        hold_minutes = (current_time - entry_time).total_seconds() / 60
        if hold_minutes >= signal.max_hold_minutes:
            exit_reason = 'time_exit'
        
        if exit_reason:
            return ScalpTrade(
                signal=signal,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=current_time,
                exit_price=current_bid,
                exit_reason=exit_reason,
                pnl_pct=pnl_pct,
                pnl_dollars=pnl_pct * entry_price / 100,  # Per contract
                hold_minutes=hold_minutes
            )
        
        return None
    
    def _force_exit(
        self,
        position: dict,
        day_data: pd.DataFrame,
        reason: str
    ) -> ScalpTrade | None:
        """Force exit position at end of day."""
        signal: ScalpSignal = position['signal']
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        
        # Get last price for the option
        option_data = day_data[day_data['symbol'] == signal.option_symbol]
        
        if option_data.empty:
            return None
        
        last_row = option_data.iloc[-1]
        exit_price = last_row['bid_px']
        exit_time = last_row['ts_event']
        
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
        elif hasattr(exit_time, 'to_pydatetime'):
            exit_time = exit_time.to_pydatetime()
        
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        hold_minutes = (exit_time - entry_time).total_seconds() / 60
        
        return ScalpTrade(
            signal=signal,
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=reason,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_pct * entry_price / 100,
            hold_minutes=hold_minutes
        )
    
    def _calculate_statistics(self, result: BacktestResult) -> None:
        """Calculate aggregate statistics from trades."""
        if not result.trades:
            return
        
        result.total_trades = len(result.trades)
        
        winners = [t for t in result.trades if t.pnl_pct > 0]
        losers = [t for t in result.trades if t.pnl_pct <= 0]
        
        result.winning_trades = len(winners)
        result.losing_trades = len(losers)
        
        result.total_pnl_pct = sum(t.pnl_pct for t in result.trades)
        result.avg_pnl_pct = result.total_pnl_pct / result.total_trades
        
        if winners:
            result.avg_winner_pct = sum(t.pnl_pct for t in winners) / len(winners)
            result.max_winner_pct = max(t.pnl_pct for t in winners)
        
        if losers:
            result.avg_loser_pct = sum(t.pnl_pct for t in losers) / len(losers)
            result.max_loser_pct = min(t.pnl_pct for t in losers)
        
        result.avg_hold_minutes = sum(t.hold_minutes for t in result.trades) / result.total_trades
        
        # Exit breakdown
        result.take_profit_exits = sum(1 for t in result.trades if t.exit_reason == 'take_profit')
        result.stop_loss_exits = sum(1 for t in result.trades if t.exit_reason == 'stop_loss')
        result.time_exits = sum(1 for t in result.trades if t.exit_reason == 'time_exit')


def main():
    """Run scalp backtest from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scalp Strategy Backtester')
    parser.add_argument('--data-dir', required=True, help='Path to DataBento data')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    # Config overrides
    parser.add_argument('--momentum-threshold', type=float, default=0.5)
    parser.add_argument('--take-profit', type=float, default=30.0)
    parser.add_argument('--stop-loss', type=float, default=15.0)
    parser.add_argument('--max-hold', type=int, default=15)
    
    args = parser.parse_args()
    
    config = ScalpConfig(
        enabled=True,
        momentum_threshold_pct=args.momentum_threshold,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
        max_hold_minutes=args.max_hold
    )
    
    backtester = ScalpBacktester(args.data_dir, config)
    
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    
    print(f"Running backtest from {start_date} to {end_date}...")
    result = backtester.run(start_date, end_date)
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Total P&L: {result.total_pnl_pct:.1f}%")
    print(f"Avg P&L per Trade: {result.avg_pnl_pct:.1f}%")
    print(f"Avg Winner: {result.avg_winner_pct:.1f}%")
    print(f"Avg Loser: {result.avg_loser_pct:.1f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Avg Hold Time: {result.avg_hold_minutes:.1f} minutes")
    print(f"\nExit Breakdown:")
    print(f"  Take Profit: {result.take_profit_exits}")
    print(f"  Stop Loss: {result.stop_loss_exits}")
    print(f"  Time Exit: {result.time_exits}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
```

---

## Part 4: Live Trading Integration

**IMPORTANT:** This section modifies existing files. Read the existing code FIRST before making changes.

### 4.0 Extend Existing Position Tracker (REQUIRED)

**Do NOT create a new position tracker.** Extend the existing one in `backend/engine/position_tracker.py`.

First, inspect the existing implementation:
```bash
cat backend/engine/position_tracker.py
```

Then add scalp-specific fields. Example extension pattern:

```python
# In backend/engine/position_tracker.py

from typing import Literal

# Add to existing TrackedPosition class or create extended version:
@dataclass
class TrackedPosition:
    # ... existing fields ...
    
    # NEW: Position type to distinguish regime vs scalp
    position_type: Literal['regime', 'scalp'] = 'regime'
    
    # NEW: Scalp-specific fields (only used when position_type='scalp')
    scalp_trigger: str | None = None  # 'momentum_burst', 'vwap_bounce', etc.
    scalp_max_hold_minutes: int = 15
    scalp_entry_velocity: float = 0.0

# Add scalp-specific exit checking method:
def check_scalp_exit(self, position: TrackedPosition, current_time: datetime) -> bool:
    """Check scalp-specific exit conditions (time-based)."""
    if position.position_type != 'scalp':
        return False
    
    hold_minutes = (current_time - position.opened_at).total_seconds() / 60
    return hold_minutes >= position.scalp_max_hold_minutes
```

### 4.1 Main.py Integration

**FIRST:** Inspect the existing main.py to understand:
- How the lifespan/startup is structured
- Where existing evaluation loops are defined
- How the WebSocket connection_manager is used

```bash
cat backend/main.py | head -100
grep -n "connection_manager\|broadcast\|websocket" backend/main.py
```

Add to existing main.py (integrate, don't duplicate):

```python
# In main.py lifespan or startup

from backend.scalping.velocity_tracker import PriceVelocityTracker
from backend.scalping.volume_analyzer import VolumeAnalyzer
from backend.scalping.technical_scalper import TechnicalScalper
from backend.scalping.signal_generator import ScalpSignalGenerator, ScalpConfig

# Initialize scalping components (alongside existing regime components)
scalp_config = ScalpConfig(
    enabled=os.getenv('SCALP_ENABLED', 'false').lower() == 'true',
    momentum_threshold_pct=float(os.getenv('SCALP_MOMENTUM_THRESHOLD', '0.5')),
    take_profit_pct=float(os.getenv('SCALP_TAKE_PROFIT', '30.0')),
    stop_loss_pct=float(os.getenv('SCALP_STOP_LOSS', '15.0')),
)

scalp_velocity = PriceVelocityTracker('TSLA')
scalp_volume = VolumeAnalyzer()
scalp_technical = TechnicalScalper('TSLA')
scalp_generator = ScalpSignalGenerator(
    symbol='TSLA',
    config=scalp_config,
    velocity_tracker=scalp_velocity,
    volume_analyzer=scalp_volume,
    technical_scalper=scalp_technical
)

# Add scalping evaluation loop
async def scalp_evaluation_loop():
    """Fast evaluation loop for scalping signals (runs every second)."""
    while True:
        try:
            await asyncio.sleep(1.0)  # 1 second interval
            
            if not scalp_config.enabled:
                continue
            
            if not is_market_hours():
                continue
            
            # Get current price from WebSocket data
            current_price = get_current_tsla_price()  # Implement based on your data source
            if current_price is None:
                continue
            
            current_time = datetime.now()
            
            # Update velocity
            scalp_velocity.add_price(current_price, current_time)
            
            # Update technicals
            scalp_technical.update(current_price, volume=0, timestamp=current_time)
            
            # Evaluate for signal
            signal = scalp_generator.evaluate(current_time, current_price)
            
            if signal:
                logger.info(f"Scalp signal generated: {signal.signal_type} - {signal.trigger}")
                
                # Broadcast to frontend
                await connection_manager.broadcast({
                    "type": "scalp_signal",
                    "data": signal.to_dict(),
                })
                
                # Auto-execute if enabled
                if auto_executor and auto_executor.scalp_enabled:
                    await auto_executor.execute_scalp(signal)
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in scalp evaluation loop: {e}")

# Add to startup tasks
asyncio.create_task(scalp_evaluation_loop())
```

### 4.2 Run Mode Flag

Add to run.py or create run_scalp.py:

```python
# run.py modifications

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['regime', 'scalp', 'both'],
        default='regime',
        help='Trading mode: regime (existing), scalp (new), or both'
    )
    args = parser.parse_args()
    
    # Set environment variables based on mode
    if args.mode == 'scalp':
        os.environ['REGIME_ENABLED'] = 'false'
        os.environ['SCALP_ENABLED'] = 'true'
    elif args.mode == 'both':
        os.environ['REGIME_ENABLED'] = 'true'
        os.environ['SCALP_ENABLED'] = 'true'
    else:  # regime
        os.environ['REGIME_ENABLED'] = 'true'
        os.environ['SCALP_ENABLED'] = 'false'
    
    # Start application
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == '__main__':
    main()
```

---

## Part 5: Configuration Reference

### 5.1 Environment Variables

```bash
# .env additions for scalping

# Enable/disable scalping module
SCALP_ENABLED=false

# Signal thresholds
SCALP_MOMENTUM_THRESHOLD=0.5   # % move to trigger
SCALP_MOMENTUM_WINDOW=30       # seconds

# Volume requirements
SCALP_VOLUME_SPIKE_RATIO=1.5   # 1.5x normal volume

# Option selection
SCALP_TARGET_DELTA=0.35
SCALP_MAX_SPREAD_PCT=8.0
SCALP_MAX_DTE=1
SCALP_MAX_CONTRACT_PRICE=5.00

# Risk management
SCALP_TAKE_PROFIT=30.0         # %
SCALP_STOP_LOSS=15.0           # %
SCALP_MAX_HOLD_MINUTES=15

# Position limits
SCALP_MAX_DAILY_TRADES=10
SCALP_MAX_CONCURRENT=1
SCALP_POSITION_SIZE_PCT=5.0

# Cooldowns
SCALP_MIN_SIGNAL_INTERVAL=60   # seconds
SCALP_LOSS_COOLDOWN=300        # seconds
```

### 5.2 Recommended Starting Parameters

For initial paper trading:

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| momentum_threshold_pct | 0.7 | 0.5 | 0.3 |
| volume_spike_ratio | 2.0 | 1.5 | 1.2 |
| take_profit_pct | 25 | 30 | 40 |
| stop_loss_pct | 10 | 15 | 20 |
| max_hold_minutes | 10 | 15 | 20 |
| max_daily_scalps | 5 | 10 | 15 |

**Start conservative, loosen based on paper trading results.**

---

## Part 6: Testing Strategy

### 6.1 Development Dataset Split

| Dataset | Date Range | Purpose |
|---------|------------|---------|
| Development | 2022-01-01 to 2023-12-31 | Build and tune strategy |
| Validation | 2024-01-01 to 2024-06-30 | Validate parameters |
| Out-of-sample | 2024-07-01 to 2025-12-31 | Final test (touch ONCE) |

### 6.2 Backtest Checklist

Before paper trading, confirm:

- [ ] Backtest on 2022 (crash) shows reasonable performance
- [ ] Backtest on 2024 (bull) shows reasonable performance  
- [ ] Win rate > 50%
- [ ] Profit factor > 1.2
- [ ] Avg winner > avg loser (absolute value)
- [ ] No single day with catastrophic loss (> 50% of trades losing)
- [ ] Signal distribution across different market hours

### 6.3 Paper Trading Checklist

First 2 weeks of paper trading:

- [ ] Signals fire at expected frequency (5-15 per day)
- [ ] Fills are within 5% of expected entry price
- [ ] Exits trigger at correct thresholds
- [ ] No technical errors or crashes
- [ ] Results roughly match backtest (within 30% of expected metrics)

---

## Part 7: File Structure

After implementation (accounting for existing code):

```
options-radar/
├── backend/
│   ├── data/
│   │   ├── databento_loader.py      # NEW: DataBento data loading
│   │   └── ... (existing clients - follow their patterns)
│   ├── engine/
│   │   ├── gates.py                 # EXISTING: Import for liquidity/spread checks
│   │   ├── position_tracker.py      # EXTEND: Add position_type='scalp'
│   │   └── ... (existing)
│   ├── models/
│   │   └── ... (CHECK: OCC parsing may already exist here)
│   ├── scalping/                    # NEW: Entire module
│   │   ├── __init__.py
│   │   ├── velocity_tracker.py
│   │   ├── volume_analyzer.py
│   │   ├── technical_scalper.py
│   │   └── signal_generator.py
│   ├── scripts/
│   │   └── scalp_backtest.py        # NEW: Backtesting script
│   ├── utils/
│   │   └── occ_parser.py            # NEW: Only if not in models/
│   ├── websocket/
│   │   └── ... (EXISTING: Use for scalp signal broadcasts)
│   └── main.py                      # MODIFY: Add scalp evaluation loop
├── data/
│   └── databento/                   # LOCAL ONLY (not in repo, add to .gitignore)
│       ├── 2022-01-03.csv.zst
│       └── ... (1007 daily files)
├── run.py                           # MODIFY: Add --mode flag
└── .env                             # MODIFY: Add SCALP_* variables
```

### Files to CHECK before creating

| File | Check Command | If Exists |
|------|---------------|-----------|
| OCC parser | `grep -r "parse_occ\|ParsedOption" backend/` | Import it, skip occ_parser.py |
| Position tracker | `cat backend/engine/position_tracker.py` | Extend with position_type field |
| Gate classes | `cat backend/engine/gates.py` | Import LiquidityGate, SpreadGate |
| WebSocket manager | `ls backend/websocket/` | Use existing broadcast method |

---

## Implementation Order

**Phase 0: Codebase Inspection (REQUIRED FIRST)**
- [ ] Run `ls -la backend/` to see full structure
- [ ] Run `cat backend/engine/gates.py` — identify reusable gates
- [ ] Run `cat backend/engine/position_tracker.py` — understand interface
- [ ] Run `grep -r "parse" backend/models/` — check for OCC parsing
- [ ] Run `cat backend/data/alpaca_client.py | head -50` — understand data client patterns
- [ ] Document findings before proceeding

1. **Phase 1: Data Infrastructure**
   - [ ] Check if OCC parsing exists; if not, implement `occ_parser.py`
   - [ ] Implement `databento_loader.py` following existing data client patterns
   - [ ] Test loading sample data files

2. **Phase 2: Core Components**
   - [ ] Implement `velocity_tracker.py`
   - [ ] Implement `volume_analyzer.py`
   - [ ] Implement `technical_scalper.py`
   - [ ] Unit test each component

3. **Phase 3: Signal Generation**
   - [ ] Implement `signal_generator.py`
   - [ ] Import and use existing gates from `backend/engine/gates.py`
   - [ ] Integration test with mock data

4. **Phase 4: Backtesting**
   - [ ] Implement `scalp_backtest.py`
   - [ ] Run on development dataset (2022-2023)
   - [ ] Tune parameters
   - [ ] Validate on 2024 H1

5. **Phase 5: Live Integration**
   - [ ] Extend `backend/engine/position_tracker.py` for scalp positions
   - [ ] Modify `main.py` to add scalp evaluation loop
   - [ ] Modify `run.py` for mode selection (--mode regime|scalp|both)
   - [ ] Add SCALP_* environment variables to .env
   - [ ] Use existing WebSocket manager for signal broadcasts
   - [ ] Paper trade

---

## Success Criteria

The scalping module is ready for live paper trading when:

| Metric | Target |
|--------|--------|
| Backtest win rate | > 50% |
| Backtest profit factor | > 1.2 |
| Backtest avg P&L per trade | > 5% |
| 2022 bear market performance | Positive or < -10% |
| 2024 bull market performance | > 20% |
| No catastrophic single-day loss | < 5 losing trades in a row |
| Out-of-sample (2025) degradation | < 30% worse than dev set |

---

*End of Implementation Guide*
