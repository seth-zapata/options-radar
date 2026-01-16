# Scalping Module Improvement Task

## Current State

We ran a Q1 2024 backtest with 260 trades. Results:

| Metric | Value |
|--------|-------|
| Win Rate | 10% (26 wins) |
| Time Exits | 85% (221 trades) — NOT reaching TP or SL |
| Stop Loss Exits | 29 trades |
| Take Profit Exits | 10 trades |
| Avg Win | $372 |
| Avg Loss | $86 |
| Net P&L | -$10,539 |

**Core Problem:** 85% of trades expire via time exit, meaning signals are not predicting real directional moves. The risk/reward ratio (4.3:1) is good — the signal quality is bad.

---

## Task: Implement Improvements in This Order

### Phase 1: Analyze the Winners FIRST (Before Any Code Changes)

Before implementing fixes, analyze what the 10 winning trades had in common:

```python
# Add to scalp_backtest.py or create analyze_winners.py

def analyze_winners(result: BacktestResult) -> dict:
    """Analyze characteristics of winning trades."""
    winners = [t for t in result.trades if t.pnl_pct > 0]
    losers = [t for t in result.trades if t.pnl_pct <= 0]
    
    analysis = {
        # Time patterns
        'winner_hours': [t.entry_time.hour for t in winners],
        'loser_hours': [t.entry_time.hour for t in losers],
        
        # Day of week
        'winner_weekdays': [t.entry_time.weekday() for t in winners],
        
        # Signal type breakdown
        'winner_triggers': Counter([t.signal.trigger for t in winners]),
        'loser_triggers': Counter([t.signal.trigger for t in losers]),
        
        # Direction
        'winner_directions': Counter([t.signal.signal_type for t in winners]),
        'loser_directions': Counter([t.signal.signal_type for t in losers]),
        
        # Velocity at entry
        'winner_velocities': [t.signal.velocity_pct for t in winners],
        'loser_velocities': [t.signal.velocity_pct for t in losers],
        
        # Option characteristics
        'winner_deltas': [t.signal.delta for t in winners],
        'winner_dtes': [t.signal.dte for t in winners],
        'winner_spreads': [t.signal.spread_pct for t in winners],
    }
    
    return analysis

# Print analysis
def print_winner_analysis(analysis: dict):
    print("\n" + "="*60)
    print("WINNER ANALYSIS (What do winning trades have in common?)")
    print("="*60)
    
    print("\nTime of Day Distribution:")
    print(f"  Winners by hour: {Counter(analysis['winner_hours'])}")
    print(f"  Losers by hour:  {Counter(analysis['loser_hours'])}")
    
    print("\nTrigger Type:")
    print(f"  Winners: {analysis['winner_triggers']}")
    print(f"  Losers:  {analysis['loser_triggers']}")
    
    print("\nDirection (SCALP_CALL vs SCALP_PUT):")
    print(f"  Winners: {analysis['winner_directions']}")
    print(f"  Losers:  {analysis['loser_directions']}")
    
    print("\nVelocity at Entry:")
    winner_vels = analysis['winner_velocities']
    loser_vels = analysis['loser_velocities']
    print(f"  Winners avg: {sum(winner_vels)/len(winner_vels):.2f}%")
    print(f"  Losers avg:  {sum(loser_vels)/len(loser_vels):.2f}%")
    print(f"  Winners min/max: {min(winner_vels):.2f}% / {max(winner_vels):.2f}%")
```

**Run this analysis and share results before proceeding.**

---

### Phase 2: Fix Underlying Price Estimation (CRITICAL)

The current implementation estimates underlying price from options data using put-call parity. This is noisy and creates phantom momentum when the "ATM" strike switches.

**Solution:** Use actual TSLA stock prices, not derived prices.

Option A: Load TSLA stock data from a separate source (if available in DataBento or fetch from Alpaca historical)

Option B: Smooth the current estimation with EMA

```python
# In velocity_tracker.py or create underlying_estimator.py

class SmoothedUnderlyingEstimator:
    """Smooth underlying price estimates to reduce noise."""
    
    def __init__(self, ema_periods: int = 5):
        self.ema_periods = ema_periods
        self.ema_multiplier = 2 / (ema_periods + 1)
        self._ema: float | None = None
        self._raw_prices: deque[float] = deque(maxlen=20)
    
    def update(self, raw_estimate: float) -> float:
        """Update with new raw estimate, return smoothed value."""
        self._raw_prices.append(raw_estimate)
        
        if self._ema is None:
            self._ema = raw_estimate
        else:
            self._ema = (raw_estimate * self.ema_multiplier) + (self._ema * (1 - self.ema_multiplier))
        
        return self._ema
    
    @property
    def current(self) -> float | None:
        return self._ema
    
    @property
    def noise_level(self) -> float:
        """Estimate current noise level (std dev of recent raw prices)."""
        if len(self._raw_prices) < 3:
            return 0.0
        return statistics.stdev(self._raw_prices)
```

**Integration:** Use smoothed price for velocity calculations, reject signals when `noise_level` is high.

---

### Phase 3: Add Time-of-Day Filter

Skip the first 30 minutes of market open (9:30-10:00 ET). This is choppy and unreliable.

```python
# In signal_generator.py - add to ScalpConfig

@dataclass
class ScalpConfig:
    # ... existing fields ...
    
    # Time filters
    skip_market_open_minutes: int = 30  # Skip first 30 min
    skip_market_close_minutes: int = 5   # Skip last 5 min
    
    # Market hours (ET)
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0

# In evaluate() method:
def _is_valid_trading_time(self, current_time: datetime) -> bool:
    """Check if current time is valid for trading."""
    # Convert to ET if needed
    hour = current_time.hour
    minute = current_time.minute
    
    # Minutes since market open
    market_open_minutes = self.config.market_open_hour * 60 + self.config.market_open_minute
    current_minutes = hour * 60 + minute
    minutes_since_open = current_minutes - market_open_minutes
    
    # Skip first N minutes
    if minutes_since_open < self.config.skip_market_open_minutes:
        return False
    
    # Minutes until market close
    market_close_minutes = self.config.market_close_hour * 60 + self.config.market_close_minute
    minutes_until_close = market_close_minutes - current_minutes
    
    # Skip last N minutes
    if minutes_until_close < self.config.skip_market_close_minutes:
        return False
    
    return True
```

---

### Phase 4: Raise Momentum Threshold

Increase from 0.5% to 0.8% and require persistence (2 consecutive readings).

```python
# In ScalpConfig
@dataclass
class ScalpConfig:
    # ... existing ...
    
    # Updated thresholds
    momentum_threshold_pct: float = 0.8  # Was 0.5
    require_confirmation: bool = True     # NEW: Require 2 readings
    confirmation_window_seconds: int = 10 # NEW: Second reading within 10s

# In signal_generator.py
class ScalpSignalGenerator:
    def __init__(self, ...):
        # ... existing ...
        self._pending_signal: dict | None = None  # For confirmation
        self._pending_signal_time: datetime | None = None
    
    def _check_momentum_burst(self, ...) -> ScalpSignal | None:
        # Check threshold
        if abs(velocity_pct) < self.config.momentum_threshold_pct:
            self._pending_signal = None  # Reset pending
            return None
        
        direction = 'SCALP_CALL' if velocity_pct > 0 else 'SCALP_PUT'
        
        # Confirmation logic
        if self.config.require_confirmation:
            if self._pending_signal is None:
                # First reading - store as pending
                self._pending_signal = {'direction': direction, 'velocity': velocity_pct}
                self._pending_signal_time = current_time
                return None  # Don't fire yet
            
            # Check if second reading confirms
            elapsed = (current_time - self._pending_signal_time).total_seconds()
            
            if elapsed > self.config.confirmation_window_seconds:
                # Too old, treat as new pending
                self._pending_signal = {'direction': direction, 'velocity': velocity_pct}
                self._pending_signal_time = current_time
                return None
            
            if self._pending_signal['direction'] != direction:
                # Direction changed, reset
                self._pending_signal = {'direction': direction, 'velocity': velocity_pct}
                self._pending_signal_time = current_time
                return None
            
            # Confirmed! Clear pending and generate signal
            self._pending_signal = None
            self._pending_signal_time = None
        
        # Continue to option selection and signal creation...
```

---

### Phase 5: Lower Take Profit Target

Reduce from 30% to 20% for 0DTE options.

```python
# In ScalpConfig - update defaults
take_profit_pct: float = 20.0  # Was 30.0
```

Also test with 15% to see impact on win rate.

---

### Phase 6: Consider Higher Delta Options

Change target delta from 0.35 to 0.45 for better underlying correlation.

```python
# In ScalpConfig
target_delta: float = 0.45     # Was 0.35
delta_tolerance: float = 0.10  # Accept 0.35-0.55
```

---

### Phase 7: Add Better Metrics

Add these to BacktestResult for better analysis:

```python
@dataclass
class BacktestResult:
    # ... existing fields ...
    
    # NEW: Time-based analysis
    pnl_by_hour: dict[int, float] = field(default_factory=dict)
    trades_by_hour: dict[int, int] = field(default_factory=dict)
    winrate_by_hour: dict[int, float] = field(default_factory=dict)
    
    # NEW: Signal quality
    avg_winner_velocity: float = 0.0
    avg_loser_velocity: float = 0.0
    
    # NEW: Drawdown
    max_drawdown_pct: float = 0.0
    
    # NEW: Streaks
    max_consecutive_losses: int = 0

# Calculate in _calculate_statistics():
def _calculate_statistics(self, result: BacktestResult) -> None:
    # ... existing code ...
    
    # P&L by hour
    for trade in result.trades:
        hour = trade.entry_time.hour
        result.pnl_by_hour[hour] = result.pnl_by_hour.get(hour, 0) + trade.pnl_dollars
        result.trades_by_hour[hour] = result.trades_by_hour.get(hour, 0) + 1
    
    # Win rate by hour
    for hour in result.trades_by_hour:
        hour_trades = [t for t in result.trades if t.entry_time.hour == hour]
        hour_wins = [t for t in hour_trades if t.pnl_pct > 0]
        result.winrate_by_hour[hour] = len(hour_wins) / len(hour_trades) if hour_trades else 0
    
    # Drawdown calculation
    cumulative = 0
    peak = 0
    max_dd = 0
    for trade in sorted(result.trades, key=lambda t: t.entry_time):
        cumulative += trade.pnl_dollars
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        max_dd = max(max_dd, drawdown)
    result.max_drawdown_pct = (max_dd / 10000) * 100 if max_dd > 0 else 0  # Assuming $10k base
    
    # Max consecutive losses
    current_streak = 0
    max_streak = 0
    for trade in sorted(result.trades, key=lambda t: t.entry_time):
        if trade.pnl_pct <= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    result.max_consecutive_losses = max_streak
```

---

## Testing Protocol

After each change, run backtest on Q1 2024 and record:

| Change | Trades | Win Rate | Net P&L | Avg Win | Avg Loss |
|--------|--------|----------|---------|---------|----------|
| Baseline | 260 | 10% | -$10,539 | $372 | $86 |
| + Winner analysis | — | — | — | — | — |
| + Smoothed underlying | ? | ? | ? | ? | ? |
| + Time filter (skip 9:30-10:00) | ? | ? | ? | ? | ? |
| + Higher threshold (0.8%) | ? | ? | ? | ? | ? |
| + Confirmation required | ? | ? | ? | ? | ? |
| + Lower TP (20%) | ? | ? | ? | ? | ? |
| + Higher delta (0.45) | ? | ? | ? | ? | ? |

**Important:** Make ONE change at a time and test. Do not bundle changes — we need to know which ones actually help.

---

## Success Criteria

The scalping module is viable when:

| Metric | Current | Target |
|--------|---------|--------|
| Win Rate | 10% | > 25% |
| Time Exits | 85% | < 50% |
| Profit Factor | ~0.4 | > 1.2 |
| Max Consecutive Losses | ? | < 10 |

If after all improvements win rate stays below 20%, we may need to reconsider the strategy premise entirely (momentum scalping on 0DTE TSLA options).

---

## Files to Modify

1. `backend/scripts/scalp_backtest.py` — Add winner analysis, better metrics
2. `backend/scalping/signal_generator.py` — Time filter, threshold, confirmation
3. `backend/scalping/velocity_tracker.py` or new `underlying_estimator.py` — Price smoothing
4. Config defaults in `ScalpConfig`

Start with Phase 1 (winner analysis) and share results before implementing code changes.
