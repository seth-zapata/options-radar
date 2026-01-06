# Scalping Module Design

## Overview

A momentum-based scalping system that runs alongside the existing regime strategy, designed to capture rapid intraday moves in TSLA options.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Existing Infrastructure                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │
│  │ Alpaca Stocks│ │Alpaca Options│ │   ORATS      │                 │
│  │   WebSocket  │ │  WebSocket   │ │  (Greeks)    │                 │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                 │
│         │                │                │                          │
│         ▼                ▼                ▼                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    DataAggregator                            │    │
│  │  (prices, quotes, greeks - all real-time)                   │    │
│  └─────────────────────────┬───────────────────────────────────┘    │
└────────────────────────────┼────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Regime Module  │ │ SCALPING MODULE │ │  Gate System    │
│  (existing)     │ │    (new)        │ │  (shared)       │
│                 │ │                 │ │                 │
│ - 7-day windows │ │ - 100ms loop    │ │ - Freshness     │
│ - Sentiment     │ │ - Velocity      │ │ - Liquidity     │
│ - Pullback/     │ │ - Volume        │ │ - Spread        │
│   bounce        │ │ - Breakouts     │ │                 │
└────────┬────────┘ └────────┬────────┘ └─────────────────┘
         │                   │
         ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     AutoExecutor                             │
│  (handles both regime and scalp signals)                    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. PriceVelocityTracker

Tracks price movement speed to detect rapid moves.

```python
@dataclass
class PriceVelocityTracker:
    """Tracks price velocity over multiple timeframes."""

    symbol: str

    # Rolling windows of price data
    _prices: deque[tuple[float, datetime]]  # (price, timestamp)

    # Configurable windows
    windows: tuple[int, ...] = (5, 15, 30, 60)  # seconds

    def add_price(self, price: float, timestamp: datetime) -> None:
        """Add a price tick."""
        self._prices.append((price, timestamp))
        self._cleanup_old_prices()

    def get_velocity(self, window_seconds: int) -> PriceVelocity:
        """Calculate price velocity over window.

        Returns:
            PriceVelocity with:
            - change_pct: Percentage change over window
            - direction: 'up', 'down', or 'flat'
            - speed: Change per second (for comparing intensity)
        """
        pass

    def detect_spike(self, threshold_pct: float = 0.5, window: int = 30) -> SpikeSignal | None:
        """Detect if price moved > threshold% in window seconds.

        Returns SpikeSignal if detected, None otherwise.
        """
        pass
```

### 2. VolumeAnalyzer

Detects unusual volume that often precedes or confirms moves.

```python
@dataclass
class VolumeAnalyzer:
    """Analyzes option volume for unusual activity."""

    # Rolling volume data
    _volume_history: dict[str, deque[tuple[int, datetime]]]  # contract -> [(volume, time)]

    # Baseline stats (updated periodically)
    _baseline_volume: dict[str, float]  # contract -> avg volume/minute

    def add_trade(self, contract: str, volume: int, timestamp: datetime) -> None:
        """Record a trade."""
        pass

    def get_volume_ratio(self, contract: str, window_minutes: int = 5) -> float:
        """Get current volume vs baseline ratio.

        Returns:
            Ratio where 1.0 = normal, 2.0 = 2x normal volume, etc.
        """
        pass

    def detect_sweep(self, min_contracts: int = 100) -> SweepSignal | None:
        """Detect large sweep orders (institutional activity).

        A sweep is a large order that takes out multiple price levels.
        """
        pass
```

### 3. TechnicalScalper

Fast technical analysis for scalping signals.

```python
@dataclass
class TechnicalScalper:
    """Technical analysis optimized for scalping."""

    symbol: str
    velocity_tracker: PriceVelocityTracker
    volume_analyzer: VolumeAnalyzer

    # VWAP tracking
    _vwap: float = 0.0
    _vwap_upper: float = 0.0  # +1 std dev
    _vwap_lower: float = 0.0  # -1 std dev

    # Support/Resistance (calculated from recent price action)
    _support_levels: list[float] = field(default_factory=list)
    _resistance_levels: list[float] = field(default_factory=list)

    # Recent high/low tracking
    _session_high: float = 0.0
    _session_low: float = float('inf')

    def update(self, price: float, volume: int, timestamp: datetime) -> None:
        """Update all technical indicators with new tick."""
        pass

    def check_vwap_rejection(self) -> ScalpSignal | None:
        """Check for VWAP rejection pattern.

        Bullish: Price touches VWAP from above, bounces
        Bearish: Price touches VWAP from below, rejects
        """
        pass

    def check_breakout(self) -> ScalpSignal | None:
        """Check for support/resistance breakout.

        Triggers when price breaks key level with volume confirmation.
        """
        pass

    def check_momentum_burst(self) -> ScalpSignal | None:
        """Check for sudden momentum burst.

        Triggers when:
        - Price velocity exceeds threshold (e.g., 0.5% in 30 seconds)
        - Volume confirms (above average)
        - Direction is clear
        """
        pass
```

### 4. ScalpSignalGenerator

Combines all inputs to generate scalp signals.

```python
@dataclass
class ScalpSignal:
    """A scalping trade signal."""

    id: str
    timestamp: datetime
    symbol: str

    # Signal details
    signal_type: Literal["SCALP_CALL", "SCALP_PUT"]
    trigger: str  # "momentum_burst", "vwap_rejection", "breakout"

    # Entry
    underlying_price: float
    entry_price_target: float  # Option price to enter at

    # Option selection
    option_symbol: str
    strike: float
    expiry: str  # Should be 0DTE or 1DTE
    delta: float

    # Risk management
    take_profit_pct: float = 30.0  # Exit at +30%
    stop_loss_pct: float = 15.0    # Exit at -15%
    max_hold_minutes: int = 15     # Time-based exit

    # Confidence/sizing
    confidence: int  # 0-100
    suggested_contracts: int


@dataclass
class ScalpSignalGenerator:
    """Generates scalping signals from technical analysis."""

    config: ScalpConfig
    technical: TechnicalScalper

    # Rate limiting
    _last_signal_time: datetime | None = None
    _min_signal_interval: float = 60.0  # Minimum seconds between signals

    # Active signal tracking (prevent duplicates)
    _active_signals: set[str] = field(default_factory=set)

    def evaluate(self) -> ScalpSignal | None:
        """Run scalping evaluation.

        Called every 100-200ms during market hours.

        Returns:
            ScalpSignal if conditions met, None otherwise.
        """
        # Check cooldown
        if self._last_signal_time:
            elapsed = (datetime.now() - self._last_signal_time).total_seconds()
            if elapsed < self._min_signal_interval:
                return None

        # Check for signals in priority order
        signal = None

        # 1. Momentum burst (highest conviction)
        signal = self.technical.check_momentum_burst()
        if signal and self._validate_signal(signal):
            return self._create_scalp_signal(signal, "momentum_burst")

        # 2. VWAP rejection
        signal = self.technical.check_vwap_rejection()
        if signal and self._validate_signal(signal):
            return self._create_scalp_signal(signal, "vwap_rejection")

        # 3. Breakout
        signal = self.technical.check_breakout()
        if signal and self._validate_signal(signal):
            return self._create_scalp_signal(signal, "breakout")

        return None

    def _select_option(self, direction: str, underlying_price: float) -> OptionSelection:
        """Select optimal 0DTE option for scalp.

        Criteria:
        - 0DTE or 1DTE expiry (max gamma)
        - Delta ~0.30-0.40 (good leverage without excessive theta)
        - Spread < 5% of option price
        - Sufficient open interest (> 500)
        """
        pass
```

### 5. ScalpConfig

Configuration for the scalping module.

```python
@dataclass(frozen=True)
class ScalpConfig:
    """Scalping module configuration."""

    # Enable/disable
    enabled: bool = False

    # Evaluation frequency
    eval_interval_ms: int = 200  # Evaluate every 200ms

    # Velocity thresholds
    momentum_threshold_pct: float = 0.5  # 0.5% move triggers momentum signal
    momentum_window_seconds: int = 30    # Over 30 second window

    # Volume thresholds
    volume_spike_ratio: float = 2.0  # 2x normal volume = spike
    min_sweep_contracts: int = 100   # Minimum for sweep detection

    # Option selection
    target_delta: float = 0.35       # Target delta for entries
    max_spread_pct: float = 5.0      # Max bid-ask spread %
    min_open_interest: int = 500     # Minimum OI
    prefer_0dte: bool = True         # Prefer same-day expiry

    # Risk management
    take_profit_pct: float = 30.0    # Take profit at +30%
    stop_loss_pct: float = 15.0      # Stop loss at -15%
    max_hold_minutes: int = 15       # Force exit after 15 min
    max_daily_scalps: int = 5        # Max scalp trades per day
    max_concurrent_scalps: int = 1   # Only 1 scalp at a time

    # Position sizing
    scalp_position_size_pct: float = 5.0  # 5% of portfolio per scalp
    max_contract_price: float = 5.0       # Don't buy options over $5

    # Cooldowns
    min_signal_interval_seconds: float = 60.0  # 1 min between signals
    cooldown_after_loss_minutes: float = 5.0   # 5 min cooldown after loss
```

## Signal Flow

```
Every 200ms:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. Update price velocity tracker with latest price                 │
│                    │                                                │
│                    ▼                                                │
│  2. Update volume analyzer with recent trades                       │
│                    │                                                │
│                    ▼                                                │
│  3. Update technical indicators (VWAP, S/R levels)                  │
│                    │                                                │
│                    ▼                                                │
│  4. Check for signals:                                              │
│     ├─ Momentum burst? (price velocity > 0.5% in 30s + volume)      │
│     ├─ VWAP rejection? (touch + bounce with volume)                 │
│     └─ Breakout? (S/R break with volume)                            │
│                    │                                                │
│                    ▼                                                │
│  5. If signal detected:                                             │
│     ├─ Validate (cooldowns, max positions, etc.)                    │
│     ├─ Select 0DTE option (delta ~0.35, tight spread)               │
│     └─ Generate ScalpSignal                                         │
│                    │                                                │
│                    ▼                                                │
│  6. AutoExecutor receives signal → Place order                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Exit Management

Scalp positions have aggressive exit rules:

```python
@dataclass
class ScalpExitManager:
    """Manages exits for scalp positions."""

    def check_exit(self, position: TrackedPosition) -> ExitSignal | None:
        """Check if scalp position should exit.

        Exit triggers (in priority order):
        1. Stop loss hit (-15%)
        2. Take profit hit (+30%)
        3. Time limit exceeded (15 minutes)
        4. Momentum reversed (optional)
        """

        # Calculate current P&L
        pnl_pct = position.unrealized_pnl_pct

        # Stop loss (highest priority)
        if pnl_pct <= -self.config.stop_loss_pct:
            return ExitSignal(
                reason="STOP_LOSS",
                message=f"Stop loss triggered at {pnl_pct:.1f}%"
            )

        # Take profit
        if pnl_pct >= self.config.take_profit_pct:
            return ExitSignal(
                reason="TAKE_PROFIT",
                message=f"Take profit triggered at {pnl_pct:.1f}%"
            )

        # Time-based exit
        hold_minutes = (datetime.now() - position.opened_at).total_seconds() / 60
        if hold_minutes >= self.config.max_hold_minutes:
            return ExitSignal(
                reason="TIME_EXIT",
                message=f"Max hold time ({self.config.max_hold_minutes}m) exceeded"
            )

        return None
```

## Integration Points

### 1. Main.py Changes

```python
# Add scalp evaluation loop (faster than regime loop)
async def scalp_evaluation_loop():
    """Fast evaluation loop for scalping signals."""
    while True:
        try:
            await asyncio.sleep(0.2)  # 200ms

            if not scalp_config.enabled:
                continue

            if not is_market_hours():
                continue

            signal = scalp_signal_generator.evaluate()
            if signal:
                # Broadcast to frontend
                await connection_manager.broadcast({
                    "type": "scalp_signal",
                    "data": signal.to_dict(),
                })

                # Auto-execute if enabled
                if auto_executor and auto_executor.enabled:
                    await auto_executor.execute_scalp(signal)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in scalp evaluation: {e}")
```

### 2. AutoExecutor Changes

```python
class AutoExecutor:
    async def execute_scalp(self, signal: ScalpSignal) -> bool:
        """Execute a scalp signal with tighter risk management."""

        # Check scalp-specific limits
        if self._daily_scalp_count >= self.config.max_daily_scalps:
            logger.info("Max daily scalps reached")
            return False

        if self._active_scalp_count >= self.config.max_concurrent_scalps:
            logger.info("Max concurrent scalps reached")
            return False

        # Place order with scalp-specific sizing
        # ... (similar to regime execution but with scalp config)
```

### 3. Frontend Changes

- New "Scalp Signals" section in dashboard
- Real-time velocity/momentum indicator
- Faster P&L updates for scalp positions
- Visual countdown for time-based exits

## Implementation Phases

### Phase 1: Foundation (MVP)
- [ ] PriceVelocityTracker with basic spike detection
- [ ] ScalpConfig with environment variable support
- [ ] Simple momentum burst signal (price velocity only)
- [ ] 0DTE option selection logic
- [ ] Integration with AutoExecutor

### Phase 2: Volume Analysis
- [ ] VolumeAnalyzer with baseline tracking
- [ ] Volume-confirmed momentum signals
- [ ] Sweep detection (optional)

### Phase 3: Technical Patterns
- [ ] VWAP calculation and tracking
- [ ] Support/resistance level detection
- [ ] VWAP rejection signals
- [ ] Breakout signals

### Phase 4: Refinement
- [ ] Backtesting framework for scalp strategies
- [ ] Parameter optimization
- [ ] Frontend enhancements
- [ ] Performance tuning (evaluation speed)

## Risk Considerations

1. **Slippage** - Fast moves mean fills may be worse than expected
2. **Overtrading** - Easy to generate too many signals; strict cooldowns needed
3. **Theta decay** - 0DTE options lose value rapidly; time exits critical
4. **Liquidity** - Even TSLA 0DTE can have moments of poor liquidity
5. **Emotional trading** - Scalping is psychologically demanding
6. **PDT rules** - Day trading restrictions apply if account < $25k

## Success Metrics

- Win rate > 55% (scalping needs higher due to costs)
- Average winner > average loser (R:R > 1.5)
- Sharpe ratio of scalp trades
- Max drawdown per day
- Execution slippage analysis
