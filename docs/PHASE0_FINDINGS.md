# Phase 0: Codebase Inspection Findings

**Date:** 2026-01-05
**Purpose:** Identify reusable components before implementing scalping module

---

## 1. OCC Symbol Parsing - ALREADY EXISTS

**Location:** `backend/models/canonical.py`

No need to create `occ_parser.py` - the codebase already has OCC parsing:

```python
from backend.models.canonical import CanonicalOptionId, parse_occ, parse_alpaca

# parse_occ("TSLA250117C00420000") returns:
CanonicalOptionId(
    underlying="TSLA",
    expiry="2025-01-17",  # ISO format
    right="C",
    strike=420.0
)
```

**Functions available:**
- `parse_occ(symbol)` - Parse OCC format (e.g., "TSLA250117C00420000")
- `parse_alpaca(symbol)` - Parse Alpaca format (same as OCC for options)
- `CanonicalOptionId.to_occ()` - Convert back to OCC string

---

## 2. Market Data Models - REUSABLE

**Location:** `backend/models/market_data.py`

All models we need for scalping already exist:

| Model | Key Fields | Use Case |
|-------|------------|----------|
| `QuoteData` | bid, ask, bid_size, ask_size, last, timestamp | Real-time quotes |
| `TradeData` | price, size, timestamp, exchange, conditions | Volume tracking |
| `GreeksData` | delta, gamma, theta, vega, iv | Option selection |
| `UnderlyingData` | symbol, price, iv_rank, timestamp | Underlying tracking |

**Useful properties:**
- `QuoteData.mid` - Mid price
- `QuoteData.spread_percent` - Spread as % of mid
- `*.age_seconds()` - All models have freshness check

---

## 3. Reusable Gates - IMPORT DIRECTLY

**Location:** `backend/engine/gates.py`

The following gates can be reused for scalping liquidity checks:

| Gate | Threshold | Purpose |
|------|-----------|---------|
| `SpreadAcceptableGate` | ≤ 10% | Spread check |
| `OpenInterestSufficientGate` | ≥ 500 | Liquidity |
| `VolumeSufficientGate` | ≥ 100 | Activity |
| `QuoteFreshGate` | ≤ 5s | Data freshness |
| `UnderlyingPriceFreshGate` | ≤ 5s | Data freshness |

**GateContext dataclass** (lines 52-176) provides:
- `underlying_price`, `option_quote`
- `open_interest`, `volume`
- `bid`, `ask`, `spread_pct`
- Timestamps for freshness

For scalping, can create `ScalpGateContext` inheriting or adapting from this.

---

## 4. Position Tracker - EXTENDABLE

**Location:** `backend/engine/position_tracker.py`

**Current TrackedPosition fields:**
```python
@dataclass
class TrackedPosition:
    id: str
    recommendation_id: str
    underlying: str
    expiry: str
    strike: float
    right: str
    action: str  # "BUY_CALL", "BUY_PUT"
    contracts: int
    fill_price: float
    entry_cost: float
    current_price: float | None
    pnl: float
    pnl_percent: float
    status: Literal["open", "closed", "exit_signal"]
    # ... more fields
```

**Extension needed:** Add `position_type: Literal["regime", "scalp"]` field

**Current Exit Config:**
```python
@dataclass
class PositionTrackerConfig:
    take_profit_percent: float = 40.0   # Regime: +40%
    stop_loss_percent: float = -20.0    # Regime: -20%
    min_dte_exit: int = 1
    max_positions: int = 3
```

**For scalping, need separate config:**
```python
@dataclass
class ScalpExitConfig:
    take_profit_percent: float = 30.0   # Scalp: +30%
    stop_loss_percent: float = -15.0    # Scalp: -15%
    max_hold_minutes: int = 15          # Time-based exit (no DTE check)
```

**SQLite schema** needs update for `position_type` column.

---

## 5. Data Client Patterns - FOLLOW CONVENTION

**Location:** `backend/data/alpaca_client.py`

Key patterns to follow:

1. **Dataclass-based client:**
```python
@dataclass
class AlpacaOptionsClient:
    config: AlpacaConfig
    on_quote: Callable[[QuoteData], None] | None = None
    on_trade: Callable[[TradeData], None] | None = None
```

2. **Callback pattern for async data:**
```python
# Set callbacks before connecting
client.on_quote = handle_quote
await client.connect()
```

3. **Connection state management:**
```python
class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
```

4. **Kill switch for repeated failures:**
```python
KILL_SWITCH_FAILURE_THRESHOLD = 5
KILL_SWITCH_WINDOW_SECONDS = 300  # 5 minutes
```

---

## 6. WebSocket Manager - SCALP MESSAGE TYPE NEEDED

**Location:** `backend/websocket/manager.py`

Current message types:
```python
class MessageType(str, Enum):
    OPTION_UPDATE = "option_update"
    UNDERLYING_UPDATE = "underlying_update"
    GATE_STATUS = "gate_status"
    ABSTAIN = "abstain"
```

**Need to add:**
- `SCALP_SIGNAL = "scalp_signal"`
- `SCALP_EXIT = "scalp_exit"`
- `VELOCITY_UPDATE = "velocity_update"` (optional, for UI)

---

## 7. Config Pattern - ADD SCALP CONFIG

**Location:** `backend/config.py`

Follow existing pattern:

```python
@dataclass(frozen=True)
class ScalpConfig:
    enabled: bool = False
    eval_interval_ms: int = 200
    momentum_threshold_pct: float = 0.5
    # ... etc from design doc
```

Load from environment:
```python
scalp=ScalpConfig(
    enabled=_get_env_bool("SCALP_ENABLED", default=False),
    # ...
)
```

---

## Summary: What to Create vs Reuse

### Reuse (import directly):
- `backend/models/canonical.py` - OCC parsing
- `backend/models/market_data.py` - QuoteData, TradeData, etc.
- `backend/engine/gates.py` - Spread, OI, Volume, Freshness gates
- `backend/data/alpaca_client.py` - Follow callback patterns

### Extend (modify existing):
- `backend/engine/position_tracker.py` - Add `position_type` field
- `backend/config.py` - Add `ScalpConfig`
- `backend/websocket/manager.py` - Add scalp message types

### Create new:
- `backend/engine/scalping/` - New module directory
  - `velocity_tracker.py` - Price velocity tracking
  - `volume_analyzer.py` - Volume analysis
  - `technical_scalper.py` - VWAP, S/R detection
  - `signal_generator.py` - Scalp signal generation
  - `exit_manager.py` - Scalp-specific exits

---

## Next Steps: Phase 1

With Phase 0 complete, proceed to Phase 1 (DataBento Integration):
1. Create `backend/data/databento_loader.py`
2. Parse CBBO-1m files from 135GB dataset
3. Build in-memory quote/trade replay system
4. Test with small date range first
