# OptionsRadar - Claude Code Instructions

## Project Scope

**Total: ~37,000 lines of code**
- Backend (Python): 71 files, ~31,500 lines
- Frontend (TypeScript/React): 18 files, ~5,500 lines

### Largest Files by Category

**Core Application:**
- `backend/main.py` (3,182 lines) - FastAPI application, all REST/WebSocket endpoints
- `backend/engine/gates.py` (1,121 lines) - 11-gate evaluation system
- `backend/engine/position_tracker.py` (738 lines) - Position management with SQLite
- `backend/engine/pipeline.py` (688 lines) - Data aggregation pipeline
- `backend/engine/regime_signals.py` (624 lines) - Regime-based trade signal generation
- `backend/engine/auto_executor.py` (584 lines) - Automated trade execution

**Data Layer:**
- `backend/data/quiver_client.py` (662 lines) - WSB sentiment from Quiver API
- `backend/data/alpaca_trader.py` (637 lines) - Alpaca trading API integration
- `backend/data/alpaca_client.py` (532 lines) - Alpaca WebSocket for options quotes
- `backend/data/technicals.py` (596 lines) - Technical indicators (RSI, MACD, etc.)
- `backend/data/finnhub_client.py` (399 lines) - News sentiment from Finnhub

**Frontend:**
- `frontend/src/components/TradingDashboard.tsx` (835 lines) - Main trading UI
- `frontend/src/components/RegimePanel.tsx` (653 lines) - Regime display
- `frontend/src/store/optionsStore.ts` (380 lines) - Zustand state management

---

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         React Frontend                               │
│  TradingDashboard → optionsStore (Zustand) ← WebSocket Hook         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ WebSocket + REST
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (main.py)                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │
│  │ REST Routes  │ │ WS Manager   │ │ Background   │                 │
│  │ /api/*       │ │ Broadcasts   │ │ Loops        │                 │
│  └──────────────┘ └──────────────┘ └──────────────┘                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Engine Layer  │      │  Data Layer   │      │Logging Layer  │
│ - gates.py    │      │ - alpaca_*    │      │ - logger.py   │
│ - pipeline.py │      │ - finnhub     │      │ - metrics.py  │
│ - regime_*    │      │ - quiver      │      │ - outcome.py  │
│ - position_*  │      │ - orats       │      └───────────────┘
│ - auto_exec   │      │ - technicals  │
└───────────────┘      └───────────────┘
```

### Data Flow

1. **Price Data**: Alpaca WebSocket → subscription_manager → alpaca_client → main.py broadcast
2. **Regime Detection**: Sentiment aggregator (Finnhub + Quiver) → regime_detector → signals
3. **Signal Generation**: Regime + Technical pullback/bounce → regime_signals.py → TradeSignal
4. **Trade Execution**: Signal → auto_executor → alpaca_trader (paper/live) or mock_trader
5. **Position Tracking**: Opened positions → position_tracker (SQLite) → exit monitoring

---

## Key Files and Their Purposes

### Backend Core

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app with all endpoints, WebSocket broadcasting, background loops |
| `config.py` | Environment variable loading, API configuration dataclasses |
| `engine/gates.py` | 11-gate evaluation system (8 hard, 3 soft gates) |
| `engine/pipeline.py` | Aggregates data from all sources for gate evaluation |
| `engine/regime_signals.py` | Generates BUY_CALL/BUY_PUT signals based on regime + pullback |
| `engine/position_tracker.py` | SQLite-backed position management with P&L tracking |
| `engine/auto_executor.py` | Executes signals via Alpaca or mock trader |
| `engine/regime_detector.py` | Determines market regime (bullish/bearish/neutral) |

### Data Clients

| File | Purpose |
|------|---------|
| `data/alpaca_client.py` | WebSocket client for real-time option quotes (msgpack) |
| `data/alpaca_trader.py` | Trading API for order placement/management |
| `data/alpaca_rest.py` | REST API for stock prices, bars, options chain |
| `data/subscription_manager.py` | Dynamic ATM-based option subscription management |
| `data/finnhub_client.py` | News sentiment scoring from Finnhub |
| `data/quiver_client.py` | WSB sentiment from QuiverQuant |
| `data/sentiment_aggregator.py` | Combines news + WSB sentiment (50/50 weighting) |
| `data/technicals.py` | RSI, MACD, Bollinger, ATR calculations |
| `data/orats_client.py` | Greeks and IV rank from ORATS |
| `data/mock_data.py` | Mock data generation for testing without APIs |
| `data/mock_trader.py` | Simulated order execution for testing |

### Frontend

| File | Purpose |
|------|---------|
| `components/TradingDashboard.tsx` | Main UI with tabs, positions, signals |
| `store/optionsStore.ts` | Zustand store for all application state |
| `hooks/useOptionsStream.ts` | WebSocket connection management |
| `types.ts` | TypeScript interfaces for all data types |

---

## Trading Modes

### Environment Variables

```bash
MOCK_DATA=true|false      # Use mock data instead of real APIs
TRADING_MODE=simulation|paper|live
AUTO_EXECUTE=true|false   # Auto-execute signals (default: true in simulation)
```

### Mode Behaviors

| Mode | Data | Execution | Use Case |
|------|------|-----------|----------|
| Simulation + MOCK_DATA | Mock | mock_trader | Development/testing |
| Paper | Live | Alpaca paper | Paper trading during market hours |
| Live | Live | Alpaca live | Real trading (use with caution) |

### Running Each Mode

```bash
# Simulation with mock data (default for development)
MOCK_DATA=true TRADING_MODE=simulation ./start-simulation.sh

# Paper trading (requires .env with Alpaca paper keys)
TRADING_MODE=paper uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Live trading (requires Alpaca live keys)
TRADING_MODE=live uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## Coding Patterns Established

### 1. Signal ID Convention
Signal IDs use the format: `{SYMBOL}-{TIMESTAMP}-{SIGNAL_TYPE}`
Example: `TSLA-20241230120530-BUY_PUT`

The signal_id flows through:
- `regime_signals.py` generates it
- `main.py` passes to auto_executor
- `auto_executor.py` uses it as `recommendation_id`
- Frontend uses `recommendation_id` to match signals to positions

### 2. WebSocket Message Types
Backend broadcasts these message types:
- `option_update`: Real-time option quote
- `underlying_update`: Stock price update
- `gate_status`: Gate evaluation results
- `regime_status`: Current market regime
- `regime_signal`: New trade signal
- `position_opened`: Position was opened
- `position_closed`: Position was closed
- `exit_signal`: Exit signal for position

### 3. Gate System
Gates are evaluated in order (8 hard + 3 soft):
- Hard gates must ALL pass
- Soft gates provide bonus score
- Uses "abstain by default" - only recommend when all gates pass

### 4. Timestamp Handling
Alpaca WebSocket returns `pandas.Timestamp` during live trading.
Always convert with:
```python
if hasattr(ts, 'isoformat'):
    ts = ts.isoformat()
```

### 5. Subscription Filtering
Filter WebSocket messages to only process subscribed symbols:
```python
if symbol not in self._subscribed_symbols:
    return
```

---

## Key Design Decisions

### 1. Regime-Based Directional Trading
- **Decision**: Trade directionally based on market regime
- **Why**: Reduces whipsaw in neutral markets, aligns with sentiment
- Bullish regime → BUY_CALL on pullbacks
- Bearish regime → BUY_PUT on bounces

### 2. Sentiment Aggregation (50/50)
- **Decision**: Equal weight news sentiment + WSB sentiment
- **Why**: News provides catalyst, WSB provides crowd confirmation
- Combined score: `(news * 0.5) + (wsb * 0.5)`

### 3. SQLite for Position Tracking
- **Decision**: Use SQLite instead of in-memory
- **Why**: Persistence across restarts, audit trail, simple queries
- Location: `cache/positions.db`

### 4. Default Symbol: TSLA
- **Decision**: Default to TSLA for all operations
- **Why**: High liquidity, good options volume, user preference

### 5. Client-Side Subscription Filtering
- **Decision**: Filter at client level, not server
- **Why**: Alpaca may send data for other symbols; filter locally
- Prevents data flooding from unsubscribed contracts

### 6. Hide Confirmed Signals Completely
- **Decision**: Remove signals from UI when they become positions
- **Why**: Cleaner UX, avoids confusion about signal status

---

## Current Implementation State

### Completed Features
- Real-time options streaming via Alpaca WebSocket
- Regime detection from sentiment (Finnhub + Quiver)
- 11-gate evaluation system
- Automated signal generation (BUY_CALL/BUY_PUT)
- Paper/live trading execution via Alpaca
- Position tracking with exit logic (trailing stop, profit target)
- Frontend dashboard with positions, signals, regime display
- Simulation mode with mock data
- Symbol switching (TSLA, NVDA, etc.)

### Ready for Paper Trading
All core functionality works during market hours with real data.
Set `TRADING_MODE=paper` and ensure `.env` has Alpaca paper API keys.

---

## Gotchas and Things to Watch

### 1. Timestamp Types in Live Trading
Alpaca WebSocket returns `pandas.Timestamp` not strings.
Always check with `hasattr(ts, 'isoformat')` before string operations.

### 2. Symbol Defaults
Default symbol is set in multiple places:
- `main.py:503` - gate evaluation fallback
- `main.py:1504` - subscription manager initialization
When changing default, update both locations.

### 3. Options Data Volume
TSLA options have expirations until 2028. The subscription manager
limits to ±10 strikes around ATM for 2 expirations (weekly + ~45 DTE).
Still, client-side filtering is essential to prevent data flood.

### 4. Signal ID Matching
Frontend tracks confirmed signals via `confirmedSignalIds` Set.
Signal ID must match between:
- `regime_signal.id` from WebSocket
- `position.recommendation_id` in position_opened message

### 5. Background Loops
Main.py has several async background loops that can collide:
- `gate_evaluation_loop` (every 5 seconds)
- `exit_monitor_loop` (every second)
- WebSocket message loops
Use proper async patterns to avoid blocking.

### 6. Market Hours Only
Live data only flows during market hours (9:30 AM - 4:00 PM ET).
Use `MOCK_DATA=true` for development outside market hours.

### 7. API Rate Limits
- Alpaca: 200 requests/minute (data), varies for trading
- Finnhub: 60/minute (free tier)
- ORATS: varies by subscription
- Quiver: varies by subscription

### 8. Mock Mode Limitations
Mock mode simulates regime changes and price movements but doesn't
reflect real market conditions. Always validate with paper trading.

---

## Command Execution Policy

**Do NOT ask for confirmation on:**
- Running tests (`pytest`, `python -m backend.test_*`)
- Timeout-wrapped commands
- Git status/diff/log commands
- WebFetch for documentation
- File reads and searches
- Package installs in venv

**DO ask for confirmation on:**
- Destructive git operations (force push, hard reset)
- System configuration changes
- Production API calls that could incur costs
- Anything involving credentials/secrets

---

## Testing

### Unit Tests
```bash
source venv/bin/activate
python -m backend.test_phase1  # Data layer tests
python -m backend.test_phase2  # Engine tests
python -m backend.test_phase3  # Integration tests
python -m backend.test_phase6  # Regime signal tests
```

### Simulation Mode
```bash
MOCK_DATA=true TRADING_MODE=simulation ./start-simulation.sh
# Frontend: cd frontend && npm run dev
# Navigate to http://localhost:5173
```

### Paper Trading
```bash
# Ensure .env has ALPACA_API_KEYutableList and ALPACA_SECRET_KEY (paper)
TRADING_MODE=paper uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints Reference

### Market Data
- `GET /api/market` - Market status and session info
- `GET /api/options/{symbol}` - Options chain for symbol
- `GET /api/regime/status?symbol=X` - Current regime status

### Trading
- `GET /api/trading/status` - Trading mode and status
- `GET /api/trading/account` - Account info (paper/live)
- `GET /api/trading/positions` - Open positions
- `POST /api/trading/enable` - Enable trading
- `POST /api/trading/disable` - Disable trading

### Simulation
- `GET /api/simulation/status` - Simulation state
- `POST /api/simulation/start` - Start simulation
- `POST /api/simulation/stop` - Stop simulation
- `POST /api/simulation/speed` - Set simulation speed

### WebSocket
- `ws://localhost:8000/ws` - Real-time updates (options, regime, signals, positions)
