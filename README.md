# OptionsRadar

A real-time options trading system with two complementary strategies:

1. **Sentiment-Based Regime Trading** - Multi-symbol strategy using news + WSB sentiment to determine market regime and generate directional signals on pullbacks/bounces
2. **Momentum Scalping (TSLA)** - High-frequency intraday strategy using price velocity and volume spikes to catch momentum bursts on 1-3 DTE options

## Quick Start

```bash
# Clone and setup
git clone https://github.com/seth-zapata/options-radar.git
cd options-radar
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Configure credentials
cp .env.example .env
# Edit .env with your API keys (see Credentials section)

# Run with the unified CLI launcher
python run.py --mode mock              # Mock data, no trading
python run.py --mode paper --scalping  # Paper trading with scalping
python run.py --mode simulation        # Fast testing with simulated trades

# Or use individual scripts
./start-mock.sh      # Terminal 1: Backend with mock data
./start-frontend.sh  # Terminal 2: Frontend
```

Open http://localhost:5173

---

## Scalping Module (TSLA Momentum Strategy)

The scalping module is a high-frequency momentum strategy optimized for TSLA options. It uses sub-second price velocity detection to catch sharp moves and exit quickly.

### Strategy Overview

| Component | Details |
|-----------|---------|
| **Symbol** | TSLA (high liquidity, volatile) |
| **DTE Range** | 1-3 DTE (never 0DTE - too much gamma risk) |
| **Signal Type** | Momentum burst (velocity threshold crossed) |
| **Entry** | Near-ATM options (0.25-0.55 delta) |
| **Exit** | Take profit (+20%), Stop loss (-15%), Momentum reversal, Time stop (5 min), Market close |

### Asymmetric Thresholds

Based on backtest analysis, PUT and CALL signals use different thresholds:

| Direction | Threshold | Window | Reasoning |
|-----------|-----------|--------|-----------|
| **PUT** | 0.4% velocity | 30s | Panic drops fast - catch early |
| **CALL** | 0.6% velocity | 30s | Rallies slower - need confirmation |

### Running Scalping Mode

```bash
# Paper trading with scalping enabled
python run.py --mode paper --scalping

# What you'll see in logs:
# [SCALP-DIAG] TSLA $425.50 | velocity=+0.52% (45pts/30s)
#   PUT 0.4%: n/a (wrong direction) | CALL 0.6%: no
# [SCALP-TRIGGER] TSLA CALL momentum=+0.65% (threshold=0.6%, margin=+0.05%, 12 options available)
# [SCALP-EXEC] Position opened: SCALP_CALL 5x TSLA250117C00426000 @ $2.15
```

### Trade Logging

Trades are logged to `scalp_trades/scalp_trades_YYYY-MM-DD.json` with:
- Entry/exit details (price, time, reason)
- Momentum metrics (velocity, threshold, margin)
- Market period (open/mid/close)
- Summary stats (win rate by signal type, exit reason, time of day)

### Backtesting

```bash
# Run backtest on DataBento options data
python run.py --backtest /path/to/databento/data --start 2024-01-01 --end 2024-12-31

# Parallel processing (4-8x faster)
python run.py --backtest /path/to/data --parallel --workers 8

# Full year with quarterly breakdown
python scripts/run_full_year_backtest.py 2024
```

### Key Configuration (ScalpConfig)

```python
# backend/scalping/config.py
ScalpConfig(
    enabled=True,
    momentum_threshold_put_pct=0.4,   # PUT trigger threshold
    momentum_threshold_call_pct=0.6,  # CALL trigger threshold
    momentum_window_seconds=30,       # Velocity calculation window
    min_dte=1,                        # Never 0DTE
    max_dte=3,                        # Up to 3 DTE
    take_profit_pct=20.0,             # Exit at +20%
    stop_loss_pct=15.0,               # Exit at -15%
    time_stop_minutes=5,              # Exit stalled trades
    max_contract_price=3.00,          # Cheap options win more
)
```

---

## Sentiment-Based Strategy (Original)

The original regime-based strategy uses news sentiment (Finnhub) and WSB social sentiment (Quiver) to determine market regime and generate signals.

### Features

- **Real-time Options Chain** - Live streaming quotes from Alpaca with Greeks from ORATS
- **Gating Engine** - 11 gates (8 hard, 3 soft) that must pass before any recommendation
- **Sentiment Integration** - News sentiment (Finnhub) + WSB social sentiment (Quiver)
- **Daily Scanner** - Identifies opportunities based on combined sentiment signals
- **Shadow Mode** - Tracks hypothetical positions without executing trades
- **Abstain by Default** - Only recommends when ALL conditions are optimal

### Regime Detection

| Regime | Sentiment Score | Action |
|--------|-----------------|--------|
| Bullish | > 0.3 | BUY_CALL on pullbacks |
| Bearish | < -0.3 | BUY_PUT on bounces |
| Neutral | -0.3 to 0.3 | No trades (abstain) |

### IV Rank "Extremes" Framework

**Novel Finding:** Traditional IV Rank logic (favor low IV, avoid high IV) doesn't apply to sentiment-driven momentum stocks. Our backtest of 111 signals revealed an "extremes" pattern:

```
IV RANK ANALYSIS (111 signals, Jan 2024 - Dec 2024)

  Traditional Framework (Premium Sellers):
    IV Rank <= 45%: 67.1% accuracy (73 signals)
    IV Rank >  45%: 65.8% accuracy (38 signals)
    Edge: +1.3% (marginal)

  Extremes Framework (Directional Momentum Buyers):
    IV Rank < 30%:  68.3% accuracy (60 signals)  --> BEST
    IV Rank > 60%:  67.7% accuracy (33 signals)  --> GOOD
    IV Rank 30-60%: 55.6% accuracy (18 signals)  --> WORST
```

| IV Rank | Modifier | Reasoning |
|---------|----------|-----------|
| **< 30%** | +5 confidence | Cheap premium, clear value |
| **> 60%** | +5 if strong sentiment (>0.3) | Retail excitement confirmed |
| **30-60%** | -5 confidence | "No man's land" - worst performance |

---

## Credentials

| Service | Variables | Required | Cost | Purpose |
|---------|-----------|----------|------|---------|
| Alpaca | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Yes | $99/mo (Algo Trader Plus) | Real-time option quotes, trading |
| ORATS | `ORATS_API_TOKEN` | Yes | $199/mo (Live) | Greeks, IV rank |
| Finnhub | `FINNHUB_API_KEY` | Optional | $50/mo | News sentiment |
| Quiver | `QUIVER_API_KEY` | Optional | $10/mo | WSB social sentiment |
| DataBento | `DATABENTO_API_KEY` | Optional | Pay-per-use | Historical options data (backtesting) |

**Getting credentials:**
- **Alpaca**: https://app.alpaca.markets/ (enable Options + Algo Trader Plus for OPRA data)
- **ORATS**: https://orats.com/ (Live subscription for real-time Greeks)
- **Finnhub**: https://finnhub.io/ (Fundamental-1 tier for news sentiment API)
- **Quiver**: https://www.quiverquant.com/ (any paid tier for WSB data)
- **DataBento**: https://databento.com/ (for historical options tick data)

---

## Project Structure

```
options-radar/
├── run.py                    # Unified CLI launcher
├── backend/
│   ├── main.py               # FastAPI app, WebSocket, background loops
│   ├── config.py             # Environment configuration
│   ├── models/               # Data models (canonical IDs, quotes, Greeks)
│   ├── data/                 # Data clients
│   │   ├── alpaca_*.py       # Alpaca streaming + trading
│   │   ├── orats_client.py   # ORATS Greeks
│   │   ├── finnhub_client.py # News sentiment
│   │   ├── quiver_client.py  # WSB sentiment
│   │   └── databento_loader.py # Historical data loading
│   ├── engine/               # Regime strategy logic
│   │   ├── gates.py          # 11 trading gates
│   │   ├── pipeline.py       # Gate orchestration
│   │   ├── regime_signals.py # Signal generation
│   │   └── position_tracker.py # SQLite position tracking
│   └── scalping/             # Scalping module
│       ├── config.py         # ScalpConfig with thresholds
│       ├── signal_generator.py # Momentum signal detection
│       ├── scalp_executor.py # Trade execution + exit monitoring
│       ├── velocity_tracker.py # Price velocity calculation
│       ├── volume_analyzer.py  # Volume spike detection
│       ├── scalp_backtester.py # Backtesting engine
│       └── replay.py         # Historical data replay
├── frontend/                 # React UI
│   ├── src/components/       # TradingDashboard, RegimePanel, etc.
│   └── src/store/            # Zustand state management
├── scripts/                  # Utility scripts
│   ├── fetch_random_weeks.py # Fetch test data
│   ├── fetch_second_bars.py  # Fetch 1-second bars
│   └── run_full_year_backtest.py # Full year backtest
├── docs/                     # Documentation
└── scalp_trades/             # Trade logs (generated)
```

---

## Trading Modes

| Mode | Data | Execution | Use Case |
|------|------|-----------|----------|
| `mock` | Mock | None | UI development |
| `simulation` | Mock | mock_trader | Fast testing |
| `paper` | Live | Alpaca paper | Paper trading (market hours) |
| `live` | Live | Alpaca live | Real trading (careful!) |

```bash
# Using run.py
python run.py --mode mock
python run.py --mode simulation --speed 10  # 10x speed
python run.py --mode paper --scalping
python run.py --mode live  # Requires confirmation

# Additional options
python run.py --mode paper --backend-only  # No frontend
python run.py --mode paper --port 8080     # Custom port
```

---

## Testing

```bash
# Phase tests
python -m backend.test_phase1  # Data layer
python -m backend.test_phase2  # Gating engine
python -m backend.test_phase3  # Integration
python -m backend.test_phase6  # Sentiment

# Unit tests
pytest

# Type checking
mypy backend

# Linting
ruff check backend
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         React Frontend                            │
│  TradingDashboard → optionsStore (Zustand) ← WebSocket Hook       │
└───────────────────────────────┬───────────────────────────────────┘
                                │ WebSocket + REST
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (main.py)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
│  │   Regime     │  │   Scalping   │  │   Position Management   │  │
│  │  Strategy    │  │   Module     │  │ (SQLite + Exit Monitor) │  │
│  │ (Sentiment)  │  │ (Momentum)   │  │                         │  │
│  └──────────────┘  └──────────────┘  └─────────────────────────┘  │
│                              │                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                        Data Layer                           │  │
│  │  Alpaca (Quotes/Trading) + ORATS (Greeks) + Sentiment APIs  │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run backend with hot reload
python run.py --mode mock --reload

# Run frontend dev server
cd frontend && npm run dev
```

## Watchlist

Default watchlist for regime strategy (configurable in backend/config.py):
- NVDA, QQQ, AAPL, TSLA, SPY, AMD, GOOGL, AMZN, META, MSFT

Scalping is currently focused on **TSLA only** due to its high liquidity and volatility.
