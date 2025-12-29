# OptionsRadar

A local, display-only options trading recommendation system for a curated watchlist of tech/AI stocks.

## Features

- **Real-time Options Chain** - Live streaming quotes from Alpaca with Greeks from ORATS
- **Gating Engine** - 11 gates (8 hard, 3 soft) that must pass before any recommendation
- **Sentiment Integration** - News sentiment (Finnhub) + WSB social sentiment (Quiver)
- **Daily Scanner** - Identifies opportunities based on combined sentiment signals
- **Shadow Mode** - Tracks hypothetical positions without executing trades
- **Abstain by Default** - Only recommends when ALL conditions are optimal

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

# Run with mock data (no API keys needed)
./start-mock.sh      # Terminal 1: Backend
./start-frontend.sh  # Terminal 2: Frontend

# Open http://localhost:5173
```

## Credentials

| Service | Variables | Required | Cost | Purpose |
|---------|-----------|----------|------|---------|
| Alpaca | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Yes | $99/mo (Algo Trader Plus) | Real-time option quotes, open interest |
| ORATS | `ORATS_API_TOKEN` | Yes | $199/mo (Live) | Greeks, IV rank (live only) |
| Finnhub | `FINNHUB_API_KEY` | Optional | $50/mo (Fundamental-1) | News sentiment |
| Quiver | `QUIVER_API_KEY` | Optional | $10/mo | WSB social sentiment |
| EODHD | `EODHD_API_KEY` | Optional | $80/mo (All-In-One) | Historical options data (backtesting) |

**Getting credentials:**
- **Alpaca**: https://app.alpaca.markets/ (enable Options + Algo Trader Plus for OPRA data)
- **ORATS**: https://orats.com/ (Live subscription for real-time Greeks) - see validation below
- **Finnhub**: https://finnhub.io/ (Fundamental-1 tier for news sentiment API)
- **Quiver**: https://www.quiverquant.com/ (any paid tier for WSB data)
- **EODHD**: https://eodhd.com/ (All-In-One tier for historical options chains)

### IV Rank Subscription Validation

The $199/mo ORATS subscription provides IV Rank for live signals. We validated whether this filter improves signal quality using EODHD historical data:

```
IV RANK VALIDATION (111 signals, Jan 2024 - Dec 2024)
  IV Rank <= 45% (PASS gate): 67.1% accuracy (73 signals)
  IV Rank >  45% (FAIL gate): 65.8% accuracy (38 signals)
  IV RANK FILTER EDGE: +1.3%
  --> VERDICT: IV Rank filter provides marginal edge. ORATS optional.
```

The IV Rank gate blocks signals when IV Rank > 45%, providing a small +1.3% edge. This marginal improvement may not justify the $199/mo ORATS cost for most users. Consider using EODHD historical IV data as an alternative.

## Testing

### Phase 1: Data Foundation

```bash
python -m backend.test_phase1
```

Tests Alpaca WebSocket streaming and ORATS Greeks fetching.

### Phase 2: Gating Engine

```bash
python -m backend.test_phase2
```

Tests gate framework (11 gates), pipeline orchestration, and abstain generation.

### Phase 3: MVP UI

```bash
python -m backend.test_phase3
```

Tests FastAPI endpoints, WebSocket manager, and React frontend structure.

### Phase 6: Sentiment & Scanner

```bash
# With live API calls (requires Finnhub + Quiver API keys)
python -m backend.test_phase6

# Mock mode (no API keys needed)
MOCK_DATA=true python -m backend.test_phase6
```

Tests:
- Finnhub news sentiment
- Quiver WSB sentiment
- Combined 50/50 weighted aggregation
- Daily opportunity scanner
- Alpaca portfolio integration
- Sentiment gates (Direction, RetailMomentum, Convergence)

**Expected output (live mode):**
```
TEST: Finnhub News Sentiment
  Fetching news sentiment for NVDA...
  Sentiment Score: 72.3
  Bullish %: 86%
  Relative Buzz: 1.24x

TEST: Quiver WSB Sentiment
  Fetching WSB sentiment for NVDA...
  Mentions (24h): 1423
  Sentiment Score: 40.0
  Rank: 7

TEST: Combined Sentiment Aggregator
  Combined Score: 56.2 (50/50 weighted)
  Signal: bullish (moderate)

TEST: Daily Opportunity Scanner
  NVDA: Score=55, Direction=bullish
    - Moderate sentiment (56)
    - WSB trending bullish
```

## Running the Full System

**Option 1: Mock Data (UI Development)**
```bash
# Terminal 1
./start-mock.sh

# Terminal 2
./start-frontend.sh
```

**Option 2: Live Data (Production)**
```bash
# Terminal 1
source venv/bin/activate
uvicorn backend.main:app --reload

# Terminal 2
./start-frontend.sh
```

Then open http://localhost:5173

## Project Structure

```
options-radar/
├── backend/
│   ├── models/          # Data models (canonical IDs, quotes, Greeks)
│   ├── data/            # Data clients
│   │   ├── alpaca_*.py  # Alpaca streaming + account
│   │   ├── orats_*.py   # ORATS Greeks
│   │   ├── finnhub_*.py # News sentiment
│   │   ├── quiver_*.py  # WSB sentiment
│   │   └── sentiment_aggregator.py
│   ├── engine/          # Core logic
│   │   ├── gates.py     # 11 trading gates
│   │   ├── pipeline.py  # Gate orchestration
│   │   ├── recommender.py
│   │   ├── scanner.py   # Daily opportunity scanner
│   │   ├── session_tracker.py
│   │   └── position_tracker.py
│   ├── websocket/       # WebSocket management
│   └── logging/         # Shadow mode logging
├── frontend/            # React UI
├── tests/               # Unit tests
├── logs/                # Evaluation logs (shadow mode)
└── docs/                # Specification
```

## Development Phases

- [x] **Phase 1: Data Foundation** - Canonical IDs, Alpaca streaming, ORATS client, aggregator
- [x] **Phase 2: Gating Engine** - Gate framework, pipeline, abstain generation
- [x] **Phase 3: MVP UI** - FastAPI WebSocket, React chain view
- [x] **Phase 4: Recommendation Logic** - Strike selection, position sizing, recommender
- [x] **Phase 5: Evaluation** - Shadow mode, session tracker, position tracker, logging
- [x] **Phase 6: Sentiment & Scanner** - Finnhub news, Quiver WSB, daily scanner

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                      Frontend (React)                     │
│ Options Chain │ Recommendations │ Abstain Panel │ Scanner │
└───────────────────────────────────────────────────────────┘
                             │
                       WebSocket/REST
                             │
┌───────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                     │
│ ┌─────────────┐ ┌─────────────┐ ┌───────────────────────┐ │
│ │   Gating    │ │ Recommender │ │   Sentiment Scanner   │ │
│ │  Pipeline   │ │   Engine    │ │     (News + WSB)      │ │
│ │ (11 Gates)  │ │             │ │                       │ │
│ └─────────────┘ └─────────────┘ └───────────────────────┘ │
│                             │                             │
│ ┌───────────────────────────────────────────────────────┐ │
│ │                    Data Aggregator                    │ │
│ │       Alpaca (Quotes) + ORATS (Greeks) + Sentiment    │ │
│ └───────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

## Backtesting

Run historical backtests using WSB sentiment signals:

```bash
# Basic backtest
python -m backend.run_backtest --symbols TSLA,NVDA,PLTR --start 2024-01-01

# With options indicators (P/C Ratio, Max Pain)
python -m backend.run_backtest --symbols TSLA,NVDA --start 2024-01-01 --include-options-indicators

# With IV Rank validation (tests ORATS subscription value)
python -m backend.run_backtest --symbols TSLA,NVDA,PLTR --start 2024-10-01 --include-iv-rank
```

**Key findings from IV Rank backtest (111 signals with IV data, 2024):**
- Overall accuracy: 67.5%
- Max Pain: +5.5% edge (informational only)
- P/C Ratio: +100.0% edge but only 2 signals (kept as +5 modifier)
- IV Rank: +1.3% edge (marginal - ORATS subscription optional)

## Development

```bash
# Run unit tests
pytest

# Type checking
mypy backend

# Linting
ruff check backend
```

## Watchlist

Default watchlist (configurable in backend/config.py):
- NVDA, QQQ, AAPL, TSLA, SPY, AMD, GOOGL, AMZN, META, MSFT
