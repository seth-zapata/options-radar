# OptionsRadar

A local, display-only options trading recommendation system for a curated watchlist of tech/AI stocks.

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure credentials
cp .env.example .env
# Edit .env with your API keys (see Credentials section below)
```

## Credentials

| Service | Variables | Required | Cost | Purpose |
|---------|-----------|----------|------|---------|
| Alpaca | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Yes | $99/mo (Algo Trader Plus) | Real-time option quotes |
| ORATS | `ORATS_API_TOKEN` | Phase 2+ | $199/mo (Live) | Greeks, IV rank |
| Finnhub | `FINNHUB_API_KEY` | Phase 6 | Free tier available | News sentiment |
| Quiver | `QUIVER_API_KEY` | Phase 6 | $20-75/mo | Social sentiment |

**Getting credentials:**
- Alpaca: https://app.alpaca.markets/ (enable Options + Algo Trader Plus for OPRA data)
- ORATS: https://orats.com/ (Live subscription for real-time Greeks)

## Testing Phase 1 (Data Foundation)

After setup, run the Phase 1 integration test:

```bash
source venv/bin/activate
python -m backend.test_phase1
```

This tests:
- Alpaca WebSocket connection and authentication
- Real-time quote streaming for NVDA options
- ORATS Greeks fetching (if credentials configured)
- Data aggregation and staleness detection

**Expected output with Alpaca only:**
```
Credentials detected:
  Alpaca: ✅ Found
  ORATS:  ⚠️  Missing (optional)

TEST: Alpaca WebSocket Connection
  Connection state: connecting
  Connection state: authenticating
  Connection state: connected
  Subscribed to 28 contracts
  ATM strike: $140.0
  Waiting 5 seconds for quotes...
  Quote: NVDA 2025-01-03 140.0C Bid: $5.20 Ask: $5.40
  ...
  ✅ Alpaca connection: PASSED
```

## Quick Demo (Quote Streaming)

```bash
source venv/bin/activate
python -m backend.demo_stream
```

Streams real-time NVDA option quotes. Press Ctrl+C to stop.

## Development

```bash
# Run unit tests
pytest

# Run with verbose output
pytest -v

# Type checking
mypy backend

# Linting
ruff check backend
```

## Project Structure

```
options-radar/
├── backend/
│   ├── models/          # Data models (canonical IDs, quotes, Greeks)
│   ├── data/            # Data clients (Alpaca, ORATS, aggregator)
│   ├── engine/          # Gating engine (Phase 2)
│   ├── websocket/       # WebSocket management (Phase 3)
│   └── logging/         # Shadow mode logging (Phase 5)
├── frontend/            # React UI (Phase 3)
├── tests/               # Unit tests
└── docs/                # Specification
```

## Development Phases

- [x] **Phase 1: Data Foundation** - Canonical IDs, Alpaca streaming, ORATS client, aggregator
- [ ] **Phase 2: Gating Engine** - Gate framework, pipeline, abstain generation
- [ ] **Phase 3: MVP UI** - FastAPI WebSocket, React chain view
- [ ] **Phase 4: Recommendation Logic** - Strike selection, position sizing
- [ ] **Phase 5: Evaluation** - Shadow mode logging, metrics
- [ ] **Phase 6: Expansion** - Sentiment, scanner, portfolio integration
