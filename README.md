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
# Edit .env with your Alpaca API keys
```

## Usage

```bash
# Activate venv (if not already active)
source venv/bin/activate

# Run the demo streaming client
python -m backend.demo_stream
```

## Development

```bash
# Run tests
pytest

# Type checking
mypy backend

# Linting
ruff check backend
```
