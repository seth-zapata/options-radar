#!/bin/bash
# Start OptionsRadar backend in mock data mode
# This generates simulated NVDA options data for UI development/testing

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Start with mock data enabled
MOCK_DATA=true uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
