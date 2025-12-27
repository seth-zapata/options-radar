#!/bin/bash
# Start the OptionsRadar backend server

cd "$(dirname "$0")"
source venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
