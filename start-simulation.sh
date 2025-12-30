#!/bin/bash
# Start OptionsRadar backend in SIMULATION MODE
# This runs accelerated auto-execution testing with mock data
# - Mock price movements cycle through regimes
# - Mock portfolio tracks simulated trades
# - Purple UI indicator shows simulation is active

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Configuration (can override with environment)
export MOCK_DATA=true
export TRADING_MODE=simulation
export AUTO_EXECUTE=true
export SIMULATION_SPEED=${SIMULATION_SPEED:-5.0}
export SIMULATION_BALANCE=${SIMULATION_BALANCE:-100000.0}

echo "=== SIMULATION MODE ==="
echo "Speed: ${SIMULATION_SPEED}x"
echo "Starting balance: \$${SIMULATION_BALANCE}"
echo ""
echo "View UI at: http://localhost:5173"
echo "(Run 'npm run dev' in frontend/ directory)"
echo ""

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
