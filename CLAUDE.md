# Claude Code Instructions for OptionsRadar

## Command Execution Policy

**Do NOT ask for confirmation on routine, non-risky commands including:**
- Running tests (`pytest`, `python -m backend.test_phase1`, etc.)
- Timeout-wrapped commands
- Git status, diff, log commands
- Fetching documentation (WebFetch)
- File reads and searches
- Running demo scripts
- Package installs in the virtual environment

**DO ask for confirmation on:**
- Destructive git operations (force push, hard reset)
- Commands that modify system configuration
- Commands that could incur costs (API calls to paid services in production)
- Anything involving credentials or secrets

## Project Context

This is an options trading recommendation system. Key points:
- Display-only (no trade execution)
- Local deployment (localhost)
- Data sources:
  - Alpaca: Real-time option quotes + portfolio read-only
  - ORATS: Greeks, IV rank
  - Finnhub: News sentiment (catalyst/trigger)
  - Quiver: WSB social sentiment (confirmation/overlay)
- 11 gates (8 hard, 3 soft) that must pass before recommendation
- "Abstain by default" philosophy - only recommend when ALL gates pass
- Sentiment uses 50/50 weighted combination of news + WSB

## Testing After Development

Always provide testing instructions after completing a phase or significant feature:
1. What command to run
2. What credentials are needed
3. Expected output
4. How to interpret results
