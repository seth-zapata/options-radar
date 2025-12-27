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
- Uses Alpaca for quotes, ORATS for Greeks
- "Abstain by default" philosophy - only recommend when ALL gates pass

## Testing After Development

Always provide testing instructions after completing a phase or significant feature:
1. What command to run
2. What credentials are needed
3. Expected output
4. How to interpret results
