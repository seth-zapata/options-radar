#!/usr/bin/env python3
"""OptionsRadar CLI runner with mode presets.

Usage:
    python run.py --mode mock        # Mock data, no trading
    python run.py --mode simulation  # Mock data + simulated trades (fast)
    python run.py --mode paper       # Live data + Alpaca paper trading
    python run.py --mode live        # Live data + real trading (careful!)

    # With scalping (0DTE/1DTE intraday momentum)
    python run.py --mode paper --scalping

    # Backtest scalping strategy
    python run.py --backtest /path/to/databento/data --start 2024-01-01 --end 2024-12-31

Options:
    --mode          Mode preset: mock, simulation, paper, live
    --scalping      Enable scalping module (0DTE/1DTE momentum trading)
    --backtest      Run scalping backtest with DataBento data
    --start/--end   Date range for backtest (YYYY-MM-DD)
    --speed         Simulation speed multiplier (default: 5.0)
    --no-auto       Disable auto-execution
    --port          Backend port (default: 8000)
    --frontend-port Frontend port (default: 5173)
    --backend-only  Only start backend (no frontend)
    --reload        Enable hot reload (default for non-live modes)
"""

import argparse
import os
import sys
import subprocess
import signal
import time
from pathlib import Path


MODE_PRESETS = {
    "mock": {
        "MOCK_DATA": "true",
        "TRADING_MODE": "off",
        "AUTO_EXECUTE": "false",
    },
    "simulation": {
        "MOCK_DATA": "true",
        "TRADING_MODE": "simulation",
        "AUTO_EXECUTE": "true",
    },
    "paper": {
        "MOCK_DATA": "false",
        "TRADING_MODE": "paper",
        "AUTO_EXECUTE": "true",
    },
    "live": {
        "MOCK_DATA": "false",
        "TRADING_MODE": "live",
        "AUTO_EXECUTE": "true",
    },
}


def run_backtest(args):
    """Run scalping backtest with DataBento data."""
    import asyncio
    from datetime import datetime
    from pathlib import Path
    import json

    # Import components
    from backend.scalping.config import ScalpConfig
    from backend.scalping.replay import QuoteReplaySystem
    from backend.scalping.signal_generator import ScalpSignalGenerator
    from backend.scalping.velocity_tracker import PriceVelocityTracker
    from backend.scalping.volume_analyzer import VolumeAnalyzer
    from backend.scalping.technical_scalper import TechnicalScalper
    from backend.scalping.scalp_backtester import ScalpBacktester
    from backend.scalping.bar_cache import BarCache, SecondBarCache
    from backend.data.databento_loader import DataBentoLoader

    # Parse dates
    start_date = datetime.fromisoformat(args.start).date() if args.start else None
    end_date = datetime.fromisoformat(args.end).date() if args.end else None

    # Setup loader and replay system
    data_path = Path(args.backtest)
    print(f"Loading data from {data_path}...")

    loader = DataBentoLoader(data_path)
    available_dates = loader.get_available_dates()
    print(f"Found {len(available_dates)} trading days ({available_dates[0]} to {available_dates[-1]})")

    # Filter to requested date range
    if start_date:
        available_dates = [d for d in available_dates if d >= start_date]
    if end_date:
        available_dates = [d for d in available_dates if d <= end_date]
    print(f"Backtest range: {len(available_dates)} days")

    # Optimized config based on 251-trade backtest analysis:
    # - Soft DTE preference: DTE=1 (63.4% WR) > DTE=2 > DTE=3, never 0DTE
    # - Time stop: exit unprofitable trades after 5 min (69% WR <5min vs 47% >5min)
    # - Confidence cap: skip conf >75 (80+ has 33% WR - overextended)
    # - Cheap options win more (avg winner $2.86)
    config = ScalpConfig(
        enabled=True,
        # DTE filtering - soft preference order
        min_dte=1,  # SKIP 0DTE - never use
        max_dte=3,  # Allow DTE 1-3
        dte_preference=(1, 2, 3),  # Prefer DTE=1 (63.4% WR, $57/trade)
        # Confidence cap - skip overextended moves
        max_confidence=75,  # 80+ conf has 33% WR (mean reversion)
        # Price filtering - cheap options win more
        max_contract_price=3.00,  # Winners avg $2.86 entry
        # Delta/spread settings
        target_delta=0.40,
        delta_tolerance=0.15,
        max_spread_pct=10.0,
        # Risk management
        take_profit_pct=20.0,  # Lowered from 30% to capture more winners
        stop_loss_pct=15.0,
        max_hold_minutes=None,  # Allow overnight holds
        time_stop_minutes=5,  # Exit unprofitable trades after 5 min
        # Signal generation
        momentum_threshold_pct=0.5,
        momentum_window_seconds=30,
    )

    # Fetch real TSLA bars for accurate underlying price tracking
    bt_start = start_date or available_dates[0]
    bt_end = end_date or available_dates[-1]

    # Use 1-second bars for accurate momentum detection (no lookahead bias)
    second_bar_cache = SecondBarCache()
    print("Fetching TSLA 1-second bars for momentum detection...")
    asyncio.run(second_bar_cache.fetch_and_cache_range("TSLA", bt_start, bt_end))

    # Also have 1-minute bars as fallback
    bar_cache = BarCache()
    print("Fetching TSLA 1-minute bars as fallback...")
    asyncio.run(bar_cache.fetch_and_cache_range("TSLA", bt_start, bt_end))

    replay = QuoteReplaySystem(
        loader,
        max_dte=config.max_dte,
        bar_cache=bar_cache,
        second_bar_cache=second_bar_cache
    )

    if len(available_dates) == 0:
        print("No data in requested date range.")
        sys.exit(1)

    # Create components
    velocity = PriceVelocityTracker("TSLA")
    volume = VolumeAnalyzer()
    technical = TechnicalScalper("TSLA")

    generator = ScalpSignalGenerator(
        symbol="TSLA",
        config=config,
        velocity_tracker=velocity,
        volume_analyzer=volume,
        technical_scalper=technical,
    )

    # Run backtest with progress logging
    print("Running backtest...")
    backtester = ScalpBacktester(config, replay, generator)

    # Progress tracking
    import time as _time
    backtest_start_time = _time.time()
    total_days = len([d for d in available_dates if bt_start <= d <= bt_end])
    days_processed = [0]  # Use list for closure

    def on_day_start(current_date):
        days_processed[0] += 1
        elapsed = _time.time() - backtest_start_time
        if days_processed[0] > 1:
            avg_per_day = elapsed / (days_processed[0] - 1)
            remaining_days = total_days - days_processed[0] + 1
            eta_seconds = avg_per_day * remaining_days
            eta_min = eta_seconds / 60
            print(f"  Processing {current_date} ({days_processed[0]}/{total_days}) - ETA: {eta_min:.1f} min", flush=True)
        else:
            print(f"  Processing {current_date} ({days_processed[0]}/{total_days})...", flush=True)

    result = backtester.run(bt_start, bt_end, on_day_start=on_day_start)

    # Print summary
    print()
    print(result.summary())

    # Save to file if requested
    if args.output:
        output_data = result.to_dict()
        output_data["trades"] = [t.to_dict() for t in result.trades]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="OptionsRadar backend + frontend runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py --mode mock              # UI development
    python run.py --mode simulation        # Fast testing with fake trades
    python run.py --mode simulation --speed 10  # 10x speed simulation
    python run.py --mode paper             # Paper trading (market hours)
    python run.py --mode paper --scalping  # Paper trading with scalping
    python run.py --mode live              # Live trading (be careful!)
    python run.py --mode mock --backend-only  # Backend only

    # Scalping backtest
    python run.py --backtest ./data/tsla --start 2024-01-01 --end 2024-12-31
    python run.py --backtest ./data/tsla -o results.json
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "simulation", "paper", "live"],
        default="mock",
        help="Trading mode preset (default: mock)",
    )
    parser.add_argument(
        "--scalping",
        action="store_true",
        help="Enable scalping module (0DTE/1DTE momentum trading)",
    )
    parser.add_argument(
        "--backtest",
        metavar="DATA_PATH",
        help="Run scalping backtest with DataBento data at this path",
    )
    parser.add_argument(
        "--start",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for backtest results",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        help="Simulation speed multiplier (default: 5.0)",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Disable auto-execution",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Backend port (default: 8000)",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=5173,
        help="Frontend port (default: 5173)",
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Only start backend (skip frontend)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=None,
        help="Enable hot reload (auto-enabled for non-live modes)",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable hot reload",
    )

    args = parser.parse_args()

    # Handle backtest mode separately
    if args.backtest:
        print("=" * 60)
        print("OptionsRadar - SCALPING BACKTEST")
        print("=" * 60)
        run_backtest(args)
        return

    # Get project root
    project_root = Path(__file__).parent.absolute()
    frontend_dir = project_root / "frontend"

    # Get preset environment variables
    env = os.environ.copy()
    preset = MODE_PRESETS[args.mode]
    env.update(preset)

    # Apply overrides
    if args.no_auto:
        env["AUTO_EXECUTE"] = "false"

    if args.scalping:
        env["SCALP_ENABLED"] = "true"

    env["SIMULATION_SPEED"] = str(args.speed)

    # Determine reload setting
    if args.no_reload:
        use_reload = False
    elif args.reload:
        use_reload = True
    else:
        # Auto-enable reload for non-live modes
        use_reload = args.mode != "live"

    # Print configuration
    print("=" * 60)
    mode_label = f"{args.mode.upper()} MODE"
    if args.scalping:
        mode_label += " + SCALPING"
    print(f"OptionsRadar - {mode_label}")
    print("=" * 60)
    print(f"  MOCK_DATA:     {env.get('MOCK_DATA')}")
    print(f"  TRADING_MODE:  {env.get('TRADING_MODE')}")
    print(f"  AUTO_EXECUTE:  {env.get('AUTO_EXECUTE')}")
    if args.scalping:
        print(f"  SCALPING:      enabled (0DTE/1DTE momentum)")
    if args.mode == "simulation":
        print(f"  SPEED:         {args.speed}x")
    print(f"  BACKEND:       http://localhost:{args.port}")
    if not args.backend_only:
        print(f"  FRONTEND:      http://localhost:{args.frontend_port}")
    print(f"  HOT RELOAD:    {use_reload}")
    print("=" * 60)

    if args.mode == "live":
        print("\n⚠️  WARNING: LIVE TRADING MODE - Real money at risk!")
        confirm = input("Type 'yes' to continue: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(1)
    elif args.mode == "paper":
        print("\n📝 Paper trading mode - requires market hours (9:30 AM - 4:00 PM ET)")

    # Build uvicorn command
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", str(args.port),
    ]
    if use_reload:
        backend_cmd.append("--reload")

    # Build frontend command
    frontend_cmd = ["npm", "run", "dev", "--", "--port", str(args.frontend_port)]

    processes = []

    def cleanup(signum=None, frame=None):
        """Terminate all child processes."""
        print("\n\nShutting down...")
        for proc in processes:
            if proc.poll() is None:  # Still running
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Start backend
        print(f"\n🚀 Starting backend on http://localhost:{args.port}")
        backend_proc = subprocess.Popen(
            backend_cmd,
            env=env,
            cwd=project_root,
        )
        processes.append(backend_proc)

        # Start frontend (unless --backend-only)
        if not args.backend_only:
            # Check if node_modules exists
            if not (frontend_dir / "node_modules").exists():
                print("\n📦 Installing frontend dependencies...")
                subprocess.run(
                    ["npm", "install"],
                    cwd=frontend_dir,
                    check=True,
                )

            # Give backend a moment to start
            time.sleep(2)

            print(f"\n🌐 Starting frontend on http://localhost:{args.frontend_port}")
            frontend_proc = subprocess.Popen(
                frontend_cmd,
                cwd=frontend_dir,
            )
            processes.append(frontend_proc)

            print(f"\n✅ OptionsRadar running!")
            print(f"   Open http://localhost:{args.frontend_port} in your browser")
        else:
            print(f"\n✅ Backend running on http://localhost:{args.port}")
            print(f"   API docs: http://localhost:{args.port}/docs")

        print("   Press Ctrl+C to stop\n")

        # Wait for processes
        while True:
            for proc in processes:
                retcode = proc.poll()
                if retcode is not None:
                    print(f"\nProcess exited with code {retcode}")
                    cleanup()
            time.sleep(1)

    except Exception as e:
        print(f"\nError: {e}")
        cleanup()


if __name__ == "__main__":
    main()
