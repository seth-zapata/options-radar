#!/usr/bin/env python3
"""OptionsRadar CLI runner with mode presets.

Usage:
    python run.py --mode mock        # Mock data, no trading
    python run.py --mode simulation  # Mock data + simulated trades (fast)
    python run.py --mode paper       # Live data + Alpaca paper trading
    python run.py --mode live        # Live data + real trading (careful!)

Options:
    --mode          Mode preset: mock, simulation, paper, live
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
    python run.py --mode live              # Live trading (be careful!)
    python run.py --mode mock --backend-only  # Backend only
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "simulation", "paper", "live"],
        default="mock",
        help="Trading mode preset (default: mock)",
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
    print(f"OptionsRadar - {args.mode.upper()} MODE")
    print("=" * 60)
    print(f"  MOCK_DATA:     {env.get('MOCK_DATA')}")
    print(f"  TRADING_MODE:  {env.get('TRADING_MODE')}")
    print(f"  AUTO_EXECUTE:  {env.get('AUTO_EXECUTE')}")
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
