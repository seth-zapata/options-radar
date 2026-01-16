"""Run full year backtest with quarterly tracking."""
import asyncio
import json
import sys
from datetime import date, datetime
from pathlib import Path

def run_backtest(year: int):
    """Run full year backtest with quarterly breakdown."""
    from backend.scalping.config import ScalpConfig
    from backend.scalping.replay import QuoteReplaySystem
    from backend.scalping.signal_generator import ScalpSignalGenerator
    from backend.scalping.velocity_tracker import PriceVelocityTracker
    from backend.scalping.volume_analyzer import VolumeAnalyzer
    from backend.scalping.technical_scalper import TechnicalScalper
    from backend.scalping.scalp_backtester import ScalpBacktester
    from backend.scalping.bar_cache import BarCache, SecondBarCache
    from backend.data.databento_loader import DataBentoLoader

    print("=" * 70)
    print(f"FULL YEAR BACKTEST: {year}")
    print("=" * 70)

    # Date range
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)

    # Load data
    data_path = Path("/mnt/e/OPRA Data/extracted")
    print(f"Loading data from {data_path}...")

    loader = DataBentoLoader(data_path)
    available_dates = loader.get_available_dates()
    available_dates = [d for d in available_dates if start_date <= d <= end_date]
    print(f"Found {len(available_dates)} trading days for {year}")

    if not available_dates:
        print(f"No data available for {year}")
        return

    # Config
    config = ScalpConfig(
        enabled=True,
        min_dte=1,
        target_dte=2,
        max_dte=3,
        max_contract_price=3.00,
        target_delta=0.40,
        delta_tolerance=0.15,
        max_spread_pct=10.0,
        take_profit_pct=20.0,
        stop_loss_pct=15.0,
        max_hold_minutes=None,
        momentum_threshold_put_pct=0.4,  # PUT: panic drops fast
        momentum_threshold_call_pct=0.6,  # CALL: rallies need confirmation
        momentum_window_seconds=30,
    )

    # Fetch 1-second bars
    print(f"\nFetching TSLA 1-second bars for {year}...")
    print("This may take 2-3 hours for a full year...")
    second_bar_cache = SecondBarCache()
    asyncio.run(second_bar_cache.fetch_and_cache_range("TSLA", start_date, end_date))

    # Fetch 1-minute bars as fallback
    print(f"\nFetching TSLA 1-minute bars as fallback...")
    bar_cache = BarCache()
    asyncio.run(bar_cache.fetch_and_cache_range("TSLA", start_date, end_date))

    # Create replay system
    replay = QuoteReplaySystem(
        loader,
        max_dte=config.max_dte,
        bar_cache=bar_cache,
        second_bar_cache=second_bar_cache
    )

    # Create components
    velocity = PriceVelocityTracker("TSLA")
    volume = VolumeAnalyzer("TSLA")
    technical = TechnicalScalper("TSLA")

    generator = ScalpSignalGenerator(
        symbol="TSLA",
        config=config,
        velocity_tracker=velocity,
        volume_analyzer=volume,
        technical_scalper=technical,
    )

    # Create backtester
    backtester = ScalpBacktester(config, replay, generator)

    # Run backtest
    print(f"\nRunning backtest for {year}...")
    print("-" * 70)

    # Progress tracking
    import time as _time
    backtest_start_time = _time.time()
    total_days = len(available_dates)
    days_processed = [0]

    def on_day_start(current_date):
        days_processed[0] += 1
        elapsed = _time.time() - backtest_start_time
        if days_processed[0] > 1:
            avg_per_day = elapsed / (days_processed[0] - 1)
            remaining_days = total_days - days_processed[0] + 1
            eta_minutes = (avg_per_day * remaining_days) / 60
            print(f"  Processing {current_date} ({days_processed[0]}/{total_days}) - ETA: {eta_minutes:.1f} min", flush=True)
        else:
            print(f"  Processing {current_date} ({days_processed[0]}/{total_days})...", flush=True)

    results = backtester.run(
        start_date=start_date,
        end_date=end_date,
        on_day_start=on_day_start,
    )

    # Save full results
    output_file = f"/tmp/backtest_{year}_full_1sec.json"
    results_dict = results.to_dict()
    # Add trades list (to_dict doesn't include it)
    results_dict["trades"] = [t.to_dict() for t in results.trades]
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    # Calculate quarterly breakdown
    print("\n" + "=" * 70)
    print(f"QUARTERLY BREAKDOWN - {year}")
    print("=" * 70)

    quarters = {
        "Q1": (date(year, 1, 1), date(year, 3, 31)),
        "Q2": (date(year, 4, 1), date(year, 6, 30)),
        "Q3": (date(year, 7, 1), date(year, 9, 30)),
        "Q4": (date(year, 10, 1), date(year, 12, 31)),
    }

    trades = results_dict["trades"]
    quarterly_stats = {}

    for q_name, (q_start, q_end) in quarters.items():
        q_trades = [
            t for t in trades
            if q_start <= datetime.fromisoformat(t["entry_time"].replace("Z", "+00:00")).date() <= q_end
        ]

        winners = [t for t in q_trades if t["pnl_dollars"] > 0]
        losers = [t for t in q_trades if t["pnl_dollars"] <= 0]
        total_pnl = sum(t["pnl_dollars"] for t in q_trades)
        win_rate = len(winners) / len(q_trades) * 100 if q_trades else 0

        quarterly_stats[q_name] = {
            "trades": len(q_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "pnl": total_pnl,
        }

        print(f"\n{q_name} ({q_start} to {q_end}):")
        print(f"  Trades: {len(q_trades)}")
        print(f"  Win Rate: {win_rate:.1f}% ({len(winners)}W / {len(losers)}L)")
        print(f"  P&L: ${total_pnl:,.2f}")

    # Save quarterly summary
    quarterly_file = f"/tmp/backtest_{year}_quarterly.json"
    with open(quarterly_file, "w") as f:
        json.dump({
            "year": year,
            "total_trades": len(trades),
            "total_pnl": sum(t["pnl_dollars"] for t in trades),
            "overall_win_rate": len([t for t in trades if t["pnl_dollars"] > 0]) / len(trades) * 100 if trades else 0,
            "quarterly": quarterly_stats,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"FULL YEAR SUMMARY - {year}")
    print("=" * 70)
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {results_dict.get('win_rate', 0) * 100:.1f}%")
    print(f"Total P&L: ${results_dict.get('total_pnl', 0):,.2f}")
    print(f"Profit Factor: {results_dict.get('profit_factor', 0):.2f}")
    print(f"\nResults saved to: {output_file}")
    print(f"Quarterly summary: {quarterly_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_full_year_backtest.py <year>")
        sys.exit(1)

    year = int(sys.argv[1])
    run_backtest(year)
