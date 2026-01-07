"""Scalp backtester for evaluating scalping strategy performance.

Uses the QuoteReplaySystem and ScalpSignalGenerator to simulate scalp
trades on historical data and calculate performance metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from backend.scalping.replay import QuoteReplaySystem, ReplayQuote
    from backend.scalping.signal_generator import ScalpSignal, ScalpSignalGenerator

from backend.scalping.config import ScalpConfig

logger = logging.getLogger(__name__)


@dataclass
class ScalpTrade:
    """Completed scalp trade record.

    Represents a trade from entry to exit with all relevant metrics.
    """

    # Identification
    signal_id: str
    symbol: str

    # Signal info
    signal_type: Literal["SCALP_CALL", "SCALP_PUT"]
    trigger: str
    confidence: int

    # Option details
    option_symbol: str
    strike: float
    expiry: str
    delta: float
    dte: int

    # Entry
    entry_time: datetime
    entry_price: float
    underlying_at_entry: float
    contracts: int = 1

    # Exit
    exit_time: datetime | None = None
    exit_price: float | None = None
    underlying_at_exit: float | None = None
    exit_reason: str | None = None  # 'take_profit', 'stop_loss', 'time_exit', 'eod'

    # P&L
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    max_gain_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    # Timing
    hold_seconds: int = 0

    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_time is None

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl_dollars > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "trigger": self.trigger,
            "confidence": self.confidence,
            "option_symbol": self.option_symbol,
            "strike": self.strike,
            "expiry": self.expiry,
            "delta": round(self.delta, 3),
            "dte": self.dte,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": round(self.entry_price, 2),
            "underlying_at_entry": round(self.underlying_at_entry, 2),
            "contracts": self.contracts,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "underlying_at_exit": (
                round(self.underlying_at_exit, 2) if self.underlying_at_exit else None
            ),
            "exit_reason": self.exit_reason,
            "pnl_dollars": round(self.pnl_dollars, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "max_gain_pct": round(self.max_gain_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "hold_seconds": self.hold_seconds,
        }

    def __repr__(self) -> str:
        status = "OPEN" if self.is_open else self.exit_reason
        return (
            f"ScalpTrade({self.signal_type} ${self.strike} "
            f"entry=${self.entry_price:.2f} -> "
            f"${self.exit_price:.2f if self.exit_price else 0:.2f}, "
            f"P&L=${self.pnl_dollars:+.2f} ({self.pnl_pct:+.1f}%), "
            f"{status})"
        )


@dataclass
class BacktestResult:
    """Aggregate backtest statistics.

    Contains all performance metrics for a backtest run.
    """

    # Period
    start_date: datetime
    end_date: datetime
    trading_days: int = 0

    # Trade counts
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    breakeven: int = 0

    # P&L
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Rates
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0

    # Per-trade averages
    avg_pnl: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_hold_seconds: float = 0.0

    # By trigger
    trades_by_trigger: dict[str, int] = field(default_factory=dict)
    pnl_by_trigger: dict[str, float] = field(default_factory=dict)
    winrate_by_trigger: dict[str, float] = field(default_factory=dict)

    # By exit reason
    trades_by_exit: dict[str, int] = field(default_factory=dict)
    pnl_by_exit: dict[str, float] = field(default_factory=dict)

    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    # Trade list
    trades: list[ScalpTrade] = field(default_factory=list)

    def summary(self) -> str:
        """Generate text summary of results."""
        # Handle both date and datetime objects
        start_str = self.start_date.isoformat() if hasattr(self.start_date, 'isoformat') else str(self.start_date)
        end_str = self.end_date.isoformat() if hasattr(self.end_date, 'isoformat') else str(self.end_date)
        lines = [
            f"{'=' * 60}",
            f"SCALP BACKTEST RESULTS",
            f"{'=' * 60}",
            f"Period: {start_str} to {end_str}",
            f"Trading Days: {self.trading_days}",
            "",
            f"{'PERFORMANCE':=^60}",
            f"Total Trades: {self.total_trades}",
            f"Win Rate: {self.win_rate:.1%} ({self.winners}W / {self.losers}L)",
            f"Total P&L: ${self.total_pnl:,.2f}",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Expectancy: ${self.expectancy:.2f}/trade",
            "",
            f"Average Win: ${self.avg_win:.2f}",
            f"Average Loss: ${self.avg_loss:.2f}",
            f"Largest Win: ${self.largest_win:.2f}",
            f"Largest Loss: ${self.largest_loss:.2f}",
            "",
            f"Avg Hold Time: {self.avg_hold_seconds:.0f}s ({self.avg_hold_seconds/60:.1f}min)",
            "",
            f"{'BY TRIGGER':=^60}",
        ]

        for trigger in sorted(self.trades_by_trigger.keys()):
            count = self.trades_by_trigger[trigger]
            pnl = self.pnl_by_trigger.get(trigger, 0)
            wr = self.winrate_by_trigger.get(trigger, 0)
            lines.append(f"  {trigger}: {count} trades, ${pnl:+,.2f}, {wr:.1%} WR")

        lines.append("")
        lines.append(f"{'BY EXIT REASON':=^60}")
        for reason in sorted(self.trades_by_exit.keys()):
            count = self.trades_by_exit[reason]
            pnl = self.pnl_by_exit.get(reason, 0)
            lines.append(f"  {reason}: {count} trades, ${pnl:+,.2f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        start_str = self.start_date.isoformat() if hasattr(self.start_date, 'isoformat') else str(self.start_date)
        end_str = self.end_date.isoformat() if hasattr(self.end_date, 'isoformat') else str(self.end_date)
        return {
            "start_date": start_str,
            "end_date": end_str,
            "trading_days": self.trading_days,
            "total_trades": self.total_trades,
            "winners": self.winners,
            "losers": self.losers,
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "avg_hold_seconds": round(self.avg_hold_seconds, 1),
            "trades_by_trigger": self.trades_by_trigger,
            "pnl_by_trigger": {k: round(v, 2) for k, v in self.pnl_by_trigger.items()},
            "winrate_by_trigger": {
                k: round(v, 4) for k, v in self.winrate_by_trigger.items()
            },
            "trades_by_exit": self.trades_by_exit,
            "pnl_by_exit": {k: round(v, 2) for k, v in self.pnl_by_exit.items()},
        }


class ScalpBacktester:
    """Backtester for scalping strategy.

    Replays historical quotes through the signal generator and simulates
    trade execution with realistic fills and exit logic.

    Usage:
        # Setup components
        config = ScalpConfig(enabled=True)
        replay = QuoteReplaySystem()
        replay.load_data(data_path, start_date, end_date)

        generator = ScalpSignalGenerator(
            symbol="TSLA",
            config=config,
            velocity_tracker=velocity,
            volume_analyzer=volume,
            technical_scalper=technical,
        )

        # Run backtest
        backtester = ScalpBacktester(config, replay, generator)
        result = backtester.run()
        print(result.summary())
    """

    def __init__(
        self,
        config: ScalpConfig,
        replay_system: "QuoteReplaySystem",
        signal_generator: "ScalpSignalGenerator",
        slippage_pct: float = 0.5,
    ):
        """Initialize backtester.

        Args:
            config: Scalping configuration
            replay_system: Loaded quote replay system
            signal_generator: Configured signal generator
            slippage_pct: Simulated slippage percentage on fills
        """
        self.config = config
        self.replay = replay_system
        self.generator = signal_generator
        self.slippage_pct = slippage_pct

        # State
        self._trades: list[ScalpTrade] = []
        self._open_trade: ScalpTrade | None = None
        self._current_quote: dict[str, "ReplayQuote"] = {}  # By option symbol
        self._underlying_price: float = 0.0

    def run(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> BacktestResult:
        """Run full backtest.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestResult with all trades and statistics
        """
        logger.info("Starting scalp backtest...")

        # Reset state
        self._trades = []
        self._open_trade = None
        self._current_quote = {}
        self.generator.reset()

        # Get date range
        if start_date is None:
            start_date = self.replay.start_time
        if end_date is None:
            end_date = self.replay.end_time

        # Track trading days
        trading_days = set()

        # Process each quote
        for tick in self.replay.replay(start_date, end_date):
            current_time = tick.timestamp

            # Track trading day
            trading_days.add(current_time.date())

            # Update quote cache
            if tick.is_underlying:
                self._underlying_price = tick.mid_price
                # Update technical indicators
                self.generator.technical.update(
                    tick.mid_price,
                    volume=tick.volume or 0,
                    timestamp=current_time,
                )
                # Update velocity tracker
                self.generator.velocity.add_price(
                    tick.mid_price,
                    current_time,
                    volume=tick.volume or 0,
                )
            else:
                # Option quote
                self._current_quote[tick.symbol] = tick
                # Update volume analyzer if we have volume
                if tick.volume:
                    option_type = "C" if "C" in tick.symbol else "P"
                    self.generator.volume.add_volume(
                        tick.symbol,
                        tick.volume,
                        current_time,
                        option_type,
                    )

            # Check for exit if we have an open trade
            if self._open_trade:
                self._check_exit(current_time)

            # Only look for new signals if no open trade
            if not self._open_trade and self._underlying_price > 0:
                # Build available options list from current quotes
                self._update_available_options()

                # Evaluate for signals
                signal = self.generator.evaluate(current_time, self._underlying_price)

                if signal:
                    self._open_trade_from_signal(signal)

        # Force close any remaining open trade
        if self._open_trade:
            self._force_exit(self.replay.end_time, "eod")

        # Calculate statistics
        result = self._calculate_statistics(start_date, end_date, len(trading_days))
        logger.info(f"Backtest complete: {len(self._trades)} trades")

        return result

    def _update_available_options(self) -> None:
        """Build options list from current quotes for signal generator."""
        options = []
        for symbol, quote in self._current_quote.items():
            # Parse option details from OCC symbol
            # Format: TSLA240105C00420000
            try:
                # Determine option type
                if "C" in symbol:
                    opt_type = "C"
                    type_idx = symbol.index("C")
                elif "P" in symbol:
                    opt_type = "P"
                    type_idx = symbol.index("P")
                else:
                    continue

                # Extract strike (last 8 chars, divide by 1000)
                strike_str = symbol[-8:]
                strike = int(strike_str) / 1000

                # Extract expiry (6 chars before type)
                expiry_str = symbol[type_idx - 6 : type_idx]
                expiry = f"20{expiry_str[:2]}-{expiry_str[2:4]}-{expiry_str[4:6]}"

                # Calculate DTE
                from datetime import date

                exp_date = date(
                    int(f"20{expiry_str[:2]}"),
                    int(expiry_str[2:4]),
                    int(expiry_str[4:6]),
                )
                dte = (exp_date - quote.timestamp.date()).days

                options.append(
                    {
                        "symbol": symbol,
                        "strike": strike,
                        "expiry": expiry,
                        "option_type": opt_type,
                        "bid_px": quote.bid_price,
                        "ask_px": quote.ask_price,
                        "dte": dte,
                        "delta": self._estimate_delta(
                            strike, self._underlying_price, opt_type, dte
                        ),
                    }
                )
            except (ValueError, IndexError):
                continue

        self.generator.update_available_options(options)

    def _estimate_delta(
        self,
        strike: float,
        underlying: float,
        option_type: str,
        dte: int,
    ) -> float:
        """Estimate option delta based on moneyness.

        Simple approximation - real implementation would use Black-Scholes.
        """
        if underlying <= 0 or dte < 0:
            return 0.5

        # Moneyness as percentage from strike
        moneyness = (underlying - strike) / underlying

        # Simple delta approximation
        if option_type == "C":
            # ITM calls have higher delta
            if moneyness > 0.05:  # ITM
                return min(0.9, 0.5 + moneyness * 5)
            elif moneyness < -0.05:  # OTM
                return max(0.1, 0.5 + moneyness * 5)
            else:  # ATM
                return 0.5
        else:  # Put
            # ITM puts (strike > underlying) have higher absolute delta
            if moneyness < -0.05:  # ITM put
                return min(-0.1, -0.5 - moneyness * 5)
            elif moneyness > 0.05:  # OTM put
                return max(-0.9, -0.5 - moneyness * 5)
            else:  # ATM
                return -0.5

    def _open_trade_from_signal(self, signal: "ScalpSignal") -> None:
        """Open a new trade from a signal."""
        # Apply slippage to entry price
        slippage_mult = 1 + (self.slippage_pct / 100)
        fill_price = signal.ask_price * slippage_mult

        self._open_trade = ScalpTrade(
            signal_id=signal.id,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            trigger=signal.trigger,
            confidence=signal.confidence,
            option_symbol=signal.option_symbol,
            strike=signal.strike,
            expiry=signal.expiry,
            delta=signal.delta,
            dte=signal.dte,
            entry_time=signal.timestamp,
            entry_price=fill_price,
            underlying_at_entry=signal.underlying_price,
            contracts=signal.suggested_contracts,
        )

        logger.debug(
            f"Opened trade: {signal.signal_type} {signal.option_symbol} "
            f"@ ${fill_price:.2f}"
        )

    def _check_exit(self, current_time: datetime) -> None:
        """Check if open trade should be exited."""
        if not self._open_trade:
            return

        trade = self._open_trade
        option_symbol = trade.option_symbol

        # Get current quote for option
        quote = self._current_quote.get(option_symbol)
        if not quote:
            return

        # Current bid is our exit price (selling)
        current_price = quote.bid_price

        if current_price <= 0:
            return

        # Calculate current P&L %
        pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100

        # Track max gain/drawdown
        if pnl_pct > trade.max_gain_pct:
            trade.max_gain_pct = pnl_pct
        if pnl_pct < trade.max_drawdown_pct:
            trade.max_drawdown_pct = pnl_pct

        # Check exit conditions
        exit_reason = None

        # 1. Take profit
        if pnl_pct >= self.config.take_profit_pct:
            exit_reason = "take_profit"

        # 2. Stop loss
        elif pnl_pct <= -self.config.stop_loss_pct:
            exit_reason = "stop_loss"

        # 3. Time exit
        hold_seconds = (current_time - trade.entry_time).total_seconds()
        max_hold_seconds = self.config.max_hold_minutes * 60
        if hold_seconds >= max_hold_seconds:
            exit_reason = "time_exit"

        # Execute exit if triggered
        if exit_reason:
            self._exit_trade(current_time, current_price, exit_reason)

    def _exit_trade(
        self,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Exit the open trade."""
        if not self._open_trade:
            return

        trade = self._open_trade

        # Apply slippage to exit (selling at bid, so negative slippage)
        slippage_mult = 1 - (self.slippage_pct / 100)
        fill_price = exit_price * slippage_mult

        # Calculate P&L
        trade.exit_time = exit_time
        trade.exit_price = fill_price
        trade.underlying_at_exit = self._underlying_price
        trade.exit_reason = exit_reason
        trade.hold_seconds = int((exit_time - trade.entry_time).total_seconds())

        # P&L per contract
        pnl_per_contract = (fill_price - trade.entry_price) * 100  # Options are 100x
        trade.pnl_dollars = pnl_per_contract * trade.contracts
        trade.pnl_pct = ((fill_price - trade.entry_price) / trade.entry_price) * 100

        # Store trade
        self._trades.append(trade)
        self._open_trade = None

        # Trigger loss cooldown if needed
        if trade.pnl_dollars < 0:
            self.generator.trigger_loss_cooldown(exit_time)

        logger.debug(
            f"Closed trade: {trade.option_symbol} @ ${fill_price:.2f}, "
            f"P&L: ${trade.pnl_dollars:+.2f} ({trade.pnl_pct:+.1f}%), "
            f"reason: {exit_reason}"
        )

    def _force_exit(self, exit_time: datetime, reason: str) -> None:
        """Force exit an open trade (e.g., end of day)."""
        if not self._open_trade:
            return

        option_symbol = self._open_trade.option_symbol
        quote = self._current_quote.get(option_symbol)

        if quote and quote.bid_price > 0:
            exit_price = quote.bid_price
        else:
            # No quote - assume worst case (entry price - stop loss)
            exit_price = self._open_trade.entry_price * (
                1 - self.config.stop_loss_pct / 100
            )

        self._exit_trade(exit_time, exit_price, reason)

    def _calculate_statistics(
        self,
        start_date: datetime,
        end_date: datetime,
        trading_days: int,
    ) -> BacktestResult:
        """Calculate aggregate statistics from trades."""
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            trades=self._trades.copy(),
        )

        if not self._trades:
            return result

        result.total_trades = len(self._trades)

        # Categorize trades
        for trade in self._trades:
            if trade.pnl_dollars > 0:
                result.winners += 1
                result.gross_profit += trade.pnl_dollars
                if trade.pnl_dollars > result.largest_win:
                    result.largest_win = trade.pnl_dollars
            elif trade.pnl_dollars < 0:
                result.losers += 1
                result.gross_loss += abs(trade.pnl_dollars)
                if trade.pnl_dollars < result.largest_loss:
                    result.largest_loss = trade.pnl_dollars
            else:
                result.breakeven += 1

            # By trigger
            trigger = trade.trigger
            result.trades_by_trigger[trigger] = (
                result.trades_by_trigger.get(trigger, 0) + 1
            )
            result.pnl_by_trigger[trigger] = (
                result.pnl_by_trigger.get(trigger, 0) + trade.pnl_dollars
            )

            # By exit reason
            if trade.exit_reason:
                reason = trade.exit_reason
                result.trades_by_exit[reason] = result.trades_by_exit.get(reason, 0) + 1
                result.pnl_by_exit[reason] = (
                    result.pnl_by_exit.get(reason, 0) + trade.pnl_dollars
                )

        # Calculate rates
        result.total_pnl = result.gross_profit - result.gross_loss

        if result.total_trades > 0:
            result.win_rate = result.winners / result.total_trades
            result.avg_pnl = result.total_pnl / result.total_trades
            result.avg_pnl_pct = (
                sum(t.pnl_pct for t in self._trades) / result.total_trades
            )
            result.avg_hold_seconds = (
                sum(t.hold_seconds for t in self._trades) / result.total_trades
            )

        if result.winners > 0:
            result.avg_win = result.gross_profit / result.winners

        if result.losers > 0:
            result.avg_loss = result.gross_loss / result.losers

        if result.gross_loss > 0:
            result.profit_factor = result.gross_profit / result.gross_loss
        elif result.gross_profit > 0:
            result.profit_factor = float("inf")

        # Expectancy: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        if result.winners > 0 and result.losers > 0:
            result.expectancy = (result.win_rate * result.avg_win) - (
                (1 - result.win_rate) * result.avg_loss
            )

        # Win rate by trigger
        for trigger in result.trades_by_trigger:
            trigger_trades = [t for t in self._trades if t.trigger == trigger]
            trigger_wins = sum(1 for t in trigger_trades if t.pnl_dollars > 0)
            if trigger_trades:
                result.winrate_by_trigger[trigger] = trigger_wins / len(trigger_trades)

        return result

    @property
    def trades(self) -> list[ScalpTrade]:
        """All completed trades."""
        return self._trades.copy()

    @property
    def open_trade(self) -> ScalpTrade | None:
        """Currently open trade, if any."""
        return self._open_trade


def main():
    """CLI entry point for running backtests."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run scalp backtest")
    parser.add_argument("data_path", help="Path to DataBento data directory")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse dates
    start_date = None
    end_date = None
    if args.start:
        start_date = datetime.fromisoformat(args.start)
    if args.end:
        end_date = datetime.fromisoformat(args.end)

    # Import components
    from backend.scalping.config import ScalpConfig
    from backend.scalping.replay import QuoteReplaySystem
    from backend.scalping.signal_generator import ScalpSignalGenerator
    from backend.scalping.velocity_tracker import PriceVelocityTracker
    from backend.scalping.volume_analyzer import VolumeAnalyzer
    from backend.scalping.technical_scalper import TechnicalScalper

    # Setup
    config = ScalpConfig(enabled=True)
    replay = QuoteReplaySystem()

    print(f"Loading data from {args.data_path}...")
    replay.load_data(Path(args.data_path), start_date, end_date)
    print(f"Loaded {replay.quote_count} quotes")

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

    # Run backtest
    backtester = ScalpBacktester(config, replay, generator)
    result = backtester.run(start_date, end_date)

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


if __name__ == "__main__":
    main()
