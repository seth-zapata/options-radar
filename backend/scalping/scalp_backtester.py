"""Scalp backtester for evaluating scalping strategy performance.

Uses the QuoteReplaySystem and ScalpSignalGenerator to simulate scalp
trades on historical data and calculate performance metrics.

Now includes full portfolio simulation with:
- Position sizing based on 2% risk per trade
- Equity curve tracking
- Risk metrics (Sharpe, Sortino, Max Drawdown)
- Monthly returns breakdown
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from backend.scalping.replay import QuoteReplaySystem, ReplayQuote
    from backend.scalping.signal_generator import ScalpSignal, ScalpSignalGenerator

from backend.scalping.config import ScalpConfig

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Portfolio configuration for backtesting.

    Controls position sizing, risk management, and equity tracking.
    """

    starting_equity: float = 100_000.0  # Initial portfolio value
    risk_per_trade_pct: float = 2.0  # Risk 2% of equity per trade
    max_contracts: int = 50  # Liquidity cap
    min_contracts: int = 1  # Minimum position size


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

    # Portfolio tracking
    equity_before: float = 0.0  # Portfolio equity before trade
    equity_after: float = 0.0  # Portfolio equity after trade

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
            "equity_before": round(self.equity_before, 2),
            "equity_after": round(self.equity_after, 2),
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

    # By direction (SCALP_CALL vs SCALP_PUT)
    trades_by_direction: dict[str, int] = field(default_factory=dict)
    pnl_by_direction: dict[str, float] = field(default_factory=dict)
    winrate_by_direction: dict[str, float] = field(default_factory=dict)

    # By DTE
    trades_by_dte: dict[int, int] = field(default_factory=dict)
    pnl_by_dte: dict[int, float] = field(default_factory=dict)
    winrate_by_dte: dict[int, float] = field(default_factory=dict)

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Portfolio tracking
    starting_equity: float = 100_000.0
    ending_equity: float = 100_000.0
    total_return_pct: float = 0.0
    cagr: float = 0.0  # Compound annual growth rate

    # Equity curve (date -> equity)
    daily_equity: dict[str, float] = field(default_factory=dict)
    high_water_mark: float = 0.0

    # Monthly returns (YYYY-MM -> return %)
    monthly_returns: dict[str, float] = field(default_factory=dict)

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
            f"{'PORTFOLIO':=^60}",
            f"Starting Equity: ${self.starting_equity:,.2f}",
            f"Ending Equity:   ${self.ending_equity:,.2f}",
            f"Total Return:    {self.total_return_pct:+.2f}%",
            f"CAGR:            {self.cagr:.2f}%",
            "",
            f"{'RISK METRICS':=^60}",
            f"Max Drawdown:    ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)",
            f"Sharpe Ratio:    {self.sharpe_ratio:.2f}",
            f"Sortino Ratio:   {self.sortino_ratio:.2f}",
            f"Calmar Ratio:    {self.calmar_ratio:.2f}",
            "",
            f"{'TRADE PERFORMANCE':=^60}",
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

        lines.append("")
        lines.append(f"{'BY DIRECTION':=^60}")
        for direction in sorted(self.trades_by_direction.keys()):
            count = self.trades_by_direction[direction]
            pnl = self.pnl_by_direction.get(direction, 0)
            wr = self.winrate_by_direction.get(direction, 0)
            lines.append(f"  {direction}: {count} trades, ${pnl:+,.2f}, {wr:.1%} WR")

        lines.append("")
        lines.append(f"{'BY DTE':=^60}")
        for dte in sorted(self.trades_by_dte.keys()):
            count = self.trades_by_dte[dte]
            pnl = self.pnl_by_dte.get(dte, 0)
            wr = self.winrate_by_dte.get(dte, 0)
            lines.append(f"  DTE={dte}: {count} trades, ${pnl:+,.2f}, {wr:.1%} WR")

        # Monthly returns
        if self.monthly_returns:
            lines.append("")
            lines.append(f"{'MONTHLY RETURNS':=^60}")
            for month in sorted(self.monthly_returns.keys()):
                ret = self.monthly_returns[month]
                lines.append(f"  {month}: {ret:+.2f}%")

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
            # Portfolio metrics
            "starting_equity": round(self.starting_equity, 2),
            "ending_equity": round(self.ending_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "cagr": round(self.cagr, 2),
            # Risk metrics
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            # Trade stats
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
            # Breakdowns
            "trades_by_trigger": self.trades_by_trigger,
            "pnl_by_trigger": {k: round(v, 2) for k, v in self.pnl_by_trigger.items()},
            "winrate_by_trigger": {
                k: round(v, 4) for k, v in self.winrate_by_trigger.items()
            },
            "trades_by_exit": self.trades_by_exit,
            "pnl_by_exit": {k: round(v, 2) for k, v in self.pnl_by_exit.items()},
            "trades_by_direction": self.trades_by_direction,
            "pnl_by_direction": {k: round(v, 2) for k, v in self.pnl_by_direction.items()},
            "winrate_by_direction": {k: round(v, 4) for k, v in self.winrate_by_direction.items()},
            "trades_by_dte": self.trades_by_dte,
            "pnl_by_dte": {str(k): round(v, 2) for k, v in self.pnl_by_dte.items()},
            "winrate_by_dte": {str(k): round(v, 4) for k, v in self.winrate_by_dte.items()},
            # Equity tracking
            "daily_equity": {k: round(v, 2) for k, v in self.daily_equity.items()},
            "monthly_returns": {k: round(v, 2) for k, v in self.monthly_returns.items()},
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
        portfolio_config: PortfolioConfig | None = None,
    ):
        """Initialize backtester.

        Args:
            config: Scalping configuration
            replay_system: Loaded quote replay system
            signal_generator: Configured signal generator
            slippage_pct: Simulated slippage percentage on fills
            portfolio_config: Portfolio configuration (defaults to $100k, 2% risk)
        """
        self.config = config
        self.replay = replay_system
        self.generator = signal_generator
        self.slippage_pct = slippage_pct
        self.portfolio_config = portfolio_config or PortfolioConfig()

        # State
        self._trades: list[ScalpTrade] = []
        self._open_trade: ScalpTrade | None = None
        self._current_quote: dict[str, "ReplayQuote"] = {}  # By option symbol
        self._underlying_price: float = 0.0
        self._current_day: date | None = None  # Track current day for cache invalidation

        # Portfolio state
        self._current_equity: float = self.portfolio_config.starting_equity
        self._high_water_mark: float = self.portfolio_config.starting_equity
        self._daily_equity: dict[str, float] = {}  # date_str -> equity
        self._daily_returns: list[float] = []  # For Sharpe/Sortino calculation

    def run(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        on_day_start: "Callable[[date], None] | None" = None,
        on_day_end: "Callable[[date, int], None] | None" = None,
    ) -> BacktestResult:
        """Run full backtest.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            on_day_start: Callback at start of each day
            on_day_end: Callback at end of each day with tick count

        Returns:
            BacktestResult with all trades and statistics
        """
        logger.info("Starting scalp backtest...")

        # Reset state
        self._trades = []
        self._open_trade = None
        self._current_quote = {}
        self._current_day = None
        self.generator.reset()

        # Reset portfolio state
        self._current_equity = self.portfolio_config.starting_equity
        self._high_water_mark = self.portfolio_config.starting_equity
        self._daily_equity = {}
        self._daily_returns = []
        prev_day_equity = self._current_equity

        # Get date range
        if start_date is None:
            start_date = self.replay.start_time
        if end_date is None:
            end_date = self.replay.end_time

        # Track trading days
        trading_days = set()
        last_time = None

        # Process each quote
        for tick in self.replay.replay(start_date, end_date, on_day_start=on_day_start, on_day_end=on_day_end):
            current_time = tick.timestamp
            last_time = current_time

            # Track trading day and clear stale quote cache on day change
            current_date = current_time.date()
            if self._current_day != current_date:
                # Record end-of-day equity for previous day
                if self._current_day is not None:
                    date_str = self._current_day.isoformat()
                    self._daily_equity[date_str] = self._current_equity
                    # Calculate daily return
                    if prev_day_equity > 0:
                        daily_return = (self._current_equity - prev_day_equity) / prev_day_equity * 100
                        self._daily_returns.append(daily_return)
                    prev_day_equity = self._current_equity
                    # Update high water mark
                    if self._current_equity > self._high_water_mark:
                        self._high_water_mark = self._current_equity

                # New trading day - clear quote cache to prevent stale/expired options
                self._current_quote.clear()
                self._current_day = current_date
            trading_days.add(current_date)

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
                self._update_available_options(current_time)

                # Evaluate for signals
                signal = self.generator.evaluate(current_time, self._underlying_price)

                if signal:
                    self._open_trade_from_signal(signal)

        # Force close any remaining open trade
        if self._open_trade and last_time:
            self._force_exit(last_time, "eod")

        # Record final day's equity
        if self._current_day is not None:
            date_str = self._current_day.isoformat()
            self._daily_equity[date_str] = self._current_equity
            if prev_day_equity > 0:
                daily_return = (self._current_equity - prev_day_equity) / prev_day_equity * 100
                self._daily_returns.append(daily_return)
            if self._current_equity > self._high_water_mark:
                self._high_water_mark = self._current_equity

        # Calculate statistics
        result = self._calculate_statistics(start_date, end_date, len(trading_days))
        logger.info(f"Backtest complete: {len(self._trades)} trades")

        return result

    def _update_available_options(self, current_time: datetime) -> None:
        """Build options list from current quotes for signal generator.

        Args:
            current_time: Current simulation time (used for DTE calculation)
        """
        options = []
        current_date = current_time.date()

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

                # Calculate expiry date
                exp_date = date(
                    int(f"20{expiry_str[:2]}"),
                    int(expiry_str[2:4]),
                    int(expiry_str[4:6]),
                )

                # CRITICAL FIX: Skip already-expired options
                if exp_date < current_date:
                    continue

                # CRITICAL FIX: Calculate DTE using current simulation time, not quote timestamp
                dte = (exp_date - current_date).days

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

    def _calculate_position_size(self, entry_price: float) -> int:
        """Calculate number of contracts based on risk management.

        Position sizing formula:
        - risk_per_trade = current_equity * risk_per_trade_pct (2%)
        - risk_per_contract = entry_price * stop_loss_pct * 100 (per contract)
        - contracts = risk_per_trade / risk_per_contract

        Args:
            entry_price: Option entry price (per share)

        Returns:
            Number of contracts to trade
        """
        pc = self.portfolio_config

        # Calculate risk amount (2% of current equity)
        risk_per_trade = self._current_equity * (pc.risk_per_trade_pct / 100)

        # Calculate risk per contract (entry * stop_loss_pct * 100 shares)
        # e.g., $1.50 entry * 15% stop = $0.225 risk per share * 100 = $22.50 per contract
        risk_per_contract = entry_price * (self.config.stop_loss_pct / 100) * 100

        if risk_per_contract <= 0:
            return pc.min_contracts

        # Calculate contracts
        contracts = int(risk_per_trade / risk_per_contract)

        # Apply limits
        contracts = max(pc.min_contracts, min(contracts, pc.max_contracts))

        return contracts

    def _open_trade_from_signal(self, signal: "ScalpSignal") -> None:
        """Open a new trade from a signal."""
        # Apply slippage to entry price
        slippage_mult = 1 + (self.slippage_pct / 100)
        fill_price = signal.ask_price * slippage_mult

        # Calculate position size based on risk management
        contracts = self._calculate_position_size(fill_price)

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
            contracts=contracts,
            equity_before=self._current_equity,
        )

        logger.debug(
            f"Opened trade: {signal.signal_type} {signal.option_symbol} "
            f"@ ${fill_price:.2f} x {contracts} contracts (equity: ${self._current_equity:,.2f})"
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

        # Calculate hold time
        hold_seconds = (current_time - trade.entry_time).total_seconds()

        # 1. Take profit
        if pnl_pct >= self.config.take_profit_pct:
            exit_reason = "take_profit"

        # 2. Stop loss
        elif pnl_pct <= -self.config.stop_loss_pct:
            exit_reason = "stop_loss"

        # 3. TIME STOP: Exit if not profitable after time_stop_minutes
        # Trades <5min have 69% WR, trades >5min drop to 47% WR
        elif self.config.time_stop_minutes > 0:
            time_stop_seconds = self.config.time_stop_minutes * 60
            if hold_seconds >= time_stop_seconds and pnl_pct <= 0:
                exit_reason = "time_stop"

        # 4. Max hold time exit (only if max_hold_minutes is set - None = allow overnight)
        elif self.config.max_hold_minutes is not None:
            max_hold_seconds = self.config.max_hold_minutes * 60
            if hold_seconds >= max_hold_seconds:
                exit_reason = "time_exit"

        # 5. Option expiration - MUST exit on or after expiry date
        if exit_reason is None and trade.expiry:
            try:
                # Parse expiry date (format: "2024-01-07")
                expiry_date = date.fromisoformat(trade.expiry[:10])
                current_date = current_time.date()

                # Exit at market close on expiry day or if somehow past expiry
                if current_date >= expiry_date:
                    exit_reason = "expiration"
                    # Option likely worthless or near-worthless at expiration
                    # Use current quote if available, otherwise assume total loss
                    if current_price <= 0:
                        current_price = 0.01  # Penny to avoid division issues
            except (ValueError, TypeError):
                pass  # Invalid expiry format, skip check

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

        # Update portfolio equity
        self._current_equity += trade.pnl_dollars
        trade.equity_after = self._current_equity

        # Store trade
        self._trades.append(trade)
        self._open_trade = None

        # Trigger loss cooldown if needed
        if trade.pnl_dollars < 0:
            self.generator.trigger_loss_cooldown(exit_time)

        logger.debug(
            f"Closed trade: {trade.option_symbol} @ ${fill_price:.2f}, "
            f"P&L: ${trade.pnl_dollars:+.2f} ({trade.pnl_pct:+.1f}%), "
            f"reason: {exit_reason}, equity: ${self._current_equity:,.2f}"
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

            # By direction (SCALP_CALL vs SCALP_PUT)
            direction = trade.signal_type
            result.trades_by_direction[direction] = (
                result.trades_by_direction.get(direction, 0) + 1
            )
            result.pnl_by_direction[direction] = (
                result.pnl_by_direction.get(direction, 0) + trade.pnl_dollars
            )

            # By DTE
            dte = trade.dte
            result.trades_by_dte[dte] = result.trades_by_dte.get(dte, 0) + 1
            result.pnl_by_dte[dte] = result.pnl_by_dte.get(dte, 0) + trade.pnl_dollars

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

        # Win rate by direction
        for direction in result.trades_by_direction:
            dir_trades = [t for t in self._trades if t.signal_type == direction]
            dir_wins = sum(1 for t in dir_trades if t.pnl_dollars > 0)
            if dir_trades:
                result.winrate_by_direction[direction] = dir_wins / len(dir_trades)

        # Win rate by DTE
        for dte in result.trades_by_dte:
            dte_trades = [t for t in self._trades if t.dte == dte]
            dte_wins = sum(1 for t in dte_trades if t.pnl_dollars > 0)
            if dte_trades:
                result.winrate_by_dte[dte] = dte_wins / len(dte_trades)

        # Portfolio metrics
        result.starting_equity = self.portfolio_config.starting_equity
        result.ending_equity = self._current_equity
        result.total_return_pct = (
            (self._current_equity - self.portfolio_config.starting_equity)
            / self.portfolio_config.starting_equity
            * 100
        )
        result.daily_equity = self._daily_equity.copy()
        result.high_water_mark = self._high_water_mark

        # Calculate max drawdown from equity curve
        if self._daily_equity:
            equity_values = list(self._daily_equity.values())
            peak = equity_values[0]
            max_dd = 0.0
            max_dd_pct = 0.0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = peak - equity
                drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
                if drawdown > max_dd:
                    max_dd = drawdown
                    max_dd_pct = drawdown_pct
            result.max_drawdown = max_dd
            result.max_drawdown_pct = max_dd_pct

        # Calculate risk metrics from daily returns
        if self._daily_returns:
            avg_return = sum(self._daily_returns) / len(self._daily_returns)

            # Standard deviation of returns
            if len(self._daily_returns) > 1:
                variance = sum((r - avg_return) ** 2 for r in self._daily_returns) / (len(self._daily_returns) - 1)
                std_dev = math.sqrt(variance)
            else:
                std_dev = 0

            # Sharpe Ratio (assuming 0% risk-free rate, annualized)
            # Annualize: multiply by sqrt(252) for daily returns
            if std_dev > 0:
                result.sharpe_ratio = (avg_return / std_dev) * math.sqrt(252)

            # Sortino Ratio (only downside deviation)
            negative_returns = [r for r in self._daily_returns if r < 0]
            if negative_returns:
                downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
                downside_dev = math.sqrt(downside_variance)
                if downside_dev > 0:
                    result.sortino_ratio = (avg_return / downside_dev) * math.sqrt(252)

        # Calmar Ratio (annual return / max drawdown)
        if result.max_drawdown_pct > 0 and trading_days > 0:
            # Annualize return
            annual_return = result.total_return_pct * (252 / trading_days)
            result.calmar_ratio = annual_return / result.max_drawdown_pct

        # CAGR (Compound Annual Growth Rate)
        if trading_days > 0:
            years = trading_days / 252
            if years > 0 and self.portfolio_config.starting_equity > 0:
                total_return = self._current_equity / self.portfolio_config.starting_equity
                if total_return > 0:
                    result.cagr = (math.pow(total_return, 1 / years) - 1) * 100

        # Monthly returns
        result.monthly_returns = self._calculate_monthly_returns()

        return result

    def _calculate_monthly_returns(self) -> dict[str, float]:
        """Calculate returns by month from daily equity curve."""
        if not self._daily_equity:
            return {}

        monthly_returns: dict[str, float] = {}
        monthly_start: dict[str, float] = {}  # First equity of month
        monthly_end: dict[str, float] = {}  # Last equity of month

        for date_str, equity in sorted(self._daily_equity.items()):
            month_key = date_str[:7]  # YYYY-MM
            if month_key not in monthly_start:
                monthly_start[month_key] = equity
            monthly_end[month_key] = equity

        for month in monthly_start:
            start_eq = monthly_start[month]
            end_eq = monthly_end[month]
            if start_eq > 0:
                monthly_returns[month] = ((end_eq - start_eq) / start_eq) * 100

        return monthly_returns

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
