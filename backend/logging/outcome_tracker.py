"""Automatic outcome tracking for recommendations.

Runs as a background task and periodically records price outcomes
for recommendations, enabling paper trading validation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.config import AlpacaConfig, load_config
from backend.data.alpaca_rest import AlpacaRestClient
from backend.logging.logger import EvaluationLogger
from backend.logging.models import Outcome, RecommendationLog

logger = logging.getLogger(__name__)


@dataclass
class OutcomeTrackerConfig:
    """Configuration for outcome tracking."""

    # Check intervals (how often to look for recommendations needing outcomes)
    check_interval_seconds: float = 60.0

    # Recording intervals (when to record prices after recommendation)
    record_at_15min: bool = True
    record_at_1hr: bool = True
    record_at_close: bool = True

    # Minimum age before first recording (to avoid premature outcomes)
    min_age_minutes: int = 15


@dataclass
class OutcomeTracker:
    """Background task for automatic outcome tracking.

    Monitors recommendations and records their outcomes at specified intervals.

    Usage:
        eval_logger = EvaluationLogger(persist_path="./logs")
        alpaca_config = load_config().alpaca
        tracker = OutcomeTracker(eval_logger, alpaca_config)

        # Start tracking in background
        await tracker.start()

        # Later, stop tracking
        await tracker.stop()
    """

    eval_logger: EvaluationLogger
    alpaca_config: AlpacaConfig
    config: OutcomeTrackerConfig = field(default_factory=OutcomeTrackerConfig)

    _running: bool = field(default=False, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)
    _rest_client: AlpacaRestClient | None = field(default=None, init=False)

    # Track which recommendations we've already processed at each interval
    _recorded_15min: set[str] = field(default_factory=set, init=False)
    _recorded_1hr: set[str] = field(default_factory=set, init=False)
    _recorded_close: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        self._rest_client = AlpacaRestClient(config=self.alpaca_config)

    async def start(self) -> None:
        """Start the outcome tracking background task."""
        if self._running:
            logger.warning("Outcome tracker already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Outcome tracker started")

    async def stop(self) -> None:
        """Stop the outcome tracking background task."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Outcome tracker stopped")

    async def _run_loop(self) -> None:
        """Main tracking loop."""
        while self._running:
            try:
                await self._check_and_record_outcomes()
            except Exception as e:
                logger.error(f"Error in outcome tracker: {e}")

            await asyncio.sleep(self.config.check_interval_seconds)

    async def _check_and_record_outcomes(self) -> None:
        """Check for recommendations needing outcomes and record them."""
        now = datetime.now(timezone.utc)

        # Get recommendations without outcomes
        logs = self.eval_logger.get_logs(
            decision_type="recommendation",
            limit=100,
        )

        # Filter to those needing outcomes
        for log in logs:
            if log.outcome is not None:
                continue

            if not log.recommendation_id:
                continue

            rec_id = log.recommendation_id
            rec_time = datetime.fromisoformat(log.timestamp.replace("Z", "+00:00"))
            age = now - rec_time

            # Check if we should record 15-min outcome
            if (
                self.config.record_at_15min
                and rec_id not in self._recorded_15min
                and age >= timedelta(minutes=15)
            ):
                await self._record_15min_outcome(log, rec_time)

            # Check if we should record 1-hr outcome
            if (
                self.config.record_at_1hr
                and rec_id not in self._recorded_1hr
                and age >= timedelta(hours=1)
            ):
                await self._record_1hr_outcome(log, rec_time)

            # Check if we should record close outcome (end of day)
            if (
                self.config.record_at_close
                and rec_id not in self._recorded_close
                and self._is_after_close(rec_time, now)
            ):
                await self._record_close_outcome(log, rec_time)

    async def _record_15min_outcome(
        self,
        log: RecommendationLog,
        rec_time: datetime,
    ) -> None:
        """Record the 15-minute outcome."""
        try:
            target_time = rec_time + timedelta(minutes=15)
            price = await self._rest_client.get_price_at_time(
                log.underlying, target_time
            )

            if price is None:
                # Try current price if historical isn't available
                price = await self._rest_client.get_latest_price(log.underlying)

            if price is not None:
                # Update or create outcome
                outcome = log.outcome or self._create_empty_outcome()
                outcome = Outcome(
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    underlying_price_at_15min=price,
                    underlying_price_at_1hr=outcome.underlying_price_at_1hr,
                    underlying_price_at_close=outcome.underlying_price_at_close,
                    would_have_profited=self._calculate_profit(log, price),
                )
                self.eval_logger.record_outcome(log.id, outcome)
                self._recorded_15min.add(log.recommendation_id)
                logger.info(
                    f"Recorded 15-min outcome for {log.recommendation_id}: "
                    f"${price:.2f}"
                )

        except Exception as e:
            logger.error(f"Error recording 15-min outcome: {e}")

    async def _record_1hr_outcome(
        self,
        log: RecommendationLog,
        rec_time: datetime,
    ) -> None:
        """Record the 1-hour outcome."""
        try:
            target_time = rec_time + timedelta(hours=1)
            price = await self._rest_client.get_price_at_time(
                log.underlying, target_time
            )

            if price is None:
                price = await self._rest_client.get_latest_price(log.underlying)

            if price is not None:
                outcome = log.outcome or self._create_empty_outcome()
                outcome = Outcome(
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    underlying_price_at_15min=outcome.underlying_price_at_15min,
                    underlying_price_at_1hr=price,
                    underlying_price_at_close=outcome.underlying_price_at_close,
                    would_have_profited=self._calculate_profit(log, price),
                )
                self.eval_logger.record_outcome(log.id, outcome)
                self._recorded_1hr.add(log.recommendation_id)
                logger.info(
                    f"Recorded 1-hr outcome for {log.recommendation_id}: "
                    f"${price:.2f}"
                )

        except Exception as e:
            logger.error(f"Error recording 1-hr outcome: {e}")

    async def _record_close_outcome(
        self,
        log: RecommendationLog,
        rec_time: datetime,
    ) -> None:
        """Record the end-of-day close outcome."""
        try:
            # Get latest price (representing close)
            price = await self._rest_client.get_latest_price(log.underlying)

            if price is not None:
                outcome = log.outcome or self._create_empty_outcome()

                # Calculate theoretical P&L
                profit_result = self._calculate_profit(log, price)
                theoretical_pnl = self._calculate_pnl(log, price)

                outcome = Outcome(
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    underlying_price_at_15min=outcome.underlying_price_at_15min,
                    underlying_price_at_1hr=outcome.underlying_price_at_1hr,
                    underlying_price_at_close=price,
                    would_have_profited=profit_result,
                    theoretical_pnl=theoretical_pnl,
                    theoretical_pnl_percent=self._calculate_pnl_percent(log, theoretical_pnl),
                )
                self.eval_logger.record_outcome(log.id, outcome)
                self._recorded_close.add(log.recommendation_id)
                logger.info(
                    f"Recorded close outcome for {log.recommendation_id}: "
                    f"${price:.2f}, profit={profit_result}"
                )

        except Exception as e:
            logger.error(f"Error recording close outcome: {e}")

    def _create_empty_outcome(self) -> Outcome:
        """Create an empty outcome object."""
        return Outcome(
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )

    def _calculate_profit(self, log: RecommendationLog, current_price: float) -> bool | None:
        """Calculate if the recommendation would have profited based on direction.

        For calls: profit if price moved up
        For puts: profit if price moved down
        """
        if log.underlying_price <= 0:
            return None

        action = log.recommendation_action
        if not action:
            return None

        price_moved_up = current_price > log.underlying_price
        price_moved_down = current_price < log.underlying_price

        if action in ("BUY_CALL", "SELL_PUT"):
            return price_moved_up
        elif action in ("BUY_PUT", "SELL_CALL"):
            return price_moved_down

        return None

    def _calculate_pnl(self, log: RecommendationLog, current_price: float) -> float | None:
        """Calculate theoretical P&L based on underlying price movement.

        This is a simplified calculation based on delta approximation.
        Real options P&L would also factor in theta, IV changes, etc.
        """
        if log.underlying_price <= 0 or log.recommendation_premium is None:
            return None

        price_change = current_price - log.underlying_price
        premium = log.recommendation_premium

        # Use delta approximation (assume delta of 0.5 for simplicity if not stored)
        # Real implementation would use stored delta from log
        delta = 0.5

        action = log.recommendation_action
        if not action:
            return None

        # Calculate P&L per contract (100 shares)
        if action == "BUY_CALL":
            pnl = (delta * price_change * 100) - (premium * 100)
        elif action == "BUY_PUT":
            pnl = (-delta * price_change * 100) - (premium * 100)
        elif action == "SELL_CALL":
            pnl = (premium * 100) - (delta * price_change * 100)
        elif action == "SELL_PUT":
            pnl = (premium * 100) + (delta * price_change * 100)
        else:
            return None

        return round(pnl, 2)

    def _calculate_pnl_percent(
        self,
        log: RecommendationLog,
        theoretical_pnl: float | None,
    ) -> float | None:
        """Calculate P&L as percentage of premium paid."""
        if theoretical_pnl is None or log.recommendation_premium is None:
            return None

        if log.recommendation_premium <= 0:
            return None

        cost = log.recommendation_premium * 100  # Per contract
        return round((theoretical_pnl / cost) * 100, 1)

    def _is_after_close(self, rec_time: datetime, now: datetime) -> bool:
        """Check if we're past market close for the recommendation's trading day.

        Market close is 4:00 PM ET.
        """
        from zoneinfo import ZoneInfo

        eastern = ZoneInfo("America/New_York")
        rec_eastern = rec_time.astimezone(eastern)
        now_eastern = now.astimezone(eastern)

        # Get close time for the recommendation's day
        close_time = rec_eastern.replace(hour=16, minute=0, second=0, microsecond=0)

        # If recommendation was after close, use next day's close
        if rec_eastern > close_time:
            close_time += timedelta(days=1)

        return now_eastern > close_time

    def get_stats(self) -> dict[str, Any]:
        """Get current tracking statistics."""
        return {
            "running": self._running,
            "recorded_15min_count": len(self._recorded_15min),
            "recorded_1hr_count": len(self._recorded_1hr),
            "recorded_close_count": len(self._recorded_close),
        }


async def create_outcome_tracker(
    eval_logger: EvaluationLogger,
) -> OutcomeTracker:
    """Create and configure an outcome tracker.

    Args:
        eval_logger: The evaluation logger to use

    Returns:
        Configured OutcomeTracker
    """
    try:
        config = load_config()
        return OutcomeTracker(
            eval_logger=eval_logger,
            alpaca_config=config.alpaca,
        )
    except ValueError:
        # Config not available (e.g., mock mode)
        logger.warning("Alpaca config not available, outcome tracking disabled")
        return None
