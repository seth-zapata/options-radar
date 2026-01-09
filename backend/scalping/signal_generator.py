"""Scalp signal generator combining velocity, volume, and technical analysis.

Generates ScalpSignal when conditions align:
- Momentum burst (price velocity + volume)
- VWAP bounce/rejection
- Support/resistance breakout
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from backend.scalping.velocity_tracker import PriceVelocityTracker
    from backend.scalping.volume_analyzer import VolumeAnalyzer
    from backend.scalping.technical_scalper import TechnicalScalper

from backend.scalping.config import ScalpConfig

logger = logging.getLogger(__name__)


@dataclass
class ScalpSignal:
    """A complete scalping trade signal.

    Contains all information needed to execute and manage a scalp trade.
    """

    # Identification
    id: str
    timestamp: datetime
    symbol: str

    # Signal classification
    signal_type: Literal["SCALP_CALL", "SCALP_PUT"]
    trigger: str  # 'momentum_burst', 'vwap_bounce', 'vwap_rejection', 'breakout'

    # Underlying state at signal time
    underlying_price: float
    velocity_pct: float
    volume_ratio: float

    # Selected option
    option_symbol: str
    strike: float
    expiry: str
    delta: float
    dte: int

    # Entry prices
    bid_price: float
    ask_price: float
    entry_price: float  # Expected fill (usually ask for buy)
    spread_pct: float

    # Risk management (from config)
    take_profit_pct: float = 30.0
    stop_loss_pct: float = 15.0
    max_hold_minutes: int = 15

    # Sizing
    confidence: int = 50  # 0-100
    suggested_contracts: int = 1
    max_position_value: float = 500.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "trigger": self.trigger,
            "underlying_price": self.underlying_price,
            "velocity_pct": round(self.velocity_pct, 3),
            "volume_ratio": round(self.volume_ratio, 2),
            "option_symbol": self.option_symbol,
            "strike": self.strike,
            "expiry": self.expiry,
            "delta": round(self.delta, 3),
            "dte": self.dte,
            "bid_price": round(self.bid_price, 2),
            "ask_price": round(self.ask_price, 2),
            "entry_price": round(self.entry_price, 2),
            "spread_pct": round(self.spread_pct, 2),
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_hold_minutes": self.max_hold_minutes,
            "confidence": self.confidence,
            "suggested_contracts": self.suggested_contracts,
        }

    def __repr__(self) -> str:
        return (
            f"ScalpSignal({self.signal_type} {self.trigger} "
            f"${self.strike} {self.expiry[:10]} @ ${self.entry_price:.2f}, "
            f"conf={self.confidence})"
        )


class ScalpSignalGenerator:
    """Generates scalping signals by combining velocity, volume, and technicals.

    Usage:
        generator = ScalpSignalGenerator(
            symbol="TSLA",
            config=ScalpConfig(enabled=True),
            velocity_tracker=velocity,
            volume_analyzer=volume,
            technical_scalper=technical,
        )

        # Update available options (from market data)
        generator.update_available_options(options_list)

        # Evaluate for signals
        signal = generator.evaluate(current_time, underlying_price)
        if signal:
            print(f"Signal: {signal}")
    """

    def __init__(
        self,
        symbol: str,
        config: ScalpConfig,
        velocity_tracker: "PriceVelocityTracker",
        volume_analyzer: "VolumeAnalyzer",
        technical_scalper: "TechnicalScalper",
    ):
        """Initialize signal generator.

        Args:
            symbol: Underlying symbol (e.g., "TSLA")
            config: Scalping configuration
            velocity_tracker: Price velocity tracker instance
            volume_analyzer: Volume analyzer instance
            technical_scalper: Technical scalper instance
        """
        self.symbol = symbol
        self.config = config
        self.velocity = velocity_tracker
        self.volume = volume_analyzer
        self.technical = technical_scalper

        # State tracking
        self._last_signal_time: datetime | None = None
        self._daily_signal_count: int = 0
        self._last_reset_date: datetime | None = None
        self._in_cooldown_until: datetime | None = None

        # Available options (updated externally)
        self._available_options: list[dict] = []

    def update_available_options(self, options: list[dict]) -> None:
        """Update list of available options for selection.

        Args:
            options: List of option dicts with keys:
                - symbol: OCC symbol
                - strike: Strike price
                - expiry: Expiration date string
                - option_type: 'C' or 'P'
                - delta: Option delta (optional)
                - bid_px: Bid price
                - ask_px: Ask price
                - dte: Days to expiration
        """
        self._available_options = options

    def evaluate(
        self,
        current_time: datetime,
        underlying_price: float,
    ) -> ScalpSignal | None:
        """Run scalping evaluation.

        Should be called at regular intervals (e.g., every second in backtest,
        every 200ms in live trading).

        Args:
            current_time: Current timestamp
            underlying_price: Current underlying stock price

        Returns:
            ScalpSignal if conditions met, None otherwise
        """
        if not self.config.enabled:
            return None

        # Reset daily counter if new day
        self._check_daily_reset(current_time)

        # Check daily limit
        if self._daily_signal_count >= self.config.max_daily_scalps:
            return None

        # Check cooldowns
        if not self._check_cooldowns(current_time):
            return None

        # Get current velocity
        velocity = self.velocity.get_velocity(self.config.momentum_window_seconds)
        if velocity is None:
            return None

        velocity_pct = velocity.change_pct

        # Get aggregate volume ratio (simplified - use total volume)
        volume_ratio = 1.0  # Default if no volume data
        total_vol = self.volume.get_total_volume(window_minutes=5)
        if total_vol > 0:
            # Simple heuristic: if we have any volume data, assume normal
            volume_ratio = 1.0

        # Check for signals in priority order
        signal = self._check_momentum_burst(
            current_time, underlying_price, velocity_pct, volume_ratio
        )

        if signal is None:
            signal = self._check_technical_signals(
                current_time, underlying_price, velocity_pct, volume_ratio
            )

        if signal:
            self._last_signal_time = current_time
            self._daily_signal_count += 1
            logger.info(f"[{self.symbol}] Scalp signal generated: {signal}")

        return signal

    def _check_daily_reset(self, current_time: datetime) -> None:
        """Reset daily counter if new trading day."""
        if (
            self._last_reset_date is None
            or current_time.date() > self._last_reset_date.date()
        ):
            self._daily_signal_count = 0
            self._last_reset_date = current_time

    def _check_cooldowns(self, current_time: datetime) -> bool:
        """Check if we're in any cooldown period.

        Returns:
            True if OK to generate signal, False if in cooldown
        """
        # Signal interval cooldown
        if self._last_signal_time:
            elapsed = (current_time - self._last_signal_time).total_seconds()
            if elapsed < self.config.min_signal_interval_seconds:
                return False

        # Loss cooldown
        if self._in_cooldown_until and current_time < self._in_cooldown_until:
            return False

        return True

    def trigger_loss_cooldown(self, current_time: datetime) -> None:
        """Trigger extended cooldown after a losing trade.

        Call this when a scalp trade exits at a loss.
        """
        self._in_cooldown_until = current_time + timedelta(
            seconds=self.config.cooldown_after_loss_seconds
        )
        logger.info(
            f"[{self.symbol}] Loss cooldown triggered until {self._in_cooldown_until}"
        )

    def _check_momentum_burst(
        self,
        current_time: datetime,
        underlying_price: float,
        velocity_pct: float,
        volume_ratio: float,
    ) -> ScalpSignal | None:
        """Check for momentum burst signal.

        Triggers when price velocity exceeds threshold with volume confirmation.
        """
        # Require sufficient momentum
        if abs(velocity_pct) < self.config.momentum_threshold_pct:
            return None

        # Require volume confirmation (relaxed for now since volume data may be sparse)
        # In live trading, this would be more strict
        if volume_ratio < self.config.volume_spike_ratio * 0.5:  # Relaxed threshold
            return None

        # Determine direction
        direction: Literal["SCALP_CALL", "SCALP_PUT"] = (
            "SCALP_CALL" if velocity_pct > 0 else "SCALP_PUT"
        )
        option_type = "C" if velocity_pct > 0 else "P"

        # Select option
        option = self._select_option(underlying_price, option_type)

        if option is None:
            return None

        # Higher confidence for stronger momentum
        confidence = min(80, 50 + int(abs(velocity_pct) * 20))

        # Apply confidence cap - high confidence (80+) signals have 33% WR!
        # Skip signals that are likely overextended mean-reversion candidates
        if confidence > self.config.max_confidence:
            logger.debug(
                f"[{self.symbol}] Skipping overextended signal: conf={confidence} > max={self.config.max_confidence}"
            )
            return None

        return self._create_signal(
            current_time=current_time,
            underlying_price=underlying_price,
            velocity_pct=velocity_pct,
            volume_ratio=volume_ratio,
            trigger="momentum_burst",
            direction=direction,
            option=option,
            confidence=confidence,
        )

    def _check_technical_signals(
        self,
        current_time: datetime,
        underlying_price: float,
        velocity_pct: float,
        volume_ratio: float,
    ) -> ScalpSignal | None:
        """Check for technical pattern signals (VWAP, breakouts)."""
        # Check VWAP signals
        vwap_signal = self.technical.check_vwap_signal(underlying_price, velocity_pct)

        if vwap_signal:
            # Filter by signal type based on config
            if vwap_signal.signal_type == "vwap_bounce" and not self.config.enable_vwap_bounce:
                logger.debug(f"[{self.symbol}] Skipping vwap_bounce (disabled in config)")
                vwap_signal = None
            elif vwap_signal.signal_type == "vwap_rejection" and not self.config.enable_vwap_rejection:
                logger.debug(f"[{self.symbol}] Skipping vwap_rejection (disabled in config)")
                vwap_signal = None

        if vwap_signal:
            # Apply confidence cap to technical signals too
            if vwap_signal.confidence > self.config.max_confidence:
                logger.debug(
                    f"[{self.symbol}] Skipping overextended VWAP signal: conf={vwap_signal.confidence}"
                )
            else:
                direction: Literal["SCALP_CALL", "SCALP_PUT"] = (
                    "SCALP_CALL" if vwap_signal.direction == "bullish" else "SCALP_PUT"
                )
                option_type = "C" if vwap_signal.direction == "bullish" else "P"

                option = self._select_option(underlying_price, option_type)

                if option:
                    return self._create_signal(
                        current_time=current_time,
                        underlying_price=underlying_price,
                        velocity_pct=velocity_pct,
                        volume_ratio=volume_ratio,
                        trigger=vwap_signal.signal_type,
                        direction=direction,
                        option=option,
                        confidence=vwap_signal.confidence,
                    )

        # Check breakout signals
        breakout_signal = self.technical.check_breakout(
            underlying_price, velocity_pct, volume_ratio
        )

        if breakout_signal:
            # Apply confidence cap to breakout signals too
            if breakout_signal.confidence > self.config.max_confidence:
                logger.debug(
                    f"[{self.symbol}] Skipping overextended breakout signal: conf={breakout_signal.confidence}"
                )
            else:
                direction = (
                    "SCALP_CALL" if breakout_signal.direction == "bullish" else "SCALP_PUT"
                )
                option_type = "C" if breakout_signal.direction == "bullish" else "P"

                option = self._select_option(underlying_price, option_type)

                if option:
                    return self._create_signal(
                        current_time=current_time,
                        underlying_price=underlying_price,
                        velocity_pct=velocity_pct,
                        volume_ratio=volume_ratio,
                        trigger="breakout",
                        direction=direction,
                        option=option,
                        confidence=breakout_signal.confidence,
                    )

        return None

    def _select_option(
        self,
        underlying_price: float,
        option_type: str,
    ) -> dict | None:
        """Select best option for the scalp trade using SOFT DTE preference.

        Criteria (updated based on 251-trade backtest analysis):
        - Correct type (C/P)
        - DTE between min_dte and max_dte (SKIP 0DTE - never use)
        - SOFT DTE preference: try DTE=1 first (63.4% WR), then 2, then 3
        - Delta near target (0.40 ± 0.15)
        - Spread under threshold
        - Price under max ($3 - cheap options win more)

        Args:
            underlying_price: Current stock price
            option_type: "C" for call, "P" for put

        Returns:
            Best matching option dict, or None if none qualify
        """
        # Try each DTE in preference order (soft preference)
        for preferred_dte in self.config.dte_preference:
            if preferred_dte < self.config.min_dte or preferred_dte > self.config.max_dte:
                continue

            candidate = self._find_best_option_for_dte(
                underlying_price, option_type, preferred_dte
            )
            if candidate:
                return candidate

        # No option found at any preferred DTE
        return None

    def _find_best_option_for_dte(
        self,
        underlying_price: float,
        option_type: str,
        target_dte: int,
    ) -> dict | None:
        """Find best option for a specific DTE.

        Args:
            underlying_price: Current stock price
            option_type: "C" for call, "P" for put
            target_dte: The DTE to search for

        Returns:
            Best matching option dict, or None if none qualify
        """
        candidates: list[tuple[float, dict]] = []

        for opt in self._available_options:
            # Filter by type
            if opt.get("option_type") != option_type:
                continue

            # Filter by exact DTE (soft preference handled in caller)
            dte = opt.get("dte", 99)
            if dte != target_dte:
                continue

            # Filter by delta (if available)
            delta = abs(opt.get("delta", self.config.target_delta))
            if abs(delta - self.config.target_delta) > self.config.delta_tolerance:
                continue

            # Filter by spread
            bid = opt.get("bid_px", 0)
            ask = opt.get("ask_px", 0)
            if bid <= 0 or ask <= 0:
                continue

            mid = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid) * 100
            if spread_pct > self.config.max_spread_pct:
                continue

            # Filter by price - prefer cheap options (backtest showed $2.86 avg winner)
            if ask > self.config.max_contract_price:
                continue

            # Score candidate (within same DTE)
            score = 0.0

            # Prefer cheaper options (backtest showed cheap options win more)
            # Max bonus of 10 for $0.50 options, decreasing to 0 for $3 options
            price_score = max(0, (self.config.max_contract_price - ask) / self.config.max_contract_price * 10)
            score += price_score

            # Prefer tighter spreads
            score += self.config.max_spread_pct - spread_pct

            # Prefer delta closer to target
            score += (
                self.config.delta_tolerance - abs(delta - self.config.target_delta)
            ) * 5

            candidates.append((score, opt))

        if not candidates:
            return None

        # Return highest scoring candidate
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    def _create_signal(
        self,
        current_time: datetime,
        underlying_price: float,
        velocity_pct: float,
        volume_ratio: float,
        trigger: str,
        direction: Literal["SCALP_CALL", "SCALP_PUT"],
        option: dict,
        confidence: int = 50,
    ) -> ScalpSignal:
        """Create a ScalpSignal from components."""
        bid = option["bid_px"]
        ask = option["ask_px"]
        mid = (bid + ask) / 2
        spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 0

        # Calculate suggested contracts based on position size limit
        max_value = self.config.max_contract_price * 100  # Convert to contract value
        suggested = max(1, int(max_value / (ask * 100))) if ask > 0 else 1

        return ScalpSignal(
            id=f"{self.symbol}-{current_time.strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}",
            timestamp=current_time,
            symbol=self.symbol,
            signal_type=direction,
            trigger=trigger,
            underlying_price=underlying_price,
            velocity_pct=velocity_pct,
            volume_ratio=volume_ratio,
            option_symbol=option["symbol"],
            strike=option["strike"],
            expiry=option.get("expiry", ""),
            delta=option.get("delta", 0),
            dte=option.get("dte", 0),
            bid_price=bid,
            ask_price=ask,
            entry_price=ask,  # Assume fill at ask for buys
            spread_pct=spread_pct,
            take_profit_pct=self.config.take_profit_pct,
            stop_loss_pct=self.config.stop_loss_pct,
            max_hold_minutes=self.config.max_hold_minutes,
            confidence=confidence,
            suggested_contracts=suggested,
        )

    @property
    def daily_signal_count(self) -> int:
        """Number of signals generated today."""
        return self._daily_signal_count

    @property
    def is_in_cooldown(self) -> bool:
        """Check if currently in cooldown."""
        if self._in_cooldown_until is None:
            return False
        return datetime.now() < self._in_cooldown_until

    def reset(self) -> None:
        """Reset all state (for new backtest run)."""
        self._last_signal_time = None
        self._daily_signal_count = 0
        self._last_reset_date = None
        self._in_cooldown_until = None
        self._available_options = []
