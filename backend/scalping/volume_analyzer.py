"""Volume analyzer for detecting unusual option activity.

Tracks option volume by contract and detects spikes that may
indicate institutional activity or momentum building.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class VolumeBar:
    """Volume observation for a time period."""

    timestamp: datetime
    volume: int
    trade_count: int = 1


@dataclass
class VolumeSpike:
    """Detected volume spike on a contract.

    Indicates current volume significantly exceeds baseline.
    """

    contract: str
    current_volume: int
    baseline_volume: float
    ratio: float  # current / baseline
    timestamp: datetime

    def __repr__(self) -> str:
        return (
            f"VolumeSpike({self.contract}: {self.current_volume} "
            f"vs baseline {self.baseline_volume:.0f}, {self.ratio:.1f}x)"
        )


@dataclass
class SweepSignal:
    """Detected sweep order (large aggressive order).

    A sweep is a large order that hits multiple price levels,
    often indicating institutional activity.
    """

    direction: Literal["call_sweep", "put_sweep"]
    contracts: int
    premium: float  # Total premium in dollars
    strikes: list[float]
    timestamp: datetime

    def __repr__(self) -> str:
        return (
            f"SweepSignal({self.direction}: {self.contracts} contracts, "
            f"${self.premium:,.0f} premium, strikes: {self.strikes})"
        )


class VolumeAnalyzer:
    """Analyzes option volume for unusual activity.

    Usage:
        analyzer = VolumeAnalyzer()

        # Record volume observations
        analyzer.add_volume("TSLA240105C00420000", 500, timestamp, "C")
        analyzer.add_volume("TSLA240105P00415000", 300, timestamp, "P")

        # Check for volume spike on specific contract
        spike = analyzer.detect_volume_spike("TSLA240105C00420000")
        if spike:
            print(f"Volume spike: {spike}")

        # Get put/call ratio
        pcr = analyzer.get_put_call_ratio()
        print(f"Put/Call ratio: {pcr:.2f}")
    """

    def __init__(
        self,
        baseline_window_minutes: int = 30,
        spike_ratio_threshold: float = 2.0,
        max_history_bars: int = 100,
    ):
        """Initialize volume analyzer.

        Args:
            baseline_window_minutes: Window for baseline calculation
            spike_ratio_threshold: Ratio above which volume is a "spike"
            max_history_bars: Max volume bars to keep per contract
        """
        self.baseline_window_minutes = baseline_window_minutes
        self.spike_ratio_threshold = spike_ratio_threshold
        self.max_history_bars = max_history_bars

        # Volume history per contract
        self._volume_history: dict[str, deque[VolumeBar]] = defaultdict(
            lambda: deque(maxlen=max_history_bars)
        )

        # Aggregate volume by option type
        self._call_volume: deque[VolumeBar] = deque(maxlen=max_history_bars)
        self._put_volume: deque[VolumeBar] = deque(maxlen=max_history_bars)

        # Aggregate volume for underlying
        self._total_volume: deque[VolumeBar] = deque(maxlen=max_history_bars)

        # Cache for baseline calculations
        self._baseline_cache: dict[str, tuple[datetime, float]] = {}
        self._cache_ttl_seconds = 60  # Refresh baseline every 60s

    def add_volume(
        self,
        contract: str,
        volume: int,
        timestamp: datetime,
        option_type: Literal["C", "P"] | None = None,
    ) -> None:
        """Record a volume observation.

        Args:
            contract: OCC option symbol
            volume: Volume (number of contracts)
            timestamp: Time of observation
            option_type: "C" for call, "P" for put
        """
        bar = VolumeBar(timestamp=timestamp, volume=volume, trade_count=1)
        self._volume_history[contract].append(bar)
        self._total_volume.append(bar)

        # Track aggregate by type
        if option_type == "C":
            self._call_volume.append(bar)
        elif option_type == "P":
            self._put_volume.append(bar)

    def add_trade(
        self,
        contract: str,
        size: int,
        price: float,
        timestamp: datetime,
        option_type: Literal["C", "P"] | None = None,
    ) -> None:
        """Record a trade (convenience method).

        Args:
            contract: OCC option symbol
            size: Trade size in contracts
            price: Trade price per contract
            timestamp: Time of trade
            option_type: "C" for call, "P" for put
        """
        self.add_volume(contract, size, timestamp, option_type)

    def get_volume_ratio(
        self,
        contract: str,
        window_minutes: int = 5,
    ) -> float:
        """Get current volume vs baseline ratio.

        Args:
            contract: OCC option symbol
            window_minutes: Window for current volume calculation

        Returns:
            Ratio where 1.0 = normal, 2.0 = 2x normal volume
        """
        history = self._volume_history.get(contract)
        if not history or len(history) < 2:
            return 1.0

        # Calculate recent volume
        current_time = history[-1].timestamp
        cutoff = current_time - timedelta(minutes=window_minutes)
        recent_volume = sum(bar.volume for bar in history if bar.timestamp >= cutoff)

        # Get baseline (volume per minute)
        baseline = self._get_baseline(contract, current_time)

        if baseline <= 0:
            return 1.0

        # Scale baseline to same time period
        baseline_scaled = baseline * window_minutes

        return recent_volume / baseline_scaled if baseline_scaled > 0 else 1.0

    def _get_baseline(self, contract: str, current_time: datetime) -> float:
        """Get cached baseline or calculate new one.

        Returns volume per minute for the contract.
        """
        # Check cache
        if contract in self._baseline_cache:
            cache_time, cached_value = self._baseline_cache[contract]
            age = (current_time - cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return cached_value

        # Calculate fresh baseline
        baseline = self._calculate_baseline(contract, current_time)
        self._baseline_cache[contract] = (current_time, baseline)
        return baseline

    def _calculate_baseline(self, contract: str, current_time: datetime) -> float:
        """Calculate baseline volume per minute for contract."""
        history = self._volume_history.get(contract)
        if not history:
            return 0.0

        cutoff = current_time - timedelta(minutes=self.baseline_window_minutes)
        baseline_bars = [bar for bar in history if bar.timestamp >= cutoff]

        if len(baseline_bars) < 2:
            return 0.0

        total_volume = sum(bar.volume for bar in baseline_bars)
        time_span_minutes = (
            baseline_bars[-1].timestamp - baseline_bars[0].timestamp
        ).total_seconds() / 60

        return total_volume / time_span_minutes if time_span_minutes > 0 else 0.0

    def detect_volume_spike(self, contract: str) -> VolumeSpike | None:
        """Detect if contract has unusual volume.

        Args:
            contract: OCC option symbol

        Returns:
            VolumeSpike if spike detected, None otherwise
        """
        ratio = self.get_volume_ratio(contract)

        if ratio < self.spike_ratio_threshold:
            return None

        history = self._volume_history.get(contract)
        if not history:
            return None

        current_time = history[-1].timestamp
        baseline = self._get_baseline(contract, current_time)

        spike = VolumeSpike(
            contract=contract,
            current_volume=history[-1].volume,
            baseline_volume=baseline,
            ratio=ratio,
            timestamp=current_time,
        )

        logger.info(f"Volume spike detected: {spike}")
        return spike

    def detect_all_spikes(self) -> list[VolumeSpike]:
        """Detect volume spikes across all tracked contracts.

        Returns:
            List of VolumeSpike for all contracts with unusual volume
        """
        spikes = []
        for contract in self._volume_history:
            spike = self.detect_volume_spike(contract)
            if spike:
                spikes.append(spike)
        return spikes

    def get_put_call_ratio(self, window_minutes: int = 5) -> float:
        """Get put/call volume ratio.

        Args:
            window_minutes: Time window for calculation

        Returns:
            Ratio where >1 means more put volume, <1 means more call volume
        """
        if not self._call_volume and not self._put_volume:
            return 1.0

        # Find most recent timestamp
        latest_call = self._call_volume[-1].timestamp if self._call_volume else None
        latest_put = self._put_volume[-1].timestamp if self._put_volume else None

        if latest_call and latest_put:
            current_time = max(latest_call, latest_put)
        elif latest_call:
            current_time = latest_call
        elif latest_put:
            current_time = latest_put
        else:
            return 1.0

        cutoff = current_time - timedelta(minutes=window_minutes)

        call_vol = sum(
            bar.volume for bar in self._call_volume if bar.timestamp >= cutoff
        )
        put_vol = sum(
            bar.volume for bar in self._put_volume if bar.timestamp >= cutoff
        )

        return put_vol / call_vol if call_vol > 0 else 1.0

    def get_total_volume(self, window_minutes: int = 5) -> int:
        """Get total option volume over window.

        Args:
            window_minutes: Time window

        Returns:
            Total volume in contracts
        """
        if not self._total_volume:
            return 0

        current_time = self._total_volume[-1].timestamp
        cutoff = current_time - timedelta(minutes=window_minutes)

        return sum(bar.volume for bar in self._total_volume if bar.timestamp >= cutoff)

    def get_volume_by_type(
        self, window_minutes: int = 5
    ) -> tuple[int, int]:
        """Get call and put volume separately.

        Args:
            window_minutes: Time window

        Returns:
            Tuple of (call_volume, put_volume)
        """
        if not self._call_volume and not self._put_volume:
            return 0, 0

        # Find most recent timestamp
        timestamps = []
        if self._call_volume:
            timestamps.append(self._call_volume[-1].timestamp)
        if self._put_volume:
            timestamps.append(self._put_volume[-1].timestamp)

        if not timestamps:
            return 0, 0

        current_time = max(timestamps)
        cutoff = current_time - timedelta(minutes=window_minutes)

        call_vol = sum(
            bar.volume for bar in self._call_volume if bar.timestamp >= cutoff
        )
        put_vol = sum(
            bar.volume for bar in self._put_volume if bar.timestamp >= cutoff
        )

        return call_vol, put_vol

    def get_active_contracts(self) -> list[str]:
        """Get list of contracts with recent volume."""
        return list(self._volume_history.keys())

    def clear(self) -> None:
        """Clear all volume history."""
        self._volume_history.clear()
        self._call_volume.clear()
        self._put_volume.clear()
        self._total_volume.clear()
        self._baseline_cache.clear()
