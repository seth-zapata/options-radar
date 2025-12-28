"""Offline replay framework for evaluation testing.

Allows recording and replaying market sessions to test gate logic
and recommendation generation without live data.

See spec section 6.3 for format details.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class MarketTick:
    """Single point-in-time market state snapshot."""
    timestamp: str

    # Quote data keyed by canonical option ID
    quotes: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Greeks data keyed by canonical option ID
    greeks: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Underlying data keyed by symbol
    underlying: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "quotes": self.quotes,
            "greeks": self.greeks,
            "underlying": self.underlying,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarketTick:
        """Create from dict."""
        return cls(
            timestamp=data["timestamp"],
            quotes=data.get("quotes", {}),
            greeks=data.get("greeks", {}),
            underlying=data.get("underlying", {}),
        )


@dataclass
class ExpectedDecision:
    """Expected decision at a specific point in time."""
    timestamp: str
    decision_type: str  # "recommendation" or "abstain"

    # For recommendations
    underlying: str | None = None
    strike: float | None = None
    expiry: str | None = None
    right: str | None = None
    action: str | None = None

    # For abstains
    abstain_reason: str | None = None
    failed_gates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "timestamp": self.timestamp,
            "decisionType": self.decision_type,
        }

        if self.decision_type == "recommendation":
            result["recommendation"] = {
                "underlying": self.underlying,
                "strike": self.strike,
                "expiry": self.expiry,
                "right": self.right,
                "action": self.action,
            }
        else:
            result["abstain"] = {
                "reason": self.abstain_reason,
                "failedGates": self.failed_gates,
            }

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectedDecision:
        """Create from dict."""
        decision = cls(
            timestamp=data["timestamp"],
            decision_type=data["decisionType"],
        )

        if data["decisionType"] == "recommendation":
            rec = data.get("recommendation", {})
            decision.underlying = rec.get("underlying")
            decision.strike = rec.get("strike")
            decision.expiry = rec.get("expiry")
            decision.right = rec.get("right")
            decision.action = rec.get("action")
        else:
            abstain = data.get("abstain", {})
            decision.abstain_reason = abstain.get("reason")
            decision.failed_gates = abstain.get("failedGates", [])

        return decision


@dataclass
class ReplaySession:
    """Complete recorded session for offline replay."""
    session_id: str
    start_time: str
    end_time: str = ""
    symbols: list[str] = field(default_factory=list)

    # Time-series of market state
    market_ticks: list[MarketTick] = field(default_factory=list)

    # Expected outputs at each decision point
    expected_decisions: list[ExpectedDecision] = field(default_factory=list)

    # Metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "sessionId": self.session_id,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "symbols": self.symbols,
            "marketTicks": [t.to_dict() for t in self.market_ticks],
            "expectedDecisions": [d.to_dict() for d in self.expected_decisions],
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplaySession:
        """Create from dict."""
        return cls(
            session_id=data["sessionId"],
            start_time=data["startTime"],
            end_time=data.get("endTime", ""),
            symbols=data.get("symbols", []),
            market_ticks=[MarketTick.from_dict(t) for t in data.get("marketTicks", [])],
            expected_decisions=[
                ExpectedDecision.from_dict(d)
                for d in data.get("expectedDecisions", [])
            ],
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )


class SessionRecorder:
    """Records live sessions for later replay.

    Usage:
        recorder = SessionRecorder(persist_path="./logs/replays")
        recorder.start_session("session123", ["SPY"])

        # Each evaluation cycle:
        recorder.record_tick(underlying, options)
        recorder.record_decision(result, recommendation)

        # End of session:
        session = recorder.end_session()
    """

    def __init__(self, persist_path: str | Path | None = None):
        """Initialize the recorder.

        Args:
            persist_path: Directory to save replay files
        """
        self._persist_path = Path(persist_path) if persist_path else None
        self._current_session: ReplaySession | None = None

        if self._persist_path:
            self._persist_path.mkdir(parents=True, exist_ok=True)

    def start_session(
        self,
        session_id: str,
        symbols: list[str],
        description: str = "",
    ) -> ReplaySession:
        """Start recording a new session.

        Args:
            session_id: Unique session identifier
            symbols: Symbols being tracked
            description: Optional description

        Returns:
            The new ReplaySession
        """
        now = datetime.now(timezone.utc)

        self._current_session = ReplaySession(
            session_id=session_id,
            start_time=now.isoformat(),
            symbols=symbols,
            description=description,
        )

        logger.info(f"Started recording session: {session_id}")
        return self._current_session

    def record_tick(
        self,
        underlying: Any,  # UnderlyingData
        options: list[Any],  # List of AggregatedOptionData
    ) -> MarketTick | None:
        """Record a market state tick.

        Args:
            underlying: Current underlying data
            options: All option data at this point

        Returns:
            The recorded MarketTick, or None if no active session
        """
        if not self._current_session:
            return None

        now = datetime.now(timezone.utc)

        # Build quote and greeks dicts
        quotes: dict[str, dict[str, Any]] = {}
        greeks: dict[str, dict[str, Any]] = {}

        for opt in options:
            cid = opt.canonical_id
            key = f"{cid.underlying}:{cid.expiry}:{cid.strike}:{cid.right}"

            quotes[key] = {
                "bid": opt.bid,
                "ask": opt.ask,
                "mid": opt.mid,
                "spread_percent": opt.spread_percent,
            }

            if opt.delta is not None:
                greeks[key] = {
                    "delta": opt.delta,
                    "gamma": opt.gamma,
                    "theta": opt.theta,
                    "vega": opt.vega,
                    "iv": opt.iv,
                }

        # Build underlying dict
        underlying_data = {
            underlying.symbol: {
                "price": underlying.price,
                "iv_rank": underlying.iv_rank,
                "iv_percentile": underlying.iv_percentile,
            }
        } if underlying else {}

        tick = MarketTick(
            timestamp=now.isoformat(),
            quotes=quotes,
            greeks=greeks,
            underlying=underlying_data,
        )

        self._current_session.market_ticks.append(tick)
        return tick

    def record_decision(
        self,
        decision_type: str,
        recommendation: Any = None,  # Recommendation
        abstain: Any = None,  # Abstain
        failed_gates: list[str] | None = None,
    ) -> ExpectedDecision | None:
        """Record a decision made by the engine.

        Args:
            decision_type: "recommendation" or "abstain"
            recommendation: The Recommendation object if applicable
            abstain: The Abstain object if applicable
            failed_gates: List of failed gate names if abstaining

        Returns:
            The recorded ExpectedDecision, or None if no active session
        """
        if not self._current_session:
            return None

        now = datetime.now(timezone.utc)

        decision = ExpectedDecision(
            timestamp=now.isoformat(),
            decision_type=decision_type,
        )

        if decision_type == "recommendation" and recommendation:
            decision.underlying = recommendation.underlying
            decision.strike = recommendation.strike
            decision.expiry = recommendation.expiry
            decision.right = recommendation.right
            decision.action = recommendation.action
        elif decision_type == "abstain" and abstain:
            reason = abstain.reason
            decision.abstain_reason = reason.value if hasattr(reason, 'value') else str(reason)
            decision.failed_gates = failed_gates or []

        self._current_session.expected_decisions.append(decision)
        return decision

    def end_session(self) -> ReplaySession | None:
        """End the current recording session.

        Returns:
            The completed ReplaySession, or None if no active session
        """
        if not self._current_session:
            return None

        now = datetime.now(timezone.utc)
        self._current_session.end_time = now.isoformat()

        session = self._current_session
        self._current_session = None

        # Persist if configured
        if self._persist_path:
            self._save_session(session)

        logger.info(
            f"Ended recording session: {session.session_id} "
            f"({len(session.market_ticks)} ticks, {len(session.expected_decisions)} decisions)"
        )

        return session

    def _save_session(self, session: ReplaySession) -> None:
        """Save session to disk."""
        if not self._persist_path:
            return

        # Organize by date
        date_str = session.start_time[:10]  # YYYY-MM-DD
        date_dir = self._persist_path / date_str
        date_dir.mkdir(exist_ok=True)

        # Save as JSON
        filepath = date_dir / f"{session.session_id}.json"
        with open(filepath, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

        logger.info(f"Saved replay session to: {filepath}")


class SessionReplayer:
    """Replays recorded sessions for testing.

    Usage:
        replayer = SessionReplayer(persist_path="./logs/replays")
        session = replayer.load_session("2024-01-15", "session123")

        for tick, expected in replayer.replay(session):
            # Feed tick to engine
            result = evaluate_gates(tick)

            # Compare to expected
            assert result matches expected
    """

    def __init__(self, persist_path: str | Path | None = None):
        """Initialize the replayer.

        Args:
            persist_path: Directory containing replay files
        """
        self._persist_path = Path(persist_path) if persist_path else None

    def load_session(self, date: str, session_id: str) -> ReplaySession | None:
        """Load a session from disk.

        Args:
            date: Date string (YYYY-MM-DD)
            session_id: Session ID

        Returns:
            The loaded ReplaySession, or None if not found
        """
        if not self._persist_path:
            return None

        filepath = self._persist_path / date / f"{session_id}.json"
        if not filepath.exists():
            logger.warning(f"Session file not found: {filepath}")
            return None

        with open(filepath) as f:
            data = json.load(f)

        return ReplaySession.from_dict(data)

    def list_sessions(self, date: str | None = None) -> list[dict[str, Any]]:
        """List available sessions.

        Args:
            date: Filter by date (YYYY-MM-DD), or None for all

        Returns:
            List of session metadata
        """
        if not self._persist_path:
            return []

        sessions = []

        if date:
            dirs = [self._persist_path / date]
        else:
            dirs = [d for d in self._persist_path.iterdir() if d.is_dir()]

        for date_dir in dirs:
            if not date_dir.exists():
                continue

            for filepath in date_dir.glob("*.json"):
                try:
                    with open(filepath) as f:
                        data = json.load(f)

                    sessions.append({
                        "sessionId": data.get("sessionId"),
                        "date": date_dir.name,
                        "startTime": data.get("startTime"),
                        "endTime": data.get("endTime"),
                        "symbols": data.get("symbols", []),
                        "tickCount": len(data.get("marketTicks", [])),
                        "decisionCount": len(data.get("expectedDecisions", [])),
                        "description": data.get("description", ""),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

        return sorted(sessions, key=lambda x: x.get("startTime", ""), reverse=True)

    def replay(
        self,
        session: ReplaySession,
        speed: float = 1.0,
    ) -> Iterator[tuple[MarketTick, ExpectedDecision | None]]:
        """Replay a session tick by tick.

        Args:
            session: The session to replay
            speed: Playback speed multiplier (1.0 = real-time, ignored for now)

        Yields:
            Tuples of (MarketTick, ExpectedDecision or None)
        """
        # Build decision lookup by timestamp
        decision_lookup: dict[str, ExpectedDecision] = {
            d.timestamp: d for d in session.expected_decisions
        }

        for tick in session.market_ticks:
            # Find closest decision (within a few seconds)
            expected = decision_lookup.get(tick.timestamp)

            # If no exact match, check for close timestamps
            if not expected:
                tick_time = datetime.fromisoformat(tick.timestamp.replace('Z', '+00:00'))
                for decision in session.expected_decisions:
                    decision_time = datetime.fromisoformat(
                        decision.timestamp.replace('Z', '+00:00')
                    )
                    diff = abs((tick_time - decision_time).total_seconds())
                    if diff < 5:  # Within 5 seconds
                        expected = decision
                        break

            yield tick, expected


@dataclass
class ReplayResult:
    """Results from running a replay comparison."""
    session_id: str
    total_ticks: int
    total_decisions: int
    matched_decisions: int
    mismatched_decisions: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        """Calculate decision match rate."""
        if self.total_decisions == 0:
            return 1.0
        return self.matched_decisions / self.total_decisions

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "sessionId": self.session_id,
            "totalTicks": self.total_ticks,
            "totalDecisions": self.total_decisions,
            "matchedDecisions": self.matched_decisions,
            "mismatchedDecisions": self.mismatched_decisions,
            "matchRate": round(self.match_rate * 100, 1),
            "mismatches": self.mismatches,
        }


class ReplayComparator:
    """Compares replay expected decisions against actual engine decisions.

    Usage:
        comparator = ReplayComparator()

        for tick, expected in replayer.replay(session):
            actual = run_engine_on_tick(tick)
            comparator.record(expected, actual)

        result = comparator.get_result(session.session_id)
    """

    def __init__(self):
        """Initialize the comparator."""
        self._matches = 0
        self._mismatches: list[dict[str, Any]] = []
        self._tick_count = 0
        self._decision_count = 0

    def record(
        self,
        expected: ExpectedDecision | None,
        actual_type: str | None,
        actual_recommendation: Any = None,
        actual_abstain: Any = None,
    ) -> bool:
        """Record a comparison between expected and actual.

        Args:
            expected: Expected decision (or None)
            actual_type: Actual decision type ("recommendation", "abstain", or None)
            actual_recommendation: Actual recommendation if applicable
            actual_abstain: Actual abstain if applicable

        Returns:
            True if matched, False if mismatched
        """
        self._tick_count += 1

        if expected is None:
            # No expected decision at this tick
            return True

        self._decision_count += 1

        # Compare decision types
        if actual_type != expected.decision_type:
            self._mismatches.append({
                "timestamp": expected.timestamp,
                "expectedType": expected.decision_type,
                "actualType": actual_type,
                "reason": "Decision type mismatch",
            })
            return False

        # Compare recommendation details
        if expected.decision_type == "recommendation" and actual_recommendation:
            if (expected.underlying != actual_recommendation.underlying or
                expected.strike != actual_recommendation.strike or
                expected.right != actual_recommendation.right):
                self._mismatches.append({
                    "timestamp": expected.timestamp,
                    "expectedType": expected.decision_type,
                    "actualType": actual_type,
                    "expected": {
                        "underlying": expected.underlying,
                        "strike": expected.strike,
                        "right": expected.right,
                    },
                    "actual": {
                        "underlying": actual_recommendation.underlying,
                        "strike": actual_recommendation.strike,
                        "right": actual_recommendation.right,
                    },
                    "reason": "Recommendation details mismatch",
                })
                return False

        # Compare abstain details
        if expected.decision_type == "abstain" and actual_abstain:
            actual_reason = actual_abstain.reason
            if hasattr(actual_reason, 'value'):
                actual_reason = actual_reason.value

            if expected.abstain_reason != str(actual_reason):
                self._mismatches.append({
                    "timestamp": expected.timestamp,
                    "expectedType": expected.decision_type,
                    "actualType": actual_type,
                    "expectedReason": expected.abstain_reason,
                    "actualReason": str(actual_reason),
                    "reason": "Abstain reason mismatch",
                })
                return False

        self._matches += 1
        return True

    def get_result(self, session_id: str) -> ReplayResult:
        """Get the comparison result.

        Args:
            session_id: Session ID for the result

        Returns:
            ReplayResult with comparison statistics
        """
        return ReplayResult(
            session_id=session_id,
            total_ticks=self._tick_count,
            total_decisions=self._decision_count,
            matched_decisions=self._matches,
            mismatched_decisions=len(self._mismatches),
            mismatches=self._mismatches,
        )

    def reset(self) -> None:
        """Reset comparator state for a new comparison."""
        self._matches = 0
        self._mismatches = []
        self._tick_count = 0
        self._decision_count = 0
