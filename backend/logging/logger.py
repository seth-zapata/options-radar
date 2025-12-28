"""Evaluation logger for capturing all recommendations and abstains.

Logs every decision with full input snapshots for later analysis.
Supports both in-memory storage and file persistence.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.logging.models import (
    GreeksSnapshot,
    InputSnapshot,
    Outcome,
    PortfolioSnapshot,
    QuoteSnapshot,
    RecommendationLog,
    UnderlyingSnapshot,
)

logger = logging.getLogger(__name__)


class EvaluationLogger:
    """Logger for capturing recommendation/abstain decisions.

    Stores all decisions in memory and optionally persists to JSON files.
    Provides access for metrics calculation and outcome recording.

    Usage:
        logger = EvaluationLogger(persist_path="./logs")

        # Log a recommendation
        logger.log_recommendation(
            recommendation=rec,
            underlying=underlying,
            options=all_options,
            portfolio_state=portfolio,
            session_id="session123",
        )

        # Log an abstain
        logger.log_abstain(
            abstain=abstain,
            underlying=underlying,
            options=all_options,
            portfolio_state=portfolio,
            session_id="session123",
        )

        # Get logs for analysis
        logs = logger.get_logs()
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        max_memory_logs: int = 1000,
    ):
        """Initialize the evaluation logger.

        Args:
            persist_path: Directory to persist logs (None for memory-only)
            max_memory_logs: Maximum logs to keep in memory
        """
        self._logs: list[RecommendationLog] = []
        self._logs_by_id: dict[str, RecommendationLog] = {}
        self._max_memory_logs = max_memory_logs
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path:
            self._persist_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Evaluation logger initialized "
            f"(persist: {self._persist_path or 'memory-only'})"
        )

    def log_recommendation(
        self,
        recommendation: Any,  # Recommendation from engine
        underlying: Any,  # UnderlyingData
        options: list[Any],  # List of AggregatedOptionData
        portfolio_state: dict[str, Any],
        session_id: str,
    ) -> RecommendationLog:
        """Log a recommendation decision.

        Args:
            recommendation: The Recommendation object
            underlying: Current underlying data
            options: All option data at decision time
            portfolio_state: Current portfolio state
            session_id: Current session ID

        Returns:
            The created RecommendationLog
        """
        now = datetime.now(timezone.utc)

        # Build input snapshot
        input_snapshot = self._build_input_snapshot(
            underlying=underlying,
            options=options,
            portfolio_state=portfolio_state,
        )

        # Create log entry
        log_entry = RecommendationLog(
            timestamp=now.isoformat(),
            decision_type="recommendation",
            underlying=recommendation.underlying,
            underlying_price=underlying.price if underlying else 0.0,
            recommendation_id=recommendation.id,
            recommendation_action=recommendation.action,
            recommendation_strike=recommendation.strike,
            recommendation_expiry=recommendation.expiry,
            recommendation_right=recommendation.right,
            recommendation_premium=recommendation.premium,
            recommendation_confidence=recommendation.confidence,
            recommendation_rationale=recommendation.rationale,
            input_snapshot=input_snapshot,
            session_id=session_id,
        )

        self._store_log(log_entry)
        logger.debug(f"Logged recommendation: {log_entry.id}")

        return log_entry

    def log_abstain(
        self,
        abstain: Any,  # Abstain from engine
        underlying: Any,  # UnderlyingData
        options: list[Any],  # List of AggregatedOptionData
        portfolio_state: dict[str, Any],
        session_id: str,
        failed_gates: list[dict[str, Any]] | None = None,
    ) -> RecommendationLog:
        """Log an abstain decision.

        Args:
            abstain: The Abstain object
            underlying: Current underlying data
            options: All option data at decision time
            portfolio_state: Current portfolio state
            session_id: Current session ID
            failed_gates: List of gates that failed

        Returns:
            The created RecommendationLog
        """
        now = datetime.now(timezone.utc)

        # Build input snapshot
        input_snapshot = self._build_input_snapshot(
            underlying=underlying,
            options=options,
            portfolio_state=portfolio_state,
        )

        # Create log entry
        log_entry = RecommendationLog(
            timestamp=now.isoformat(),
            decision_type="abstain",
            underlying=underlying.symbol if underlying else "",
            underlying_price=underlying.price if underlying else 0.0,
            abstain_reason=abstain.reason.value if hasattr(abstain.reason, 'value') else str(abstain.reason),
            abstain_resume_condition=abstain.resume_condition,
            failed_gates=failed_gates or [],
            input_snapshot=input_snapshot,
            session_id=session_id,
        )

        self._store_log(log_entry)
        logger.debug(f"Logged abstain: {log_entry.id}")

        return log_entry

    def record_outcome(
        self,
        log_id: str,
        outcome: Outcome,
    ) -> bool:
        """Record outcome for a previous log entry.

        Args:
            log_id: ID of the log entry
            outcome: Outcome data to record

        Returns:
            True if successful, False if log not found
        """
        log_entry = self._logs_by_id.get(log_id)
        if not log_entry:
            logger.warning(f"Log entry not found: {log_id}")
            return False

        log_entry.outcome = outcome

        # Re-persist if using file storage
        if self._persist_path:
            self._persist_log(log_entry)

        logger.debug(f"Recorded outcome for: {log_id}")
        return True

    def get_log(self, log_id: str) -> RecommendationLog | None:
        """Get a specific log entry by ID."""
        return self._logs_by_id.get(log_id)

    def get_logs(
        self,
        session_id: str | None = None,
        decision_type: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[RecommendationLog]:
        """Get log entries with optional filtering.

        Args:
            session_id: Filter by session ID
            decision_type: Filter by "recommendation" or "abstain"
            since: Only logs after this time
            limit: Maximum number to return

        Returns:
            List of matching log entries (newest first)
        """
        logs = self._logs

        if session_id:
            logs = [log for log in logs if log.session_id == session_id]

        if decision_type:
            logs = [log for log in logs if log.decision_type == decision_type]

        if since:
            since_str = since.isoformat() if isinstance(since, datetime) else since
            logs = [log for log in logs if log.timestamp >= since_str]

        # Sort by timestamp descending
        logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)

        if limit:
            logs = logs[:limit]

        return logs

    def get_logs_needing_outcome(
        self,
        min_age_minutes: int = 15,
    ) -> list[RecommendationLog]:
        """Get logs that need outcome recording.

        Args:
            min_age_minutes: Minimum age before recording outcome

        Returns:
            List of logs needing outcomes
        """
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - (min_age_minutes * 60)

        return [
            log for log in self._logs
            if log.outcome is None
            and datetime.fromisoformat(log.timestamp.replace('Z', '+00:00')).timestamp() < cutoff
        ]

    def _build_input_snapshot(
        self,
        underlying: Any,
        options: list[Any],
        portfolio_state: dict[str, Any],
    ) -> InputSnapshot:
        """Build a complete input snapshot."""
        now = datetime.now(timezone.utc)

        # Build quote snapshots
        quotes = []
        greeks = []

        for opt in options:
            cid = opt.canonical_id

            # Handle timestamps that may be datetime or string
            quote_ts = opt.quote_timestamp
            if quote_ts and hasattr(quote_ts, 'isoformat'):
                quote_ts = quote_ts.isoformat()
            elif not quote_ts:
                quote_ts = now.isoformat()

            quotes.append(QuoteSnapshot(
                underlying=cid.underlying,
                expiry=cid.expiry,
                strike=cid.strike,
                right=cid.right,
                bid=opt.bid,
                ask=opt.ask,
                mid=opt.mid,
                spread_percent=opt.spread_percent,
                timestamp=quote_ts,
            ))

            if opt.delta is not None:
                greeks_ts = opt.greeks_timestamp
                if greeks_ts and hasattr(greeks_ts, 'isoformat'):
                    greeks_ts = greeks_ts.isoformat()
                elif not greeks_ts:
                    greeks_ts = now.isoformat()

                greeks.append(GreeksSnapshot(
                    underlying=cid.underlying,
                    expiry=cid.expiry,
                    strike=cid.strike,
                    right=cid.right,
                    delta=opt.delta,
                    gamma=opt.gamma,
                    theta=opt.theta,
                    vega=opt.vega,
                    iv=opt.iv,
                    timestamp=greeks_ts,
                ))

        # Build underlying snapshot
        underlying_snapshot = UnderlyingSnapshot(
            symbol=underlying.symbol if underlying else "",
            price=underlying.price if underlying else 0.0,
            iv_rank=underlying.iv_rank if underlying else None,
            iv_percentile=underlying.iv_percentile if underlying else None,
            timestamp=underlying.timestamp if underlying else now.isoformat(),
        )

        # Build portfolio snapshot
        portfolio_snapshot = PortfolioSnapshot(
            total_exposure=portfolio_state.get("total_exposure", 0.0),
            open_position_count=portfolio_state.get("open_position_count", 0),
            cash_available=portfolio_state.get("cash_available", 5000.0),
            positions=portfolio_state.get("positions", []),
        )

        return InputSnapshot(
            quotes=quotes,
            greeks=greeks,
            underlying=underlying_snapshot,
            portfolio=portfolio_snapshot,
            timestamp=now.isoformat(),
        )

    def _store_log(self, log_entry: RecommendationLog) -> None:
        """Store a log entry in memory and optionally persist."""
        self._logs.append(log_entry)
        self._logs_by_id[log_entry.id] = log_entry

        # Trim memory if needed
        if len(self._logs) > self._max_memory_logs:
            old_log = self._logs.pop(0)
            if old_log.id in self._logs_by_id:
                del self._logs_by_id[old_log.id]

        # Persist if configured
        if self._persist_path:
            self._persist_log(log_entry)

    def _persist_log(self, log_entry: RecommendationLog) -> None:
        """Persist a log entry to disk."""
        if not self._persist_path:
            return

        # Organize by date
        date_str = log_entry.timestamp[:10]  # YYYY-MM-DD
        date_dir = self._persist_path / date_str
        date_dir.mkdir(exist_ok=True)

        # Write to session file
        session_file = date_dir / f"{log_entry.session_id}.jsonl"
        with open(session_file, "a") as f:
            f.write(json.dumps(log_entry.to_dict()) + "\n")

    def load_logs_from_disk(
        self,
        date: str | None = None,
        session_id: str | None = None,
    ) -> list[RecommendationLog]:
        """Load logs from disk storage.

        Args:
            date: Date string (YYYY-MM-DD) to load
            session_id: Session ID to load

        Returns:
            List of loaded log entries
        """
        if not self._persist_path:
            return []

        logs = []

        # Determine which directories to scan
        if date:
            dirs = [self._persist_path / date]
        else:
            dirs = list(self._persist_path.iterdir())

        for date_dir in dirs:
            if not date_dir.is_dir():
                continue

            # Determine which files to read
            if session_id:
                files = [date_dir / f"{session_id}.jsonl"]
            else:
                files = list(date_dir.glob("*.jsonl"))

            for log_file in files:
                if not log_file.exists():
                    continue

                with open(log_file) as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            # Convert back to RecommendationLog
                            # (simplified - would need full deserialization)
                            logs.append(data)
                        except json.JSONDecodeError:
                            continue

        return logs

    @property
    def log_count(self) -> int:
        """Total logs in memory."""
        return len(self._logs)

    @property
    def recommendation_count(self) -> int:
        """Count of recommendations in memory."""
        return len([log for log in self._logs if log.decision_type == "recommendation"])

    @property
    def abstain_count(self) -> int:
        """Count of abstains in memory."""
        return len([log for log in self._logs if log.decision_type == "abstain"])
