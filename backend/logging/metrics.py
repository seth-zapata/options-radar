"""Metrics calculation for evaluation.

Calculates key performance metrics from logged recommendations/abstains.
See spec section 6.2 for metric definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.logging.models import RecommendationLog


@dataclass
class EvaluationMetrics:
    """Aggregate metrics calculated from log entries.

    See spec section 6.2 for definitions and targets.
    """
    # Time range
    period_start: str = ""
    period_end: str = ""
    duration_hours: float = 0.0

    # Volume metrics
    total_decisions: int = 0
    recommendation_count: int = 0
    abstain_count: int = 0

    # Rate metrics
    recommendations_per_hour: float = 0.0
    abstention_rate: float = 0.0  # Target: 30-70%

    # Quality metrics (require outcomes)
    outcomes_recorded: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0  # Target: > 50%
    false_positive_rate: float = 0.0  # Target: < 50%

    # Regret metrics (abstains that would have profited)
    regret_count: int = 0
    regret_rate: float = 0.0  # Track, minimize

    # Profitability metrics
    total_theoretical_pnl: float = 0.0
    avg_win_pnl: float = 0.0
    avg_loss_pnl: float = 0.0
    profit_factor: float = 0.0  # Total wins / Total losses

    # Data quality metrics
    avg_spread_at_rec: float = 0.0  # Target: < 5%
    stale_data_abstains: int = 0
    stale_data_rate: float = 0.0

    # Gate analysis
    gate_failure_counts: dict[str, int] = field(default_factory=dict)
    most_failed_gate: str = ""

    # Confidence calibration
    avg_confidence: float = 0.0
    high_confidence_win_rate: float = 0.0  # Win rate for confidence >= 80
    low_confidence_win_rate: float = 0.0  # Win rate for confidence < 60

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "periodStart": self.period_start,
            "periodEnd": self.period_end,
            "durationHours": round(self.duration_hours, 2),
            "volume": {
                "totalDecisions": self.total_decisions,
                "recommendations": self.recommendation_count,
                "abstains": self.abstain_count,
            },
            "rates": {
                "recommendationsPerHour": round(self.recommendations_per_hour, 2),
                "abstentionRate": round(self.abstention_rate * 100, 1),
            },
            "outcomes": {
                "recorded": self.outcomes_recorded,
                "wins": self.win_count,
                "losses": self.loss_count,
                "winRate": round(self.win_rate * 100, 1),
                "falsePositiveRate": round(self.false_positive_rate * 100, 1),
            },
            "regret": {
                "count": self.regret_count,
                "rate": round(self.regret_rate * 100, 1),
            },
            "profitability": {
                "totalTheoreticalPnl": round(self.total_theoretical_pnl, 2),
                "avgWinPnl": round(self.avg_win_pnl, 2),
                "avgLossPnl": round(self.avg_loss_pnl, 2),
                "profitFactor": round(self.profit_factor, 2),
            },
            "dataQuality": {
                "avgSpreadAtRec": round(self.avg_spread_at_rec * 100, 2),
                "staleDataAbstains": self.stale_data_abstains,
                "staleDataRate": round(self.stale_data_rate * 100, 1),
            },
            "gateAnalysis": {
                "failureCounts": self.gate_failure_counts,
                "mostFailed": self.most_failed_gate,
            },
            "confidenceCalibration": {
                "avgConfidence": round(self.avg_confidence, 1),
                "highConfidenceWinRate": round(self.high_confidence_win_rate * 100, 1),
                "lowConfidenceWinRate": round(self.low_confidence_win_rate * 100, 1),
            },
        }


class MetricsCalculator:
    """Calculates evaluation metrics from log entries.

    Usage:
        calculator = MetricsCalculator()
        metrics = calculator.calculate(logs)
    """

    def calculate(
        self,
        logs: list[RecommendationLog],
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> EvaluationMetrics:
        """Calculate metrics from log entries.

        Args:
            logs: List of RecommendationLog entries
            period_start: Start of period (default: earliest log)
            period_end: End of period (default: latest log)

        Returns:
            EvaluationMetrics with calculated values
        """
        if not logs:
            return EvaluationMetrics()

        # Sort by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.timestamp)

        # Determine period
        if period_start is None:
            period_start = datetime.fromisoformat(
                sorted_logs[0].timestamp.replace('Z', '+00:00')
            )
        if period_end is None:
            period_end = datetime.fromisoformat(
                sorted_logs[-1].timestamp.replace('Z', '+00:00')
            )

        duration_hours = (period_end - period_start).total_seconds() / 3600
        if duration_hours == 0:
            duration_hours = 1 / 60  # At least 1 minute

        metrics = EvaluationMetrics(
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
            duration_hours=duration_hours,
        )

        # Volume metrics
        metrics.total_decisions = len(logs)
        metrics.recommendation_count = len([
            log for log in logs if log.decision_type == "recommendation"
        ])
        metrics.abstain_count = len([
            log for log in logs if log.decision_type == "abstain"
        ])

        # Rate metrics
        metrics.recommendations_per_hour = (
            metrics.recommendation_count / duration_hours
            if duration_hours > 0 else 0
        )
        metrics.abstention_rate = (
            metrics.abstain_count / metrics.total_decisions
            if metrics.total_decisions > 0 else 0
        )

        # Process recommendations with outcomes
        recs_with_outcomes = [
            log for log in logs
            if log.decision_type == "recommendation" and log.outcome is not None
        ]

        metrics.outcomes_recorded = len(recs_with_outcomes)

        wins = [log for log in recs_with_outcomes if log.outcome.would_have_profited]
        losses = [log for log in recs_with_outcomes if log.outcome.would_have_profited is False]

        metrics.win_count = len(wins)
        metrics.loss_count = len(losses)
        metrics.win_rate = (
            len(wins) / len(recs_with_outcomes)
            if recs_with_outcomes else 0
        )
        metrics.false_positive_rate = (
            len(losses) / len(recs_with_outcomes)
            if recs_with_outcomes else 0
        )

        # Regret metrics (abstains with missed opportunities)
        abstains_with_outcomes = [
            log for log in logs
            if log.decision_type == "abstain" and log.outcome is not None
        ]
        regrets = [
            log for log in abstains_with_outcomes
            if log.outcome.missed_opportunity
        ]
        metrics.regret_count = len(regrets)
        metrics.regret_rate = (
            len(regrets) / len(abstains_with_outcomes)
            if abstains_with_outcomes else 0
        )

        # Profitability
        total_wins = sum(
            log.outcome.theoretical_pnl or 0
            for log in wins
        )
        total_losses = abs(sum(
            log.outcome.theoretical_pnl or 0
            for log in losses
        ))

        metrics.total_theoretical_pnl = total_wins - total_losses
        metrics.avg_win_pnl = total_wins / len(wins) if wins else 0
        metrics.avg_loss_pnl = -total_losses / len(losses) if losses else 0
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Data quality - average spread at recommendation time
        rec_logs = [log for log in logs if log.decision_type == "recommendation"]
        spreads = []
        for log in rec_logs:
            if log.input_snapshot:
                for quote in log.input_snapshot.quotes:
                    if (
                        quote.strike == log.recommendation_strike
                        and quote.right == log.recommendation_right
                        and quote.spread_percent is not None
                    ):
                        spreads.append(quote.spread_percent / 100)
                        break

        metrics.avg_spread_at_rec = sum(spreads) / len(spreads) if spreads else 0

        # Stale data analysis
        stale_abstains = [
            log for log in logs
            if log.decision_type == "abstain"
            and any("stale" in gate.get("name", "").lower() or "fresh" in gate.get("name", "").lower()
                    for gate in log.failed_gates)
        ]
        metrics.stale_data_abstains = len(stale_abstains)
        metrics.stale_data_rate = (
            len(stale_abstains) / metrics.abstain_count
            if metrics.abstain_count > 0 else 0
        )

        # Gate failure analysis
        gate_counts: dict[str, int] = {}
        for log in logs:
            if log.decision_type == "abstain":
                for gate in log.failed_gates:
                    name = gate.get("name", "unknown")
                    gate_counts[name] = gate_counts.get(name, 0) + 1

        metrics.gate_failure_counts = gate_counts
        if gate_counts:
            metrics.most_failed_gate = max(gate_counts, key=gate_counts.get)

        # Confidence calibration
        confidences = [
            log.recommendation_confidence
            for log in rec_logs
            if log.recommendation_confidence is not None
        ]
        metrics.avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        high_conf_recs = [
            log for log in recs_with_outcomes
            if log.recommendation_confidence is not None
            and log.recommendation_confidence >= 80
        ]
        high_conf_wins = [log for log in high_conf_recs if log.outcome.would_have_profited]
        metrics.high_confidence_win_rate = (
            len(high_conf_wins) / len(high_conf_recs)
            if high_conf_recs else 0
        )

        low_conf_recs = [
            log for log in recs_with_outcomes
            if log.recommendation_confidence is not None
            and log.recommendation_confidence < 60
        ]
        low_conf_wins = [log for log in low_conf_recs if log.outcome.would_have_profited]
        metrics.low_confidence_win_rate = (
            len(low_conf_wins) / len(low_conf_recs)
            if low_conf_recs else 0
        )

        return metrics

    def calculate_rolling(
        self,
        logs: list[RecommendationLog],
        window_hours: float = 1.0,
    ) -> list[EvaluationMetrics]:
        """Calculate rolling metrics over time windows.

        Args:
            logs: List of RecommendationLog entries
            window_hours: Size of each window

        Returns:
            List of metrics for each window
        """
        if not logs:
            return []

        sorted_logs = sorted(logs, key=lambda x: x.timestamp)

        # Group logs by window
        windows: list[list[RecommendationLog]] = []
        current_window: list[RecommendationLog] = []
        window_start = None

        for log in sorted_logs:
            log_time = datetime.fromisoformat(log.timestamp.replace('Z', '+00:00'))

            if window_start is None:
                window_start = log_time
                current_window = [log]
            elif (log_time - window_start).total_seconds() / 3600 < window_hours:
                current_window.append(log)
            else:
                if current_window:
                    windows.append(current_window)
                window_start = log_time
                current_window = [log]

        if current_window:
            windows.append(current_window)

        # Calculate metrics for each window
        return [self.calculate(window) for window in windows]
