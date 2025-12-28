"""Session-based portfolio tracking for recommendations.

Tracks cumulative exposure from recommendations within a server session.
Resets on server restart - not persisted.

This is display-only tracking for the MVP. No actual trades are executed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from backend.engine.recommender import Recommendation

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for session tracking.

    Attributes:
        max_session_exposure: Maximum cumulative exposure per session ($)
        max_single_position: Maximum single position size ($)
        max_recommendations: Maximum recommendations to keep in history
        exposure_warning_threshold: Warn when approaching max (0-1)
    """
    max_session_exposure: float = 25000.0  # $25,000 total (50% of $50k portfolio)
    max_single_position: float = 10000.0   # $10,000 per position (20% of $50k portfolio)
    max_recommendations: int = 100        # Keep last 100 recommendations
    exposure_warning_threshold: float = 0.8  # Warn at 80%


@dataclass
class SessionStats:
    """Current session statistics.

    Attributes:
        session_id: Unique session identifier
        started_at: Session start time (ISO)
        recommendation_count: Total recommendations generated
        total_exposure: Cumulative exposure from all recommendations ($)
        exposure_remaining: How much exposure is available ($)
        exposure_percent: Percentage of max exposure used
        is_at_limit: True if no more exposure available
        is_warning: True if approaching limit
        recommendations_by_symbol: Count per underlying
        last_recommendation_at: Time of most recent recommendation
    """
    session_id: str
    started_at: str
    recommendation_count: int
    total_exposure: float
    exposure_remaining: float
    exposure_percent: float
    is_at_limit: bool
    is_warning: bool
    recommendations_by_symbol: Dict[str, int]
    last_recommendation_at: str | None


class SessionTracker:
    """Tracks recommendations and cumulative exposure within a session.

    A session starts when the server starts and ends when it stops.
    All tracking is in-memory and resets on restart.

    Usage:
        tracker = SessionTracker()

        # Check if recommendation is allowed
        if tracker.can_add_recommendation(recommendation):
            tracker.add_recommendation(recommendation)
            # Broadcast recommendation to clients

        # Get stats for UI
        stats = tracker.get_stats()
    """

    def __init__(self, config: SessionConfig | None = None):
        self.config = config or SessionConfig()

        # Generate session ID
        import uuid
        self._session_id = str(uuid.uuid4())[:8]
        self._started_at = datetime.now(timezone.utc)

        # Tracking state
        self._recommendations: List[Recommendation] = []
        self._total_exposure: float = 0.0
        self._by_symbol: Dict[str, int] = {}

        logger.info(f"Session tracker started: {self._session_id}")

    def can_add_recommendation(
        self,
        recommendation: Recommendation,
        check_session_limit: bool = True,
        check_position_limit: bool = True,
    ) -> tuple[bool, str | None]:
        """Check if a recommendation can be added to the session.

        Args:
            recommendation: The recommendation to check
            check_session_limit: Check against session max exposure
            check_position_limit: Check against single position limit

        Returns:
            Tuple of (allowed: bool, reason: str | None)
        """
        cost = recommendation.total_cost

        # Check single position limit
        if check_position_limit and cost > self.config.max_single_position:
            return (
                False,
                f"Position ${cost:.0f} exceeds single position limit "
                f"${self.config.max_single_position:.0f}"
            )

        # Check session exposure limit
        if check_session_limit:
            new_total = self._total_exposure + cost
            if new_total > self.config.max_session_exposure:
                return (
                    False,
                    f"Would exceed session limit: ${new_total:.0f} > "
                    f"${self.config.max_session_exposure:.0f}"
                )

        return (True, None)

    def add_recommendation(self, recommendation: Recommendation) -> bool:
        """Add a recommendation to the session.

        Args:
            recommendation: The recommendation to add

        Returns:
            True if added, False if rejected
        """
        allowed, reason = self.can_add_recommendation(recommendation)

        if not allowed:
            logger.warning(f"Recommendation rejected: {reason}")
            return False

        # Add to history
        self._recommendations.append(recommendation)
        self._total_exposure += recommendation.total_cost

        # Track by symbol
        symbol = recommendation.underlying
        self._by_symbol[symbol] = self._by_symbol.get(symbol, 0) + 1

        # Trim history if needed
        if len(self._recommendations) > self.config.max_recommendations:
            # Remove oldest, but don't reduce exposure (already "spent")
            self._recommendations = self._recommendations[-self.config.max_recommendations:]

        logger.info(
            f"Recommendation added: {recommendation.action} "
            f"{recommendation.underlying} ${recommendation.strike} "
            f"(${recommendation.total_cost:.0f}). "
            f"Session exposure: ${self._total_exposure:.0f}"
        )

        return True

    def get_stats(self) -> SessionStats:
        """Get current session statistics."""
        exposure_remaining = max(
            0,
            self.config.max_session_exposure - self._total_exposure
        )
        exposure_percent = (
            self._total_exposure / self.config.max_session_exposure * 100
            if self.config.max_session_exposure > 0 else 100
        )

        is_at_limit = exposure_remaining <= 0
        is_warning = (
            exposure_percent >= self.config.exposure_warning_threshold * 100
            and not is_at_limit
        )

        last_rec_at = None
        if self._recommendations:
            last_rec_at = self._recommendations[-1].generated_at

        return SessionStats(
            session_id=self._session_id,
            started_at=self._started_at.isoformat(),
            recommendation_count=len(self._recommendations),
            total_exposure=self._total_exposure,
            exposure_remaining=exposure_remaining,
            exposure_percent=round(exposure_percent, 1),
            is_at_limit=is_at_limit,
            is_warning=is_warning,
            recommendations_by_symbol=dict(self._by_symbol),
            last_recommendation_at=last_rec_at,
        )

    def get_recent_recommendations(self, limit: int = 10) -> List[Recommendation]:
        """Get the most recent recommendations.

        Args:
            limit: Maximum number to return

        Returns:
            List of recommendations, newest first
        """
        return list(reversed(self._recommendations[-limit:]))

    def reset(self) -> None:
        """Reset session tracking (for testing)."""
        import uuid
        self._session_id = str(uuid.uuid4())[:8]
        self._started_at = datetime.now(timezone.utc)
        self._recommendations = []
        self._total_exposure = 0.0
        self._by_symbol = {}
        logger.info(f"Session tracker reset: {self._session_id}")
