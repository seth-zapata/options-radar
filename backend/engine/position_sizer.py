"""Dynamic position sizing based on signal conviction.

Sizes positions based on:
1. Regime strength (strong/moderate/weak)
2. Technical confirmations (0-3 indicators)
3. VIX regime modifier
4. Other confidence factors

This improves risk-adjusted returns by concentrating capital
where edge is strongest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""

    base_size: float  # Base position size (e.g., 0.10 for 10%)
    final_size: float  # Final adjusted size
    regime_modifier: float  # Modifier from regime strength
    tech_modifier: float  # Modifier from technical confirmations
    vix_modifier: float  # Modifier from VIX regime
    message: str  # Human-readable explanation


class DynamicPositionSizer:
    """Calculates dynamic position sizes based on signal conviction.

    Signal Strength Mapping:
    | Regime Strength | Tech Confirmations | Base Modifier |
    |-----------------|-------------------|---------------|
    | Strong          | 3/3 technicals    | 1.2 (120%)    |
    | Strong          | 2/3 technicals    | 1.1 (110%)    |
    | Moderate        | 3/3 technicals    | 1.0 (100%)    |
    | Moderate        | 2/3 technicals    | 0.9 (90%)     |
    | Moderate        | 1/3 technicals    | 0.7 (70%)     |
    | Weak            | Any               | 0.5 (50%)     |

    Combined with VIX modifier:
    final_position_size = base_size * regime_modifier * tech_modifier * vix_modifier

    Example:
    - Base size: 10%
    - Strong regime: × 1.15
    - 3/3 technicals: × 1.05
    - VIX elevated: × 0.5
    - Final size: 10% × 1.15 × 1.05 × 0.5 = 6.0%
    """

    # Regime strength multipliers
    REGIME_MULTIPLIERS = {
        "strong": 1.15,
        "moderate": 1.0,
        "weak": 0.7,
    }

    # Technical confirmation multipliers
    TECH_MULTIPLIERS = {
        3: 1.05,  # All 3 technicals confirm
        2: 1.0,   # 2 technicals confirm
        1: 0.85,  # Only 1 technical
        0: 0.5,   # No confirmation (shouldn't trade, but just in case)
    }

    def __init__(
        self,
        base_size: float = 0.10,
        max_size: float = 0.15,
        min_size: float = 0.05,
        enabled: bool = True,
    ):
        """Initialize dynamic position sizer.

        Args:
            base_size: Base position size as fraction (0.10 = 10%)
            max_size: Maximum position size cap (0.15 = 15%)
            min_size: Minimum position size floor (0.05 = 5%)
            enabled: Whether dynamic sizing is enabled
        """
        self.base_size = base_size
        self.max_size = max_size
        self.min_size = min_size
        self.enabled = enabled

    def calculate(
        self,
        regime_strength: Optional[str] = None,
        technical_confirmations: int = 0,
        vix_modifier: float = 1.0,
    ) -> PositionSizeResult:
        """Calculate final position size.

        Args:
            regime_strength: "strong", "moderate", or "weak"
            technical_confirmations: Number of confirming technicals (0-3)
            vix_modifier: Modifier from VIX regime (0.0-1.0)

        Returns:
            PositionSizeResult with final size and breakdown
        """
        if not self.enabled:
            return PositionSizeResult(
                base_size=self.base_size,
                final_size=self.base_size,
                regime_modifier=1.0,
                tech_modifier=1.0,
                vix_modifier=1.0,
                message=f"Dynamic sizing disabled, using base size {self.base_size:.1%}",
            )

        # Get regime modifier
        regime_mod = self.REGIME_MULTIPLIERS.get(
            regime_strength or "moderate",
            1.0,
        )

        # Get technical modifier
        tech_mod = self.TECH_MULTIPLIERS.get(
            min(technical_confirmations, 3),
            1.0,
        )

        # Calculate final size
        final_size = self.base_size * regime_mod * tech_mod * vix_modifier

        # Apply caps
        if final_size > 0:
            final_size = max(min(final_size, self.max_size), self.min_size)

        # Build explanation message
        parts = [f"{self.base_size:.0%} base"]
        if regime_strength:
            parts.append(f"{regime_strength} regime ×{regime_mod:.2f}")
        if technical_confirmations > 0:
            parts.append(f"{technical_confirmations}/3 tech ×{tech_mod:.2f}")
        if vix_modifier < 1.0:
            parts.append(f"VIX ×{vix_modifier:.1f}")

        message = f"Position size: {final_size:.1%} ({', '.join(parts)})"

        logger.debug(message)

        return PositionSizeResult(
            base_size=self.base_size,
            final_size=final_size,
            regime_modifier=regime_mod,
            tech_modifier=tech_mod,
            vix_modifier=vix_modifier,
            message=message,
        )

    def calculate_contracts(
        self,
        portfolio_value: float,
        option_price: float,
        regime_strength: Optional[str] = None,
        technical_confirmations: int = 0,
        vix_modifier: float = 1.0,
    ) -> tuple[int, PositionSizeResult]:
        """Calculate number of contracts to buy.

        Args:
            portfolio_value: Total portfolio value
            option_price: Price per option contract (not per share)
            regime_strength: "strong", "moderate", or "weak"
            technical_confirmations: Number of confirming technicals (0-3)
            vix_modifier: Modifier from VIX regime (0.0-1.0)

        Returns:
            Tuple of (number_of_contracts, PositionSizeResult)
        """
        result = self.calculate(
            regime_strength=regime_strength,
            technical_confirmations=technical_confirmations,
            vix_modifier=vix_modifier,
        )

        # Calculate target dollar amount
        target_amount = portfolio_value * result.final_size

        # Each contract is 100 shares
        contract_cost = option_price * 100

        # Calculate number of contracts (minimum 1 if we can afford it)
        if contract_cost <= 0:
            return 0, result

        num_contracts = max(1, int(target_amount / contract_cost))

        return num_contracts, result

    def get_status(self) -> dict:
        """Get current configuration status."""
        return {
            "enabled": self.enabled,
            "base_size": self.base_size,
            "max_size": self.max_size,
            "min_size": self.min_size,
            "regime_multipliers": self.REGIME_MULTIPLIERS,
            "tech_multipliers": self.TECH_MULTIPLIERS,
        }
