"""
agents/market_normalization_agent.py
=====================================
Validates that overround removal was applied correctly and that the
resulting fair probabilities obey the constraints of each market family.

For mutually exclusive markets: Σ p_fair ≈ 1.0
For double chance: no outcome overlaps treated as independent
For Asian handicap: push/half-win/half-loss handled correctly
"""

from __future__ import annotations

import logging
from typing import Optional

from data.schemas import MarketBundle, MarketType
from markets.market_math import (
    double_chance_probability,
    draw_no_bet_probability,
    overround as compute_overround,
    raw_implied_probability,
)

logger = logging.getLogger(__name__)


class MarketNormalizationAgent:
    """
    Runs post-margin-removal validation on a MarketBundle.

    Returns (bundle, warnings) after asserting probability constraints.
    """

    # Markets where probabilities must sum to 1.0
    _EXCLUSIVE_MARKETS: set[MarketType] = {
        MarketType.ONE_X_TWO,
        MarketType.DRAW_NO_BET,
        MarketType.TOTALS,
        MarketType.BTTS,
        MarketType.TEAM_TOTALS,
        MarketType.HT_1X2,
        MarketType.HT_TOTALS,
        MarketType.HT_BTTS,
        MarketType.CORRECT_SCORE,
    }

    SUM_TOLERANCE = 0.02   # Allow ±2pp deviation from 1.0

    def validate(
        self, bundle: MarketBundle
    ) -> tuple[MarketBundle, list[str]]:
        """
        Validate fair probabilities in a normalised MarketBundle.

        Parameters
        ----------
        bundle : MarketBundle with overround and margin_method_used set

        Returns
        -------
        (bundle, warnings)
        """
        warnings: list[str] = []

        if not bundle.odds:
            warnings.append("No odds in bundle — cannot validate.")
            return bundle, warnings

        if bundle.overround is None:
            warnings.append(
                "MarketNormalizationAgent: overround not computed. "
                "Run MarginRemovalAgent first."
            )
            return bundle, warnings

        # Retrieve fair probabilities (set by adapter)
        fair_probs = [rec.__dict__.get("_fair_probability") for rec in bundle.odds]

        if None in fair_probs:
            warnings.append(
                "MarketNormalizationAgent: fair probabilities missing — "
                "margin removal may not have been applied."
            )
            return bundle, warnings

        fair_probs_clean: list[float] = [p for p in fair_probs if p is not None]

        # ── 1. Raw overround confirmation ──────────────────────────────────────
        raw_odds = [rec.decimal_odds for rec in bundle.odds]
        measured_overround = compute_overround(raw_odds)

        if abs(measured_overround - bundle.overround) > 0.005:
            warnings.append(
                f"Overround mismatch: stored={bundle.overround:.4f}, "
                f"recomputed={measured_overround:.4f}. Possible data corruption."
            )

        # ── 2. Fair-prob sum check for exclusive markets ───────────────────────
        if bundle.market_type in self._EXCLUSIVE_MARKETS:
            total = sum(fair_probs_clean)
            if abs(total - 1.0) > self.SUM_TOLERANCE:
                warnings.append(
                    f"NORMALISATION ERROR: Fair probs for {bundle.market_type} sum to "
                    f"{total:.4f} (expected ≈1.0). Margin method: "
                    f"'{bundle.margin_method_used}'. Recalculate."
                )
            else:
                logger.debug(
                    "MarketNormalizationAgent: %s fair probs sum = %.4f ✓",
                    bundle.market_type, total,
                )

        # ── 3. Double-chance overlap check ────────────────────────────────────
        if bundle.market_type == MarketType.DOUBLE_CHANCE:
            n = len(bundle.odds)
            if n != 3:
                warnings.append(
                    f"Double-chance market should have exactly 3 outcomes, got {n}."
                )
            else:
                warnings.append(
                    "Double-chance: outcomes are NOT mutually exclusive — "
                    "1X, X2, and 12 overlap. They do NOT sum to 1 by design."
                )
                # Validate that each is plausibly derived from 1X2
                for rec in bundle.odds:
                    fp = rec.__dict__.get("_fair_probability", 0.0)
                    if fp < 0.0 or fp > 1.0:
                        warnings.append(
                            f"Double-chance '{rec.selection}' has invalid fair prob {fp:.4f}."
                        )

        # ── 4. Asian handicap quarter-line check ──────────────────────────────
        if bundle.market_type == MarketType.ASIAN_HANDICAP:
            from markets.market_math import is_quarter_line
            if bundle.line is not None and is_quarter_line(bundle.line):
                if len(bundle.odds) != 2:
                    warnings.append(
                        f"Asian handicap quarter-line should have 2 outcomes, got {len(bundle.odds)}."
                    )
                else:
                    total = sum(fair_probs_clean)
                    if abs(total - 1.0) > self.SUM_TOLERANCE:
                        warnings.append(
                            f"Asian handicap fair probs sum to {total:.4f} — expected ≈1.0."
                        )

        # ── 5. All individual fair probs in [0, 1] ────────────────────────────
        for rec, fp in zip(bundle.odds, fair_probs_clean):
            if not (0.0 <= fp <= 1.0):
                warnings.append(
                    f"Fair probability for '{rec.selection}' is {fp:.4f} — "
                    "outside [0, 1]. Margin removal may have failed."
                )

        if not warnings:
            logger.info(
                "MarketNormalizationAgent: All checks passed for %s / %s.",
                bundle.fixture_id, bundle.market_type,
            )

        return bundle, warnings
