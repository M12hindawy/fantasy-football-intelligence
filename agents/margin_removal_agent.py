"""
agents/margin_removal_agent.py
==============================
Selects and applies the appropriate margin-removal method for each
market type, records which method was used, and refuses to silently
average incompatible methods.

Method selection logic follows the master spec Section 4 and the
1xBet adapter's auto-method mapping.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from data.schemas import MarketBundle, MarketType
from markets.market_math import MARGIN_METHODS, overround, raw_implied_probability

logger = logging.getLogger(__name__)

MarginMethod = Literal["proportional", "additive", "shin", "power", "auto"]


class MarginRemovalAgent:
    """
    Applies margin removal to a MarketBundle and records metadata.

    Parameters
    ----------
    default_method : method to use when no market-specific rule applies
    """

    # Market-specific preferred methods
    _METHOD_PREFERENCE: dict[MarketType, str] = {
        MarketType.CORRECT_SCORE: "additive",
        MarketType.ASIAN_HANDICAP: "power",
        MarketType.EUROPEAN_HANDICAP: "power",
        MarketType.TOTALS: "proportional",
        MarketType.BTTS: "proportional",
        MarketType.TEAM_TOTALS: "proportional",
        MarketType.PLAYER_PROP: "proportional",
    }

    # Default for all other markets
    _DEFAULT_METHOD = "shin"

    def __init__(self, default_method: str = "shin") -> None:
        if default_method not in MARGIN_METHODS:
            raise ValueError(
                f"Unknown margin method '{default_method}'. "
                f"Valid options: {list(MARGIN_METHODS)}"
            )
        self._default_method = default_method

    def apply(
        self,
        bundle: MarketBundle,
        method: MarginMethod = "auto",
    ) -> tuple[MarketBundle, str, list[str]]:
        """
        Remove the bookmaker margin from a MarketBundle.

        Parameters
        ----------
        bundle : incoming MarketBundle (odds must be populated)
        method : "auto" selects market-specific method; otherwise forces one

        Returns
        -------
        (bundle, method_used, warnings)
        """
        warnings: list[str] = []

        if not bundle.odds:
            warnings.append("MarginRemovalAgent: no odds to process.")
            return bundle, "none", warnings

        odds_values = [rec.decimal_odds for rec in bundle.odds]

        # Determine method
        if method == "auto":
            chosen = self._METHOD_PREFERENCE.get(
                bundle.market_type, self._default_method
            )
        else:
            if method not in MARGIN_METHODS:
                warnings.append(
                    f"Requested method '{method}' is invalid. "
                    f"Falling back to '{self._default_method}'."
                )
                chosen = self._default_method
            else:
                chosen = method

        # Compute overround before removal
        vig = overround(odds_values)
        bundle.overround = round(vig, 6)
        bundle.margin_method_used = chosen

        # Log margin info
        vig_pct = vig * 100.0
        logger.info(
            "MarginRemovalAgent: %s market overround = %.2f%% | method = %s",
            bundle.market_type, vig_pct, chosen,
        )

        if vig_pct > 15.0:
            warnings.append(
                f"HIGH OVERROUND: {vig_pct:.1f}% on {bundle.market_type}. "
                "Edge calculations become unreliable above ~10%."
            )
        elif vig_pct < 0.0:
            warnings.append(
                f"NEGATIVE OVERROUND ({vig_pct:.2f}%): possible data error or "
                "arbitrage opportunity — verify odds source."
            )

        # Apply removal
        fn = MARGIN_METHODS[chosen]
        fair_probs = fn(odds_values)

        # Attach fair probs to records
        for rec, fp in zip(bundle.odds, fair_probs):
            rec.__dict__["_fair_probability"] = round(fp, 6)
            rec.__dict__["_margin_method"] = chosen

        if not warnings:
            logger.info(
                "MarginRemovalAgent: Margin removed from %s / %s using '%s'.",
                bundle.fixture_id, bundle.market_type, chosen,
            )

        return bundle, chosen, warnings

    # ── Convenience: extract fair probs after removal ─────────────────────────

    @staticmethod
    def get_fair_probabilities(bundle: MarketBundle) -> dict[str, float]:
        """Return {selection: fair_probability} after margin removal."""
        return {
            rec.selection: rec.__dict__.get("_fair_probability", 0.0)
            for rec in bundle.odds
        }

    @staticmethod
    def get_raw_probabilities(bundle: MarketBundle) -> dict[str, float]:
        """Return {selection: raw_implied_probability}."""
        return {
            rec.selection: raw_implied_probability(rec.decimal_odds)
            for rec in bundle.odds
        }
