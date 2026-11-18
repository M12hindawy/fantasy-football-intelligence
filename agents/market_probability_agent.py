"""
agents/market_probability_agent.py
===================================
Translates ensemble model predictions into per-market model probabilities.

Receives a PredictionOutput from the ensemble and a MarketBundle from the
odds pipeline, and outputs the model probability that corresponds to the
specific selection being evaluated.

Supports all market families in the spec (Section 3).
"""

from __future__ import annotations

import logging
from typing import Optional

from data.schemas import MarketBundle, MarketType, OddsRecord, PredictionOutput
from markets.market_math import (
    double_chance_probability,
    draw_no_bet_probability,
)
from models.dixon_coles import (
    matrix_to_btts,
    matrix_to_over_under,
    matrix_to_over_under as dc_over_under,
    score_probability_matrix,
)

logger = logging.getLogger(__name__)


class MarketProbabilityAgent:
    """
    Maps model output to a model probability for a specific market selection.

    Parameters
    ----------
    prediction : PredictionOutput from the ensemble
    bundle     : MarketBundle (after margin removal)
    selection  : name of the selection to evaluate (e.g., "Over 2.5", "Home")
    """

    def get_model_probability(
        self,
        prediction: PredictionOutput,
        bundle: MarketBundle,
        selection: str,
    ) -> tuple[Optional[float], list[str]]:
        """
        Return (model_probability, notes) for the given selection.

        Returns (None, notes) if the market is not supported or data is
        insufficient.
        """
        mt = bundle.market_type
        notes: list[str] = []

        # ── 1X2 ───────────────────────────────────────────────────────────────
        if mt == MarketType.ONE_X_TWO:
            return self._resolve_1x2(prediction, selection, notes)

        # ── Double Chance ────────────────────────────────────────────────────
        if mt == MarketType.DOUBLE_CHANCE:
            sel_upper = selection.upper().replace(" ", "")
            # Normalise common labels
            if sel_upper in {"1X", "HOMEDRAW", "DRAWOHOME"}:
                dc_sel = "1X"
            elif sel_upper in {"X2", "DRAWAWAY", "AWAYDRAW"}:
                dc_sel = "X2"
            elif sel_upper in {"12", "HOMEAWAY", "AWAYHOME"}:
                dc_sel = "12"
            else:
                notes.append(f"Unrecognised double-chance selection '{selection}'.")
                return None, notes
            try:
                p = double_chance_probability(
                    prediction.p_home_win,
                    prediction.p_draw,
                    prediction.p_away_win,
                    dc_sel,
                )
                return round(p, 6), notes
            except Exception as exc:
                notes.append(str(exc))
                return None, notes

        # ── Draw No Bet ───────────────────────────────────────────────────────
        if mt == MarketType.DRAW_NO_BET:
            sel_lower = selection.lower()
            bet_on_win = "home" in sel_lower or "1" in sel_lower
            if "away" in sel_lower or "2" in sel_lower:
                bet_on_win = False
            try:
                p = draw_no_bet_probability(
                    prediction.p_home_win if bet_on_win else prediction.p_away_win,
                    prediction.p_draw,
                    prediction.p_away_win if bet_on_win else prediction.p_home_win,
                    bet_on_win=True,
                )
                return round(p, 6), notes
            except Exception as exc:
                notes.append(str(exc))
                return None, notes

        # ── Totals (Over/Under) ───────────────────────────────────────────────
        if mt in {MarketType.TOTALS, MarketType.HT_TOTALS}:
            return self._resolve_totals(prediction, bundle, selection, notes)

        # ── BTTS ─────────────────────────────────────────────────────────────
        if mt in {MarketType.BTTS, MarketType.HT_BTTS}:
            if prediction.p_btts is None:
                notes.append("BTTS probability not available in model output.")
                return None, notes
            sel_lower = selection.lower()
            if "yes" in sel_lower:
                return round(prediction.p_btts, 6), notes
            elif "no" in sel_lower:
                return round(1.0 - prediction.p_btts, 6), notes
            notes.append(f"Unrecognised BTTS selection '{selection}'.")
            return None, notes

        # ── Asian Handicap ────────────────────────────────────────────────────
        if mt == MarketType.ASIAN_HANDICAP:
            return self._resolve_asian_handicap(prediction, bundle, selection, notes)

        # ── Correct Score ─────────────────────────────────────────────────────
        if mt == MarketType.CORRECT_SCORE:
            return self._resolve_correct_score(prediction, selection, notes)

        # ── Half-time 1X2 ─────────────────────────────────────────────────────
        if mt == MarketType.HT_1X2:
            notes.append(
                "Half-time 1X2: model uses full-match parameters scaled to first-half "
                "rates. This is an approximation — confidence is lower than for FT 1X2."
            )
            # Approximate: first-half xG ≈ 55% of full-match xG
            ht_xg_h = prediction.expected_goals_home * 0.55
            ht_xg_a = prediction.expected_goals_away * 0.55
            mat = score_probability_matrix(ht_xg_h, ht_xg_a, -0.05, max_goals=5)
            from models.dixon_coles import matrix_to_1x2
            p_home, p_draw, p_away = matrix_to_1x2(mat)
            return self._resolve_1x2_from_probs(p_home, p_draw, p_away, selection, notes)

        # ── Player props ──────────────────────────────────────────────────────
        if mt == MarketType.PLAYER_PROP:
            notes.append(
                "PASS_PENDING_LINEUP: Player prop probabilities require confirmed "
                "lineup and expected minutes. Cannot compute without this data."
            )
            return None, notes

        # ── Live ─────────────────────────────────────────────────────────────
        if mt == MarketType.LIVE:
            notes.append(
                "LIVE_UNAVAILABLE: Live market probability requires a real-time "
                "event feed with current score, minute, and cards."
            )
            return None, notes

        notes.append(
            f"MarketProbabilityAgent: Market type '{mt}' is not fully implemented. "
            "Returning None."
        )
        return None, notes

    # ── Private resolvers ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_1x2(
        prediction: PredictionOutput, selection: str, notes: list[str]
    ) -> tuple[Optional[float], list[str]]:
        sel_lower = selection.lower()
        if any(k in sel_lower for k in ("home", "1")):
            return round(prediction.p_home_win, 6), notes
        if any(k in sel_lower for k in ("draw", "x")):
            return round(prediction.p_draw, 6), notes
        if any(k in sel_lower for k in ("away", "2")):
            return round(prediction.p_away_win, 6), notes
        notes.append(f"Could not map 1X2 selection '{selection}' to home/draw/away.")
        return None, notes

    @staticmethod
    def _resolve_1x2_from_probs(
        p_home: float,
        p_draw: float,
        p_away: float,
        selection: str,
        notes: list[str],
    ) -> tuple[Optional[float], list[str]]:
        sel_lower = selection.lower()
        if any(k in sel_lower for k in ("home", "1")):
            return round(p_home, 6), notes
        if any(k in sel_lower for k in ("draw", "x")):
            return round(p_draw, 6), notes
        if any(k in sel_lower for k in ("away", "2")):
            return round(p_away, 6), notes
        notes.append(f"Could not map selection '{selection}'.")
        return None, notes

    @staticmethod
    def _resolve_totals(
        prediction: PredictionOutput,
        bundle: MarketBundle,
        selection: str,
        notes: list[str],
    ) -> tuple[Optional[float], list[str]]:
        line = bundle.line
        if line is None:
            # Try to parse from selection name e.g. "Over 2.5"
            import re
            m = re.search(r"(\d+\.?\d*)", selection)
            if m:
                line = float(m.group(1))
            else:
                notes.append("Cannot determine totals line from selection or bundle.")
                return None, notes

        sel_lower = selection.lower()
        # Use the score matrix when available
        if prediction.score_matrix is not None:
            import numpy as np
            mat = np.array(prediction.score_matrix.matrix)
            p_over, p_under = dc_over_under(mat, line)
        else:
            # Fallback: use cached over probability from prediction
            cache = {
                0.5: prediction.p_over_05,
                1.5: prediction.p_over_15,
                2.5: prediction.p_over_25,
                3.5: prediction.p_over_35,
                4.5: prediction.p_over_45,
            }
            p_over_cached = cache.get(line)
            if p_over_cached is None:
                notes.append(f"No pre-computed over probability for line {line}.")
                return None, notes
            p_over = p_over_cached
            p_under = 1.0 - p_over

        if "over" in sel_lower:
            return round(p_over, 6), notes
        elif "under" in sel_lower:
            return round(p_under, 6), notes
        notes.append(f"Cannot determine Over/Under direction from selection '{selection}'.")
        return None, notes

    @staticmethod
    def _resolve_asian_handicap(
        prediction: PredictionOutput,
        bundle: MarketBundle,
        selection: str,
        notes: list[str],
    ) -> tuple[Optional[float], list[str]]:
        """
        Compute AH probability using the score matrix.

        For quarter lines: split and average.
        For half lines: sum cells where adjusted margin > 0.
        """
        from markets.market_math import is_quarter_line, split_quarter_line, asian_handicap_result
        import numpy as np

        line = bundle.line
        if line is None:
            notes.append("Asian handicap line is None — cannot compute probability.")
            return None, notes

        if prediction.score_matrix is None:
            notes.append("Score matrix unavailable for Asian handicap computation.")
            return None, notes

        mat = np.array(prediction.score_matrix.matrix)
        n = mat.shape[0]
        sel_lower = selection.lower()
        bet_on_home = "home" in sel_lower or "1" in sel_lower

        if is_quarter_line(line):
            lo, hi = split_quarter_line(line)
            notes.append(
                f"Quarter-line AH {line:+.2f} split: "
                f"50% on AH {lo:+.2f} and 50% on AH {hi:+.2f}."
            )
            # Compute for each sub-line
            p_lo = _ah_win_prob(mat, lo, bet_on_home, n)
            p_hi = _ah_win_prob(mat, hi, bet_on_home, n)
            # Effective probability = average (since stake is split 50/50)
            p = (p_lo + p_hi) / 2.0
        else:
            p = _ah_win_prob(mat, line, bet_on_home, n)

        return round(p, 6), notes


def _ah_win_prob(mat, handicap: float, bet_on_home: bool, n: int) -> float:
    """Sum matrix cells that result in a 'win' for the chosen side."""
    import numpy as np
    total = 0.0
    for i in range(n):
        for j in range(n):
            result = asian_handicap_result(i, j, handicap, bet_on_home)
            if result == "win":
                total += mat[i, j]
            elif result == "half_win":
                total += mat[i, j] * 0.5
            # push and half_loss contribute 0 or -0.5 (not relevant for win prob)
    return float(total)
