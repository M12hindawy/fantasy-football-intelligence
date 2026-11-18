"""
models/elo_model.py
===================
Dynamic Elo rating system for football team strength estimation.

Home advantage, promotion/relegation, and season carry-over are
handled explicitly.  The model is lightweight — it serves as the
second member of the ensemble alongside Dixon-Coles.

Reference
---------
Hvattum, L. M., & Arntzen, H. (2010). Using ELO ratings for match
result prediction in association football. International Journal of
Forecasting, 26(3), 460-470.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from data.schemas import PredictionOutput

MODEL_VERSION = "elo-dynamic-1.0"

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_RATING: float = 1500.0
HOME_ADVANTAGE: float = 65.0      # ELO points added to home team for prediction
K_FACTOR: float = 32.0            # Standard K-factor
K_FACTOR_PROMOTED: float = 40.0   # Higher K for newly promoted teams
SEASON_CARRY_OVER: float = 0.70   # Fraction of rating maintained between seasons

# Goal-difference multiplier table (increases K for blowouts)
# Inspired by FiveThirtyEight's NFL Elo implementation
_GOAL_DIFF_MULTIPLIERS = {
    1: 1.00,
    2: 1.25,
    3: 1.50,
    4: 1.75,
    5: 2.00,
}


def _goal_diff_multiplier(goal_diff: int) -> float:
    """Multiplier for the K-factor based on absolute goal difference."""
    gd = abs(goal_diff)
    if gd in _GOAL_DIFF_MULTIPLIERS:
        return _GOAL_DIFF_MULTIPLIERS[gd]
    return min(2.5, 1.0 + (gd - 1) * 0.25)


@dataclass
class EloRating:
    team_id: str
    team_name: str
    rating: float = DEFAULT_RATING
    matches_played: int = 0
    last_updated: Optional[datetime] = None
    season: Optional[str] = None


@dataclass
class DynamicEloModel:
    """
    Dynamic Elo model that updates ratings match-by-match.

    Parameters
    ----------
    home_advantage : ELO-point home advantage used in win-probability
    k_factor       : base K-factor for rating updates
    season_carry   : fraction of rating retained at season start
    """

    home_advantage: float = HOME_ADVANTAGE
    k_factor: float = K_FACTOR
    season_carry: float = SEASON_CARRY_OVER

    ratings: dict[str, EloRating] = field(default_factory=dict)
    is_fitted: bool = False
    n_matches_trained: int = 0
    fitted_at: Optional[datetime] = None

    # Real league-average goals, accumulated during fit() — replaces the
    # previous hardcoded 1.35/1.05 placeholder constants with numbers
    # actually computed from the historical matches this model was fit on.
    _total_home_goals: int = 0
    _total_away_goals: int = 0

    # ── Rating management ─────────────────────────────────────────────────────

    def get_rating(self, team_id: str, team_name: str = "") -> float:
        """Return current rating; initialise to default if not seen."""
        if team_id not in self.ratings:
            self.ratings[team_id] = EloRating(
                team_id=team_id,
                team_name=team_name or team_id,
                rating=DEFAULT_RATING,
            )
        return self.ratings[team_id].rating

    def set_rating(self, team_id: str, rating: float) -> None:
        if team_id in self.ratings:
            self.ratings[team_id].rating = rating
        else:
            self.ratings[team_id] = EloRating(team_id=team_id, team_name=team_id, rating=rating)

    def apply_season_reset(self, team_ids: Optional[list[str]] = None) -> None:
        """
        Apply regression-to-mean at the start of a new season.
        rating_new = carry × rating_old + (1 − carry) × DEFAULT
        """
        targets = team_ids or list(self.ratings.keys())
        for tid in targets:
            if tid in self.ratings:
                old = self.ratings[tid].rating
                self.ratings[tid].rating = (
                    self.season_carry * old + (1 - self.season_carry) * DEFAULT_RATING
                )

    # ── Win probability ───────────────────────────────────────────────────────

    def win_probability(
        self,
        home_rating: float,
        away_rating: float,
        home_advantage: Optional[float] = None,
    ) -> tuple[float, float, float]:
        """
        Compute P(home win), P(draw), P(away win) using the Elo formula.

        This uses the Poisson-based conversion from Maher (1982) adapted
        for Elo:  expected score = 1 / (1 + 10^(−d/400))

        Draw probability is estimated from the expected score.

        Returns
        -------
        (p_home_win, p_draw, p_away_win)
        """
        ha = home_advantage if home_advantage is not None else self.home_advantage
        d = home_rating + ha - away_rating
        e_home = 1.0 / (1.0 + 10.0 ** (-d / 400.0))

        # Approximate draw probability (empirically tuned for football)
        # Based on the method in Hvattum & Arntzen (2010)
        p_draw = max(0.05, 0.35 - 0.5 * abs(d) / 400.0)

        # Distribute remainder between win and loss proportional to e_home
        p_home = (1.0 - p_draw) * e_home
        p_away = (1.0 - p_draw) * (1.0 - e_home)

        # Ensure they sum to 1.0
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        match_records: list[dict],
    ) -> "DynamicEloModel":
        """
        Update ratings chronologically through historical matches.

        match_records : list of dicts with keys:
            home_id, home_name, away_id, away_name,
            home_goals, away_goals, date (datetime),
            is_promoted_home (bool, optional),
            is_promoted_away (bool, optional)
        """
        # Sort chronologically
        records = sorted(match_records, key=lambda m: m["date"])

        for m in records:
            home_id = m["home_id"]
            away_id = m["away_id"]
            hg = int(m["home_goals"])
            ag = int(m["away_goals"])

            r_h = self.get_rating(home_id, m.get("home_name", home_id))
            r_a = self.get_rating(away_id, m.get("away_name", away_id))

            # Actual result from home perspective: 1=win, 0.5=draw, 0=loss
            if hg > ag:
                s_h = 1.0
            elif hg == ag:
                s_h = 0.5
            else:
                s_h = 0.0

            # Expected result
            e_h, _, _ = self.win_probability(r_h, r_a)

            # K-factor adjustment
            k = (
                K_FACTOR_PROMOTED
                if m.get("is_promoted_home") or m.get("is_promoted_away")
                else self.k_factor
            )
            gdm = _goal_diff_multiplier(hg - ag)

            delta = k * gdm * (s_h - e_h)
            self.ratings[home_id].rating = r_h + delta
            self.ratings[away_id].rating = r_a - delta
            self.ratings[home_id].matches_played += 1
            self.ratings[away_id].matches_played += 1
            self.ratings[home_id].last_updated = m["date"]
            self.ratings[away_id].last_updated = m["date"]
            self.n_matches_trained += 1
            self._total_home_goals += hg
            self._total_away_goals += ag

        self.is_fitted = True
        self.fitted_at = datetime.now(timezone.utc)
        return self

    @property
    def league_avg_home_goals(self) -> float:
        """Real average home goals per match across everything fit() has
        seen so far. Falls back to a documented, labeled default only when
        nothing has been fit yet — never silently reused as if it were a
        real computed value."""
        if self.n_matches_trained == 0:
            return 1.35  # widely-cited long-run top-flight average; explicitly a fallback, not a fit result
        return self._total_home_goals / self.n_matches_trained

    @property
    def league_avg_away_goals(self) -> float:
        if self.n_matches_trained == 0:
            return 1.10
        return self._total_away_goals / self.n_matches_trained

    def predict(
        self,
        home_id: str,
        away_id: str,
        fixture_id: str = "unknown",
        home_name: str = "",
        away_name: str = "",
    ) -> PredictionOutput:
        """
        Predict probabilities for a fixture.

        Returns a PredictionOutput (xG fields set to Elo-implied estimates).
        """
        r_h = self.get_rating(home_id, home_name)
        r_a = self.get_rating(away_id, away_name)
        p_home, p_draw, p_away = self.win_probability(r_h, r_a)

        # Elo cannot directly produce xG (it's a win/draw/loss-only rating
        # system), so we use the real league-average goals computed from
        # whatever this model was actually fit on (see league_avg_home_goals/
        # league_avg_away_goals above) as a stand-in. The Ensemble's primary
        # xG figure comes from Dixon-Coles, which models goals directly —
        # this is a secondary, clearly-labeled fallback, not the headline number.
        xg_home = self.league_avg_home_goals
        xg_away = self.league_avg_away_goals

        return PredictionOutput(
            fixture_id=fixture_id,
            model_name="DynamicElo",
            model_version=MODEL_VERSION,
            generated_at=datetime.now(timezone.utc),
            p_home_win=round(p_home, 6),
            p_draw=round(p_draw, 6),
            p_away_win=round(p_away, 6),
            expected_goals_home=xg_home,
            expected_goals_away=xg_away,
        )

    def rating_summary(self) -> list[dict]:
        """Return a sorted list of all team ratings."""
        return sorted(
            [
                {
                    "team_id": tid,
                    "team_name": r.team_name,
                    "rating": round(r.rating, 1),
                    "matches": r.matches_played,
                }
                for tid, r in self.ratings.items()
            ],
            key=lambda x: x["rating"],
            reverse=True,
        )
