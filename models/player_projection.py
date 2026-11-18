"""
models/player_projection.py
============================
Projects fantasy points for a player for a given fixture using:
  - Expected minutes (from ExpectedMinutesModel)
  - Player xG/xA per 90 (from PlayerFeatureExtractor)
  - Match simulation output (team-level goal expectations)
  - FPL scoring system (configurable)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from agents.availability_agent import AvailabilityResult
from features.player_features import PlayerFeatures
from models.expected_minutes import ExpectedMinutesModel, ExpectedMinutesResult

# FPL scoring constants
_FPL_APPEARANCE = 1          # for playing any minutes
_FPL_60_MIN_BONUS = 1        # extra for 60+ mins
_FPL_GOAL_FWD = 4
_FPL_GOAL_MID = 5
_FPL_GOAL_DEF = 6
_FPL_GOAL_GK  = 6
_FPL_ASSIST   = 3
_FPL_CLEAN_SHEET_DEF = 4
_FPL_CLEAN_SHEET_GK  = 4
_FPL_CLEAN_SHEET_MID = 1
_FPL_SAVE_3   = 1            # per 3 saves
_FPL_YELLOW   = -1
_FPL_RED      = -3


def _goal_points(position: str) -> int:
    return {"FWD": _FPL_GOAL_FWD, "MID": _FPL_GOAL_MID,
            "DEF": _FPL_GOAL_DEF, "GK": _FPL_GOAL_GK}.get(position, _FPL_GOAL_FWD)

def _cs_points(position: str) -> float:
    return {"GK": _FPL_CLEAN_SHEET_GK, "DEF": _FPL_CLEAN_SHEET_DEF,
            "MID": _FPL_CLEAN_SHEET_MID, "FWD": 0}.get(position, 0)


@dataclass
class FantasyProjection:
    player_id: str
    position: str
    expected_minutes: float
    starting_probability: float
    expected_goals: float
    expected_assists: float
    expected_clean_sheet_prob: float
    expected_saves: float
    expected_bonus: float
    expected_points: float
    ceiling: float
    floor: float
    is_nailed: bool

    def as_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "position": self.position,
            "expected_minutes": round(self.expected_minutes, 1),
            "starting_probability": round(self.starting_probability, 3),
            "expected_goals": round(self.expected_goals, 3),
            "expected_assists": round(self.expected_assists, 3),
            "expected_points": round(self.expected_points, 2),
            "ceiling": round(self.ceiling, 2),
            "floor": round(self.floor, 2),
            "is_nailed": self.is_nailed,
        }


class PlayerProjectionModel:
    """
    Projects FPL fantasy points for one player for one fixture.

    Parameters
    ----------
    team_p_clean_sheet : team-level clean sheet probability from Monte Carlo
    team_mu_goals      : team-level expected goals from Dixon-Coles/ensemble
    """

    def __init__(self):
        self._mins_model = ExpectedMinutesModel()

    def project(
        self,
        player_features: PlayerFeatures,
        position: str,
        availability: AvailabilityResult,
        team_p_clean_sheet: float = 0.30,
        team_mu_goals: float = 1.50,
        avg_saves_per_game: float = 3.0,
    ) -> FantasyProjection:
        mins_result = self._mins_model.predict(player_features, availability)
        exp_mins = mins_result.expected_minutes
        start_p = mins_result.starting_probability

        # Scale participation fraction 0-1
        part = exp_mins / 90.0

        # Goal/assist expectations
        exp_goals = player_features.xg_90 * part
        exp_assists = player_features.xa_90 * part

        # Clean sheet prob scales with participation (full game needed)
        cs_prob = team_p_clean_sheet * (1.0 - abs(exp_mins - 90.0) / 90.0)
        cs_prob = max(0.0, cs_prob)

        # Saves (GK only)
        exp_saves = avg_saves_per_game * part if position == "GK" else 0.0

        # ── Expected Points ──────────────────────────────────────────────
        gp = _goal_points(position)
        pts = 0.0

        if exp_mins > 0:
            pts += _FPL_APPEARANCE                          # any minutes
        if exp_mins >= 60:
            pts += _FPL_60_MIN_BONUS * (exp_mins / 90.0)   # 60+ mins (scaled expectation)

        pts += exp_goals * gp
        pts += exp_assists * _FPL_ASSIST
        pts += cs_prob * _cs_points(position)
        pts += (exp_saves / 3.0) * _FPL_SAVE_3
        pts += _FPL_YELLOW * 0.05  # small yellow card penalty
        pts += exp_goals * 0.10    # small bonus approximation
        pts = max(0.0, pts)  # never negative

        # ── Ceiling / Floor (optimistic / pessimistic scenario) ──────────
        ceiling = pts + (gp * 0.5) + _FPL_ASSIST + (3.0 if cs_prob > 0.3 else 0.0)
        floor = max(0.0, pts - (gp * 0.3) - 1.0) if exp_mins > 0 else 0.0

        return FantasyProjection(
            player_id=player_features.player_id,
            position=position,
            expected_minutes=exp_mins,
            starting_probability=start_p,
            expected_goals=round(exp_goals, 4),
            expected_assists=round(exp_assists, 4),
            expected_clean_sheet_prob=round(cs_prob, 4),
            expected_saves=round(exp_saves, 3),
            expected_bonus=round(exp_goals * 0.10, 3),
            expected_points=round(pts, 3),
            ceiling=round(ceiling, 3),
            floor=round(floor, 3),
            is_nailed=mins_result.is_nailed,
        )
