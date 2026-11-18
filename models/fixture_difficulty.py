"""
models/fixture_difficulty.py
=============================
Rates fixture difficulty (1-5) for FPL transfers and captaincy decisions.
Uses team ELO difference, league position, and form to determine difficulty.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FixtureDifficultyResult:
    team_id: str
    opponent_id: str
    is_home: bool
    difficulty: int        # 1 (very easy) → 5 (very hard)
    difficulty_label: str  # VERY_EASY, EASY, MEDIUM, HARD, VERY_HARD
    elo_diff: float
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "team_id": self.team_id,
            "opponent_id": self.opponent_id,
            "is_home": self.is_home,
            "difficulty": self.difficulty,
            "difficulty_label": self.difficulty_label,
            "elo_diff": round(self.elo_diff, 1),
            "notes": self.notes,
        }


_BANDS = [
    (200.0,  1, "VERY_EASY"),
    ( 75.0,  2, "EASY"),
    (-75.0,  3, "MEDIUM"),
    (-200.0, 4, "HARD"),
    (float("-inf"), 5, "VERY_HARD"),
]

_HOME_ADV_ELO = 60.0   # Typical ELO home advantage equivalent


class FixtureDifficultyRater:
    """
    Rates fixtures from the perspective of the attacking team
    (i.e., how hard is it for THIS team to score / win).
    """

    def rate(
        self,
        team_id: str,
        team_elo: float,
        opponent_id: str,
        opponent_elo: float,
        is_home: bool,
        opponent_form_w5: float = 1.5,
    ) -> FixtureDifficultyResult:
        # Positive elo_diff = team is stronger (EASIER fixture)
        adj_opponent_elo = opponent_elo - (_HOME_ADV_ELO if not is_home else 0.0)
        elo_diff = team_elo - adj_opponent_elo

        # Adjust for opponent recent form
        form_adj = (opponent_form_w5 - 1.5) * 30.0  # hot form = harder
        effective_diff = elo_diff - form_adj

        label = "MEDIUM"
        difficulty = 3
        for threshold, level, lbl in _BANDS:
            if effective_diff > threshold:
                difficulty = level
                label = lbl
                break

        notes = []
        if not is_home:
            notes.append("Away fixture (+difficulty).")
        if opponent_form_w5 > 2.0:
            notes.append(f"Opponent in red-hot form (W5={opponent_form_w5:.1f}).")
        elif opponent_form_w5 < 0.8:
            notes.append(f"Opponent in poor form (W5={opponent_form_w5:.1f}).")

        return FixtureDifficultyResult(
            team_id=team_id,
            opponent_id=opponent_id,
            is_home=is_home,
            difficulty=difficulty,
            difficulty_label=label,
            elo_diff=effective_diff,
            notes=" ".join(notes),
        )
