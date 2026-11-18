"""
fantasy/gameweek_engine.py
==========================
Orchestrates all player projections for a given FPL gameweek.
For each fixture, runs player feature extraction, availability check,
expected minutes, and FPL fantasy points projection.
Returns a ranked player list ready for optimizer input.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from agents.availability_agent import AvailabilityAgent
from features.player_features import PlayerFeatureExtractor
from models.player_projection import PlayerProjectionModel, FantasyProjection
from models.fixture_difficulty import FixtureDifficultyRater
from fantasy.fpl_client import FPLClient
from fantasy.optimizer import PlayerOption

logger = logging.getLogger(__name__)


@dataclass
class GameweekReport:
    gameweek: int
    generated_at: datetime
    player_projections: list[FantasyProjection]
    ranked_players: list[dict]   # sorted by expected_points desc
    fixtures: list[dict]

    def top_picks(self, n: int = 20) -> list[dict]:
        return self.ranked_players[:n]


class GameweekEngine:
    def __init__(self):
        self._availability_agent = AvailabilityAgent()
        self._feature_extractor = PlayerFeatureExtractor()
        self._projection_model = PlayerProjectionModel()
        self._fdr = FixtureDifficultyRater()
        self._fpl = FPLClient()

    def run(
        self,
        gameweek: int,
        player_match_records: dict[str, list[dict]],
        injuries: list[dict],
        suspensions: list[dict],
        team_clean_sheet_probs: dict[str, float],
        team_mu_goals: dict[str, float],
        team_matches_played: dict[str, int],
        player_positions: dict[str, str],
    ) -> GameweekReport:
        """
        Parameters
        ----------
        player_match_records  : {player_id: [match_record_dicts]}
        injuries              : list of injury dicts with player_id key
        suspensions           : list of suspension dicts with player_id key
        team_clean_sheet_probs: {team_id: float 0-1}
        team_mu_goals         : {team_id: float expected goals in next match}
        team_matches_played   : {team_id: int} for start_pct calculation
        player_positions      : {player_id: "GK"|"DEF"|"MID"|"FWD"}
        """
        projections: list[FantasyProjection] = []

        for player_id, match_records in player_match_records.items():
            position = player_positions.get(player_id, "FWD")
            
            # Feature extraction
            team_n = team_matches_played.get(
                match_records[0].get("team_id", "") if match_records else "", 0
            )
            features = self._feature_extractor.extract(player_id, match_records, team_n)

            # Availability
            availability = self._availability_agent.assess(player_id, injuries, suspensions)

            # Team context
            team_id = match_records[0].get("team_id", "") if match_records else ""
            cs_prob = team_clean_sheet_probs.get(team_id, 0.25)
            mu_goals = team_mu_goals.get(team_id, 1.30)

            # Project
            proj = self._projection_model.project(
                player_features=features,
                position=position,
                availability=availability,
                team_p_clean_sheet=cs_prob,
                team_mu_goals=mu_goals,
            )
            projections.append(proj)

        # Rank
        ranked = sorted(
            [p.as_dict() for p in projections],
            key=lambda x: x["expected_points"],
            reverse=True,
        )

        return GameweekReport(
            gameweek=gameweek,
            generated_at=datetime.now(timezone.utc),
            player_projections=projections,
            ranked_players=ranked,
            fixtures=[],
        )

    def to_optimizer_candidates(
        self,
        report: GameweekReport,
        player_prices: dict[str, float],
        player_names: dict[str, str],
        player_clubs: dict[str, str],
    ) -> list[PlayerOption]:
        """Convert projections to optimizer-ready PlayerOption objects."""
        candidates = []
        for proj in report.player_projections:
            if proj.expected_minutes < 10:
                continue   # skip players unlikely to play
            candidates.append(PlayerOption(
                player_id=proj.player_id,
                name=player_names.get(proj.player_id, proj.player_id),
                position=proj.position,
                club_id=player_clubs.get(proj.player_id, "UNK"),
                price=player_prices.get(proj.player_id, 5.0),
                expected_points=proj.expected_points,
                is_nailed=proj.is_nailed,
            ))
        return candidates

gameweek_engine = GameweekEngine()
