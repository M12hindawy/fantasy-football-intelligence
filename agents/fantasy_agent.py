"""
agents/fantasy_agent.py
========================
Top-level Fantasy agent. Pulls live FPL data, runs the gameweek engine,
calls the optimizer, and returns a structured recommendation.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from fantasy.fpl_client import FPLClient, fpl_client
from fantasy.gameweek_engine import GameweekEngine, GameweekReport
from fantasy.optimizer import FPLOptimizer, OptimizedSquad, PlayerOption

logger = logging.getLogger(__name__)


@dataclass
class FantasyRecommendation:
    gameweek: int
    squad: Optional[OptimizedSquad]
    top_picks: list[dict]
    captain_recommendation: Optional[str]
    generated_at: datetime
    notes: list[str]

    def as_dict(self) -> dict:
        return {
            "gameweek": self.gameweek,
            "captain": self.captain_recommendation,
            "top_picks": self.top_picks[:15],
            "squad": self.squad.summary() if self.squad else None,
            "notes": self.notes,
            "generated_at": self.generated_at.isoformat(),
        }


class FantasyAgent:
    """
    Orchestrates the full fantasy analysis pipeline:
      FPL API → player features → projections → optimization → recommendation
    """

    def __init__(self):
        self._client = fpl_client
        self._engine = GameweekEngine()
        self._optimizer = FPLOptimizer()

    def recommend(
        self,
        player_match_records: dict[str, list[dict]],
        injuries: list[dict],
        suspensions: list[dict],
        team_clean_sheet_probs: dict[str, float],
        team_mu_goals: dict[str, float],
        team_matches_played: dict[str, int],
        player_positions: dict[str, str],
        player_prices: Optional[dict[str, float]] = None,
        player_names: Optional[dict[str, str]] = None,
        player_clubs: Optional[dict[str, str]] = None,
        gameweek: Optional[int] = None,
    ) -> FantasyRecommendation:

        notes = []

        # Determine gameweek
        gw = gameweek
        if gw is None:
            try:
                gw = self._client.get_next_gameweek() or 1
                notes.append(f"Auto-detected gameweek: GW{gw}")
            except Exception:
                gw = 1
                notes.append("Could not detect gameweek from FPL API — defaulting to GW1.")

        # Run gameweek engine
        report = self._engine.run(
            gameweek=gw,
            player_match_records=player_match_records,
            injuries=injuries,
            suspensions=suspensions,
            team_clean_sheet_probs=team_clean_sheet_probs,
            team_mu_goals=team_mu_goals,
            team_matches_played=team_matches_played,
            player_positions=player_positions,
        )

        # Build optimizer candidates
        candidates: list[PlayerOption] = []
        if player_prices and player_names and player_clubs:
            candidates = self._engine.to_optimizer_candidates(
                report,
                player_prices=player_prices,
                player_names=player_names,
                player_clubs=player_clubs,
            )

        # Optimize squad
        squad = None
        if len(candidates) >= 15:
            squad = self._optimizer.optimize(candidates)
            notes.append("Squad optimization complete.")
        else:
            notes.append(f"Insufficient candidates ({len(candidates)}) for optimization — showing top picks only.")

        captain = squad.captain.name if (squad and squad.captain) else (
            report.ranked_players[0].get("player_id") if report.ranked_players else None
        )

        return FantasyRecommendation(
            gameweek=gw,
            squad=squad,
            top_picks=report.top_picks(20),
            captain_recommendation=captain,
            generated_at=datetime.now(timezone.utc),
            notes=notes,
        )

fantasy_agent = FantasyAgent()
