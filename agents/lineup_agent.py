"""
agents/lineup_agent.py
======================
Assesses confirmed vs predicted lineups and identifies missing key players.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class LineupAssessment:
    team_id: str
    is_confirmed: bool
    starting_xis: list[str]
    missing_key_players: list[str]

class LineupAgent:
    def assess(self, team_id: str, confirmed_lineup: list[dict], predicted_lineup: list[dict], key_players: list[str]) -> LineupAssessment:
        is_conf = len(confirmed_lineup) > 0
        active = confirmed_lineup if is_conf else predicted_lineup
        
        starters = [p["player_id"] for p in active if p.get("is_starter", True)]
        missing = [kp for kp in key_players if kp not in starters]
        
        return LineupAssessment(
            team_id=team_id,
            is_confirmed=is_conf,
            starting_xis=starters,
            missing_key_players=missing
        )
