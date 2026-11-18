"""
agents/league_context_agent.py
==============================
Analyzes league table context to predict motivation levels and rotation risks.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class LeagueContext:
    team_id: str
    motivation_level: str  # HIGH, NORMAL, LOW
    rotation_risk: str     # HIGH, NORMAL, LOW
    notes: str

class LeagueContextAgent:
    def assess(self, team_id: str, current_position: int, matches_remaining: int, points_from_safety: int, points_from_title: int) -> LeagueContext:
        if matches_remaining <= 5:
            if points_from_title <= 3 or points_from_safety <= 3:
                return LeagueContext(team_id, "HIGH", "LOW", "High stakes: End of season title/relegation battle")
            elif points_from_title > 15 and points_from_safety > 15:
                return LeagueContext(team_id, "LOW", "HIGH", "Low stakes: Nothing to play for, high rotation risk")
                
        return LeagueContext(team_id, "NORMAL", "NORMAL", "Standard league match context")
