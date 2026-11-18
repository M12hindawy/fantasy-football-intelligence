"""
simulation/relegation_engine.py
===============================
Calculates probabilities for Title, UCL, and Relegation from season sim data.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SeasonOutcomeProbabilities:
    team_id: str
    title_prob: float
    top_4_prob: float
    relegation_prob: float
    expected_position: float

class SeasonOutcomesEngine:
    def analyze(
        self, 
        simulated_ranks: dict[str, list[int]], 
        relegation_spots: int = 3, 
        total_teams: int = 20
    ) -> dict[str, SeasonOutcomeProbabilities]:
        
        outcomes = {}
        for team_id, ranks in simulated_ranks.items():
            n_sims = len(ranks)
            title = sum(1 for r in ranks if r == 1) / n_sims
            top4 = sum(1 for r in ranks if r <= 4) / n_sims
            relegated = sum(1 for r in ranks if r > (total_teams - relegation_spots)) / n_sims
            exp_pos = sum(ranks) / n_sims
            
            outcomes[team_id] = SeasonOutcomeProbabilities(
                team_id=team_id,
                title_prob=round(title, 4),
                top_4_prob=round(top4, 4),
                relegation_prob=round(relegated, 4),
                expected_position=round(exp_pos, 1)
            )
        return outcomes
