"""
simulation/season_simulator.py
==============================
Monte Carlo season simulator. Runs thousands of season simulations
to determine title, qualification, and relegation probabilities.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Re-export so callers can do: from simulation.season_simulator import SeasonOutcomesEngine
from simulation.relegation_engine import SeasonOutcomesEngine  # noqa: F401

@dataclass
class TeamSeasonContext:
    team_id: str
    points: int
    goal_diff: int
    goals_for: int
    attack_strength: float
    defense_strength: float

class SeasonSimulator:
    def __init__(self, n_simulations: int = 1000):
        self.n_sims = n_simulations

    def simulate_season(
        self, 
        current_standings: list[TeamSeasonContext],
        remaining_fixtures: list[tuple[str, str]], 
        home_advantage: float = 0.3
    ) -> dict[str, list[int]]:
        """
        Returns a dictionary mapping team_id to a list of their final
        league positions across all simulations.
        """
        teams = {t.team_id: i for i, t in enumerate(current_standings)}
        n_teams = len(teams)
        
        pts = np.tile(np.array([t.points for t in current_standings]), (self.n_sims, 1))
        gd = np.tile(np.array([t.goal_diff for t in current_standings]), (self.n_sims, 1))
        gf = np.tile(np.array([t.goals_for for t in current_standings]), (self.n_sims, 1))
        
        atk = np.array([t.attack_strength for t in current_standings])
        dfn = np.array([t.defense_strength for t in current_standings])
        
        rng = np.random.default_rng(42)
        
        for home_id, away_id in remaining_fixtures:
            if home_id not in teams or away_id not in teams:
                continue
                
            h_idx = teams[home_id]
            a_idx = teams[away_id]
            
            # Simple Poisson generation using log-linear expected goals
            mu_h = np.exp(atk[h_idx] - dfn[a_idx] + home_advantage)
            mu_a = np.exp(atk[a_idx] - dfn[h_idx])
            
            sim_h_goals = rng.poisson(mu_h, self.n_sims)
            sim_a_goals = rng.poisson(mu_a, self.n_sims)
            
            gf[:, h_idx] += sim_h_goals
            gf[:, a_idx] += sim_a_goals
            gd[:, h_idx] += (sim_h_goals - sim_a_goals)
            gd[:, a_idx] += (sim_a_goals - sim_h_goals)
            
            home_wins = sim_h_goals > sim_a_goals
            away_wins = sim_a_goals > sim_h_goals
            draws = sim_h_goals == sim_a_goals
            
            pts[:, h_idx] += (home_wins * 3) + draws
            pts[:, a_idx] += (away_wins * 3) + draws
            
        ranks = np.zeros((self.n_sims, n_teams), dtype=int)
        for i in range(self.n_sims):
            # Sort by points, then GD, then GF (lexsort sorts right-to-left)
            order = np.lexsort((-gf[i], -gd[i], -pts[i]))
            for pos, team_idx in enumerate(order):
                ranks[i, team_idx] = pos + 1
                
        results = {}
        for t_id, idx in teams.items():
            results[t_id] = ranks[:, idx].tolist()
            
        return results
