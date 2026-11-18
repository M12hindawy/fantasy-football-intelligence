"""
simulation/monte_carlo.py
==========================
Monte Carlo match simulation engine.

Runs N independent simulations of a football match using
the Dixon-Coles score probability matrix as the sampling distribution.

Key design decisions:
- Fixed random seed for reproducibility (configurable in settings)
- Simulation count configurable: 1_000 / 10_000 / 50_000
- Returns full distribution, not just mean
- Never fabricates data — all probabilities come from fitted models

Usage
-----
    from simulation.monte_carlo import MonteCarloEngine
    engine = MonteCarloEngine(n_simulations=10_000, seed=42)
    result = engine.simulate_match(mu_home=1.72, mu_away=0.84, rho=-0.10)
    print(result.p_home_win, result.p_draw, result.p_away_win)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from config.settings import settings
from models.dixon_coles import score_probability_matrix, matrix_to_1x2, matrix_to_over_under, matrix_to_btts


@dataclass
class MatchSimulationResult:
    """Results from a Monte Carlo match simulation."""

    fixture_id: str
    n_simulations: int
    seed: int
    mu_home: float
    mu_away: float
    rho: float

    # 1X2 probabilities (from simulation)
    p_home_win: float = 0.0
    p_draw: float = 0.0
    p_away_win: float = 0.0

    # Goals
    mean_home_goals: float = 0.0
    mean_away_goals: float = 0.0
    median_home_goals: float = 0.0
    median_away_goals: float = 0.0

    # Derived markets
    p_btts: float = 0.0
    p_over_15: float = 0.0
    p_over_25: float = 0.0
    p_over_35: float = 0.0
    p_over_45: float = 0.0
    p_clean_sheet_home: float = 0.0
    p_clean_sheet_away: float = 0.0

    # Score distribution (top 10 most likely scores)
    top_scorelines: list[dict] = field(default_factory=list)

    # Uncertainty (std deviation across simulations)
    std_home_goals: float = 0.0
    std_away_goals: float = 0.0

    # Analytic reference (from score matrix)
    analytic_p_home: float = 0.0
    analytic_p_draw: float = 0.0
    analytic_p_away: float = 0.0

    generated_at: Optional[datetime] = None

    def as_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "n_simulations": self.n_simulations,
            "seed": self.seed,
            "mu_home": self.mu_home,
            "mu_away": self.mu_away,
            "rho": self.rho,
            "p_home_win": round(self.p_home_win, 4),
            "p_draw": round(self.p_draw, 4),
            "p_away_win": round(self.p_away_win, 4),
            "mean_home_goals": round(self.mean_home_goals, 3),
            "mean_away_goals": round(self.mean_away_goals, 3),
            "p_btts": round(self.p_btts, 4),
            "p_over_15": round(self.p_over_15, 4),
            "p_over_25": round(self.p_over_25, 4),
            "p_over_35": round(self.p_over_35, 4),
            "p_clean_sheet_home": round(self.p_clean_sheet_home, 4),
            "p_clean_sheet_away": round(self.p_clean_sheet_away, 4),
            "top_scorelines": self.top_scorelines[:10],
            "analytic_reference": {
                "p_home": round(self.analytic_p_home, 4),
                "p_draw": round(self.analytic_p_draw, 4),
                "p_away": round(self.analytic_p_away, 4),
            },
            "simulation_vs_analytic_delta": round(
                abs(self.p_home_win - self.analytic_p_home), 4
            ),
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


class MonteCarloEngine:
    """
    Monte Carlo match simulation using Poisson sampling.

    The simulation samples home and away goals from independent
    Poisson distributions and applies the Dixon-Coles correlation
    correction via rejection sampling for low-score outcomes.

    Parameters
    ----------
    n_simulations : Number of match simulations (default from settings)
    seed          : Random seed for reproducibility (default from settings)
    """

    def __init__(
        self,
        n_simulations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_simulations = n_simulations or settings.simulation_count
        self.seed = seed if seed is not None else settings.simulation_seed

    def simulate_match(
        self,
        mu_home: float,
        mu_away: float,
        rho: float = -0.10,
        fixture_id: str = "unknown",
    ) -> MatchSimulationResult:
        """
        Simulate a match N times and return probability distribution.

        Uses the score probability matrix (analytic) as the sampling
        distribution for speed. For large N (50k+), uses numpy vectorized
        sampling from the score matrix.

        Parameters
        ----------
        mu_home : Expected home goals (from Dixon-Coles or ensemble)
        mu_away : Expected away goals
        rho     : Dixon-Coles correlation correction
        """
        rng = np.random.default_rng(seed=self.seed)

        # Compute score probability matrix (analytic)
        mat = score_probability_matrix(mu_home, mu_away, rho)
        analytic_p_home, analytic_p_draw, analytic_p_away = matrix_to_1x2(mat)

        # Flatten matrix to 1D probability distribution for sampling
        max_goals = mat.shape[0]
        flat_probs = mat.flatten()
        flat_probs = flat_probs / flat_probs.sum()  # Ensure sums to 1

        n_outcomes = len(flat_probs)
        indices = rng.choice(n_outcomes, size=self.n_simulations, p=flat_probs, replace=True)

        # Convert linear indices back to (home_goals, away_goals)
        home_goals_sim = indices // max_goals
        away_goals_sim = indices % max_goals

        # Aggregate results
        home_wins = np.sum(home_goals_sim > away_goals_sim)
        draws = np.sum(home_goals_sim == away_goals_sim)
        away_wins = np.sum(away_goals_sim > home_goals_sim)

        total_goals_sim = home_goals_sim + away_goals_sim

        # Scoreline distribution
        from collections import Counter
        scoreline_counts = Counter(
            zip(home_goals_sim.tolist(), away_goals_sim.tolist())
        )
        top_scorelines = [
            {
                "home_goals": h,
                "away_goals": a,
                "probability": round(cnt / self.n_simulations, 4),
                "label": f"{h}-{a}",
            }
            for (h, a), cnt in scoreline_counts.most_common(15)
        ]

        n = self.n_simulations
        result = MatchSimulationResult(
            fixture_id=fixture_id,
            n_simulations=n,
            seed=self.seed,
            mu_home=mu_home,
            mu_away=mu_away,
            rho=rho,
            p_home_win=float(home_wins / n),
            p_draw=float(draws / n),
            p_away_win=float(away_wins / n),
            mean_home_goals=float(np.mean(home_goals_sim)),
            mean_away_goals=float(np.mean(away_goals_sim)),
            median_home_goals=float(np.median(home_goals_sim)),
            median_away_goals=float(np.median(away_goals_sim)),
            std_home_goals=float(np.std(home_goals_sim)),
            std_away_goals=float(np.std(away_goals_sim)),
            p_btts=float(np.mean((home_goals_sim > 0) & (away_goals_sim > 0))),
            p_over_15=float(np.mean(total_goals_sim > 1)),
            p_over_25=float(np.mean(total_goals_sim > 2)),
            p_over_35=float(np.mean(total_goals_sim > 3)),
            p_over_45=float(np.mean(total_goals_sim > 4)),
            p_clean_sheet_home=float(np.mean(away_goals_sim == 0)),
            p_clean_sheet_away=float(np.mean(home_goals_sim == 0)),
            top_scorelines=top_scorelines,
            analytic_p_home=analytic_p_home,
            analytic_p_draw=analytic_p_draw,
            analytic_p_away=analytic_p_away,
            generated_at=datetime.now(timezone.utc),
        )
        return result

    def validate_against_analytic(
        self, result: MatchSimulationResult, tolerance: float = 0.02
    ) -> bool:
        """
        Verify simulation probabilities are close to analytic values.
        Large deviations indicate a bug or insufficient N.
        """
        delta_home = abs(result.p_home_win - result.analytic_p_home)
        delta_draw = abs(result.p_draw - result.analytic_p_draw)
        delta_away = abs(result.p_away_win - result.analytic_p_away)
        return max(delta_home, delta_draw, delta_away) <= tolerance
