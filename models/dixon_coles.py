"""
models/dixon_coles.py
=====================
Dixon-Coles bivariate Poisson model for football score prediction.

Reference
---------
Dixon, M. J., & Coles, S. G. (1997). Modelling association football
scores and inefficiencies in the football betting market.
Applied Statistics, 46(2), 265-280.

The model fits team-specific attack (α) and defence (δ) parameters,
a home advantage (γ), and a Dixon-Coles correction factor (ρ) for
low-scoring outcomes (0-0, 1-0, 0-1, 1-1).

Usage
-----
    model = DixonColesModel()
    model.fit(match_records)          # list of dicts with goals
    probs = model.predict(home_id, away_id)
    print(probs.p_home_win, probs.p_draw, probs.p_away_win)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from data.schemas import PredictionOutput, ScoreMatrix, SourceRecord, TeamRecord

MODEL_VERSION = "dixon-coles-1.1"
MAX_GOALS = 8        # Matrix truncation; ≥8 covers >99.9% of outcomes


# ── Low-score correction ──────────────────────────────────────────────────────

def _tau(home: int, away: int, mu_home: float, mu_away: float, rho: float) -> float:
    """
    Dixon-Coles correction factor τ for low-score outcomes.

    τ(0,0) = 1 − μ_h · μ_a · ρ
    τ(1,0) = 1 + μ_a · ρ
    τ(0,1) = 1 + μ_h · ρ
    τ(1,1) = 1 − ρ
    otherwise = 1
    """
    if home == 0 and away == 0:
        return 1.0 - mu_home * mu_away * rho
    elif home == 1 and away == 0:
        return 1.0 + mu_away * rho
    elif home == 0 and away == 1:
        return 1.0 + mu_home * rho
    elif home == 1 and away == 1:
        return 1.0 - rho
    return 1.0


# ── Score matrix ──────────────────────────────────────────────────────────────

def score_probability_matrix(
    mu_home: float,
    mu_away: float,
    rho: float,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """
    Compute P(home=i, away=j) for all i,j ∈ [0, max_goals].

    Returns
    -------
    np.ndarray of shape (max_goals+1, max_goals+1)
    """
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_h = poisson.pmf(i, mu_home)
            p_a = poisson.pmf(j, mu_away)
            corr = _tau(i, j, mu_home, mu_away, rho)
            matrix[i, j] = p_h * p_a * corr

    # Normalise so matrix sums exactly to 1
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


# ── Derived market probabilities ──────────────────────────────────────────────

def matrix_to_1x2(matrix: np.ndarray) -> tuple[float, float, float]:
    """Extract P(home win), P(draw), P(away win) from a score matrix."""
    n = matrix.shape[0]
    p_home = float(np.sum(np.tril(matrix, -1)))   # home > away (below diagonal)
    p_draw = float(np.trace(matrix))
    p_away = float(np.sum(np.triu(matrix, 1)))    # away > home (above diagonal)
    return p_home, p_draw, p_away


def matrix_to_over_under(matrix: np.ndarray, line: float) -> tuple[float, float]:
    """
    Calculate P(total goals > line) and P(total goals ≤ line).

    Handles half-ball lines (1.5, 2.5, …) and whole-ball push lines.
    """
    n = matrix.shape[0]
    p_over = 0.0
    for i in range(n):
        for j in range(n):
            total = i + j
            if total > line:
                p_over += matrix[i, j]
    p_under = 1.0 - p_over
    return p_over, p_under


def matrix_to_btts(matrix: np.ndarray) -> float:
    """P(both teams score) = 1 − P(home=0) − P(away=0) + P(0-0)."""
    p_home_zero = matrix[0, :].sum()
    p_away_zero = matrix[:, 0].sum()
    p_00 = matrix[0, 0]
    return float(1.0 - p_home_zero - p_away_zero + p_00)


# ── Model class ───────────────────────────────────────────────────────────────

@dataclass
class TeamParams:
    attack: float = 1.0
    defence: float = 1.0


@dataclass
class DixonColesModel:
    """
    Fitted Dixon-Coles model.

    Parameters
    ----------
    time_weight_xi : decay parameter for time-weighting (0 = no decay,
                     0.0065 is typical for weekly data)
    """

    time_weight_xi: float = 0.0065
    max_goals: int = MAX_GOALS

    # Fitted parameters (set after fit())
    teams: dict[str, TeamParams] = field(default_factory=dict)
    home_advantage: float = 0.25   # log-scale additive home attack boost
    rho: float = -0.1              # Dixon-Coles correction (typically < 0)
    intercept: float = 0.0
    is_fitted: bool = False
    n_matches_trained: int = 0
    fitted_at: Optional[datetime] = None

    def _time_weight(self, match_date: datetime, as_of: datetime) -> float:
        """Exponential time weight; more recent matches weighted higher."""
        if match_date.tzinfo is None:
            match_date = match_date.replace(tzinfo=timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        days = max(0, (as_of - match_date).days)
        return math.exp(-self.time_weight_xi * days)

    def _expected_goals(self, home_id: str, away_id: str) -> tuple[float, float]:
        """
        Compute μ_home and μ_away for a fixture.

        μ_h = exp(intercept + α_h − δ_a + γ)
        μ_a = exp(intercept + α_a − δ_h)
        """
        h = self.teams.get(home_id, TeamParams())
        a = self.teams.get(away_id, TeamParams())
        mu_h = math.exp(self.intercept + h.attack - a.defence + self.home_advantage)
        mu_a = math.exp(self.intercept + a.attack - h.defence)
        return mu_h, mu_a

    def fit(
        self,
        match_records: list[dict],
        as_of: Optional[datetime] = None,
    ) -> "DixonColesModel":
        """
        Fit the model using maximum likelihood estimation.

        match_records : list of dicts with keys:
            home_id, away_id, home_goals, away_goals, date (datetime)

        Returns self (for chaining).
        """
        if not match_records:
            raise ValueError("Need at least one match record to fit.")

        if as_of is None:
            as_of = datetime.now(timezone.utc)

        # Collect all team IDs
        team_ids = sorted(
            {m["home_id"] for m in match_records} |
            {m["away_id"] for m in match_records}
        )
        n_teams = len(team_ids)
        team_idx = {t: i for i, t in enumerate(team_ids)}

        # Parameter vector: [attack_0..n, defence_0..n, home_advantage, rho, intercept]
        # Constrain: attack[0] = 0 for identifiability
        n_params = n_teams * 2 + 3

        def _neg_log_likelihood(params: np.ndarray) -> float:
            attacks = np.concatenate([[0.0], params[:n_teams - 1]])
            defences = params[n_teams - 1: 2 * n_teams - 1]
            ha = params[2 * n_teams - 1]
            rho_ = params[2 * n_teams]
            intercept_ = params[2 * n_teams + 1]

            ll = 0.0
            for m in match_records:
                w = self._time_weight(m["date"], as_of)
                hi = team_idx[m["home_id"]]
                ai = team_idx[m["away_id"]]
                mu_h = math.exp(intercept_ + attacks[hi] - defences[ai] + ha)
                mu_a = math.exp(intercept_ + attacks[ai] - defences[hi])
                hg = int(m["home_goals"])
                ag = int(m["away_goals"])
                corr = _tau(hg, ag, mu_h, mu_a, rho_)
                if corr <= 0 or mu_h <= 0 or mu_a <= 0:
                    return 1e12
                ll += w * (
                    math.log(corr)
                    + poisson.logpmf(hg, mu_h)
                    + poisson.logpmf(ag, mu_a)
                )
            return -ll

        x0 = np.zeros(n_params)
        x0[2 * n_teams - 1] = 0.25  # home advantage
        x0[2 * n_teams] = -0.1       # rho

        result = minimize(
            _neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        # Unpack
        p = result.x
        attacks = np.concatenate([[0.0], p[:n_teams - 1]])
        defences = p[n_teams - 1: 2 * n_teams - 1]

        for i, tid in enumerate(team_ids):
            self.teams[tid] = TeamParams(
                attack=float(attacks[i]),
                defence=float(defences[i]),
            )
        self.home_advantage = float(p[2 * n_teams - 1])
        self.rho = float(p[2 * n_teams])
        self.intercept = float(p[2 * n_teams + 1])
        self.is_fitted = True
        self.n_matches_trained = len(match_records)
        self.fitted_at = datetime.now(timezone.utc)
        return self

    def predict(
        self,
        home_id: str,
        away_id: str,
        fixture_id: str = "unknown",
        competition: str = "",
    ) -> PredictionOutput:
        """
        Predict match outcome probabilities.

        Returns a fully populated PredictionOutput.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        mu_h, mu_a = self._expected_goals(home_id, away_id)
        mat = score_probability_matrix(mu_h, mu_a, self.rho, self.max_goals)

        p_home, p_draw, p_away = matrix_to_1x2(mat)
        p_btts = matrix_to_btts(mat)
        p_o05, _ = matrix_to_over_under(mat, 0.5)
        p_o15, _ = matrix_to_over_under(mat, 1.5)
        p_o25, _ = matrix_to_over_under(mat, 2.5)
        p_o35, _ = matrix_to_over_under(mat, 3.5)
        p_o45, _ = matrix_to_over_under(mat, 4.5)

        # Build ScoreMatrix schema object
        matrix_list = mat.tolist()
        score_mat = ScoreMatrix(max_goals=self.max_goals, matrix=matrix_list)

        return PredictionOutput(
            fixture_id=fixture_id,
            model_name="DixonColes",
            model_version=MODEL_VERSION,
            generated_at=datetime.now(timezone.utc),
            p_home_win=round(p_home, 6),
            p_draw=round(p_draw, 6),
            p_away_win=round(p_away, 6),
            expected_goals_home=round(mu_h, 4),
            expected_goals_away=round(mu_a, 4),
            p_btts=round(p_btts, 6),
            p_over_05=round(p_o05, 6),
            p_over_15=round(p_o15, 6),
            p_over_25=round(p_o25, 6),
            p_over_35=round(p_o35, 6),
            p_over_45=round(p_o45, 6),
            score_matrix=score_mat,
        )

    def predict_raw(
        self, home_id: str, away_id: str
    ) -> tuple[float, float, float, float]:
        """
        Quick prediction: (p_home, p_draw, p_away, mu_home, mu_away).
        Does not build the full PredictionOutput object.
        """
        mu_h, mu_a = self._expected_goals(home_id, away_id)
        mat = score_probability_matrix(mu_h, mu_a, self.rho, self.max_goals)
        p_home, p_draw, p_away = matrix_to_1x2(mat)
        return p_home, p_draw, p_away, mu_h, mu_a
