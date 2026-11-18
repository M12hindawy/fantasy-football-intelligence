"""
models/ensemble.py
==================
Calibrated ensemble that combines Dixon-Coles and Dynamic Elo predictions.

Weights are learned via walk-forward cross-validation.  Before sufficient
data is available, the default weights from settings.py are used.

The ensemble reports:
- point-estimate probabilities (weighted average)
- 90% bootstrap confidence intervals
- model agreement score (0 = full disagreement, 1 = perfect agreement)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from config.settings import settings
from data.schemas import PredictionOutput
from models.calibration import IsotonicCalibrator
from models.dixon_coles import DixonColesModel
from models.elo_model import DynamicEloModel

# v3: XGBoost optional leg — safe import, no crash when not installed
try:
    from models.xgboost_model import XGBoostMatchModel as _XGBMatchModel
    _XGB_IMPORT_OK = True
except ImportError:
    _XGB_IMPORT_OK = False

MODEL_VERSION = "ensemble-2.0"
BOOTSTRAP_N = 500   # Number of bootstrap samples for interval estimation


@dataclass
class EnsembleModel:
    """
    Weighted ensemble of DixonColes + DynamicElo (+ optional XGBoost)
    with isotonic calibration.

    v3: XGBoost is a third leg controlled by ENABLE_XGBOOST flag.
    When disabled (default), ensemble behaves exactly as v2.

    Parameters
    ----------
    dc_weight  : weight assigned to Dixon-Coles output (0–1)
    elo_weight : weight assigned to Elo output (0–1)
    xgb_weight : weight for XGBoost leg (active only when ENABLE_XGBOOST=True)
    """

    dc_weight: float = field(default_factory=lambda: settings.dixon_coles_weight)
    elo_weight: float = field(default_factory=lambda: settings.elo_weight)
    xgb_weight: float = field(default_factory=lambda: settings.xgboost_weight)
    dc_model: Optional[DixonColesModel] = None
    elo_model: Optional[DynamicEloModel] = None
    xgb_model: Optional[object] = None   # XGBoostMatchModel or None

    # Per-market calibrators (keyed by "home_win", "draw", "away_win", "over_25", etc.)
    calibrators: dict[str, IsotonicCalibrator] = field(default_factory=dict)

    n_matches_trained: int = 0
    weights_source: str = "default_settings"  # "default_settings" | "walk_forward"
    fitted_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        # Instantiate XGBoost if enabled and importable
        if settings.enable_xgboost and _XGB_IMPORT_OK and self.xgb_model is None:
            m = _XGBMatchModel()
            loaded = m.load()   # No-op (returns False) when no model file exists
            self.xgb_model = m if loaded else None
        self._rebalance_weights()

    def _rebalance_weights(self) -> None:
        """Normalise weights so DC+Elo+XGB sum to 1.0."""
        xgb_active = self.xgb_model is not None and getattr(self.xgb_model, "_trained", False)
        if xgb_active:
            total = self.dc_weight + self.elo_weight + self.xgb_weight
        else:
            total = self.dc_weight + self.elo_weight
        if abs(total) < 1e-9:
            total = 1.0
        factor = 1.0 / total
        self.dc_weight  *= factor
        self.elo_weight *= factor
        if xgb_active:
            self.xgb_weight *= factor
        else:
            self.xgb_weight = 0.0

    # ── Weight learning ───────────────────────────────────────────────────────

    def set_weights_from_validation(
        self, dc_brier: float, elo_brier: float, xgb_brier: Optional[float] = None
    ) -> None:
        """
        Set ensemble weights inversely proportional to validation Brier scores.
        Lower Brier score → higher weight.
        """
        eps = 1e-6
        inv_dc  = 1.0 / (dc_brier  + eps)
        inv_elo = 1.0 / (elo_brier  + eps)

        if xgb_brier is not None and self.xgb_model is not None:
            inv_xgb = 1.0 / (xgb_brier + eps)
            total = inv_dc + inv_elo + inv_xgb
            self.xgb_weight = inv_xgb / total
        else:
            inv_xgb = 0.0
            total = inv_dc + inv_elo

        self.dc_weight  = inv_dc  / total
        self.elo_weight = inv_elo / total
        self.weights_source = "walk_forward"

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        home_id: str,
        away_id: str,
        fixture_id: str = "unknown",
        home_name: str = "",
        away_name: str = "",
        home_features: Optional[dict] = None,
        away_features: Optional[dict] = None,
    ) -> PredictionOutput:
        """
        Produce an ensemble prediction.

        Returns a PredictionOutput with calibrated probabilities and
        bootstrap confidence intervals.
        
        v3: home_features / away_features are optional. When provided and
        XGBoost is active, a third model leg is blended in.
        """
        if self.dc_model is None or self.elo_model is None:
            raise RuntimeError(
                "Both dc_model and elo_model must be set before calling predict()."
            )

        dc_pred  = self.dc_model.predict(home_id, away_id, fixture_id)
        elo_pred = self.elo_model.predict(home_id, away_id, fixture_id, home_name, away_name)

        # ── Weighted combination ──────────────────────────────────────────
        w_dc  = self.dc_weight
        w_elo = self.elo_weight

        p_home = w_dc * dc_pred.p_home_win + w_elo * elo_pred.p_home_win
        p_draw = w_dc * dc_pred.p_draw     + w_elo * elo_pred.p_draw
        p_away = w_dc * dc_pred.p_away_win + w_elo * elo_pred.p_away_win

        # XGBoost third leg (only when trained + features provided)
        xgb_probs = None
        xgb_active = (
            self.xgb_model is not None
            and getattr(self.xgb_model, "_trained", False)
            and home_features is not None
            and away_features is not None
        )
        if xgb_active:
            xgb_probs = self.xgb_model.predict(home_features, away_features)

        if xgb_probs is not None and self.xgb_weight > 0:
            xp_h, xp_d, xp_a = xgb_probs
            p_home = p_home + self.xgb_weight * xp_h
            p_draw = p_draw + self.xgb_weight * xp_d
            p_away = p_away + self.xgb_weight * xp_a

        # Normalise (floating point)
        total = p_home + p_draw + p_away
        p_home /= total
        p_draw /= total
        p_away /= total

        # Apply calibration if available
        if "home_win" in self.calibrators:
            p_home = self.calibrators["home_win"].apply(p_home)
        if "draw" in self.calibrators:
            p_draw = self.calibrators["draw"].apply(p_draw)
        if "away_win" in self.calibrators:
            p_away = self.calibrators["away_win"].apply(p_away)

        # Re-normalise after calibration
        total = p_home + p_draw + p_away
        p_home /= total
        p_draw /= total
        p_away /= total

        # Goal-related fields come primarily from Dixon-Coles
        xg_h = dc_pred.expected_goals_home if self.dc_model.is_fitted else 1.35
        xg_a = dc_pred.expected_goals_away if self.dc_model.is_fitted else 1.05

        # Bootstrap confidence interval on p_home_win
        ci_low, ci_high = self._bootstrap_interval(
            dc_pred.p_home_win, elo_pred.p_home_win
        )

        # Model agreement (1 − max pairwise difference across three outcomes)
        disagreement = max(
            abs(dc_pred.p_home_win - elo_pred.p_home_win),
            abs(dc_pred.p_draw - elo_pred.p_draw),
            abs(dc_pred.p_away_win - elo_pred.p_away_win),
        )
        agreement = round(1.0 - disagreement, 4)

        # Combine derived market probs from DC (more reliable than Elo for these)
        p_over_25 = dc_pred.p_over_25
        if p_over_25 and "over_25" in self.calibrators:
            p_over_25 = self.calibrators["over_25"].apply(p_over_25)

        model_name = "Ensemble(DC+Elo+XGB)" if (xgb_probs is not None) else "Ensemble(DC+Elo)"

        return PredictionOutput(
            fixture_id=fixture_id,
            model_name=model_name,
            model_version=MODEL_VERSION,
            generated_at=datetime.now(timezone.utc),
            p_home_win=round(p_home, 6),
            p_draw=round(p_draw, 6),
            p_away_win=round(p_away, 6),
            expected_goals_home=round(xg_h, 4),
            expected_goals_away=round(xg_a, 4),
            p_btts=dc_pred.p_btts,
            p_over_05=dc_pred.p_over_05,
            p_over_15=dc_pred.p_over_15,
            p_over_25=p_over_25,
            p_over_35=dc_pred.p_over_35,
            p_over_45=dc_pred.p_over_45,
            score_matrix=dc_pred.score_matrix,
            p_home_win_low=round(ci_low, 4),
            p_home_win_high=round(ci_high, 4),
        ), agreement

    def _bootstrap_interval(
        self,
        dc_prob: float,
        elo_prob: float,
        alpha: float = 0.10,
    ) -> tuple[float, float]:
        """
        Estimate a (1-alpha) confidence interval for the ensemble probability
        using bootstrap sampling over the two model outputs.

        This is a simplified parametric bootstrap — the two models are treated
        as independent samples of a Dirichlet-like distribution centred at the
        weighted mean.
        """
        rng = np.random.default_rng(seed=42)
        samples = rng.choice(
            [dc_prob, elo_prob],
            size=BOOTSTRAP_N,
            p=[self.dc_weight, self.elo_weight],
            replace=True,
        )
        noise = rng.normal(0, 0.02, size=BOOTSTRAP_N)
        samples = np.clip(samples + noise, 0.0, 1.0)

        low = float(np.percentile(samples, (alpha / 2) * 100))
        high = float(np.percentile(samples, (1 - alpha / 2) * 100))
        return low, high

    # ── Calibrator management ─────────────────────────────────────────────────

    def add_calibrator(self, market: str, calibrator: IsotonicCalibrator) -> None:
        """Register a pre-fitted calibrator for a market key."""
        self.calibrators[market] = calibrator

    def calibration_summary(self) -> dict[str, str]:
        return {
            market: cal.status
            for market, cal in self.calibrators.items()
        }
