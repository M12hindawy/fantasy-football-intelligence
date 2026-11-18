"""
models/calibration.py
=====================
Probability calibration utilities.

Wraps scikit-learn's isotonic regression and Platt scaling, and
computes calibration metrics (ECE, reliability curves) used by the
ensemble and the decision engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ── Expected Calibration Error ────────────────────────────────────────────────

def expected_calibration_error(
    probs: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> float:
    """
    Compute the Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b| / n) × |acc(B_b) − conf(B_b)|

    Parameters
    ----------
    probs    : predicted probabilities ∈ [0, 1]
    outcomes : binary observed outcomes (1 = event occurred)
    n_bins   : number of equal-width probability bins

    Returns
    -------
    float ∈ [0, 1]; lower is better (0 = perfectly calibrated)
    """
    probs_arr = np.array(probs)
    outcomes_arr = np.array(outcomes)
    n = len(probs_arr)
    if n == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_arr >= lo) & (probs_arr < hi)
        if mask.sum() == 0:
            continue
        bin_probs = probs_arr[mask]
        bin_outcomes = outcomes_arr[mask]
        acc = bin_outcomes.mean()
        conf = bin_probs.mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_curve(
    probs: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> tuple[list[float], list[float]]:
    """
    Return (fraction_of_positives, mean_predicted_probability) for plotting.

    Uses sklearn's calibration_curve under the hood.
    """
    frac_pos, mean_pred = calibration_curve(
        outcomes, probs, n_bins=n_bins, strategy="uniform"
    )
    return frac_pos.tolist(), mean_pred.tolist()


# ── Isotonic calibrator ───────────────────────────────────────────────────────

@dataclass
class IsotonicCalibrator:
    """
    Wraps sklearn IsotonicRegression with a consistent fit/apply interface.

    Must be fitted on a held-out calibration set (NOT the training set).
    """

    _model: IsotonicRegression = field(default_factory=IsotonicRegression, repr=False)
    is_fitted: bool = False
    calibration_ece: Optional[float] = None
    n_calibration_samples: int = 0

    def fit(self, raw_probs: list[float], outcomes: list[int]) -> "IsotonicCalibrator":
        """
        Fit the calibrator.

        Parameters
        ----------
        raw_probs : uncalibrated model probabilities
        outcomes  : binary outcomes (1 = event occurred)
        """
        X = np.array(raw_probs).reshape(-1, 1)
        y = np.array(outcomes)
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(X.ravel(), y)
        self.is_fitted = True
        self.n_calibration_samples = len(outcomes)

        # Evaluate ECE on calibration set (in-sample; used for reporting only)
        cal_probs = self._model.predict(X.ravel()).tolist()
        self.calibration_ece = expected_calibration_error(cal_probs, list(y))
        return self

    def apply(self, raw_prob: float) -> float:
        """
        Apply calibration to a single raw probability.

        Returns
        -------
        float ∈ [0, 1]
        """
        if not self.is_fitted:
            return raw_prob  # Pass-through when not fitted
        cal = float(self._model.predict([raw_prob])[0])
        return max(0.0, min(1.0, cal))

    def apply_batch(self, raw_probs: list[float]) -> list[float]:
        """Apply calibration to a list of raw probabilities."""
        if not self.is_fitted:
            return raw_probs
        cal = self._model.predict(np.array(raw_probs)).tolist()
        return [max(0.0, min(1.0, p)) for p in cal]

    @property
    def status(self) -> str:
        if not self.is_fitted:
            return "not_fitted"
        ece = self.calibration_ece or 0.0
        if ece < 0.03:
            return "well_calibrated"
        elif ece < 0.07:
            return "acceptably_calibrated"
        else:
            return "poorly_calibrated"


# ── Platt (logistic) calibrator ───────────────────────────────────────────────

@dataclass
class PlattCalibrator:
    """
    Platt scaling: fits a logistic regression on the raw model score.

    Useful when the raw model output is a log-odds score rather than
    a probability.
    """

    _model: LogisticRegression = field(
        default_factory=lambda: LogisticRegression(C=1.0), repr=False
    )
    is_fitted: bool = False
    calibration_ece: Optional[float] = None

    def fit(self, raw_scores: list[float], outcomes: list[int]) -> "PlattCalibrator":
        X = np.array(raw_scores).reshape(-1, 1)
        y = np.array(outcomes)
        self._model.fit(X, y)
        self.is_fitted = True
        cal = self._model.predict_proba(X)[:, 1].tolist()
        self.calibration_ece = expected_calibration_error(cal, list(y))
        return self

    def apply(self, raw_score: float) -> float:
        if not self.is_fitted:
            return raw_score
        prob = float(self._model.predict_proba([[raw_score]])[0, 1])
        return max(0.0, min(1.0, prob))


# ── Convenience function ──────────────────────────────────────────────────────

def calibration_status_label(ece: Optional[float]) -> str:
    """Return a human-readable label for a calibration ECE score."""
    if ece is None:
        return "unknown"
    if ece < 0.03:
        return "well_calibrated (ECE < 3%)"
    elif ece < 0.07:
        return "acceptable (ECE 3–7%)"
    else:
        return f"poor (ECE = {ece:.1%})"
