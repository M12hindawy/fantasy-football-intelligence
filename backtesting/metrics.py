"""
backtesting/metrics.py
======================
All scoring metrics required by spec Section 13.

- Brier score
- Log loss
- Expected Calibration Error (ECE)
- Accuracy
- Mean Absolute Error for goals
- Closing Line Value (CLV)
- ROI / Yield
- Maximum drawdown
- Turnover
- Selection count

All functions are pure (stateless, no I/O).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ── Brier Score ───────────────────────────────────────────────────────────────

def BrierScore(probs: list[float], outcomes: list[int]) -> float:
    """
    Mean squared probability error.

    BS = (1/n) Σ (p_i - o_i)²

    Lower is better.  Random classifier: 0.25.  Perfect: 0.0.
    """
    if not probs:
        return float("nan")
    arr_p = np.array(probs, dtype=float)
    arr_o = np.array(outcomes, dtype=float)
    return float(np.mean((arr_p - arr_o) ** 2))


# ── Log Loss ──────────────────────────────────────────────────────────────────

def LogLoss(probs: list[float], outcomes: list[int], eps: float = 1e-9) -> float:
    """
    Binary cross-entropy log loss.

    LL = -(1/n) Σ [o_i * log(p_i) + (1-o_i) * log(1-p_i)]

    Lower is better.
    """
    if not probs:
        return float("nan")
    arr_p = np.clip(np.array(probs, dtype=float), eps, 1 - eps)
    arr_o = np.array(outcomes, dtype=float)
    return float(-np.mean(arr_o * np.log(arr_p) + (1 - arr_o) * np.log(1 - arr_p)))


# ── Accuracy ──────────────────────────────────────────────────────────────────

def Accuracy(probs: list[float], outcomes: list[int], threshold: float = 0.5) -> float:
    """Fraction of predictions that match outcomes when rounded at threshold."""
    if not probs:
        return float("nan")
    arr_p = np.array(probs)
    arr_o = np.array(outcomes)
    preds = (arr_p >= threshold).astype(int)
    return float(np.mean(preds == arr_o))


# ── Expected Calibration Error ────────────────────────────────────────────────

def compute_calibration_ece(
    probs: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Re-exported here for use in the backtester without circular imports.
    """
    from models.calibration import expected_calibration_error
    return expected_calibration_error(probs, outcomes, n_bins)


# ── MAE for goals ─────────────────────────────────────────────────────────────

def GoalMAE(predicted_xg: list[float], actual_goals: list[int]) -> float:
    """
    Mean Absolute Error between predicted expected goals and actual goals.
    """
    if not predicted_xg:
        return float("nan")
    arr_pred = np.array(predicted_xg, dtype=float)
    arr_act = np.array(actual_goals, dtype=float)
    return float(np.mean(np.abs(arr_pred - arr_act)))


# ── Closing Line Value ────────────────────────────────────────────────────────

def compute_clv(model_prob: float, closing_prob: float) -> float:
    """
    Closing Line Value (CLV).

    CLV = model_prob − closing_prob

    Positive CLV means the model identified value that the closing market
    also recognised (and removed by shortening the price).

    Parameters
    ----------
    model_prob   : model probability at prediction time
    closing_prob : fair probability derived from closing odds

    Note: closing_prob should be the MARGIN-ADJUSTED closing fair probability.
    """
    return model_prob - closing_prob


def mean_clv(model_probs: list[float], closing_probs: list[float]) -> float:
    """Mean CLV over a set of predictions."""
    if not model_probs:
        return float("nan")
    clvs = [compute_clv(m, c) for m, c in zip(model_probs, closing_probs)]
    return float(np.mean(clvs))


# ── ROI / Yield ───────────────────────────────────────────────────────────────

def compute_roi(
    net_returns: list[float],
    turnover: Optional[float] = None,
) -> float:
    """
    Return on Investment as a percentage.

    ROI = (total_profit / turnover) * 100

    If turnover is not provided, assumes 1 unit staked per bet.

    Parameters
    ----------
    net_returns : list of per-bet net return (e.g., 0.85 means +85% on a win,
                  -1.0 means losing the stake)
    turnover    : total staked (if None, uses len(net_returns))
    """
    if not net_returns:
        return float("nan")
    total_profit = sum(net_returns)
    t = turnover if turnover is not None else float(len(net_returns))
    if t == 0:
        return float("nan")
    return round(total_profit / t * 100.0, 4)


def compute_yield(
    net_returns: list[float],
    stakes: Optional[list[float]] = None,
) -> float:
    """
    Yield = total_profit / total_staked * 100 (same as ROI when stakes = 1).
    """
    return compute_roi(net_returns, sum(stakes) if stakes else None)


# ── Maximum Drawdown ──────────────────────────────────────────────────────────

def compute_max_drawdown(cumulative_returns: list[float]) -> float:
    """
    Maximum peak-to-trough decline in cumulative profit.

    Parameters
    ----------
    cumulative_returns : running cumulative profit/loss (e.g., after each bet)

    Returns
    -------
    Maximum drawdown as a positive number (e.g., 5.0 means −5 units).
    """
    if not cumulative_returns:
        return 0.0
    arr = np.array(cumulative_returns)
    peak = np.maximum.accumulate(arr)
    drawdowns = peak - arr
    return float(np.max(drawdowns))


# ── Summary report ────────────────────────────────────────────────────────────

def summarise_metrics(
    probs: list[float],
    outcomes: list[int],
    net_returns: Optional[list[float]] = None,
    xg_preds: Optional[list[float]] = None,
    actual_goals: Optional[list[int]] = None,
    closing_probs: Optional[list[float]] = None,
    confidence_intervals: bool = True,
) -> dict:
    """
    Compute all metrics and return as a dict.

    Uses bootstrap for confidence intervals when confidence_intervals=True.
    """
    n = len(probs)
    result = {
        "n_predictions": n,
        "brier_score": BrierScore(probs, outcomes),
        "log_loss": LogLoss(probs, outcomes),
        "ece": compute_calibration_ece(probs, outcomes),
        "accuracy": Accuracy(probs, outcomes),
    }

    if net_returns is not None:
        cumulative = list(np.cumsum(net_returns))
        result["roi_pct"] = compute_roi(net_returns)
        result["yield_pct"] = compute_yield(net_returns)
        result["max_drawdown"] = compute_max_drawdown(cumulative)
        result["turnover"] = float(len(net_returns))
        result["selection_count"] = len(net_returns)

    if xg_preds is not None and actual_goals is not None:
        result["goal_mae"] = GoalMAE(xg_preds, actual_goals)

    if closing_probs is not None:
        result["mean_clv"] = mean_clv(probs, closing_probs)

    # Bootstrap confidence intervals (1 000 samples)
    if confidence_intervals and n >= 30:
        rng = np.random.default_rng(42)
        bs_briers = []
        for _ in range(1_000):
            idx = rng.integers(0, n, size=n)
            bp = [probs[i] for i in idx]
            bo = [outcomes[i] for i in idx]
            bs_briers.append(BrierScore(bp, bo))
        result["brier_ci_95"] = [
            round(float(np.percentile(bs_briers, 2.5)), 4),
            round(float(np.percentile(bs_briers, 97.5)), 4),
        ]

    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in result.items()}
