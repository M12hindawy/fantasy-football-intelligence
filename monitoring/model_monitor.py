"""
monitoring/model_monitor.py
===========================
Evaluates model predictions against actual outcomes.
Calculates Brier Score, Log Loss, and Expected Calibration Error (ECE).
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    brier_score: float
    log_loss: float
    accuracy: float
    sample_size: int
    home_win_ece: float
    draw_ece: float
    away_win_ece: float
    overall_ece: float

class ModelMonitor:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def evaluate(self, predictions: list[tuple[float, float, float]], actuals: list[int]) -> Optional[EvaluationMetrics]:
        """
        predictions: list of (p_home, p_draw, p_away)
        actuals: list of outcomes (0=Home, 1=Draw, 2=Away)
        """
        n = len(actuals)
        if n == 0 or len(predictions) != n:
            return None

        brier_sum = 0.0
        log_loss_sum = 0.0
        correct = 0

        # For ECE
        bins_h = {i: {"conf": 0.0, "acc": 0.0, "count": 0} for i in range(self.n_bins)}
        bins_d = {i: {"conf": 0.0, "acc": 0.0, "count": 0} for i in range(self.n_bins)}
        bins_a = {i: {"conf": 0.0, "acc": 0.0, "count": 0} for i in range(self.n_bins)}

        for (ph, pd, pa), actual in zip(predictions, actuals):
            # Normalize just in case
            total = ph + pd + pa
            ph, pd, pa = ph/total, pd/total, pa/total

            # Brier Score (multi-class)
            y_h = 1.0 if actual == 0 else 0.0
            y_d = 1.0 if actual == 1 else 0.0
            y_a = 1.0 if actual == 2 else 0.0
            brier_sum += (ph - y_h)**2 + (pd - y_d)**2 + (pa - y_a)**2

            # Log loss
            pred_prob = ph if actual == 0 else (pd if actual == 1 else pa)
            pred_prob = max(1e-15, min(1 - 1e-15, pred_prob))
            log_loss_sum += -math.log(pred_prob)

            # Accuracy
            if (actual == 0 and ph >= pd and ph >= pa) or \
               (actual == 1 and pd >= ph and pd >= pa) or \
               (actual == 2 and pa >= ph and pa >= pd):
                correct += 1

            # ECE Accumulation
            self._accum_bin(bins_h, ph, y_h)
            self._accum_bin(bins_d, pd, y_d)
            self._accum_bin(bins_a, pa, y_a)

        ece_h = self._calc_ece(bins_h, n)
        ece_d = self._calc_ece(bins_d, n)
        ece_a = self._calc_ece(bins_a, n)
        overall_ece = (ece_h + ece_d + ece_a) / 3.0

        return EvaluationMetrics(
            brier_score=brier_sum / n,
            log_loss=log_loss_sum / n,
            accuracy=correct / n,
            sample_size=n,
            home_win_ece=ece_h,
            draw_ece=ece_d,
            away_win_ece=ece_a,
            overall_ece=overall_ece,
        )

    def _accum_bin(self, bins: dict, prob: float, actual: float) -> None:
        bin_idx = min(int(prob * self.n_bins), self.n_bins - 1)
        bins[bin_idx]["conf"] += prob
        bins[bin_idx]["acc"] += actual
        bins[bin_idx]["count"] += 1

    def _calc_ece(self, bins: dict, total_samples: int) -> float:
        ece = 0.0
        for b in bins.values():
            if b["count"] > 0:
                avg_conf = b["conf"] / b["count"]
                avg_acc = b["acc"] / b["count"]
                weight = b["count"] / total_samples
                ece += weight * abs(avg_acc - avg_conf)
        return ece
