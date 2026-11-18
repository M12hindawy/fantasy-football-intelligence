"""
monitoring/drift_detector.py
============================
Detects statistical drift in model performance.
Alerts when recent performance degrades significantly compared to historical baseline.
"""
from __future__ import annotations
import logging
from config.settings import settings
from monitoring.model_monitor import EvaluationMetrics

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, threshold: float = None):
        self.threshold = threshold or settings.model_drift_threshold

    def check_drift(self, baseline: EvaluationMetrics, recent: EvaluationMetrics) -> dict:
        """
        Checks if 'recent' metrics are significantly worse than 'baseline'.
        Returns a dict of drift flags and degraded metrics.
        """
        is_drifted = False
        degraded = {}

        # Brier Score increases when performance degrades
        if recent.brier_score > baseline.brier_score * (1 + self.threshold):
            is_drifted = True
            degraded["brier_score"] = {"base": baseline.brier_score, "recent": recent.brier_score}
        
        # Log Loss increases when performance degrades
        if recent.log_loss > baseline.log_loss * (1 + self.threshold):
            is_drifted = True
            degraded["log_loss"] = {"base": baseline.log_loss, "recent": recent.log_loss}

        # Accuracy decreases when performance degrades
        if recent.accuracy < baseline.accuracy * (1 - self.threshold):
            is_drifted = True
            degraded["accuracy"] = {"base": baseline.accuracy, "recent": recent.accuracy}

        # ECE increases when calibration degrades
        if recent.overall_ece > baseline.overall_ece + (self.threshold / 2): # absolute threshold for ECE
            is_drifted = True
            degraded["ece"] = {"base": baseline.overall_ece, "recent": recent.overall_ece}

        if is_drifted:
            logger.warning("Model drift detected! Degraded metrics: %s", degraded)

        return {
            "drift_detected": is_drifted,
            "degraded_metrics": degraded,
            "threshold": self.threshold
        }
