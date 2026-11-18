"""
tests/test_monitoring.py
========================
Tests for the monitoring and drift detection modules.
"""
from __future__ import annotations
import pytest
from monitoring.model_monitor import ModelMonitor, EvaluationMetrics
from monitoring.drift_detector import DriftDetector
from monitoring.alert_manager import AlertManager

class TestModelMonitor:
    def test_perfect_predictions(self):
        monitor = ModelMonitor()
        preds = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        actuals = [0, 1, 2]
        metrics = monitor.evaluate(preds, actuals)
        
        assert metrics is not None
        assert metrics.accuracy == 1.0
        assert metrics.brier_score == 0.0
        assert metrics.log_loss < 1e-5
        assert metrics.overall_ece == 0.0

    def test_imperfect_predictions(self):
        monitor = ModelMonitor()
        preds = [(0.7, 0.2, 0.1), (0.1, 0.6, 0.3), (0.2, 0.2, 0.6)]
        actuals = [0, 2, 2] # 2nd prediction was wrong
        metrics = monitor.evaluate(preds, actuals)
        
        assert metrics is not None
        assert metrics.accuracy == pytest.approx(0.666, abs=0.01)
        assert metrics.brier_score > 0.0
        assert metrics.log_loss > 0.0
        assert metrics.overall_ece > 0.0

    def test_empty_input(self):
        monitor = ModelMonitor()
        metrics = monitor.evaluate([], [])
        assert metrics is None

class TestDriftDetector:
    def test_no_drift(self):
        detector = DriftDetector(threshold=0.05)
        base = EvaluationMetrics(0.5, 0.6, 0.6, 100, 0.05, 0.05, 0.05, 0.05)
        recent = EvaluationMetrics(0.51, 0.61, 0.59, 50, 0.06, 0.05, 0.05, 0.053)
        
        result = detector.check_drift(base, recent)
        assert not result["drift_detected"]
        assert len(result["degraded_metrics"]) == 0

    def test_drift_detected_on_brier(self):
        detector = DriftDetector(threshold=0.05)
        base = EvaluationMetrics(0.5, 0.6, 0.6, 100, 0.05, 0.05, 0.05, 0.05)
        recent = EvaluationMetrics(0.6, 0.61, 0.59, 50, 0.06, 0.05, 0.05, 0.053)
        
        result = detector.check_drift(base, recent)
        assert result["drift_detected"]
        assert "brier_score" in result["degraded_metrics"]

    def test_drift_detected_on_accuracy(self):
        detector = DriftDetector(threshold=0.05)
        base = EvaluationMetrics(0.5, 0.6, 0.6, 100, 0.05, 0.05, 0.05, 0.05)
        recent = EvaluationMetrics(0.5, 0.6, 0.5, 50, 0.05, 0.05, 0.05, 0.05)
        
        result = detector.check_drift(base, recent)
        assert result["drift_detected"]
        assert "accuracy" in result["degraded_metrics"]

class TestAlertManager:
    def test_trigger_drift_alert(self):
        manager = AlertManager()
        manager.trigger_drift_alert("v2.0", {"brier_score": {"base": 0.5, "recent": 0.6}})
        alerts = manager.get_recent_alerts()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "MODEL_DRIFT"
        assert alerts[0]["model_version"] == "v2.0"

    def test_trigger_dq_alert(self):
        manager = AlertManager()
        manager.trigger_dq_alert("FIX-123", 0.35, "D")
        alerts = manager.get_recent_alerts()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "DATA_QUALITY_DROP"
        assert alerts[0]["fixture_id"] == "FIX-123"
        assert alerts[0]["grade"] == "D"

    def test_clear_alerts(self):
        manager = AlertManager()
        manager.trigger_dq_alert("FIX-123", 0.35, "D")
        manager.clear_alerts()
        assert len(manager.get_recent_alerts()) == 0
