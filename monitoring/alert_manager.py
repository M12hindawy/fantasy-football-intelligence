"""
monitoring/alert_manager.py
===========================
Manages notifications for model drift and data quality drops.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self):
        self.alerts = []

    def trigger_drift_alert(self, model_version: str, drift_info: dict) -> None:
        alert = {
            "type": "MODEL_DRIFT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version,
            "details": drift_info
        }
        self.alerts.append(alert)
        logger.error("ALERT TRIGGERED: Model Drift - %s", alert)

    def trigger_dq_alert(self, fixture_id: str, dq_score: float, grade: str) -> None:
        alert = {
            "type": "DATA_QUALITY_DROP",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fixture_id": fixture_id,
            "score": dq_score,
            "grade": grade
        }
        self.alerts.append(alert)
        logger.warning("ALERT TRIGGERED: Data Quality - %s", alert)
    
    def get_recent_alerts(self, n: int = 10) -> list[dict]:
        return self.alerts[-n:]
    
    def clear_alerts(self) -> None:
        self.alerts.clear()

alert_manager = AlertManager()
