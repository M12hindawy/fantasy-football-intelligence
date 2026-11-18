"""
features/feature_pipeline.py
Centralized feature engineering pipeline.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional
from features.team_features import TeamFeatureExtractor
from data.quality.data_quality_engine import data_quality_engine

logger = logging.getLogger(__name__)


class FeaturePipeline:
    def __init__(self):
        self._team_extractor = TeamFeatureExtractor()

    def extract_match_features(self, home_id: str, away_id: str,
                               match_records: list[dict],
                               as_of: Optional[datetime] = None,
                               home_elo: float = 1500.0, away_elo: float = 1500.0,
                               home_position: Optional[int] = None,
                               away_position: Optional[int] = None,
                               odds_age_seconds: Optional[float] = None,
                               stats_age_seconds: Optional[float] = None,
                               lineup_confirmed: bool = False,
                               injury_source: str = "unknown",
                               provider_name: str = "unknown",
                               calibration_ece: Optional[float] = None) -> dict:
        if as_of is None:
            as_of = datetime.now(timezone.utc)
        home_feat = self._team_extractor.extract(home_id, match_records, as_of,
                                                  home_elo, home_position)
        away_feat = self._team_extractor.extract(away_id, match_records, as_of,
                                                  away_elo, away_position)
        missing = []
        if home_feat.xg_per_game is None: missing.append("home_xg")
        if away_feat.xg_per_game is None: missing.append("away_xg")
        min_n = min(home_feat.sample_size, away_feat.sample_size)
        dq = data_quality_engine.assess(
            lineup_confirmed=lineup_confirmed,
            odds_age_seconds=odds_age_seconds,
            stats_age_seconds=stats_age_seconds,
            sample_size=min_n,
            injury_source=injury_source,
            provider_name=provider_name,
            calibration_ece=calibration_ece,
            missing_fields=missing,
        )
        logger.info("Features: home=%s(n=%d,elo=%.0f) away=%s(n=%d,elo=%.0f) DQ=%.2f",
                    home_id, home_feat.sample_size, home_elo,
                    away_id, away_feat.sample_size, away_elo, dq.score)
        return {"home_features": home_feat, "away_features": away_feat,
                "dq_report": dq, "as_of": as_of}


feature_pipeline = FeaturePipeline()
