"""
data/quality/data_quality_engine.py
Data Quality Engine - produces DQ score 0.0-1.0 for any dataset.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from config.settings import settings

logger = logging.getLogger(__name__)

_WEIGHTS = {
    "lineup_confirmed":     0.20,
    "odds_freshness":       0.15,
    "stats_freshness":      0.15,
    "sample_size":          0.15,
    "no_source_conflict":   0.10,
    "injury_reliability":   0.10,
    "provider_reliability": 0.10,
    "calibration_ok":       0.05,
}

@dataclass
class DataQualityReport:
    score: float = 0.0
    completeness: float = 0.0
    freshness: float = 0.0
    source_reliability: float = 0.0
    lineup_confirmed: bool = False
    odds_freshness_ok: bool = False
    stats_freshness_ok: bool = False
    sample_size_ok: bool = False
    no_source_conflicts: bool = False
    injury_reliability_ok: bool = False
    provider_reliability_ok: bool = True
    calibration_ok: bool = False
    sample_size: int = 0
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    generated_at: Optional[datetime] = None

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4), "grade": self.grade,
            "completeness": round(self.completeness, 4),
            "freshness": round(self.freshness, 4),
            "source_reliability": round(self.source_reliability, 4),
            "lineup_confirmed": self.lineup_confirmed,
            "sample_size": self.sample_size,
            "warnings": self.warnings, "errors": self.errors,
        }

    @property
    def passes_threshold(self) -> bool:
        return self.score >= settings.min_data_quality

    @property
    def grade(self) -> str:
        if self.score >= 0.85: return "A"
        if self.score >= 0.70: return "B"
        if self.score >= 0.55: return "C"
        if self.score >= 0.40: return "D"
        return "F"


class DataQualityEngine:
    def assess(self, *, lineup_confirmed=False, odds_age_seconds=None,
               stats_age_seconds=None, sample_size=0, has_source_conflict=False,
               injury_source="unknown", provider_name="unknown",
               calibration_ece=None, missing_fields=None, extra_warnings=None):
        report = DataQualityReport(generated_at=datetime.now(timezone.utc))
        report.sample_size = sample_size
        warnings = list(extra_warnings or [])
        errors = []

        report.lineup_confirmed = lineup_confirmed
        if not lineup_confirmed:
            warnings.append("Lineup not confirmed - prediction variance increases.")

        odds_max = settings.prematch_odds_max_age_s
        if odds_age_seconds is None:
            report.odds_freshness_ok = False
            warnings.append("Odds timestamp unknown.")
        elif odds_age_seconds > odds_max:
            report.odds_freshness_ok = False
            warnings.append(f"Odds are {odds_age_seconds/60:.1f} min old (max {odds_max/60:.0f} min).")
        else:
            report.odds_freshness_ok = True

        stats_max = settings.team_news_max_age_s
        if stats_age_seconds is None:
            report.stats_freshness_ok = False
        elif stats_age_seconds > stats_max:
            report.stats_freshness_ok = False
            warnings.append(f"Team stats {stats_age_seconds/3600:.1f}h old.")
        else:
            report.stats_freshness_ok = True

        min_sz = settings.min_sample_size
        report.sample_size_ok = sample_size >= min_sz
        if not report.sample_size_ok:
            warnings.append(f"Only {sample_size} matches available (min {min_sz}).")

        report.no_source_conflicts = not has_source_conflict
        if has_source_conflict:
            warnings.append("DATA_CONFLICT: Sources disagree.")

        report.injury_reliability_ok = injury_source in {"official", "club"}
        if injury_source == "unknown":
            warnings.append("Injury data from unknown source.")

        known_reliable = {"football-data.org", "api-football", "sofascore", "opta"}
        report.provider_reliability_ok = provider_name.lower() in known_reliable

        if calibration_ece is None:
            report.calibration_ok = False
            warnings.append("Model not calibrated.")
        elif calibration_ece < 0.07:
            report.calibration_ok = True
        else:
            report.calibration_ok = False
            warnings.append(f"Calibration ECE={calibration_ece:.3f} > 0.07.")

        missing = missing_fields or []
        missing_penalty = min(0.30, len(missing) * 0.05)
        if missing:
            errors.append(f"Missing fields: {', '.join(missing)}")

        comp = {
            "lineup_confirmed":     1.0 if report.lineup_confirmed else 0.0,
            "odds_freshness":       1.0 if report.odds_freshness_ok else 0.3,
            "stats_freshness":      1.0 if report.stats_freshness_ok else 0.3,
            "sample_size":          min(1.0, sample_size / max(min_sz, 1)),
            "no_source_conflict":   1.0 if report.no_source_conflicts else 0.5,
            "injury_reliability":   1.0 if report.injury_reliability_ok else 0.4,
            "provider_reliability": 1.0 if report.provider_reliability_ok else 0.6,
            "calibration_ok":       1.0 if report.calibration_ok else 0.5,
        }
        raw = sum(_WEIGHTS[k] * comp[k] for k in _WEIGHTS)
        report.score = max(0.0, min(1.0, raw - missing_penalty))
        report.freshness = comp["odds_freshness"]*0.5 + comp["stats_freshness"]*0.5
        report.completeness = max(0.0, 1.0 - missing_penalty - (0.1 if not lineup_confirmed else 0.0))
        report.source_reliability = comp["no_source_conflict"]*0.5 + comp["provider_reliability"]*0.5
        report.warnings = warnings
        report.errors = errors
        return report

data_quality_engine = DataQualityEngine()
