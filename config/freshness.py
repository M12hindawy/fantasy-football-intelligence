"""
config/freshness.py
===================
Staleness thresholds and a FreshnessChecker that every agent must call
before trusting cached data.

Thresholds are drawn from the master spec (Section 5) and are expressed
in seconds.  The checker returns a FreshnessStatus enum that agents use
to decide whether to proceed, warn, or refuse to produce output.
"""

from __future__ import annotations

import enum
from datetime import datetime, timedelta, timezone
from typing import Optional


class FreshnessStatus(str, enum.Enum):
    FRESH = "fresh"
    WARNING = "warning"     # Within 2× the threshold — proceed with caution
    STALE = "stale"         # Beyond 2× the threshold — lower confidence
    EXPIRED = "expired"     # Beyond 5× the threshold — refuse or warn strongly


# ── Threshold registry (seconds) ─────────────────────────────────────────────

THRESHOLDS: dict[str, int] = {
    # Live / in-play data
    "live_score": 45,
    "live_cards": 45,
    "live_event_state": 45,

    # Near-kickoff lineup data
    "starting_lineup": 900,          # 15 minutes
    "player_availability": 900,

    # Pre-match odds (price-sensitive)
    "prematch_odds": 300,            # 5 minutes
    "prematch_odds_comparison": 60,  # 1 minute when actively comparing

    # Team news and contextual data
    "team_news": 21_600,             # 6 hours
    "injury_report": 21_600,

    # Historical stats (season-scoped; always label with season)
    "historical_team_stats": 86_400,  # 24 hours
    "historical_player_stats": 86_400,
    "competition_standings": 3_600,   # 1 hour
}


class FreshnessChecker:
    """
    Checks whether a data record is within acceptable freshness bounds.

    Usage::

        checker = FreshnessChecker()
        status, age = checker.check("prematch_odds", retrieved_at=ts)
    """

    def __init__(self, custom_thresholds: Optional[dict[str, int]] = None) -> None:
        self._thresholds: dict[str, int] = {**THRESHOLDS}
        if custom_thresholds:
            self._thresholds.update(custom_thresholds)

    def threshold(self, data_type: str) -> int:
        """Return the threshold in seconds for *data_type*, or a safe default."""
        return self._thresholds.get(data_type, 3_600)

    def check(
        self,
        data_type: str,
        retrieved_at: datetime,
        as_of: Optional[datetime] = None,
    ) -> tuple[FreshnessStatus, float]:
        """
        Return (FreshnessStatus, age_in_seconds).

        Parameters
        ----------
        data_type:    key in THRESHOLDS
        retrieved_at: when the data was last fetched (timezone-aware preferred)
        as_of:        reference time (default: now UTC)
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)

        # Normalise to UTC-aware
        if retrieved_at.tzinfo is None:
            retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        age_s = (as_of - retrieved_at).total_seconds()
        thr = self.threshold(data_type)

        if age_s <= thr:
            return FreshnessStatus.FRESH, age_s
        elif age_s <= thr * 2:
            return FreshnessStatus.WARNING, age_s
        elif age_s <= thr * 5:
            return FreshnessStatus.STALE, age_s
        else:
            return FreshnessStatus.EXPIRED, age_s

    def human_readable(self, age_s: float) -> str:
        """Convert age in seconds to a human-readable string."""
        delta = timedelta(seconds=age_s)
        total_minutes = int(delta.total_seconds() // 60)
        if total_minutes < 60:
            return f"{total_minutes}m ago"
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours}h {minutes}m ago"


# Module-level singleton
freshness_checker = FreshnessChecker()
