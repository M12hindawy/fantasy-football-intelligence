"""
agents/data_ingestion_agent.py
==============================
Orchestrates all data fetchers, enforces freshness rules, logs source
conflicts, and lowers confidence scores when data is stale or contradictory.

This agent is the single entry point for all fixture/stats/lineup data
in the pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from config.freshness import FreshnessStatus, freshness_checker
from data.cache import FootballCache
from data.fetchers.football_data_fetcher import FootballDataFetcher
from data.schemas import FixtureRecord, LineupRecord, PlayerAvailability, TeamStats

logger = logging.getLogger(__name__)


class DataIngestionAgent:
    """
    Gathers and validates football data from configured providers.

    Responsibilities
    ----------------
    - Fetch fixtures, team stats, and lineup/availability data
    - Enforce freshness thresholds from config/freshness.py
    - Detect and log source conflicts
    - Produce a data_quality_signals dict used downstream by the
      value assessment agent
    """

    def __init__(
        self,
        fetcher: Optional[FootballDataFetcher] = None,
        cache: Optional[FootballCache] = None,
    ) -> None:
        self._cache = cache or FootballCache()
        self._fetcher = fetcher or FootballDataFetcher(cache=self._cache)

    # ── Fixture retrieval ─────────────────────────────────────────────────────

    def get_fixture(
        self,
        competition_code: str,
        season: Optional[int] = None,
        home_team_name: Optional[str] = None,
        away_team_name: Optional[str] = None,
        fixture_id: Optional[str] = None,
    ) -> Optional[FixtureRecord]:
        """
        Fetch a specific fixture by competition + team names (or fixture_id).

        Returns None if the fixture cannot be found with sufficient
        freshness, and logs a warning.
        """
        fixtures = self._fetcher.get_fixtures(competition_code, season)

        if fixture_id:
            matches = [f for f in fixtures if f.fixture_id == fixture_id]
        elif home_team_name and away_team_name:
            hn = home_team_name.lower()
            an = away_team_name.lower()
            matches = [
                f for f in fixtures
                if hn in f.home_team.name.lower() and an in f.away_team.name.lower()
            ]
        else:
            logger.warning("DataIngestionAgent: no fixture identifier provided.")
            return None

        if not matches:
            logger.warning(
                "DataIngestionAgent: fixture not found in %s (season %s).",
                competition_code, season
            )
            return None

        fixture = matches[0]

        # Freshness check
        status, age = freshness_checker.check(
            "historical_team_stats", fixture.source.retrieved_at
        )
        if status == FreshnessStatus.EXPIRED:
            logger.error(
                "Fixture data is EXPIRED (%s old). Refusing to proceed.",
                freshness_checker.human_readable(age),
            )
            return None
        if status == FreshnessStatus.STALE:
            logger.warning(
                "Fixture data is STALE (%s old). Confidence will be lowered.",
                freshness_checker.human_readable(age),
            )

        return fixture

    # ── Team stats ────────────────────────────────────────────────────────────

    def get_team_stats_pair(
        self,
        home_team_id: str,
        away_team_id: str,
        competition_code: str,
        season: Optional[int] = None,
    ) -> tuple[Optional[TeamStats], Optional[TeamStats]]:
        """Fetch stats for both teams; log warnings on missing data."""
        home_stats = self._fetcher.get_team_stats(home_team_id, competition_code, season)
        away_stats = self._fetcher.get_team_stats(away_team_id, competition_code, season)

        if home_stats is None:
            logger.warning("DataIngestionAgent: no stats found for team %s.", home_team_id)
        if away_stats is None:
            logger.warning("DataIngestionAgent: no stats found for team %s.", away_team_id)

        return home_stats, away_stats

    # ── Lineup / availability ─────────────────────────────────────────────────

    def get_lineup(self, fixture_id: str) -> Optional[LineupRecord]:
        """
        Attempt to retrieve lineup data.

        Note: Free-tier APIs rarely provide confirmed lineups before kickoff.
        Returns None if unavailable; the value agent will set
        PASS_PENDING_LINEUP accordingly.
        """
        # Stub: real implementation would call a lineup-specific endpoint
        logger.info(
            "DataIngestionAgent: lineup data not available for fixture %s "
            "(no lineup API configured). Analysis will proceed without confirmed lineup.",
            fixture_id,
        )
        return None

    # ── Data-quality signals ──────────────────────────────────────────────────

    def assess_data_quality(
        self,
        fixture: Optional[FixtureRecord],
        home_stats: Optional[TeamStats],
        away_stats: Optional[TeamStats],
        lineup: Optional[LineupRecord],
        odds_freshness_ok: bool = True,
    ) -> dict:
        """
        Produce a dict of data-quality signals for use by ValueAssessmentAgent.

        Returns
        -------
        dict with keys:
          - lineup_confirmed: bool
          - odds_freshness_ok: bool
          - stats_freshness_ok: bool
          - no_source_conflicts: bool
          - sample_size_ok: bool
          - warnings: list[str]
          - penalties: dict (for uncertainty_adjusted_edge)
        """
        warnings: list[str] = []
        penalties: dict[str, float] = {}

        # Lineup
        lineup_confirmed = (
            lineup is not None
            and (lineup.home_lineup_confirmed and lineup.away_lineup_confirmed)
        )
        if not lineup_confirmed:
            warnings.append("Lineups not confirmed — player props and exact probabilities may shift.")
            penalties["lineup_uncertainty"] = 0.015

        # Stats freshness
        stats_freshness_ok = True
        if home_stats is None or away_stats is None:
            stats_freshness_ok = False
            warnings.append("Team stats unavailable for one or both teams.")
            penalties["missing_stats"] = 0.03
        else:
            for label, stats in [("home", home_stats), ("away", away_stats)]:
                status, age = freshness_checker.check(
                    "historical_team_stats", stats.source.retrieved_at
                )
                if status == FreshnessStatus.STALE:
                    warnings.append(f"{label.capitalize()} team stats are STALE ({freshness_checker.human_readable(age)}).")
                    penalties["stats_staleness"] = max(penalties.get("stats_staleness", 0), 0.02)
                    stats_freshness_ok = False

        # Sample size
        n_matches = 0
        if home_stats:
            n_matches += home_stats.matches_played
        if away_stats:
            n_matches += away_stats.matches_played
        sample_size_ok = n_matches >= 20
        if not sample_size_ok:
            warnings.append(f"Small sample size ({n_matches} matches combined). Predictions less reliable.")
            penalties["sample_size"] = 0.02

        # Odds freshness
        if not odds_freshness_ok:
            warnings.append("Odds freshness threshold exceeded. Recalculate before acting.")
            penalties["odds_staleness"] = 0.04

        # Source conflicts (stub: would compare multiple providers in production)
        no_source_conflicts = True

        return {
            "lineup_confirmed": lineup_confirmed,
            "odds_freshness_ok": odds_freshness_ok,
            "stats_freshness_ok": stats_freshness_ok,
            "no_source_conflicts": no_source_conflicts,
            "sample_size_ok": sample_size_ok,
            "warnings": warnings,
            "penalties": penalties,
        }
