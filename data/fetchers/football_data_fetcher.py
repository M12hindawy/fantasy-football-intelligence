"""
data/fetchers/football_data_fetcher.py
======================================
Data fetcher for the Football-Data.org free API (v4).

Endpoints used:
  GET /competitions/{code}/matches
  GET /competitions/{code}/standings
  GET /teams/{id}/matches

Authentication: X-Auth-Token header (free tier).

All responses are cached using data/cache.py with appropriate TTL.
Source attribution is added to every record.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from config.freshness import freshness_checker
from config.settings import settings
from data.cache import FootballCache
from data.schemas import (
    FixtureRecord,
    FreshnessStatusEnum,
    MatchStatus,
    SourceRecord,
    TeamRecord,
    TeamStats,
)


BASE_URL = "https://api.football-data.org/v4"
PROVIDER = "football-data.org"


def _status_map(raw: str) -> MatchStatus:
    mapping = {
        "SCHEDULED": MatchStatus.PRE_MATCH,
        "TIMED": MatchStatus.PRE_MATCH,
        "IN_PLAY": MatchStatus.LIVE,
        "PAUSED": MatchStatus.LIVE,
        "FINISHED": MatchStatus.FINISHED,
        "POSTPONED": MatchStatus.POSTPONED,
        "CANCELLED": MatchStatus.CANCELLED,
        "SUSPENDED": MatchStatus.CANCELLED,
    }
    return mapping.get(raw.upper(), MatchStatus.UNKNOWN)


class FootballDataFetcher:
    """
    Fetches fixtures, standings, and team stats from Football-Data.org.

    Parameters
    ----------
    api_key : Football-Data.org X-Auth-Token (reads from settings if empty)
    cache   : FootballCache instance (created if None)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[FootballCache] = None,
    ) -> None:
        self._api_key = api_key or settings.football_data_api_key
        self._cache = cache or FootballCache(
            db_path=settings.cache_db_path,
            enabled=settings.cache_enabled,
        )
        self._session = requests.Session()
        self._session.headers.update({
            "X-Auth-Token": self._api_key,
            "Accept": "application/json",
        })

    # ── Internal HTTP helper ──────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """
        HTTP GET with caching, retries, and error handling.

        Raises requests.HTTPError on non-2xx responses.
        """
        url = f"{BASE_URL}{path}"
        cache_key = f"fd:{url}:{str(params)}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.payload

        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            raise RuntimeError(f"Request timed out: {url}")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                raise RuntimeError("Football-Data.org rate limit exceeded.  Wait 1 minute.")
            raise

        # Determine cache data_type from path
        if "matches" in path:
            data_type = "historical_team_stats"
        elif "standings" in path:
            data_type = "competition_standings"
        else:
            data_type = "historical_team_stats"

        self._cache.set(cache_key, data_type, data)
        return data

    # ── Source record helper ──────────────────────────────────────────────────

    def _make_source(self, data_type: str = "historical_team_stats") -> SourceRecord:
        now = datetime.now(timezone.utc)
        status, _ = freshness_checker.check(data_type, now)
        return SourceRecord(
            provider=PROVIDER,
            source_url=BASE_URL,
            retrieved_at=now,
            freshness_status=FreshnessStatusEnum(status.value),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_fixtures(
        self,
        competition_code: str,
        season: Optional[int] = None,
        status: Optional[str] = None,
    ) -> list[FixtureRecord]:
        """
        Fetch upcoming or historical fixtures for a competition.

        Parameters
        ----------
        competition_code : e.g. "PL" (Premier League), "PD" (La Liga)
        season           : year (e.g. 2024 for 2024/25 season)
        status           : "SCHEDULED", "FINISHED", etc.

        Returns
        -------
        list[FixtureRecord]
        """
        params: dict[str, Any] = {}
        if season:
            params["season"] = season
        if status:
            params["status"] = status

        data = self._get(f"/competitions/{competition_code}/matches", params)
        source = self._make_source("historical_team_stats")
        fixtures = []

        for match in data.get("matches", []):
            try:
                kickoff_str = match.get("utcDate", "")
                kickoff = (
                    datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
                    if kickoff_str
                    else datetime.now(timezone.utc)
                )
                home = match.get("homeTeam", {})
                away = match.get("awayTeam", {})
                score = match.get("score", {})
                full_time = score.get("fullTime", {})

                fixtures.append(FixtureRecord(
                    fixture_id=str(match.get("id", "")),
                    home_team=TeamRecord(
                        id=str(home.get("id", "")),
                        name=home.get("name", "Unknown"),
                        short_name=home.get("shortName"),
                    ),
                    away_team=TeamRecord(
                        id=str(away.get("id", "")),
                        name=away.get("name", "Unknown"),
                        short_name=away.get("shortName"),
                    ),
                    competition=data.get("competition", {}).get("name", competition_code),
                    season=str(season or ""),
                    kickoff=kickoff,
                    status=_status_map(match.get("status", "UNKNOWN")),
                    venue=match.get("venue"),
                    referee=match.get("referees", [{}])[0].get("name") if match.get("referees") else None,
                    source=source,
                    home_score=full_time.get("home"),
                    away_score=full_time.get("away"),
                    minute=match.get("minute"),
                ))
            except Exception as exc:
                # Log and skip malformed records rather than crashing
                print(f"[FootballDataFetcher] Skipped malformed match record: {exc}")

        return fixtures

    def get_team_stats(
        self,
        team_id: str,
        competition_code: str,
        season: Optional[int] = None,
        last_n: int = 10,
    ) -> Optional[TeamStats]:
        """
        Compute basic team stats from recent match results.

        Returns None if insufficient data is available.
        """
        params: dict[str, Any] = {"limit": max(last_n, 10)}
        if season:
            params["season"] = season
        params["status"] = "FINISHED"

        data = self._get(f"/teams/{team_id}/matches", params)
        matches = data.get("matches", [])
        if not matches:
            return None

        # Parse team metadata from first match
        first = matches[0]
        home_team = first.get("homeTeam", {})
        away_team = first.get("awayTeam", {})
        is_home_in_first = str(home_team.get("id", "")) == team_id
        team_data = home_team if is_home_in_first else away_team

        wins = draws = losses = goals_for = goals_against = 0
        form: list[str] = []

        for m in matches[-last_n:]:
            ht = m.get("homeTeam", {})
            at = m.get("awayTeam", {})
            score = m.get("score", {}).get("fullTime", {})
            h_goals = score.get("home")
            a_goals = score.get("away")
            if h_goals is None or a_goals is None:
                continue
            is_home = str(ht.get("id", "")) == team_id
            gf = h_goals if is_home else a_goals
            ga = a_goals if is_home else h_goals
            goals_for += gf
            goals_against += ga
            if gf > ga:
                wins += 1
                form.append("W")
            elif gf == ga:
                draws += 1
                form.append("D")
            else:
                losses += 1
                form.append("L")

        n_matches = wins + draws + losses
        source = self._make_source("historical_team_stats")

        return TeamStats(
            team=TeamRecord(
                id=team_id,
                name=team_data.get("name", "Unknown"),
                short_name=team_data.get("shortName"),
            ),
            season=str(season or ""),
            competition=competition_code,
            matches_played=n_matches,
            wins=wins,
            draws=draws,
            losses=losses,
            goals_scored=goals_for,
            goals_conceded=goals_against,
            form_results=list(reversed(form)),  # most recent first
            source=source,
        )

    def get_standings(self, competition_code: str, season: Optional[int] = None) -> dict:
        """
        Fetch competition standings.

        Returns raw standings dict (provider-specific structure preserved
        for flexibility; caller parses as needed).
        """
        params = {}
        if season:
            params["season"] = season
        return self._get(f"/competitions/{competition_code}/standings", params)
