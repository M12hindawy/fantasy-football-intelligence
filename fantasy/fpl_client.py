"""
fantasy/fpl_client.py
=====================
Lightweight FPL (Fantasy Premier League) API client.
Fetches live gameweek, player, and fixture data from the official FPL API.
No auth required for public endpoints.
"""
from __future__ import annotations
import logging
import time
from typing import Optional
import urllib.request
import json

logger = logging.getLogger(__name__)

FPL_BASE = "https://fantasy.premierleague.com/api"
_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 300  # 5 minutes


def _fetch(path: str, ttl: int = _CACHE_TTL) -> dict:
    url = f"{FPL_BASE}{path}"
    now = time.time()
    if url in _CACHE:
        ts, data = _CACHE[url]
        if now - ts < ttl:
            return data
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        _CACHE[url] = (now, data)
        return data
    except Exception as exc:
        logger.warning("FPL API error for %s: %s", url, exc)
        return {}


class FPLClient:
    """Thin wrapper around the public FPL REST API."""

    def get_bootstrap(self) -> dict:
        """Main bootstrap: players, teams, gameweeks, element_types."""
        return _fetch("/bootstrap-static/")

    def get_player_summary(self, player_fpl_id: int) -> dict:
        """Per-player history and fixtures."""
        return _fetch(f"/element-summary/{player_fpl_id}/")

    def get_fixtures(self) -> list[dict]:
        """All fixtures for current season."""
        data = _fetch("/fixtures/")
        return data if isinstance(data, list) else []

    def get_gameweek_fixtures(self, gw: int) -> list[dict]:
        data = _fetch(f"/fixtures/?event={gw}", ttl=60)
        return data if isinstance(data, list) else []

    def get_live_gameweek(self, gw: int) -> dict:
        """Live points and bonus data for a gameweek."""
        return _fetch(f"/event/{gw}/live/", ttl=60)

    def get_my_team(self, team_id: int, gw: int) -> dict:
        """Requires auth — returns empty if not authenticated."""
        logger.info("get_my_team requires auth — not implemented in public client.")
        return {}

    def get_players(self) -> list[dict]:
        """Returns player list from bootstrap."""
        b = self.get_bootstrap()
        return b.get("elements", [])

    def get_teams(self) -> list[dict]:
        b = self.get_bootstrap()
        return b.get("teams", [])

    def get_current_gameweek(self) -> Optional[int]:
        b = self.get_bootstrap()
        for gw in b.get("events", []):
            if gw.get("is_current"):
                return gw["id"]
        return None

    def get_next_gameweek(self) -> Optional[int]:
        b = self.get_bootstrap()
        for gw in b.get("events", []):
            if gw.get("is_next"):
                return gw["id"]
        return None

fpl_client = FPLClient()
