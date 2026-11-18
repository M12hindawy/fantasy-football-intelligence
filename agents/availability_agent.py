"""
agents/availability_agent.py
============================
Real-time availability agent using FPL API.
Fetches live injury/suspension data for all PL players.
"""
from __future__ import annotations
import logging
import urllib.request
import json
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

STATUS_MAP = {
    "a": ("AVAILABLE", 1.0),
    "d": ("DOUBTFUL", 0.5),
    "i": ("INJURED", 0.0),
    "s": ("SUSPENDED", 0.0),
    "u": ("UNAVAILABLE", 0.0),
    "n": ("NOT_AVAILABLE", 0.0),
}

@dataclass
class AvailabilityResult:
    player_id: str
    name: str
    status: str
    probability_of_playing: float
    chance_next_round: Optional[int]
    news: str
    reason: Optional[str] = None

class AvailabilityAgent:
    def __init__(self):
        self._cache: Optional[list] = None

    def _load_fpl_players(self) -> list:
        if self._cache is not None:
            return self._cache
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        self._cache = data.get("elements", [])
        return self._cache

    def get_team_availability(self, fpl_team_id: int) -> list[AvailabilityResult]:
        """Get availability for all players in a FPL team (by team ID)."""
        players = self._load_fpl_players()
        results = []
        for p in players:
            if p.get("team") != fpl_team_id:
                continue
            raw_status = p.get("status", "a")
            status_str, default_prob = STATUS_MAP.get(raw_status, ("UNKNOWN", 0.5))
            chance = p.get("chance_of_playing_next_round")
            prob = (chance / 100.0) if chance is not None else default_prob
            results.append(AvailabilityResult(
                player_id=str(p["id"]),
                name=p.get("web_name", "Unknown"),
                status=status_str,
                probability_of_playing=prob,
                chance_next_round=chance,
                news=p.get("news", ""),
                reason=p.get("news", "") or None
            ))
        return results

    def get_injury_alerts(self, fpl_team_id: int) -> list[AvailabilityResult]:
        """Only return non-available players (injured, doubtful, suspended)."""
        all_players = self.get_team_availability(fpl_team_id)
        return [p for p in all_players if p.status != "AVAILABLE"]

    def get_all_injuries(self) -> list[AvailabilityResult]:
        """Return all non-available players across ALL teams."""
        players = self._load_fpl_players()
        results = []
        for p in players:
            raw_status = p.get("status", "a")
            if raw_status == "a":
                continue
            status_str, default_prob = STATUS_MAP.get(raw_status, ("UNKNOWN", 0.5))
            chance = p.get("chance_of_playing_next_round")
            prob = (chance / 100.0) if chance is not None else default_prob
            results.append(AvailabilityResult(
                player_id=str(p["id"]),
                name=p.get("web_name", "Unknown"),
                status=status_str,
                probability_of_playing=prob,
                chance_next_round=chance,
                news=p.get("news", ""),
            ))
        return results

    def invalidate_cache(self):
        self._cache = None

    def assess(self, player_id: str, injuries: list[dict], suspensions: list[dict]) -> AvailabilityResult:
        """Backwards-compatible method for local injury/suspension dicts."""
        for s in suspensions:
            if s.get("player_id") == player_id:
                return AvailabilityResult(player_id, player_id, "SUSPENDED", 0.0, None, s.get("reason", ""))
        for i in injuries:
            if i.get("player_id") == player_id:
                status = i.get("status", "unknown").upper()
                prob = i.get("probability_of_playing")
                if status == "INJURED":
                    return AvailabilityResult(player_id, player_id, "INJURED", prob if prob is not None else 0.0, None, i.get("injury_type", ""))
                elif status in ("DOUBTFUL", "UNKNOWN"):
                    return AvailabilityResult(player_id, player_id, "DOUBTFUL", prob if prob is not None else 0.5, None, i.get("injury_type", ""))
        return AvailabilityResult(player_id, player_id, "AVAILABLE", 1.0, None, "")

availability_agent = AvailabilityAgent()
