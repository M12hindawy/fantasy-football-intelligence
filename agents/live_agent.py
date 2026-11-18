"""
agents/live_agent.py
====================
Connects LIVE to FPL API to get real-time match data.
Supports lookup by full team name (e.g. "Aston Villa") not just short codes.
"""
import urllib.request
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

@dataclass
class LiveMatchData:
    is_live: bool = False
    is_finished: bool = False
    minute: int = 0
    home_score: int = 0
    away_score: int = 0
    home_team_name: str = ""
    away_team_name: str = ""
    live_stats: List[Dict] = field(default_factory=list)

class LiveMatchAgent:
    def __init__(self):
        self.teams_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        self.fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"

    def _load_teams(self):
        req = urllib.request.Request(self.teams_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read())["teams"]

    def _find_team_id(self, teams_data, name: str):
        """
        Find team ID from FPL teams list.
        Tries: exact name match, short_name match, substring match (case-insensitive).
        """
        name_lower = name.strip().lower()
        for t in teams_data:
            if t["name"].lower() == name_lower:
                return t["id"], t["name"]
        for t in teams_data:
            if t["short_name"].lower() == name_lower:
                return t["id"], t["name"]
        for t in teams_data:
            if name_lower in t["name"].lower() or t["name"].lower() in name_lower:
                return t["id"], t["name"]
        # Partial word match
        for t in teams_data:
            parts = name_lower.split()
            if any(p in t["name"].lower() for p in parts if len(p) > 3):
                return t["id"], t["name"]
        return None, None

    def fetch_live_data(self, home_name: str, away_name: str) -> Optional[LiveMatchData]:
        """
        Fetch live match data for any two teams.
        Accepts full names (e.g. "Aston Villa"), short codes (e.g. "AVL"), or any fuzzy match.
        """
        try:
            teams_data = self._load_teams()

            home_id, home_fpl_name = self._find_team_id(teams_data, home_name)
            away_id, away_fpl_name = self._find_team_id(teams_data, away_name)

            if not home_id or not away_id:
                logger.debug(f"Teams not found in FPL: {home_name}, {away_name}")
                return None

            req_f = urllib.request.Request(self.fixtures_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_f, timeout=8) as r:
                fixtures = json.loads(r.read())

            for f in fixtures:
                if f["team_h"] == home_id and f["team_a"] == away_id:
                    return LiveMatchData(
                        is_live=bool(f.get("started")) and not bool(f.get("finished")),
                        is_finished=bool(f.get("finished")),
                        minute=f.get("minutes", 0),
                        home_score=f.get("team_h_score") or 0,
                        away_score=f.get("team_a_score") or 0,
                        home_team_name=home_fpl_name,
                        away_team_name=away_fpl_name,
                        live_stats=f.get("stats", [])
                    )
            return None
        except Exception as e:
            logger.error(f"Live API error: {e}")
            return None

live_agent = LiveMatchAgent()
