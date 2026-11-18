"""
data/providers/fpl_provider.py
==============================
FPL implementation of the BaseFootballProvider interface.
"""
import urllib.request
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FPLProvider:
    """Provider for Fantasy Premier League official API."""
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        
    def _fetch(self, endpoint: str) -> dict:
        url = f"{self.base_url}/{endpoint}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
            
    def get_bootstrap_static(self) -> dict:
        return self._fetch("bootstrap-static/")
        
    def get_teams(self) -> list[dict]:
        data = self.get_bootstrap_static()
        return data.get("teams", [])
        
    def get_players(self) -> list[dict]:
        data = self.get_bootstrap_static()
        return data.get("elements", [])
        
    def get_fixtures(self, event: Optional[int] = None) -> list[dict]:
        endpoint = f"fixtures/?event={event}" if event else "fixtures/"
        return self._fetch(endpoint)
        
    def get_live_fixtures(self) -> list[dict]:
        fixtures = self.get_fixtures()
        return [f for f in fixtures if f.get("started") and not f.get("finished")]
        
    def get_team_by_name(self, name: str) -> Optional[dict]:
        teams = self.get_teams()
        name_lower = name.lower()
        for t in teams:
            if t["name"].lower() == name_lower or t["short_name"].lower() == name_lower:
                return t
        for t in teams:
            if name_lower in t["name"].lower():
                return t
        return None

fpl_provider = FPLProvider()
