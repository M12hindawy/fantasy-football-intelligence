"""
data/competition_registry.py
============================
Registry for all supported competitions and their provider codes.
"""
import logging
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CompetitionDef:
    name: str
    country: str
    fd_code: str  # football-data.co.uk code
    tier: int

class CompetitionRegistry:
    def __init__(self):
        self.competitions = {
            "E0": CompetitionDef("Premier League", "England", "E0", 1),
            "SP1": CompetitionDef("La Liga", "Spain", "SP1", 1),
            "I1": CompetitionDef("Serie A", "Italy", "I1", 1),
            "D1": CompetitionDef("Bundesliga", "Germany", "D1", 1),
            "F1": CompetitionDef("Ligue 1", "France", "F1", 1),
            "N1": CompetitionDef("Eredivisie", "Netherlands", "N1", 1),
            "P1": CompetitionDef("Primeira Liga", "Portugal", "P1", 1),
            "B1": CompetitionDef("Belgian Pro League", "Belgium", "B1", 1),
            "SC0": CompetitionDef("Scottish Premiership", "Scotland", "SC0", 1),
            "T1": CompetitionDef("Turkish Super Lig", "Turkey", "T1", 1),
        }
        
    def get_seasons_url(self, competition_code: str, season_code: str) -> str:
        """Returns the football-data URL for a specific comp and season (e.g. 2324)."""
        return f"https://www.football-data.co.uk/mmz4281/{season_code}/{competition_code}.csv"
        
    def discover_available_seasons(self, competition_code: str) -> list[str]:
        """Probes recent seasons to see which ones are available."""
        available = []
        # Probe last 5 seasons
        seasons_to_check = ["2021", "2122", "2223", "2324", "2425", "2526", "2627"]
        for s in seasons_to_check:
            url = self.get_seasons_url(competition_code, s)
            try:
                req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=3) as r:
                    if r.status == 200:
                        available.append(s)
            except Exception:
                pass
        return available
        
    def get_all_competitions(self) -> list[dict]:
        return [{"code": k, "name": v.name, "country": v.country} for k, v in self.competitions.items()]

competition_registry = CompetitionRegistry()
