"""
data/providers/football_data_org.py
===================================
Implementation of FootballDataProvider for Football-Data.org.
Wraps the existing FootballDataFetcher for backward compatibility.
"""
from datetime import date, datetime
from typing import Optional

from data.providers.base_provider import FootballDataProvider
from data.fetchers.football_data_fetcher import FootballDataFetcher

class FootballDataOrgProvider(FootballDataProvider):
    def __init__(self, api_key: Optional[str] = None):
        self._fetcher = FootballDataFetcher(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "football-data.org"
    
    @property
    def supports_player_stats(self) -> bool: return False
    
    @property
    def supports_lineups(self) -> bool: return False
    
    @property
    def supports_injuries(self) -> bool: return False
    
    @property
    def rate_limit_per_minute(self) -> int: return 10

    def get_competitions(self) -> list[dict]:
        return []

    def get_seasons(self, competition_code: str) -> list[dict]:
        return []

    def get_matches(self, competition_code: str, season: Optional[int] = None, date_from: Optional[date] = None, date_to: Optional[date] = None, status: Optional[str] = None) -> list[dict]:
        return []

    def get_match_details(self, match_id: str) -> Optional[dict]:
        return None

    def get_team(self, team_id: str) -> Optional[dict]:
        return None
    
    def get_team_stats(self, team_id: str, competition_code: str, season: Optional[int] = None, last_n: int = 10) -> Optional[dict]:
        stats = self._fetcher.get_team_stats(team_id, competition_code, season, last_n)
        return stats.model_dump() if stats else None

    def get_player(self, player_id: str) -> Optional[dict]:
        return None

    def get_player_stats(self, player_id: str, season: Optional[int] = None, competition_code: Optional[str] = None) -> Optional[dict]:
        return None

    def get_player_match_stats(self, player_id: str, match_id: str) -> Optional[dict]:
        return None

    def get_lineups(self, match_id: str) -> Optional[dict]:
        return None

    def get_injuries(self, team_id: str) -> list[dict]:
        return []

    def get_suspensions(self, team_id: str) -> list[dict]:
        return []
    
    def get_standings(self, competition_code: str, season: Optional[int] = None) -> list[dict]:
        return self._fetcher.get_standings(competition_code, season)
        
    def get_fixtures(self, competition_code: str, season: Optional[int] = None, status: Optional[str] = None) -> list[dict]:
        fixs = self._fetcher.get_fixtures(competition_code, season, status)
        return [f.model_dump() for f in fixs]
