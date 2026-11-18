"""
data/providers/base_provider.py
================================
Abstract base class for all football data providers.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Optional


class FootballDataProvider(ABC):
    """Abstract interface — all concrete providers must implement this."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def supports_player_stats(self) -> bool: ...

    @property
    @abstractmethod
    def supports_lineups(self) -> bool: ...

    @property
    @abstractmethod
    def supports_injuries(self) -> bool: ...

    @property
    @abstractmethod
    def rate_limit_per_minute(self) -> int: ...

    @abstractmethod
    def get_competitions(self) -> list[dict]: ...

    @abstractmethod
    def get_seasons(self, competition_code: str) -> list[dict]: ...

    @abstractmethod
    def get_matches(
        self,
        competition_code: str,
        season: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        status: Optional[str] = None,
    ) -> list[dict]: ...

    @abstractmethod
    def get_match_details(self, match_id: str) -> Optional[dict]: ...

    @abstractmethod
    def get_team(self, team_id: str) -> Optional[dict]: ...

    @abstractmethod
    def get_team_stats(
        self, team_id: str, competition_code: str,
        season: Optional[int] = None, last_n: int = 10,
    ) -> Optional[dict]: ...

    @abstractmethod
    def get_player(self, player_id: str) -> Optional[dict]: ...

    @abstractmethod
    def get_player_stats(
        self, player_id: str,
        season: Optional[int] = None,
        competition_code: Optional[str] = None,
    ) -> Optional[dict]: ...

    @abstractmethod
    def get_player_match_stats(self, player_id: str, match_id: str) -> Optional[dict]: ...

    @abstractmethod
    def get_lineups(self, match_id: str) -> Optional[dict]: ...

    @abstractmethod
    def get_injuries(self, team_id: str) -> list[dict]: ...

    @abstractmethod
    def get_suspensions(self, team_id: str) -> list[dict]: ...

    @abstractmethod
    def get_standings(self, competition_code: str, season: Optional[int] = None) -> list[dict]: ...

    @abstractmethod
    def get_fixtures(
        self, competition_code: str,
        season: Optional[int] = None,
        status: Optional[str] = None,
    ) -> list[dict]: ...

    def get_transfers(self, team_id: str, season: Optional[int] = None) -> list[dict]:
        raise NotImplementedError(f"{self.provider_name} does not support transfers.")

    def health_check(self) -> dict:
        return {"provider": self.provider_name, "status": "unknown",
                "checked_at": datetime.utcnow().isoformat()}
