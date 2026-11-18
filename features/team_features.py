"""
features/team_features.py
==========================
Extracts team-level features for prediction models.
All features computed from historical match records only (no future data).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from config.settings import settings


@dataclass
class TeamFeatures:
    team_id: str
    season: str = ""
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    form_w3: float = 0.0
    form_w5: float = 0.0
    form_w10: float = 0.0
    goals_scored_per_game: float = 0.0
    goals_conceded_per_game: float = 0.0
    xg_per_game: Optional[float] = None
    xga_per_game: Optional[float] = None
    home_goals_per_game: float = 0.0
    home_conceded_per_game: float = 0.0
    away_goals_per_game: float = 0.0
    away_conceded_per_game: float = 0.0
    days_since_last_match: Optional[int] = None
    matches_in_last_14_days: int = 0
    matches_in_last_30_days: int = 0
    elo_rating: float = 1500.0
    league_position: Optional[int] = None
    points: int = 0
    matches_played: int = 0
    sample_size: int = 0
    data_completeness: float = 0.0

    def to_dict(self) -> dict:
        return {
            "team_id": self.team_id, "sample_size": self.sample_size,
            "form_w3": self.form_w3, "form_w5": self.form_w5, "form_w10": self.form_w10,
            "goals_scored_per_game": self.goals_scored_per_game,
            "goals_conceded_per_game": self.goals_conceded_per_game,
            "xg_per_game": self.xg_per_game, "xga_per_game": self.xga_per_game,
            "home_goals_per_game": self.home_goals_per_game,
            "away_goals_per_game": self.away_goals_per_game,
            "days_since_last_match": self.days_since_last_match,
            "matches_in_last_14_days": self.matches_in_last_14_days,
            "elo_rating": self.elo_rating,
            "league_position": self.league_position,
            "data_completeness": round(self.data_completeness, 4),
        }


class TeamFeatureExtractor:
    def extract(self, team_id: str, match_records: list[dict],
                as_of: Optional[datetime] = None, elo_rating: float = 1500.0,
                league_position: Optional[int] = None, points: int = 0,
                season: str = "") -> TeamFeatures:
        if as_of is None:
            as_of = datetime.now(timezone.utc)

        team_matches = []
        for m in match_records:
            if m.get("home_id") != team_id and m.get("away_id") != team_id:
                continue
            md = m["date"]
            if md.tzinfo is None:
                md = md.replace(tzinfo=timezone.utc)
            if md >= as_of:
                continue
            team_matches.append({**m, "_date_tz": md})
        team_matches.sort(key=lambda x: x["_date_tz"])
        n = len(team_matches)

        feat = TeamFeatures(team_id=team_id, season=season, elo_rating=elo_rating,
                            league_position=league_position, points=points,
                            sample_size=n,
                            data_completeness=min(1.0, n / max(settings.min_sample_size, 1)))
        if n == 0:
            return feat

        last_date = team_matches[-1]["_date_tz"]
        feat.days_since_last_match = (as_of - last_date).days
        feat.matches_in_last_14_days = sum(1 for m in team_matches if (as_of - m["_date_tz"]).days <= 14)
        feat.matches_in_last_30_days = sum(1 for m in team_matches if (as_of - m["_date_tz"]).days <= 30)

        def gf_ga(m):
            ih = m["home_id"] == team_id
            return (int(m["home_goals"]) if ih else int(m["away_goals"]),
                    int(m["away_goals"]) if ih else int(m["home_goals"]))

        def pts(m):
            gf, ga = gf_ga(m)
            return 3.0 if gf > ga else (1.0 if gf == ga else 0.0)

        def ppg(ms):
            return round(sum(pts(m) for m in ms) / len(ms), 4) if ms else 0.0

        feat.form_w3 = ppg(team_matches[-3:])
        feat.form_w5 = ppg(team_matches[-5:])
        feat.form_w10 = ppg(team_matches[-10:])

        tgf = tga = hgf = hga = hn = agf = aga = an = 0
        xg_list, xga_list = [], []
        for m in team_matches:
            ih = m["home_id"] == team_id
            gf, ga = gf_ga(m)
            tgf += gf; tga += ga
            if ih:
                hgf += gf; hga += ga; hn += 1
                if m.get("home_xg") is not None: xg_list.append(float(m["home_xg"]))
                if m.get("away_xg") is not None: xga_list.append(float(m["away_xg"]))
            else:
                agf += gf; aga += ga; an += 1
                if m.get("away_xg") is not None: xg_list.append(float(m["away_xg"]))
                if m.get("home_xg") is not None: xga_list.append(float(m["home_xg"]))

        feat.goals_scored_per_game = round(tgf / n, 4)
        feat.goals_conceded_per_game = round(tga / n, 4)
        if hn: feat.home_goals_per_game = round(hgf / hn, 4)
        if hn: feat.home_conceded_per_game = round(hga / hn, 4)
        if an: feat.away_goals_per_game = round(agf / an, 4)
        if an: feat.away_conceded_per_game = round(aga / an, 4)
        if xg_list: feat.xg_per_game = round(sum(xg_list) / len(xg_list), 4)
        if xga_list: feat.xga_per_game = round(sum(xga_list) / len(xga_list), 4)
        feat.matches_played = n
        return feat
