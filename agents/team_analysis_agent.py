"""
agents/team_analysis_agent.py
==============================
Full team analysis with opponent-adjusted form, xG, xGA, and Elo.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class TeamAnalysisResult:
    team_id: str
    matches_analyzed: int
    avg_xg: float
    avg_xga: float
    avg_shots_on_target: float
    avg_corners: float
    form_5: str
    form_10: str
    home_win_rate: float
    away_win_rate: float
    clean_sheets: int
    elo_rating: float
    opponent_adj_form: float     # 0-1, higher = better recent wins vs strong opposition
    attack_strength: float       # relative to league avg
    defense_strength: float      # relative to league avg (lower = better defense)

    def as_dict(self):
        import dataclasses
        return dataclasses.asdict(self)
    key_injuries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TeamAnalysisAgent:
    def analyze(
        self,
        team_id: str,
        match_records: list[dict],
        as_home: bool = True,
        elo_rating: float = 1500.0,
        key_injuries: Optional[list[str]] = None,
        n_recent: int = 10,
    ) -> TeamAnalysisResult:
        team_records = []
        for r in match_records:
            if as_home and r.get("home_id") == team_id:
                team_records.append({
                    "scored": r.get("home_goals", 0) or 0,
                    "conceded": r.get("away_goals", 0) or 0,
                    "xg": r.get("home_xg", 0) or 0,
                    "xga": r.get("away_xg", 0) or 0,
                    "sot": r.get("home_shots_target", 0) or 0,
                    "corners": r.get("home_corners", 0) or 0,
                    "opp_id": r.get("away_id", ""),
                    "result": "W" if r["home_goals"] > r["away_goals"] else ("D" if r["home_goals"] == r["away_goals"] else "L"),
                })
            elif not as_home and r.get("away_id") == team_id:
                team_records.append({
                    "scored": r.get("away_goals", 0) or 0,
                    "conceded": r.get("home_goals", 0) or 0,
                    "xg": r.get("away_xg", 0) or 0,
                    "xga": r.get("home_xg", 0) or 0,
                    "sot": r.get("away_shots_target", 0) or 0,
                    "corners": r.get("away_corners", 0) or 0,
                    "opp_id": r.get("home_id", ""),
                    "result": "W" if r["away_goals"] > r["home_goals"] else ("D" if r["away_goals"] == r["home_goals"] else "L"),
                })

        if not team_records:
            return TeamAnalysisResult(
                team_id=team_id, matches_analyzed=0,
                avg_xg=1.5, avg_xga=1.5, avg_shots_on_target=4.0, avg_corners=5.0,
                form_5="—", form_10="—", home_win_rate=0.5, away_win_rate=0.5,
                clean_sheets=0, elo_rating=elo_rating,
                opponent_adj_form=0.5, attack_strength=1.0, defense_strength=1.0,
                warnings=["No match data found for this team."]
            )

        recent = team_records[-n_recent:]
        n = len(recent)
        league_avg = 1.5

        def _avg(lst, key):
            vals = [x[key] for x in lst if x.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        avg_xg = _avg(recent, "xg") or _avg(recent, "scored")
        avg_xga = _avg(recent, "xga") or _avg(recent, "conceded")
        avg_sot = _avg(recent, "sot")
        avg_corners = _avg(recent, "corners")

        results = [r["result"] for r in recent]
        form_5 = "".join(results[-5:]) if len(results) >= 5 else "".join(results)
        form_10 = "".join(results[-10:]) if len(results) >= 10 else "".join(results)

        wins = results.count("W")
        draws = results.count("D")
        clean_sheets = sum(1 for r in recent if r["conceded"] == 0)

        home_records = [r for r in team_records if as_home]
        away_records = [r for r in team_records if not as_home]
        home_wr = home_records.count("W") / len(home_records) if home_records else 0.5
        away_wr = away_records.count("W") / len(away_records) if away_records else 0.5

        # Opponent-adjusted form: weight wins against stronger opponents higher
        # Proxy: we use xG of opponents as their strength indicator
        opp_adj_scores = []
        for r in recent:
            if r["result"] == "W":
                opp_strength = r.get("xga", 1.5) / league_avg
                opp_adj_scores.append(min(1.0, 0.5 + 0.3 * opp_strength))
            elif r["result"] == "D":
                opp_adj_scores.append(0.3)
            else:
                opp_adj_scores.append(0.0)
        opp_adj_form = round(sum(opp_adj_scores) / len(opp_adj_scores), 3) if opp_adj_scores else 0.5

        attack_strength = round(avg_xg / league_avg, 3)
        defense_strength = round(avg_xga / league_avg, 3)

        warnings = []
        if n < 5:
            warnings.append(f"Only {n} matches available — limited confidence.")
        if key_injuries:
            warnings.append(f"Key injuries: {', '.join(key_injuries)}")

        return TeamAnalysisResult(
            team_id=team_id,
            matches_analyzed=n,
            avg_xg=avg_xg,
            avg_xga=avg_xga,
            avg_shots_on_target=avg_sot,
            avg_corners=avg_corners,
            form_5=form_5,
            form_10=form_10,
            home_win_rate=home_wr,
            away_win_rate=away_wr,
            clean_sheets=clean_sheets,
            elo_rating=elo_rating,
            opponent_adj_form=opp_adj_form,
            attack_strength=attack_strength,
            defense_strength=defense_strength,
            key_injuries=key_injuries or [],
            warnings=warnings,
        )

team_analysis_agent = TeamAnalysisAgent()
