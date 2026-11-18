"""
fantasy/multi_gw_engine.py
===========================
Multi-Gameweek Fantasy analysis.
Fetches the next N gameweeks' fixture difficulty and player form to
rank transfer targets and captain picks across multiple GWs.
"""
from __future__ import annotations
import logging
import urllib.request
import json
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MultiGWPlayer:
    player_id: int
    name: str
    team_name: str
    position: str
    price: float
    form: float
    total_points: int
    xgi_per_90: float
    ep_next: float
    fixtures: list[dict] = field(default_factory=list)   # next N fixtures with FDR
    avg_fdr: float = 3.0
    multi_gw_score: float = 0.0
    captain_potential: float = 0.0
    risk: str = "MEDIUM"
    starting_prob: float = 0.9
    value_rank: int = 0


class MultiGWEngine:
    POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    def __init__(self):
        self._bootstrap: Optional[dict] = None
        self._fixtures: Optional[list] = None

    def _load_bootstrap(self) -> dict:
        if self._bootstrap is None:
            url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self._bootstrap = json.loads(r.read())
        return self._bootstrap

    def _load_fixtures(self) -> list:
        if self._fixtures is None:
            url = "https://fantasy.premierleague.com/api/fixtures/"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self._fixtures = json.loads(r.read())
        return self._fixtures

    def analyze(self, n_gws: int = 5, position: Optional[str] = None, max_price: float = 20.0) -> list[MultiGWPlayer]:
        bootstrap = self._load_bootstrap()
        all_fixtures = self._load_fixtures()

        teams = {t["id"]: t for t in bootstrap["teams"]}
        events = bootstrap["events"]

        # Find the current/next GW
        next_event = next((e for e in events if e["is_next"]), None)
        current_event = next((e for e in events if e["is_current"]), None)
        start_gw = (next_event or current_event or events[0])["id"]

        # Build team -> list of upcoming fixtures (with FDR)
        team_fixtures: dict[int, list[dict]] = {t_id: [] for t_id in teams}
        for f in all_fixtures:
            if f.get("finished"):
                continue
            gw = f.get("event")
            if gw is None or gw < start_gw or gw > start_gw + n_gws - 1:
                continue
            h_id, a_id = f["team_h"], f["team_a"]
            h_fdr = f.get("team_h_difficulty", 3)
            a_fdr = f.get("team_a_difficulty", 3)
            team_fixtures[h_id].append({"gw": gw, "opponent": teams.get(a_id, {}).get("short_name", "?"), "fdr": h_fdr, "home": True})
            team_fixtures[a_id].append({"gw": gw, "opponent": teams.get(h_id, {}).get("short_name", "?"), "fdr": a_fdr, "home": False})

        results = []
        for p in bootstrap["elements"]:
            pos_str = self.POS_MAP.get(p.get("element_type"), "MID")
            if position and pos_str != position:
                continue
            price = (p.get("now_cost", 50)) / 10.0
            if price > max_price:
                continue
            if p.get("status") not in ("a", "d"):
                continue
            chance = p.get("chance_of_playing_next_round")
            if chance is not None and chance < 50:
                continue

            team_id = p.get("team")
            team_name = teams.get(team_id, {}).get("short_name", "?")
            fixtures_for_team = sorted(team_fixtures.get(team_id, []), key=lambda x: x["gw"])[:n_gws]

            form = float(p.get("form", 0) or 0)
            xgi_90 = float(p.get("expected_goal_involvements_per_90", 0) or 0)
            ep_next = float(p.get("ep_next", 0) or 0)
            ict = float(p.get("ict_index", 0) or 0)
            total_pts = int(p.get("total_points", 0))

            # Average fixture difficulty for next N GWs (lower = easier)
            avg_fdr = sum(f["fdr"] for f in fixtures_for_team) / len(fixtures_for_team) if fixtures_for_team else 3.0

            # Multi-GW score: weights fixture ease + form + xGI
            fixture_bonus = (5.0 - avg_fdr) / 4.0    # 0-1, higher for easier fixtures
            multi_gw_score = (
                ep_next * 0.35
                + form * 0.20
                + fixture_bonus * 2.5 * 0.25
                + xgi_90 * 1.5 * 0.10
                + (ict / 10.0) * 0.10
            )

            # Captain potential
            penalty_taker = bool(p.get("penalties_order") == 1)
            setpiece_taker = bool(p.get("corners_and_indirect_freekicks_order") == 1)
            captain_bonus = 0.5 if penalty_taker else (0.3 if setpiece_taker else 0.0)
            captain_potential = min(10.0, multi_gw_score + captain_bonus)

            # Risk assessment
            if chance is not None and chance < 75:
                risk = "HIGH"
            elif form < 2.0 and ep_next < 3.0:
                risk = "HIGH"
            elif form > 6.0 and avg_fdr < 2.5:
                risk = "LOW"
            else:
                risk = "MEDIUM"

            # Starting probability proxy
            starts = p.get("starts", 0)
            minutes = p.get("minutes", 0)
            expected_matches = max(1, total_pts // 2)
            starting_prob = min(0.99, (starts / expected_matches)) if expected_matches > 0 else 0.9

            results.append(MultiGWPlayer(
                player_id=p["id"],
                name=f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
                team_name=team_name,
                position=pos_str,
                price=price,
                form=form,
                total_points=total_pts,
                xgi_per_90=xgi_90,
                ep_next=ep_next,
                fixtures=fixtures_for_team,
                avg_fdr=round(avg_fdr, 2),
                multi_gw_score=round(multi_gw_score, 3),
                captain_potential=round(captain_potential, 3),
                risk=risk,
                starting_prob=round(starting_prob, 2),
            ))

        # Sort by multi_gw_score
        results.sort(key=lambda x: x.multi_gw_score, reverse=True)
        # Assign value rank
        for i, r in enumerate(results):
            r.value_rank = i + 1
        return results

    def get_captain_picks(self, n_gws: int = 3) -> list[MultiGWPlayer]:
        players = self.analyze(n_gws=n_gws)
        return sorted(players, key=lambda x: x.captain_potential, reverse=True)[:10]

    def get_differential_picks(self, n_gws: int = 3, max_ownership: float = 10.0) -> list[MultiGWPlayer]:
        """Players with good scores but low ownership (differentials)."""
        players = self.analyze(n_gws=n_gws)
        diffs = [p for p in players if float(self._bootstrap["elements"][p.player_id - 1].get("selected_by_percent", 0) or 0) < max_ownership]
        return diffs[:10]


multi_gw_engine = MultiGWEngine()
