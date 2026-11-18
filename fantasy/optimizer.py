"""
fantasy/optimizer.py
====================
FPL squad optimizer. Given a list of player projections, selects the
highest-expected-points squad within FPL budget and squad structure rules.

FPL Rules enforced:
  - 15 players total: 2 GK, 5 DEF, 5 MID, 3 FWD
  - Max 3 players from same club
  - Budget: 100.0m (default)
  - Starting 11 must have valid formation (min 1 GK, 3 DEF, 2 MID, 1 FWD)
  - Captain selection: highest EV player

Uses real integer linear programming (scipy.optimize.milp) to select the
squad that maximizes total expected points subject to budget/position/club
constraints simultaneously. A greedy value-ratio heuristic (sort by
xP/price, take what fits) is NOT used here: greedy is provably suboptimal
in general knapsack-with-multiple-constraints problems — it can lock in an
early "good value" pick that forecloses a much better combination later,
and it has no way to prove the squad it returns is actually the best
budget allows. MILP finds the provably optimal squad for the given
projections (or proves none exists, e.g. if the budget is too tight).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

logger = logging.getLogger(__name__)

SQUAD_RULES = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
BUDGET = 100.0
MAX_PER_CLUB = 3


@dataclass
class PlayerOption:
    player_id: str
    name: str
    position: str
    club_id: str
    price: float
    expected_points: float
    is_nailed: bool
    ownership_pct: float = 0.0


@dataclass
class OptimizedSquad:
    players: list[PlayerOption]
    captain: Optional[PlayerOption]
    vice_captain: Optional[PlayerOption]
    total_cost: float
    total_expected_points: float
    bench: list[PlayerOption]

    def summary(self) -> dict:
        return {
            "total_cost": round(self.total_cost, 1),
            "total_xp": round(self.total_expected_points, 2),
            "captain": self.captain.name if self.captain else None,
            "vice_captain": self.vice_captain.name if self.vice_captain else None,
            "squad": [
                {
                    "name": p.name, "position": p.position, "club": p.club_id,
                    "price": p.price, "xp": round(p.expected_points, 2),
                    "nailed": p.is_nailed,
                }
                for p in self.players + self.bench
            ],
        }


class FPLOptimizer:
    def __init__(self, budget: float = BUDGET, max_per_club: int = MAX_PER_CLUB):
        self.budget = budget
        self.max_per_club = max_per_club

    def optimize(self, candidates: list[PlayerOption]) -> Optional[OptimizedSquad]:
        """Real MILP squad selection: maximize sum(expected_points) subject
        to budget, exact position counts, and per-club cap — all satisfied
        simultaneously, not approximated by a sorted greedy pass."""
        eligible = [c for c in candidates if c.position in SQUAD_RULES]
        if not eligible:
            logger.warning("Optimizer: no candidates with a recognized position.")
            return None

        n = len(eligible)
        prices = np.array([c.price for c in eligible])
        points = np.array([c.expected_points for c in eligible])

        c = -points  # milp minimizes; maximize points == minimize -points

        constraints = [
            LinearConstraint(prices, -np.inf, self.budget),
            LinearConstraint(np.ones(n), 15, 15),
        ]
        for pos, required in SQUAD_RULES.items():
            mask = np.array([1.0 if p.position == pos else 0.0 for p in eligible])
            constraints.append(LinearConstraint(mask, required, required))

        clubs = sorted(set(p.club_id for p in eligible))
        for club in clubs:
            mask = np.array([1.0 if p.club_id == club else 0.0 for p in eligible])
            constraints.append(LinearConstraint(mask, -np.inf, self.max_per_club))

        integrality = np.ones(n)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

        if not result.success:
            logger.warning("Optimizer: no feasible 15-man squad under budget=%.1f, "
                            "max_per_club=%d, with %d eligible candidates — %s",
                            self.budget, self.max_per_club, n, result.message)
            return None

        selected = [eligible[i] for i in range(n) if result.x[i] > 0.5]
        total_cost = sum(p.price for p in selected)

        starters_pool = sorted(selected, key=lambda p: p.expected_points, reverse=True)
        captain = starters_pool[0] if starters_pool else None
        vice_captain = starters_pool[1] if len(starters_pool) > 1 else None

        bench = self._pick_bench(selected)
        starters = [p for p in selected if p not in bench]

        return OptimizedSquad(
            players=starters,
            captain=captain,
            vice_captain=vice_captain,
            total_cost=round(total_cost, 1),
            total_expected_points=round(sum(p.expected_points for p in starters), 2),
            bench=bench,
        )

    def _pick_bench(self, squad: list[PlayerOption]) -> list[PlayerOption]:
        """Pick 4 bench players ensuring valid starting XI formation."""
        bench = []
        pos_in_starting: dict[str, int] = {p: 0 for p in SQUAD_RULES}
        sorted_by_xp = sorted(squad, key=lambda p: p.expected_points, reverse=True)
        
        # First ensure minimum positional requirements met in starting XI
        remaining = list(sorted_by_xp)
        for player in sorted_by_xp:
            pos = player.position
            if pos_in_starting.get(pos, 0) < STARTING_XI_MIN.get(pos, 0):
                pos_in_starting[pos] = pos_in_starting.get(pos, 0) + 1
                remaining.remove(player)
        
        # Fill starting XI to 11 by best xP
        starters_count = sum(pos_in_starting.values())
        for player in sorted_by_xp:
            if player not in remaining:
                continue
            if starters_count >= 11:
                break
            pos_in_starting[player.position] = pos_in_starting.get(player.position, 0) + 1
            remaining.remove(player)
            starters_count += 1

        bench = remaining[:4]
        return bench
