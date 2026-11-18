"""
simulation/what_if_engine.py
============================
Injects hypothetical scenarios ("What If?") and adjustments into expected goals.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class WhatIfScenario:
    type: str  # "KEY_PLAYER_OUT", "NEW_MANAGER", "FATIGUED"
    target_team: str  # "HOME" or "AWAY"
    severity: float = 1.0  # 1.0 = standard impact

class WhatIfEngine:
    def apply_scenarios(
        self, 
        mu_home: float, 
        mu_away: float, 
        scenarios: list[WhatIfScenario]
    ) -> tuple[float, float]:
        adj_home = mu_home
        adj_away = mu_away
        
        for s in scenarios:
            impact = s.severity
            
            if s.type == "KEY_PLAYER_OUT":
                if s.target_team == "HOME":
                    adj_home *= (1.0 - (0.15 * impact))
                else:
                    adj_away *= (1.0 - (0.15 * impact))
                    
            elif s.type == "DEFENDER_OUT":
                if s.target_team == "HOME":
                    adj_away *= (1.0 + (0.10 * impact))
                else:
                    adj_home *= (1.0 + (0.10 * impact))
                    
            elif s.type == "NEW_MANAGER":
                if s.target_team == "HOME":
                    adj_home *= (1.0 + (0.10 * impact))
                    adj_away *= (1.0 - (0.05 * impact))
                else:
                    adj_away *= (1.0 + (0.10 * impact))
                    adj_home *= (1.0 - (0.05 * impact))
                    
            elif s.type == "FATIGUED":
                if s.target_team == "HOME":
                    adj_home *= (1.0 - (0.08 * impact))
                    adj_away *= (1.0 + (0.05 * impact))
                else:
                    adj_away *= (1.0 - (0.08 * impact))
                    adj_home *= (1.0 + (0.05 * impact))
                    
            else:
                logger.warning(f"Unknown scenario type: {s.type}")
        
        return max(0.1, round(adj_home, 3)), max(0.1, round(adj_away, 3))
