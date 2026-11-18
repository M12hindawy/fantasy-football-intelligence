"""
features/player_features.py
===========================
Extracts player-level features for Expected Minutes and Fantasy models.
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class PlayerFeatures:
    player_id: str
    minutes_played: int = 0
    starts: int = 0
    appearances: int = 0
    goals: int = 0
    assists: int = 0
    xg: float = 0.0
    xa: float = 0.0
    
    # Derived per 90
    xg_90: float = 0.0
    xa_90: float = 0.0
    goals_90: float = 0.0
    assists_90: float = 0.0
    
    start_pct: float = 0.0
    subbed_on_pct: float = 0.0
    subbed_off_pct: float = 0.0

class PlayerFeatureExtractor:
    def extract(self, player_id: str, match_records: list[dict], team_matches_count: int) -> PlayerFeatures:
        feat = PlayerFeatures(player_id=player_id)
        sub_on = 0
        sub_off = 0
        
        for m in match_records:
            mins = m.get("minutes", 0)
            feat.minutes_played += mins
            
            if m.get("started", False):
                feat.starts += 1
                if mins < 89:
                    sub_off += 1
            else:
                if mins > 0:
                    sub_on += 1
                    
            if mins > 0:
                feat.appearances += 1
                
            feat.goals += m.get("goals", 0)
            feat.assists += m.get("assists", 0)
            feat.xg += m.get("xg", 0.0)
            feat.xa += m.get("xa", 0.0)
            
        if feat.minutes_played > 0:
            ninety = feat.minutes_played / 90.0
            feat.xg_90 = round(feat.xg / ninety, 4)
            feat.xa_90 = round(feat.xa / ninety, 4)
            feat.goals_90 = round(feat.goals / ninety, 4)
            feat.assists_90 = round(feat.assists / ninety, 4)
            
        if team_matches_count > 0:
            feat.start_pct = round(feat.starts / team_matches_count, 4)
            feat.subbed_on_pct = round(sub_on / team_matches_count, 4)
            feat.subbed_off_pct = round(sub_off / max(1, feat.starts), 4)
            
        return feat
