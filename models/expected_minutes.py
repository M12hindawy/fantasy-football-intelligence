"""
models/expected_minutes.py
==========================
Predicts expected minutes (0-90) based on availability and historical rotation.
Crucial for Fantasy Expected Points calculations.
"""
from __future__ import annotations
from dataclasses import dataclass
from features.player_features import PlayerFeatures
from agents.availability_agent import AvailabilityResult

@dataclass
class ExpectedMinutesResult:
    expected_minutes: float
    starting_probability: float
    is_nailed: bool

class ExpectedMinutesModel:
    def predict(self, features: PlayerFeatures, availability: AvailabilityResult) -> ExpectedMinutesResult:
        if availability.probability_of_playing <= 0.0:
            return ExpectedMinutesResult(0.0, 0.0, False)
            
        start_p = features.start_pct * availability.probability_of_playing
        
        # Heuristic model
        avg_start_mins = 80.0 if features.subbed_off_pct > 0.5 else 90.0
        avg_sub_mins = 20.0
        
        exp_mins = (start_p * avg_start_mins) + ((1.0 - start_p) * features.subbed_on_pct * avg_sub_mins)
        exp_mins *= availability.probability_of_playing
        
        return ExpectedMinutesResult(
            expected_minutes=round(min(90.0, max(0.0, exp_mins)), 2),
            starting_probability=round(min(1.0, max(0.0, start_p)), 4),
            is_nailed=(start_p > 0.85)
        )
