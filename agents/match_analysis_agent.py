"""
agents/match_analysis_agent.py
==============================
Coordinates team analysis, feature extraction, and Monte Carlo simulation
to produce a complete Match Analysis Context.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from agents.team_analysis_agent import TeamAnalysisAgent, TeamAnalysisResult
from data.quality.data_quality_engine import DataQualityReport
from features.feature_pipeline import feature_pipeline
from simulation.monte_carlo import MonteCarloEngine, MatchSimulationResult

@dataclass
class MatchAnalysisContext:
    fixture_id: str
    home_team_id: str
    away_team_id: str
    dq_report: DataQualityReport
    home_analysis: TeamAnalysisResult
    away_analysis: TeamAnalysisResult
    simulation: MatchSimulationResult

    def as_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "data_quality": self.dq_report.as_dict(),
            "home_analysis": self.home_analysis.as_dict(),
            "away_analysis": self.away_analysis.as_dict(),
            "simulation": self.simulation.as_dict(),
        }

class MatchAnalysisAgent:
    def __init__(self):
        self._team_agent = TeamAnalysisAgent()
        self._mc_engine = MonteCarloEngine()
        
    def analyze(
        self,
        fixture_id: str,
        home_id: str,
        away_id: str,
        match_records: list[dict],
        home_injuries: Optional[list[dict]] = None,
        away_injuries: Optional[list[dict]] = None,
        home_suspensions: Optional[list[dict]] = None,
        away_suspensions: Optional[list[dict]] = None,
        mu_home: float = 1.5,
        mu_away: float = 1.0,
        rho: float = -0.10,
    ) -> MatchAnalysisContext:
        
        # 1. Feature Extraction & DQ
        pipeline_out = feature_pipeline.extract_match_features(
            home_id=home_id, away_id=away_id,
            match_records=match_records
        )
        
        # 2. Team Analysis
        home_analysis = self._team_agent.analyze(
            pipeline_out["home_features"],
            home_injuries or [],
            home_suspensions or []
        )
        away_analysis = self._team_agent.analyze(
            pipeline_out["away_features"],
            away_injuries or [],
            away_suspensions or []
        )
        
        # 3. Monte Carlo Simulation
        sim_result = self._mc_engine.simulate_match(
            mu_home=mu_home,
            mu_away=mu_away,
            rho=rho,
            fixture_id=fixture_id
        )
        
        return MatchAnalysisContext(
            fixture_id=fixture_id,
            home_team_id=home_id,
            away_team_id=away_id,
            dq_report=pipeline_out["dq_report"],
            home_analysis=home_analysis,
            away_analysis=away_analysis,
            simulation=sim_result
        )
