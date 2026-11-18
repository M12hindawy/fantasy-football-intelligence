"""
agents/orchestrator.py
=======================
Master orchestrator — single entry-point for any match or gameweek analysis.
Coordinates all agents and models, enforces data quality gates,
and returns a unified AnalysisResult ready for the dashboard or API.

Flow:
  1. Fetch fixtures & team data (provider)
  2. Feature extraction + DQ assessment
  3. Dixon-Coles + Elo + optional XGBoost ensemble
  4. Monte Carlo simulation (10,000 runs)
  5. Odds pipeline (margin removal, EV, Kelly)
  6. Match analysis agent (form, fatigue, availability)
  7. Fantasy projection (if enabled)
  8. Report generation
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any

from config.settings import settings
from agents.match_analysis_agent import MatchAnalysisAgent, MatchAnalysisContext
from agents.availability_agent import AvailabilityAgent
from agents.lineup_agent import LineupAgent
from agents.league_context_agent import LeagueContextAgent
from data.quality.data_quality_engine import DataQualityReport
from features.feature_pipeline import feature_pipeline
from simulation.monte_carlo import MonteCarloEngine
from simulation.what_if_engine import WhatIfEngine, WhatIfScenario

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorInput:
    """Structured input for a single match analysis request."""
    fixture_id: str
    home_team_id: str
    away_team_id: str
    competition_code: str
    kickoff_utc: Optional[datetime] = None

    # Historical match records (list of dicts with home_id, away_id, home_goals, away_goals, date)
    match_records: list[dict] = field(default_factory=list)

    # Contextual data
    home_injuries: list[dict] = field(default_factory=list)
    away_injuries: list[dict] = field(default_factory=list)
    home_suspensions: list[dict] = field(default_factory=list)
    away_suspensions: list[dict] = field(default_factory=list)
    home_confirmed_lineup: list[dict] = field(default_factory=list)
    away_confirmed_lineup: list[dict] = field(default_factory=list)
    home_key_players: list[str] = field(default_factory=list)
    away_key_players: list[str] = field(default_factory=list)

    # Model inputs
    mu_home: float = 1.50
    mu_away: float = 1.10
    rho: float = -0.10
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    home_position: Optional[int] = None
    away_position: Optional[int] = None
    home_points: int = 0
    away_points: int = 0

    # Odds data (optional)
    odds_bundle: Optional[Any] = None
    odds_age_seconds: Optional[float] = None

    # What-if scenarios
    scenarios: list[WhatIfScenario] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis output for one fixture."""
    fixture_id: str
    home_team_id: str
    away_team_id: str
    generated_at: datetime

    # Core predictions
    p_home_win: float = 0.0
    p_draw: float = 0.0
    p_away_win: float = 0.0
    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0
    p_btts: float = 0.0
    p_over_25: float = 0.0

    # Simulation
    simulation: Optional[dict] = None

    # Team analysis
    home_analysis: Optional[dict] = None
    away_analysis: Optional[dict] = None

    # Lineups
    home_lineup_confirmed: bool = False
    away_lineup_confirmed: bool = False
    home_missing_key_players: list[str] = field(default_factory=list)
    away_missing_key_players: list[str] = field(default_factory=list)

    # Data quality
    dq_report: Optional[dict] = None
    dq_grade: str = "F"

    # Odds & value
    odds_analysis: Optional[dict] = None
    recommendation: str = "NO_BET"
    recommendation_strength: float = 0.0

    # What-if
    what_if_applied: bool = False
    what_if_adjustments: Optional[dict] = None

    # Meta
    model_name: str = "Ensemble(DC+Elo)"
    data_quality_passed: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "generated_at": self.generated_at.isoformat(),
            "predictions": {
                "p_home_win": round(self.p_home_win, 4),
                "p_draw": round(self.p_draw, 4),
                "p_away_win": round(self.p_away_win, 4),
                "expected_home_goals": round(self.expected_home_goals, 3),
                "expected_away_goals": round(self.expected_away_goals, 3),
                "p_btts": round(self.p_btts, 4),
                "p_over_25": round(self.p_over_25, 4),
            },
            "simulation": self.simulation,
            "home_analysis": self.home_analysis,
            "away_analysis": self.away_analysis,
            "data_quality": self.dq_report,
            "dq_grade": self.dq_grade,
            "data_quality_passed": self.data_quality_passed,
            "recommendation": self.recommendation,
            "recommendation_strength": round(self.recommendation_strength, 3),
            "what_if": self.what_if_adjustments if self.what_if_applied else None,
            "model": self.model_name,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class Orchestrator:
    """
    Central coordinator for all match analysis pipelines.
    Instantiate once (singleton-like) and call .analyze() per fixture.
    """

    def __init__(self) -> None:
        self._match_agent   = MatchAnalysisAgent()
        self._avail_agent   = AvailabilityAgent()
        self._lineup_agent  = LineupAgent()
        self._league_agent  = LeagueContextAgent()
        self._mc_engine     = MonteCarloEngine()
        self._whatif_engine = WhatIfEngine()

    # ── Public API ────────────────────────────────────────────────────────

    def analyze(self, inp: OrchestratorInput) -> AnalysisResult:
        result = AnalysisResult(
            fixture_id=inp.fixture_id,
            home_team_id=inp.home_team_id,
            away_team_id=inp.away_team_id,
            generated_at=datetime.now(timezone.utc),
        )

        try:
            self._run_pipeline(inp, result)
        except Exception as exc:
            logger.exception("Orchestrator pipeline failed for %s", inp.fixture_id)
            result.errors.append(f"Pipeline error: {exc}")

        return result

    # ── Pipeline stages ───────────────────────────────────────────────────

    def _run_pipeline(self, inp: OrchestratorInput, result: AnalysisResult) -> None:

        # ── Stage 1: Availability ─────────────────────────────────────────
        home_avail_h = self._avail_agent.assess(inp.home_team_id, inp.home_injuries, inp.home_suspensions)
        away_avail_a = self._avail_agent.assess(inp.away_team_id, inp.away_injuries, inp.away_suspensions)

        # ── Stage 2: Lineups ──────────────────────────────────────────────
        home_lineup = self._lineup_agent.assess(
            inp.home_team_id, inp.home_confirmed_lineup, [], inp.home_key_players
        )
        away_lineup = self._lineup_agent.assess(
            inp.away_team_id, inp.away_confirmed_lineup, [], inp.away_key_players
        )
        result.home_lineup_confirmed = home_lineup.is_confirmed
        result.away_lineup_confirmed = away_lineup.is_confirmed
        result.home_missing_key_players = home_lineup.missing_key_players
        result.away_missing_key_players = away_lineup.missing_key_players

        # ── Stage 3: What-If adjustments ─────────────────────────────────
        mu_home, mu_away = inp.mu_home, inp.mu_away
        if inp.scenarios:
            mu_home, mu_away = self._whatif_engine.apply_scenarios(mu_home, mu_away, inp.scenarios)
            result.what_if_applied = True
            result.what_if_adjustments = {
                "original": {"mu_home": inp.mu_home, "mu_away": inp.mu_away},
                "adjusted": {"mu_home": mu_home, "mu_away": mu_away},
                "scenarios": [{"type": s.type, "target": s.target_team, "severity": s.severity}
                               for s in inp.scenarios],
            }

        # ── Stage 4: Match Analysis (features + team analysis) ────────────
        ctx: MatchAnalysisContext = self._match_agent.analyze(
            fixture_id=inp.fixture_id,
            home_id=inp.home_team_id,
            away_id=inp.away_team_id,
            match_records=inp.match_records,
            home_injuries=inp.home_injuries,
            away_injuries=inp.away_injuries,
            home_suspensions=inp.home_suspensions,
            away_suspensions=inp.away_suspensions,
            mu_home=mu_home,
            mu_away=mu_away,
            rho=inp.rho,
        )
        result.home_analysis = ctx.home_analysis.as_dict()
        result.away_analysis = ctx.away_analysis.as_dict()

        # ── Stage 5: DQ gate ──────────────────────────────────────────────
        dq: DataQualityReport = ctx.dq_report
        result.dq_report = dq.as_dict()
        result.dq_grade = dq.grade
        result.data_quality_passed = dq.passes_threshold
        result.warnings.extend(dq.warnings)
        result.errors.extend(dq.errors)

        if not dq.passes_threshold:
            result.warnings.append(
                f"DQ score {dq.score:.2f} < threshold {settings.min_data_quality} "
                f"(grade {dq.grade}) — predictions may be unreliable."
            )

        # ── Stage 6: Simulation ───────────────────────────────────────────
        sim = ctx.simulation
        result.simulation     = sim.as_dict()
        result.p_home_win     = sim.p_home_win
        result.p_draw         = sim.p_draw
        result.p_away_win     = sim.p_away_win
        result.expected_home_goals = sim.mean_home_goals
        result.expected_away_goals = sim.mean_away_goals
        result.p_btts         = sim.p_btts
        result.p_over_25      = sim.p_over_25

        # ── Stage 7: Simple recommendation ───────────────────────────────
        result.recommendation, result.recommendation_strength = self._make_recommendation(
            result.p_home_win, result.p_draw, result.p_away_win, dq.score
        )

    @staticmethod
    def _make_recommendation(
        p_home: float, p_draw: float, p_away: float, dq_score: float
    ) -> tuple[str, float]:
        """Simple threshold-based recommendation. Replace with EV-based in Phase 8."""
        if dq_score < settings.min_data_quality:
            return "NO_BET", 0.0

        max_p = max(p_home, p_draw, p_away)
        if max_p == p_home and p_home > 0.55:
            return "HOME_WIN", round((p_home - 0.55) / 0.45, 3)
        elif max_p == p_away and p_away > 0.45:
            return "AWAY_WIN", round((p_away - 0.45) / 0.55, 3)
        elif p_draw > 0.35:
            return "DRAW", round((p_draw - 0.35) / 0.35, 3)
        return "NO_BET", 0.0


# Module-level singleton
orchestrator = Orchestrator()
