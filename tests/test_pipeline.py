"""
tests/test_pipeline.py
======================
Integration test: runs the full analysis pipeline with synthetic data
to verify every agent produces valid output.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.schemas import MarketType, MatchStatus
from data.fetchers.odds_api_fetcher import OddsApiFetcher
from agents.odds_ingestion_agent import OddsIngestionAgent
from agents.margin_removal_agent import MarginRemovalAgent
from agents.market_normalization_agent import MarketNormalizationAgent
from agents.market_probability_agent import MarketProbabilityAgent
from agents.value_assessment_agent import ValueAssessmentAgent
from agents.responsible_use_guard_agent import ResponsibleUseGuardAgent
from models.dixon_coles import score_probability_matrix, matrix_to_1x2, matrix_to_btts, matrix_to_over_under
from data.schemas import PredictionOutput, ScoreMatrix
from output.formatter import format_report
from output.json_contract import AnalysisOutputJSON


KICKOFF = datetime(2025, 12, 15, 15, 0, 0, tzinfo=timezone.utc)
ODDS_TS = datetime(2025, 12, 14, 20, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def standard_1x2_bundle():
    """A standard 1X2 bundle with typical Premier League odds."""
    return OddsApiFetcher.from_manual_input(
        fixture_id="arsenal_chelsea_test",
        bookmaker="1xBet",
        market_type=MarketType.ONE_X_TWO,
        selections=[("Home", 2.10), ("Draw", 3.30), ("Away", 3.50)],
        odds_timestamp=ODDS_TS,
        event_start_time=KICKOFF,
    )


@pytest.fixture
def totals_bundle():
    """An Over/Under 2.5 bundle."""
    return OddsApiFetcher.from_manual_input(
        fixture_id="arsenal_chelsea_test",
        bookmaker="1xBet",
        market_type=MarketType.TOTALS,
        selections=[("Over 2.5", 1.85), ("Under 2.5", 1.95)],
        line=2.5,
        odds_timestamp=ODDS_TS,
        event_start_time=KICKOFF,
    )


@pytest.fixture
def baseline_prediction():
    """League-average Dixon-Coles prediction."""
    mu_h, mu_a, rho = 1.35, 1.10, -0.10
    mat = score_probability_matrix(mu_h, mu_a, rho)
    p_home, p_draw, p_away = matrix_to_1x2(mat)
    p_btts = matrix_to_btts(mat)
    p_o25, _ = matrix_to_over_under(mat, 2.5)
    return PredictionOutput(
        fixture_id="arsenal_chelsea_test",
        model_name="TestDixonColes",
        model_version="test-1.0",
        generated_at=datetime.now(timezone.utc),
        p_home_win=round(p_home, 6),
        p_draw=round(p_draw, 6),
        p_away_win=round(p_away, 6),
        expected_goals_home=mu_h,
        expected_goals_away=mu_a,
        p_btts=round(p_btts, 6),
        p_over_25=round(p_o25, 6),
        score_matrix=ScoreMatrix(matrix=mat.tolist()),
    )


class TestOddsIngestion:
    def test_valid_bundle_passes(self, standard_1x2_bundle):
        agent = OddsIngestionAgent()
        _, issues = agent.validate(standard_1x2_bundle, KICKOFF)
        # Future kickoff — should have no INVALID ODDS issues
        invalid_issues = [i for i in issues if "INVALID" in i.upper()]
        assert not invalid_issues

    def test_freshness_penalty(self, standard_1x2_bundle):
        penalty = OddsIngestionAgent.freshness_penalty(standard_1x2_bundle)
        assert 0.0 <= penalty <= 0.10


class TestMarginRemoval:
    def test_fair_probs_attached(self, standard_1x2_bundle):
        agent = MarginRemovalAgent()
        bundle, method, warnings = agent.apply(standard_1x2_bundle, "shin")
        fair_probs = [r.__dict__.get("_fair_probability") for r in bundle.odds]
        assert all(p is not None for p in fair_probs)
        assert abs(sum(fair_probs) - 1.0) < 0.01
        assert method == "shin"

    def test_overround_computed(self, standard_1x2_bundle):
        agent = MarginRemovalAgent()
        bundle, _, _ = agent.apply(standard_1x2_bundle, "shin")
        assert bundle.overround is not None
        assert bundle.overround > 0.0


class TestNormalisation:
    def test_passes_for_valid_bundle(self, standard_1x2_bundle):
        # Apply margin removal first
        margin_agent = MarginRemovalAgent()
        bundle, _, _ = margin_agent.apply(standard_1x2_bundle, "shin")
        norm_agent = MarketNormalizationAgent()
        _, warnings = norm_agent.validate(bundle)
        error_warnings = [w for w in warnings if "ERROR" in w]
        assert not error_warnings


class TestMarketProbability:
    def test_1x2_home(self, standard_1x2_bundle, baseline_prediction):
        margin_agent = MarginRemovalAgent()
        bundle, _, _ = margin_agent.apply(standard_1x2_bundle, "shin")
        mp_agent = MarketProbabilityAgent()
        prob, notes = mp_agent.get_model_probability(baseline_prediction, bundle, "Home")
        assert prob is not None
        assert 0.0 < prob < 1.0

    def test_totals_over(self, totals_bundle, baseline_prediction):
        margin_agent = MarginRemovalAgent()
        bundle, _, _ = margin_agent.apply(totals_bundle, "proportional")
        mp_agent = MarketProbabilityAgent()
        prob, notes = mp_agent.get_model_probability(baseline_prediction, bundle, "Over 2.5")
        assert prob is not None
        assert 0.0 < prob < 1.0


class TestValueAssessment:
    def test_produces_result(self, standard_1x2_bundle, baseline_prediction):
        margin_agent = MarginRemovalAgent()
        bundle, _, _ = margin_agent.apply(standard_1x2_bundle, "shin")

        mp_agent = MarketProbabilityAgent()
        model_prob, _ = mp_agent.get_model_probability(baseline_prediction, bundle, "Home")

        target = next(r for r in bundle.odds if r.selection == "Home")
        va_agent = ValueAssessmentAgent()
        result = va_agent.assess(
            home_team="Arsenal",
            away_team="Chelsea",
            competition="Premier League",
            kickoff=KICKOFF,
            status=MatchStatus.PRE_MATCH,
            bundle=bundle,
            odds_record=target,
            prediction=baseline_prediction,
            model_probability=model_prob,
        )
        # Decision must be one of the valid states
        from data.schemas import DecisionStatus
        assert result.decision_status in list(DecisionStatus)

        # Model probability in [0, 1]
        if result.model_probability is not None:
            assert 0.0 <= result.model_probability <= 1.0

        # Responsible gambling notice always present
        assert result.responsible_gambling_notice


class TestResponsibleUseGuard:
    def test_notice_injected(self, standard_1x2_bundle, baseline_prediction):
        margin_agent = MarginRemovalAgent()
        bundle, _, _ = margin_agent.apply(standard_1x2_bundle, "shin")
        mp_agent = MarketProbabilityAgent()
        model_prob, _ = mp_agent.get_model_probability(baseline_prediction, bundle, "Home")
        target = next(r for r in bundle.odds if r.selection == "Home")
        va_agent = ValueAssessmentAgent()
        result = va_agent.assess(
            home_team="Arsenal", away_team="Chelsea",
            competition="PL", kickoff=KICKOFF, status=MatchStatus.PRE_MATCH,
            bundle=bundle, odds_record=target,
            prediction=baseline_prediction, model_probability=model_prob,
        )
        guard = ResponsibleUseGuardAgent()
        guard_result = guard.check(result)
        assert guard_result.modified_result.responsible_gambling_notice

    def test_forbidden_language_detected(self):
        guard = ResponsibleUseGuardAgent()
        violations = guard.check_text("This is a guaranteed win, a sure bet!")
        assert len(violations) > 0

    def test_clean_text_passes(self):
        guard = ResponsibleUseGuardAgent()
        violations = guard.check_text("The model probability is 57.4% with medium confidence.")
        assert len(violations) == 0


class TestOutput:
    def test_formatter_produces_string(self, standard_1x2_bundle, baseline_prediction):
        margin_agent = MarginRemovalAgent()
        bundle, _, _ = margin_agent.apply(standard_1x2_bundle, "shin")
        mp_agent = MarketProbabilityAgent()
        model_prob, _ = mp_agent.get_model_probability(baseline_prediction, bundle, "Home")
        target = next(r for r in bundle.odds if r.selection == "Home")
        va_agent = ValueAssessmentAgent()
        result = va_agent.assess(
            home_team="Arsenal", away_team="Chelsea",
            competition="PL", kickoff=KICKOFF, status=MatchStatus.PRE_MATCH,
            bundle=bundle, odds_record=target,
            prediction=baseline_prediction, model_probability=model_prob,
        )
        report = format_report(result)
        assert "Arsenal" in report
        assert "RESPONSIBLE GAMBLING" in report
        assert "DECISION" in report

    def test_json_contract_valid(self, standard_1x2_bundle, baseline_prediction):
        margin_agent = MarginRemovalAgent()
        bundle, _, _ = margin_agent.apply(standard_1x2_bundle, "shin")
        mp_agent = MarketProbabilityAgent()
        model_prob, _ = mp_agent.get_model_probability(baseline_prediction, bundle, "Home")
        target = next(r for r in bundle.odds if r.selection == "Home")
        va_agent = ValueAssessmentAgent()
        result = va_agent.assess(
            home_team="Arsenal", away_team="Chelsea",
            competition="PL", kickoff=KICKOFF, status=MatchStatus.PRE_MATCH,
            bundle=bundle, odds_record=target,
            prediction=baseline_prediction, model_probability=model_prob,
        )
        json_out = AnalysisOutputJSON.from_result(result)
        d = json_out.to_dict()
        assert "event" in d
        assert "market" in d
        assert "prediction" in d
        assert "decision" in d
        assert "quality" in d
        assert "responsible_gambling_notice" in d
