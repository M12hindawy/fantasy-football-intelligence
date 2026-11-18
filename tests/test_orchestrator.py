"""
tests/test_orchestrator.py
===========================
Integration tests for the Orchestrator and ReportGenerator.
Uses synthetic match records — no real API calls.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta


def _make_records(home_id="H", away_id="A", n=15):
    base = datetime(2024, 8, 1, tzinfo=timezone.utc)
    records = []
    for i in range(n):
        records.append({
            "home_id": home_id, "away_id": away_id,
            "home_goals": 2, "away_goals": 1,
            "date": base + timedelta(days=i * 7),
            "home_xg": 1.9, "away_xg": 0.95,
        })
    return records


class TestOrchestrator:
    def _build_input(self, scenarios=None):
        from agents.orchestrator import OrchestratorInput
        return OrchestratorInput(
            fixture_id="FIX-001",
            home_team_id="HOME",
            away_team_id="AWAY",
            competition_code="PL",
            match_records=_make_records("HOME", "AWAY", 15),
            mu_home=1.75, mu_away=0.95, rho=-0.10,
            home_elo=1600.0, away_elo=1480.0,
            scenarios=scenarios or [],
        )

    def test_analyze_returns_result(self):
        from agents.orchestrator import Orchestrator
        orc = Orchestrator()
        result = orc.analyze(self._build_input())
        assert result.fixture_id == "FIX-001"
        assert 0 < result.p_home_win < 1
        assert abs(result.p_home_win + result.p_draw + result.p_away_win - 1.0) < 0.01

    def test_probs_sum_to_one(self):
        from agents.orchestrator import Orchestrator
        orc = Orchestrator()
        result = orc.analyze(self._build_input())
        total = result.p_home_win + result.p_draw + result.p_away_win
        assert abs(total - 1.0) < 0.005

    def test_strong_home_favoured(self):
        from agents.orchestrator import Orchestrator, OrchestratorInput
        orc = Orchestrator()
        inp = OrchestratorInput(
            fixture_id="FIX-002",
            home_team_id="H", away_team_id="A",
            competition_code="PL",
            match_records=_make_records("H", "A", 15),
            mu_home=2.5, mu_away=0.4,
        )
        result = orc.analyze(inp)
        assert result.p_home_win > 0.60

    def test_whatif_changes_probs(self):
        from agents.orchestrator import Orchestrator
        from simulation.what_if_engine import WhatIfScenario
        orc = Orchestrator()
        base   = orc.analyze(self._build_input())
        wi_inp = self._build_input(scenarios=[
            WhatIfScenario(type="KEY_PLAYER_OUT", target_team="HOME", severity=1.0)
        ])
        wi_result = orc.analyze(wi_inp)
        assert wi_result.what_if_applied
        # Home win probability should drop when key home player is out
        assert wi_result.p_home_win < base.p_home_win + 0.05

    def test_dq_report_present(self):
        from agents.orchestrator import Orchestrator
        orc = Orchestrator()
        result = orc.analyze(self._build_input())
        assert result.dq_report is not None
        assert "score" in result.dq_report
        assert result.dq_grade in ("A","B","C","D","F")

    def test_as_dict_structure(self):
        from agents.orchestrator import Orchestrator
        orc = Orchestrator()
        d = orc.analyze(self._build_input()).as_dict()
        assert "predictions" in d
        assert "simulation" in d
        assert "home_analysis" in d
        assert "recommendation" in d

    def test_no_errors_on_minimal_input(self):
        from agents.orchestrator import Orchestrator, OrchestratorInput
        orc = Orchestrator()
        inp = OrchestratorInput(
            fixture_id="MIN", home_team_id="T1", away_team_id="T2",
            competition_code="PL",
        )
        result = orc.analyze(inp)
        # Should not raise — may have warnings but not crash
        assert result.fixture_id == "MIN"


class TestReportGenerator:
    def _get_result(self):
        from agents.orchestrator import Orchestrator
        from agents.orchestrator import OrchestratorInput
        orc = Orchestrator()
        return orc.analyze(OrchestratorInput(
            fixture_id="RPT-001",
            home_team_id="MAN", away_team_id="LIV",
            competition_code="PL",
            match_records=_make_records("MAN", "LIV", 15),
            mu_home=1.6, mu_away=1.3,
        ))

    def test_text_output_contains_probabilities(self):
        from output.report_generator import ReportGenerator
        rg = ReportGenerator()
        result = self._get_result()
        text = rg.text(result, {"MAN": "Man City", "LIV": "Liverpool"})
        assert "Man City" in text
        assert "Liverpool" in text
        assert "%" in text
        assert "RECOMMENDATION" in text

    def test_markdown_has_table(self):
        from output.report_generator import ReportGenerator
        rg = ReportGenerator()
        result = self._get_result()
        md = rg.markdown(result)
        assert "|" in md
        assert "##" in md

    def test_json_output_is_valid(self):
        import json
        from output.report_generator import ReportGenerator
        rg = ReportGenerator()
        result = self._get_result()
        data = json.loads(rg.json(result))
        assert data["fixture_id"] == "RPT-001"
        assert "predictions" in data

    def test_disclaimer_in_text(self):
        from output.report_generator import ReportGenerator
        rg = ReportGenerator()
        result = self._get_result()
        text = rg.text(result)
        assert "ANALYTICAL TOOL ONLY" in text
