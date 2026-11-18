"""
tests/test_v3_components.py
============================
v3 test suite — covers DQ engine, team features, Monte Carlo,
player projections, FDR, season simulator, availability, and fantasy optimizer.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta


# ─── Data Quality Engine ──────────────────────────────────────────────────────

class TestDataQualityEngine:
    def setup_method(self):
        from data.quality.data_quality_engine import DataQualityEngine
        self.engine = DataQualityEngine()

    def test_perfect_score(self):
        r = self.engine.assess(
            lineup_confirmed=True, odds_age_seconds=60.0,
            stats_age_seconds=3600.0, sample_size=30,
            has_source_conflict=False, injury_source="official",
            provider_name="football-data.org", calibration_ece=0.04,
        )
        assert r.score > 0.85
        assert r.grade == "A"
        assert r.passes_threshold

    def test_stale_odds_reduces_score(self):
        r = self.engine.assess(
            lineup_confirmed=True, odds_age_seconds=9000.0,  # >2h
            stats_age_seconds=3600.0, sample_size=30,
            provider_name="football-data.org", calibration_ece=0.04,
        )
        assert r.odds_freshness_ok is False
        assert r.score < 0.90

    def test_missing_fields_penalty(self):
        r = self.engine.assess(
            lineup_confirmed=False, sample_size=5,
            missing_fields=["home_xg", "away_xg", "away_goals"],
        )
        assert r.score < 0.60

    def test_grade_f_for_very_low_score(self):
        r = self.engine.assess(
            lineup_confirmed=False, odds_age_seconds=99999,
            stats_age_seconds=99999, sample_size=0,
            has_source_conflict=True, injury_source="unknown",
            calibration_ece=0.25, missing_fields=["a","b","c","d","e"],
        )
        assert r.grade in ("D", "F")


# ─── Team Features ────────────────────────────────────────────────────────────

class TestTeamFeatures:
    def _make_records(self, n=10, home_id="H", away_id="A"):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        records = []
        for i in range(n):
            records.append({
                "home_id": home_id, "away_id": away_id,
                "home_goals": 2, "away_goals": 1,
                "date": base + timedelta(days=i * 7),
                "home_xg": 1.8, "away_xg": 0.9,
            })
        return records

    def test_form_computed(self):
        from features.team_features import TeamFeatureExtractor
        ex = TeamFeatureExtractor()
        records = self._make_records(10)
        as_of = datetime(2025, 4, 1, tzinfo=timezone.utc)
        feat = ex.extract("H", records, as_of)
        assert feat.form_w3 == pytest.approx(3.0)   # all wins
        assert feat.goals_scored_per_game == pytest.approx(2.0)
        assert feat.xg_per_game == pytest.approx(1.8)

    def test_no_future_leakage(self):
        from features.team_features import TeamFeatureExtractor
        ex = TeamFeatureExtractor()
        records = self._make_records(5)
        as_of = records[2]["date"]  # midway
        feat = ex.extract("H", records, as_of)
        assert feat.sample_size == 2  # only past matches


# ─── Monte Carlo Engine ───────────────────────────────────────────────────────

class TestMonteCarloEngine:
    def setup_method(self):
        from simulation.monte_carlo import MonteCarloEngine
        self.mc = MonteCarloEngine(n_simulations=5000, seed=42)

    def test_probabilities_sum_to_one(self):
        r = self.mc.simulate_match(1.8, 1.1)
        total = r.p_home_win + r.p_draw + r.p_away_win
        assert abs(total - 1.0) < 0.001

    def test_strong_team_more_likely_to_win(self):
        r = self.mc.simulate_match(mu_home=2.5, mu_away=0.5)
        assert r.p_home_win > 0.70

    def test_analytic_vs_simulation_close(self):
        r = self.mc.simulate_match(1.5, 1.2)
        assert self.mc.validate_against_analytic(r, tolerance=0.03)

    def test_btts_reasonable(self):
        r = self.mc.simulate_match(1.6, 1.3)
        assert 0.30 < r.p_btts < 0.75

    def test_as_dict_keys_present(self):
        r = self.mc.simulate_match(1.5, 1.0)
        d = r.as_dict()
        for key in ("p_home_win", "p_draw", "p_away_win", "p_over_25", "p_btts", "top_scorelines"):
            assert key in d


# ─── Player Projection ────────────────────────────────────────────────────────

class TestPlayerProjection:
    def _make_features(self, xg90=0.3, xa90=0.2, start_pct=0.9):
        from features.player_features import PlayerFeatures
        f = PlayerFeatures(player_id="P1")
        f.xg_90 = xg90; f.xa_90 = xa90
        f.start_pct = start_pct; f.subbed_on_pct = 0.05
        f.subbed_off_pct = 0.2; f.minutes_played = 810
        return f

    def _avail(self, prob=1.0):
        from agents.availability_agent import AvailabilityResult
        return AvailabilityResult("P1", "AVAILABLE", prob)

    def test_nailed_player_gets_points(self):
        from models.player_projection import PlayerProjectionModel
        model = PlayerProjectionModel()
        proj = model.project(self._make_features(), "FWD", self._avail())
        assert proj.expected_points > 3.0
        assert proj.is_nailed

    def test_suspended_player_zero_points(self):
        from models.player_projection import PlayerProjectionModel
        from agents.availability_agent import AvailabilityResult
        model = PlayerProjectionModel()
        avail = AvailabilityResult("P1", "SUSPENDED", 0.0)
        proj = model.project(self._make_features(), "FWD", avail)
        assert proj.expected_points == pytest.approx(0.0, abs=0.01)

    def test_gk_cs_bonus(self):
        from models.player_projection import PlayerProjectionModel
        model = PlayerProjectionModel()
        proj = model.project(self._make_features(xg90=0.0, xa90=0.0), "GK", self._avail(),
                             team_p_clean_sheet=0.80)
        assert proj.expected_clean_sheet_prob > 0.3
        assert proj.expected_points > 2.0

    def test_ceiling_above_expected(self):
        from models.player_projection import PlayerProjectionModel
        model = PlayerProjectionModel()
        proj = model.project(self._make_features(), "MID", self._avail())
        assert proj.ceiling >= proj.expected_points


# ─── Fixture Difficulty Rater ─────────────────────────────────────────────────

class TestFixtureDifficultyRater:
    def setup_method(self):
        from models.fixture_difficulty import FixtureDifficultyRater
        self.rater = FixtureDifficultyRater()

    def test_strong_home_vs_weak_easy(self):
        r = self.rater.rate("H", 1800, "A", 1400, is_home=True)
        assert r.difficulty <= 2

    def test_weak_away_vs_strong_hard(self):
        r = self.rater.rate("H", 1400, "A", 1800, is_home=False)
        assert r.difficulty >= 4

    def test_evenly_matched_medium(self):
        r = self.rater.rate("H", 1550, "A", 1500, is_home=True)
        assert r.difficulty in (2, 3)

    def test_output_dict_keys(self):
        r = self.rater.rate("H", 1500, "A", 1500, is_home=True)
        d = r.as_dict()
        assert "difficulty" in d and "difficulty_label" in d


# ─── Season Simulator ─────────────────────────────────────────────────────────

class TestSeasonSimulator:
    def _make_standings(self):
        from simulation.season_simulator import TeamSeasonContext
        return [
            TeamSeasonContext("T1", 60, 20, 60, 0.3, -0.1),
            TeamSeasonContext("T2", 55, 15, 55, 0.2, -0.1),
            TeamSeasonContext("T3", 30, -10, 35, 0.0, 0.1),
            TeamSeasonContext("T4", 25, -25, 28, -0.1, 0.2),
        ]

    def test_ranks_sum_correctly(self):
        from simulation.season_simulator import SeasonSimulator
        sim = SeasonSimulator(n_simulations=200)
        standings = self._make_standings()
        fixtures = [("T1","T3"), ("T2","T4"), ("T3","T1"), ("T4","T2")]
        results = sim.simulate_season(standings, fixtures)
        for tid, ranks in results.items():
            assert len(ranks) == 200
            for r in ranks:
                assert 1 <= r <= 4

    def test_strong_team_usually_finishes_top(self):
        from simulation.season_simulator import SeasonSimulator, SeasonOutcomesEngine
        sim = SeasonSimulator(n_simulations=500)
        standings = self._make_standings()
        fixtures = [("T1","T3"), ("T2","T4")]
        ranks = sim.simulate_season(standings, fixtures)
        engine = SeasonOutcomesEngine()
        outcomes = engine.analyze(ranks, total_teams=4, relegation_spots=1)
        assert outcomes["T1"].title_prob > outcomes["T3"].title_prob


# ─── Availability Agent ───────────────────────────────────────────────────────

class TestAvailabilityAgent:
    def setup_method(self):
        from agents.availability_agent import AvailabilityAgent
        self.agent = AvailabilityAgent()

    def test_no_issues_available(self):
        r = self.agent.assess("P1", [], [])
        assert r.status == "AVAILABLE"
        assert r.probability_of_playing == 1.0

    def test_injured_player(self):
        r = self.agent.assess("P1", [{"player_id": "P1", "status": "INJURED", "injury_type": "hamstring"}], [])
        assert r.status == "INJURED"
        assert r.probability_of_playing == 0.0

    def test_suspended_player(self):
        r = self.agent.assess("P1", [], [{"player_id": "P1", "reason": "yellow card accumulation"}])
        assert r.status == "SUSPENDED"
        assert r.probability_of_playing == 0.0

    def test_doubtful_player(self):
        r = self.agent.assess("P2", [{"player_id": "P2", "status": "DOUBTFUL",
                                       "probability_of_playing": 0.4, "injury_type": "knock"}], [])
        assert r.status == "DOUBTFUL"
        assert r.probability_of_playing == pytest.approx(0.4)
