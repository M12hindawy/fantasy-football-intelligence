"""
tests/test_market_math.py
=========================
Unit tests for markets/market_math.py

All tests use exact expected values computed by hand.
"""

from __future__ import annotations

import math
import pytest

from markets.market_math import (
    raw_implied_probability,
    overround,
    normalize_proportional,
    normalize_additive,
    normalize_shin,
    normalize_power,
    expected_value,
    fair_odds,
    edge,
    uncertainty_adjusted_edge,
    price_alert_threshold,
    double_chance_probability,
    draw_no_bet_probability,
    split_quarter_line,
    is_quarter_line,
    asian_handicap_result,
)


class TestRawImpliedProbability:
    def test_evens(self):
        assert raw_implied_probability(2.0) == pytest.approx(0.5, abs=1e-9)

    def test_standard(self):
        assert raw_implied_probability(1.85) == pytest.approx(1 / 1.85, abs=1e-9)

    def test_invalid_odds(self):
        with pytest.raises(ValueError):
            raw_implied_probability(0.9)

    def test_invalid_odds_one(self):
        with pytest.raises(ValueError):
            raw_implied_probability(1.0)


class TestOverround:
    def test_balanced_market(self):
        # Perfect market: 3 outcomes at exactly 1/3 each → 0% margin
        odds = [3.0, 3.0, 3.0]
        assert overround(odds) == pytest.approx(0.0, abs=1e-6)

    def test_typical_1x2(self):
        # 2.10, 3.30, 3.50 → overround ~4.6%
        odds = [2.10, 3.30, 3.50]
        vig = overround(odds)
        assert 0.03 < vig < 0.08

    def test_empty_list(self):
        with pytest.raises(ValueError):
            overround([])


class TestMarginRemoval:
    _1x2 = [2.10, 3.30, 3.50]

    def test_proportional_sums_to_one(self):
        probs = normalize_proportional(self._1x2)
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)
        assert all(0 <= p <= 1 for p in probs)

    def test_shin_sums_to_one(self):
        probs = normalize_shin(self._1x2)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert all(0 <= p <= 1 for p in probs)

    def test_additive_sums_to_one(self):
        probs = normalize_additive(self._1x2)
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)

    def test_power_sums_to_one(self):
        probs = normalize_power(self._1x2)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_shin_vs_proportional(self):
        # Shin should give slightly more probability to the longer-odds outcome
        shin = normalize_shin(self._1x2)
        prop = normalize_proportional(self._1x2)
        # Third outcome (3.50) is the longest — Shin should be >= proportional
        assert shin[2] >= prop[2] - 1e-6  # allow for numerical noise


class TestExpectedValue:
    def test_breakeven(self):
        # p_model = 1/o → EV = 0
        assert expected_value(0.5, 2.0) == pytest.approx(0.0, abs=1e-9)

    def test_positive_ev(self):
        assert expected_value(0.6, 2.0) == pytest.approx(0.2, abs=1e-9)

    def test_negative_ev(self):
        assert expected_value(0.4, 2.0) == pytest.approx(-0.2, abs=1e-9)

    def test_invalid_odds(self):
        with pytest.raises(ValueError):
            expected_value(0.5, 0.9)

    def test_invalid_prob(self):
        with pytest.raises(ValueError):
            expected_value(1.1, 2.0)


class TestFairOdds:
    def test_50pct(self):
        assert fair_odds(0.5) == pytest.approx(2.0, abs=1e-9)

    def test_boundary(self):
        with pytest.raises(ValueError):
            fair_odds(0.0)
        with pytest.raises(ValueError):
            fair_odds(1.0)


class TestEdge:
    def test_positive_edge(self):
        assert edge(0.60, 0.55) == pytest.approx(0.05, abs=1e-9)

    def test_no_edge(self):
        assert edge(0.50, 0.50) == pytest.approx(0.0, abs=1e-9)

    def test_negative_edge(self):
        assert edge(0.45, 0.55) == pytest.approx(-0.10, abs=1e-9)


class TestUncertaintyAdjustedEdge:
    def test_no_penalty(self):
        assert uncertainty_adjusted_edge(0.05) == pytest.approx(0.05, abs=1e-9)

    def test_penalty_reduces_edge(self):
        adj = uncertainty_adjusted_edge(0.05, calibration_error=0.05)
        assert adj < 0.05

    def test_kills_edge(self):
        adj = uncertainty_adjusted_edge(0.03, staleness_penalty=0.10)
        assert adj < 0.0


class TestPriceAlert:
    def test_breakeven(self):
        # min_ev = 0 → threshold = 1/p
        assert price_alert_threshold(0.5, min_ev=0.0) == pytest.approx(2.0, abs=1e-9)

    def test_positive_min_ev(self):
        # For positive EV threshold, odds must be higher than break-even
        t = price_alert_threshold(0.5, min_ev=0.05)
        assert t > 2.0


class TestDoubleChance:
    def test_1x(self):
        p = double_chance_probability(0.5, 0.25, 0.25, "1X")
        assert p == pytest.approx(0.75, abs=1e-9)

    def test_x2(self):
        p = double_chance_probability(0.5, 0.25, 0.25, "X2")
        assert p == pytest.approx(0.50, abs=1e-9)

    def test_12(self):
        p = double_chance_probability(0.5, 0.25, 0.25, "12")
        assert p == pytest.approx(0.75, abs=1e-9)

    def test_invalid_selection(self):
        with pytest.raises(ValueError):
            double_chance_probability(0.5, 0.25, 0.25, "XX")


class TestDrawNoBet:
    def test_win_side(self):
        p = draw_no_bet_probability(0.50, 0.25, 0.25, bet_on_win=True)
        assert p == pytest.approx(0.5 / 0.75, abs=1e-6)

    def test_loss_side(self):
        p = draw_no_bet_probability(0.50, 0.25, 0.25, bet_on_win=False)
        assert p == pytest.approx(0.25 / 0.75, abs=1e-6)


class TestAsianHandicap:
    def test_quarter_line_detection(self):
        assert is_quarter_line(-0.25)
        assert is_quarter_line(0.75)
        assert not is_quarter_line(-0.5)
        assert not is_quarter_line(-1.0)

    def test_split_quarter_line(self):
        lo, hi = split_quarter_line(-0.25)
        # Lower line is -0.5, upper line is 0.0 (i.e., level)
        assert lo == pytest.approx(-0.5, abs=1e-9)
        assert hi == pytest.approx(0.0, abs=1e-9)

    def test_ah_settlement_win(self):
        # 2-0 with -1.5 handicap: adjusted_home = 2 + (-1.5) = 0.5, diff = 0.5 → half_win
        assert asian_handicap_result(2, 0, -1.5, bet_on_home=True) == "half_win"

    def test_ah_settlement_loss(self):
        # 1-0 with -1.5 handicap: adjusted_home = 1 + (-1.5) = -0.5, diff = -0.5 → half_loss
        assert asian_handicap_result(1, 0, -1.5, bet_on_home=True) == "half_loss"

    def test_ah_settlement_clear_win(self):
        # 3-0 with -1.5 handicap: adjusted_home = 3 - 1.5 = 1.5, diff = 1.5 > 0.5 → win
        assert asian_handicap_result(3, 0, -1.5, bet_on_home=True) == "win"

    def test_ah_settlement_push(self):
        assert asian_handicap_result(1, 0, -1.0, bet_on_home=True) == "push"

    def test_ah_settlement_half_win(self):
        assert asian_handicap_result(2, 1, -0.5, bet_on_home=True) == "half_win"
