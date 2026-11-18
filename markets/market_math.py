"""
markets/market_math.py
======================
Pure-function mathematics for bookmaker market analysis.

All functions are stateless and tested independently.  No side effects,
no I/O, no model dependencies.

References
----------
- Dixon & Coles (1997) for score distribution
- Shin (1993) for vig allocation
- Joseph Buchdahl "Squares & Sharps, Suckers & Sharks" for overround
"""

from __future__ import annotations

import math
from typing import Literal


# ── Raw implied probability ────────────────────────────────────────────────────

def raw_implied_probability(decimal_odds: float) -> float:
    """
    Convert decimal odds to raw (bookmaker-inflated) implied probability.

    p_raw = 1 / o

    Parameters
    ----------
    decimal_odds : float  Must be > 1.0

    Returns
    -------
    float  ∈ (0, 1)
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


# ── Overround / margin ────────────────────────────────────────────────────────

def overround(odds_list: list[float]) -> float:
    """
    Calculate bookmaker overround (margin) for a mutually exclusive market.

    overround = Σ(1/o_i) − 1

    A value of 0.05 means 5% bookmaker margin.

    Parameters
    ----------
    odds_list : list of decimal odds for all outcomes in the market

    Returns
    -------
    float  ≥ 0 (negative overround would imply an arbitrage opportunity)
    """
    if not odds_list:
        raise ValueError("odds_list must not be empty")
    return sum(raw_implied_probability(o) for o in odds_list) - 1.0


def overround_pct(odds_list: list[float]) -> float:
    """Return overround expressed as a percentage (e.g., 5.0 for 5%)."""
    return overround(odds_list) * 100.0


# ── Margin removal methods ────────────────────────────────────────────────────

def normalize_proportional(odds_list: list[float]) -> list[float]:
    """
    Proportional margin removal (simplest method).

    p_fair_i = p_raw_i / Σ(p_raw_j)

    Distributes the bookmaker's margin proportionally to each raw
    implied probability.  Slightly favours the favourite.

    Returns
    -------
    list of fair probabilities summing to 1.0
    """
    raw = [raw_implied_probability(o) for o in odds_list]
    total = sum(raw)
    return [p / total for p in raw]


def normalize_additive(odds_list: list[float]) -> list[float]:
    """
    Additive margin removal.

    Subtracts an equal absolute amount from each raw probability so
    they sum to 1.  More appropriate when the bookmaker margin is
    distributed evenly (e.g., correct-score markets).

    p_fair_i = p_raw_i − overround / n
    """
    raw = [raw_implied_probability(o) for o in odds_list]
    n = len(raw)
    vig = overround(odds_list)
    adjusted = [p - vig / n for p in raw]
    # Clamp to [0, 1]
    adjusted = [max(0.0, min(1.0, p)) for p in adjusted]
    total = sum(adjusted)
    return [p / total for p in adjusted]


def normalize_shin(odds_list: list[float]) -> list[float]:
    """
    Shin (1993) margin removal.

    The Shin model assumes that some fraction z of bettors have inside
    information.  It produces fairer implied probabilities than
    proportional normalisation, especially for longer-odds outcomes.

    Algorithm
    ---------
    1. Let r_i = 1/o_i (raw probs).
    2. Solve for z via Newton-Raphson:
           Σ sqrt(z² + 4(1−z) r_i²) = 2 − z(n−1) + (n-1)z
       (simplified closed-form approximation used here)
    3. Fair prob: p_i = (sqrt(z² + 4(1−z) r_i) − z) / (2(1−z))

    We use an approximate solution that is accurate for typical sports
    market overrounds (≤ 15%).

    Returns
    -------
    list of Shin-adjusted fair probabilities summing to 1.0
    """
    raw = [raw_implied_probability(o) for o in odds_list]
    vig = overround(odds_list)          # ≈ z for small overrounds

    # Iterative solution for z
    z = vig / (1.0 + vig)
    for _ in range(20):
        numerators = [math.sqrt(z * z + 4 * (1 - z) * p) - z for p in raw]
        total = sum(numerators)
        denom = 2 * (1 - z)
        probs = [n_ / denom for n_ in numerators]
        # Re-estimate z
        new_z = (sum(probs) - 1.0) / (sum(probs))
        if abs(new_z - z) < 1e-8:
            break
        z = new_z

    # Final fair probs
    denom = 2 * (1 - z) if z < 1 else 1e-9
    fair = [(math.sqrt(z * z + 4 * (1 - z) * p) - z) / denom for p in raw]
    total = sum(fair)
    return [p / total for p in fair]


def normalize_power(odds_list: list[float]) -> list[float]:
    """
    Power / multiplicative margin removal.

    Find exponent k such that Σ (p_raw_i)^(1/k) = 1.
    Uses binary search.

    Returns
    -------
    list of power-normalised fair probabilities summing to 1.0
    """
    raw = [raw_implied_probability(o) for o in odds_list]

    def _residual(k: float) -> float:
        return sum(p ** (1.0 / k) for p in raw) - 1.0

    lo, hi = 0.5, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _residual(mid) > 0:
            lo = mid
        else:
            hi = mid

    k = (lo + hi) / 2.0
    fair = [p ** (1.0 / k) for p in raw]
    total = sum(fair)
    return [p / total for p in fair]


MARGIN_METHODS = {
    "proportional": normalize_proportional,
    "additive": normalize_additive,
    "shin": normalize_shin,
    "power": normalize_power,
}


def remove_margin(
    odds_list: list[float],
    method: Literal["proportional", "additive", "shin", "power"] = "shin",
) -> tuple[list[float], str]:
    """
    Apply the chosen margin-removal method.

    Returns
    -------
    (fair_probabilities, method_used)
    """
    if method not in MARGIN_METHODS:
        raise ValueError(f"Unknown margin method '{method}'. Choose from {list(MARGIN_METHODS)}")
    return MARGIN_METHODS[method](odds_list), method


# ── Fair odds ─────────────────────────────────────────────────────────────────

def fair_odds(p_model: float) -> float:
    """
    Convert model probability to fair decimal odds (no margin).

    fair_odds = 1 / p_model

    Parameters
    ----------
    p_model : float  Model probability ∈ (0, 1)
    """
    if not (0.0 < p_model < 1.0):
        raise ValueError(f"Model probability must be in (0, 1), got {p_model}")
    return 1.0 / p_model


# ── Expected value ────────────────────────────────────────────────────────────

def expected_value(p_model: float, decimal_odds: float) -> float:
    """
    Net expected value per unit stake.

    EV = p_model × (o − 1) − (1 − p_model)
       = p_model × o − 1

    A positive EV indicates a mathematically favourable bet at these odds,
    assuming the model probability is accurate.  This is NOT a guarantee
    of profit.

    Parameters
    ----------
    p_model      : model probability of the outcome
    decimal_odds : decimal odds offered by the bookmaker

    Returns
    -------
    float  (e.g., 0.06 means +6% per unit stake in expectation)
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"Odds must be > 1.0, got {decimal_odds}")
    if not (0.0 < p_model < 1.0):
        raise ValueError(f"Probability must be in (0, 1), got {p_model}")
    return p_model * decimal_odds - 1.0


def expected_return(p_model: float, decimal_odds: float) -> float:
    """
    Gross return per unit stake: p_model × o.
    Returns 1.0 at break-even.
    """
    return p_model * decimal_odds


# ── Edge ──────────────────────────────────────────────────────────────────────

def edge(p_model: float, p_fair_market: float) -> float:
    """
    Edge in probability percentage points.

    edge = p_model − p_fair_market

    Positive edge means the model considers the outcome more likely than
    the market's margin-adjusted estimate.

    Parameters
    ----------
    p_model       : model probability
    p_fair_market : market fair probability (after margin removal)
    """
    return p_model - p_fair_market


def uncertainty_adjusted_edge(
    raw_edge: float,
    calibration_error: float = 0.0,
    interval_width: float = 0.0,
    model_disagreement: float = 0.0,
    staleness_penalty: float = 0.0,
    sample_size_penalty: float = 0.0,
) -> float:
    """
    Reduce the raw edge by penalties for uncertainty.

    adjusted_edge = raw_edge − Σ penalties

    Each penalty is non-negative.  If adjusted_edge ≤ 0, the selection
    should be NO_BET regardless of the raw edge sign.

    Parameters (all ∈ [0, 1], interpreted as probability-point penalties)
    ----------
    calibration_error    : ECE or similar calibration metric
    interval_width       : (p_high − p_low) / 2  of the model interval
    model_disagreement   : 1 − agreement score across ensemble members
    staleness_penalty    : derived from freshness status
    sample_size_penalty  : 1 − min(n / 500, 1) for small samples
    """
    total_penalty = (
        calibration_error * 0.5
        + interval_width * 0.3
        + model_disagreement * 0.2
        + staleness_penalty * 0.4
        + sample_size_penalty * 0.2
    )
    return raw_edge - total_penalty


# ── Price alert threshold ─────────────────────────────────────────────────────

def price_alert_threshold(p_model: float, min_ev: float = 0.0) -> float:
    """
    Minimum decimal odds at which the bet retains the required EV.

    From EV = p × o − 1 ≥ min_ev:
        o_min = (1 + min_ev) / p_model

    Parameters
    ----------
    p_model : model probability
    min_ev  : minimum acceptable expected value per unit (default 0 = break-even)
    """
    if not (0.0 < p_model < 1.0):
        raise ValueError(f"p_model must be in (0, 1), got {p_model}")
    return (1.0 + min_ev) / p_model


# ── Asian handicap helpers ────────────────────────────────────────────────────

def split_quarter_line(line: float) -> tuple[float, float]:
    """
    Split a quarter-ball Asian handicap line into two half-ball lines.

    Examples
    --------
    -0.25 → (-0.0, -0.5)   # half stake on 0, half on -0.5
    +0.75 → (+0.5, +1.0)
    """
    lower = math.floor(line * 2) / 2
    upper = math.ceil(line * 2) / 2
    return lower, upper


def is_quarter_line(line: float) -> bool:
    """True if *line* is a quarter-ball handicap (e.g., -0.25, +0.75)."""
    return (line * 4) % 2 == 1


def asian_handicap_result(
    home_goals: int,
    away_goals: int,
    handicap: float,
    bet_on_home: bool,
) -> Literal["win", "half_win", "push", "half_loss", "loss"]:
    """
    Determine the settlement of an Asian handicap bet.

    Parameters
    ----------
    home_goals  : actual home goals
    away_goals  : actual away goals
    handicap    : the handicap applied to the home team (e.g., -1.5)
    bet_on_home : True if the bet is on the home team

    Returns
    -------
    Settlement: "win" | "half_win" | "push" | "half_loss" | "loss"
    """
    adjusted_home = home_goals + handicap
    diff = adjusted_home - away_goals   # from home team perspective

    if not bet_on_home:
        diff = -diff

    if diff > 0.5:
        return "win"
    elif diff == 0.5:
        return "half_win"
    elif diff == 0.0:
        return "push"
    elif diff == -0.5:
        return "half_loss"
    else:
        return "loss"


# ── Double-chance helper ──────────────────────────────────────────────────────

def double_chance_probability(
    p_home: float, p_draw: float, p_away: float, selection: str
) -> float:
    """
    Calculate double-chance probability from 1X2 probs.

    selection : "1X" | "X2" | "12"
    """
    if abs(p_home + p_draw + p_away - 1.0) > 0.02:
        raise ValueError("1X2 probs must sum to ~1.0")
    if selection == "1X":
        return p_home + p_draw
    elif selection == "X2":
        return p_draw + p_away
    elif selection == "12":
        return p_home + p_away
    else:
        raise ValueError(f"Invalid double-chance selection '{selection}'")


# ── Draw no bet helper ────────────────────────────────────────────────────────

def draw_no_bet_probability(
    p_win: float, p_draw: float, p_loss: float, bet_on_win: bool
) -> float:
    """
    Fair probability for a Draw No Bet market.

    The draw refunds the stake, so the effective market is binary:
    p_dnb_win = p_win / (p_win + p_loss)

    Parameters
    ----------
    p_win      : raw probability of the team winning
    p_draw     : raw probability of draw (voided outcome)
    p_loss     : raw probability of the team losing
    bet_on_win : True → return win probability; False → return loss prob
    """
    binary_sum = p_win + p_loss
    if binary_sum <= 0.0:
        raise ValueError("p_win + p_loss must be > 0")
    return (p_win if bet_on_win else p_loss) / binary_sum


# ── Data-quality score helpers ────────────────────────────────────────────────

def compute_data_quality_score(
    lineup_confirmed: bool,
    odds_freshness_ok: bool,
    stats_freshness_ok: bool,
    no_source_conflicts: bool,
    sample_size_ok: bool,
    calibration_ok: bool,
) -> float:
    """
    Simple weighted data-quality score in [0, 100].

    Weights reflect spec priority ordering.
    """
    weights = {
        "lineup_confirmed": 20,
        "odds_freshness": 25,
        "stats_freshness": 20,
        "no_conflicts": 15,
        "sample_size": 10,
        "calibration": 10,
    }
    score = 0.0
    score += weights["lineup_confirmed"] * lineup_confirmed
    score += weights["odds_freshness"] * odds_freshness_ok
    score += weights["stats_freshness"] * stats_freshness_ok
    score += weights["no_conflicts"] * no_source_conflicts
    score += weights["sample_size"] * sample_size_ok
    score += weights["calibration"] * calibration_ok
    return round(score, 1)
