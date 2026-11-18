"""
agents/risk_and_stake_agent.py
==============================
Optional stake-sizing module — activated ONLY when the user explicitly
provides bankroll, currency, risk profile, and legal context.

Never infers bankroll from any account information.
Never recommends a stake that would materially endanger essential finances.
Always offers a "stake not required" alternative.

Supports:
- Flat stake (percentage of bankroll)
- Capped fractional Kelly
- Daily and per-event exposure limits
- Stop-loss reminders
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import settings
from data.schemas import (
    BankrollContext,
    ConfidenceLevel,
    DecisionStatus,
    MarketAnalysisResult,
    StakeMethod,
)

logger = logging.getLogger(__name__)

RESPONSIBLE_NOTICE = (
    "Betting involves financial risk. Only use money you can afford to lose, "
    "follow local laws and age requirements, and consider setting limits or "
    "stopping if gambling is becoming difficult to control."
)


@dataclass
class StakeRecommendation:
    method: StakeMethod
    stake_amount: Optional[float]
    stake_pct_of_bankroll: Optional[float]
    kelly_fraction_used: Optional[float]
    kelly_raw: Optional[float]
    kelly_capped: Optional[float]
    cap_applied: bool
    daily_exposure_remaining: Optional[float]
    reasoning: str
    warning: str = ""


class RiskAndStakeAgent:
    """
    Computes a stake recommendation given a MarketAnalysisResult and
    a BankrollContext.

    Parameters
    ----------
    kelly_fraction      : fractional Kelly (0.25 = quarter Kelly)
    max_stake_pct       : hard cap on single stake as fraction of bankroll
    max_daily_pct       : daily exposure cap as fraction of bankroll
    """

    def __init__(
        self,
        kelly_fraction: float = settings.kelly_fraction,
        max_stake_pct: float = settings.max_stake_pct_of_bankroll,
        max_daily_pct: float = settings.max_daily_exposure_pct,
    ) -> None:
        self._kelly_f = max(0.0, min(kelly_fraction, 1.0))
        self._max_stake_pct = max(0.001, min(max_stake_pct, 0.10))
        self._max_daily_pct = max(0.001, min(max_daily_pct, 0.20))

    def recommend(
        self,
        result: MarketAnalysisResult,
        bankroll_ctx: BankrollContext,
        today_stakes_total: float = 0.0,
    ) -> StakeRecommendation:
        """
        Produce a stake recommendation.

        Parameters
        ----------
        result           : MarketAnalysisResult from ValueAssessmentAgent
        bankroll_ctx     : user-provided bankroll context
        today_stakes_total : sum of stakes already placed today (same currency)

        Returns
        -------
        StakeRecommendation — always includes reasoning and a notice.
        """
        # ── Guard: only act on VALUE_CANDIDATE ───────────────────────────────
        if result.decision_status != DecisionStatus.VALUE_CANDIDATE:
            return StakeRecommendation(
                method=StakeMethod.NONE,
                stake_amount=None,
                stake_pct_of_bankroll=None,
                kelly_fraction_used=None,
                kelly_raw=None,
                kelly_capped=None,
                cap_applied=False,
                daily_exposure_remaining=None,
                reasoning=(
                    f"Decision is {result.decision_status.value} — "
                    "no stake recommended. 'No bet' is a valid result."
                ),
                warning=RESPONSIBLE_NOTICE,
            )

        # ── Guard: high uncertainty kills stake ──────────────────────────────
        if result.confidence == ConfidenceLevel.LOW:
            return StakeRecommendation(
                method=StakeMethod.NONE,
                stake_amount=None,
                stake_pct_of_bankroll=None,
                kelly_fraction_used=None,
                kelly_raw=None,
                kelly_capped=None,
                cap_applied=False,
                daily_exposure_remaining=None,
                reasoning=(
                    "Confidence is LOW — model uncertainty is too high to size a stake. "
                    "Setting stake to zero."
                ),
                warning=RESPONSIBLE_NOTICE,
            )

        bankroll = bankroll_ctx.bankroll
        daily_limit = bankroll * self._max_daily_pct
        exposure_used = today_stakes_total
        exposure_remaining = max(0.0, daily_limit - exposure_used)

        if exposure_remaining <= 0.0:
            return StakeRecommendation(
                method=StakeMethod.NONE,
                stake_amount=None,
                stake_pct_of_bankroll=None,
                kelly_fraction_used=None,
                kelly_raw=None,
                kelly_capped=None,
                cap_applied=False,
                daily_exposure_remaining=0.0,
                reasoning=f"Daily exposure limit reached ({self._max_daily_pct:.1%} of bankroll). No more stakes today.",
                warning=RESPONSIBLE_NOTICE,
            )

        risk_profile = bankroll_ctx.risk_profile.lower()

        # ── Risk-profile modifier ─────────────────────────────────────────────
        profile_multiplier = {"conservative": 0.5, "moderate": 1.0, "aggressive": 1.5}.get(
            risk_profile, 1.0
        )

        # ── Fractional Kelly ─────────────────────────────────────────────────
        p = result.model_probability
        o = result.decimal_odds

        if p is None or o is None:
            return StakeRecommendation(
                method=StakeMethod.NONE,
                stake_amount=None,
                stake_pct_of_bankroll=None,
                kelly_fraction_used=None,
                kelly_raw=None,
                kelly_capped=None,
                cap_applied=False,
                daily_exposure_remaining=round(exposure_remaining, 2),
                reasoning="Model probability or odds unavailable. Cannot size stake.",
                warning=RESPONSIBLE_NOTICE,
            )

        # Kelly fraction: f = (p*(o-1) - (1-p)) / (o-1)
        o_net = o - 1.0
        if o_net <= 0:
            kelly_raw = 0.0
        else:
            kelly_raw = (p * o_net - (1.0 - p)) / o_net

        kelly_raw = max(0.0, kelly_raw)

        # Apply fractional Kelly
        kelly_fractional = kelly_raw * self._kelly_f * profile_multiplier

        # Apply hard cap
        max_kelly = self._max_stake_pct
        cap_applied = kelly_fractional > max_kelly
        kelly_capped = min(kelly_fractional, max_kelly)

        stake_amount = round(kelly_capped * bankroll, 2)
        stake_amount = min(stake_amount, exposure_remaining)

        # ── Flat alternative ──────────────────────────────────────────────────
        flat_pct = self._max_stake_pct * 0.5 * profile_multiplier
        flat_stake = round(bankroll * flat_pct, 2)

        reasoning = (
            f"Kelly formula: f_raw = {kelly_raw:.4f}, "
            f"fractional({self._kelly_f}) = {kelly_fractional:.4f}, "
            f"capped at {max_kelly:.2%} = {kelly_capped:.4f} × {bankroll_ctx.currency} {bankroll:.2f} "
            f"= {bankroll_ctx.currency} {stake_amount:.2f}. "
            f"Risk profile: {risk_profile} (×{profile_multiplier}). "
            f"Alternative: flat {flat_pct:.2%} = {bankroll_ctx.currency} {flat_stake:.2f}. "
            "Remember: a stake is NOT required. NO_BET is always valid."
        )

        return StakeRecommendation(
            method=StakeMethod.FRACTIONAL_KELLY_CAPPED,
            stake_amount=stake_amount,
            stake_pct_of_bankroll=round(kelly_capped, 4),
            kelly_fraction_used=self._kelly_f,
            kelly_raw=round(kelly_raw, 4),
            kelly_capped=round(kelly_capped, 4),
            cap_applied=cap_applied,
            daily_exposure_remaining=round(exposure_remaining - stake_amount, 2),
            reasoning=reasoning,
            warning=RESPONSIBLE_NOTICE,
        )

    def apply_to_result(
        self,
        result: MarketAnalysisResult,
        stake: StakeRecommendation,
    ) -> MarketAnalysisResult:
        """Attach stake recommendation to the analysis result (mutates a copy)."""
        import copy
        r = copy.copy(result)
        r.stake = stake.stake_amount
        r.stake_method = stake.method
        return r
