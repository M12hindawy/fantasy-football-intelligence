"""
agents/value_assessment_agent.py
=================================
Core decision engine.

For every market selection, this agent:
1. Computes edge, adjusted edge, and EV from model and market inputs
2. Applies the uncertainty penalty
3. Emits one of: NO_BET | LEAN | VALUE_CANDIDATE |
                 PASS_PENDING_LINEUP | LIVE_UNAVAILABLE
4. Sets a price-alert threshold
5. Never forces a recommendation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from config.settings import settings
from data.schemas import (
    ConfidenceLevel,
    DecisionStatus,
    MarketAnalysisResult,
    MarketBundle,
    MarketType,
    MatchStatus,
    OddsRecord,
    PredictionOutput,
    RiskLevel,
    StakeMethod,
)
from markets.market_math import (
    compute_data_quality_score,
    expected_value,
    fair_odds,
    price_alert_threshold,
    uncertainty_adjusted_edge,
)

logger = logging.getLogger(__name__)


class ValueAssessmentAgent:
    """
    Produces a MarketAnalysisResult for a specific selection.

    Parameters
    ----------
    min_edge        : minimum adjusted edge for VALUE_CANDIDATE (pp)
    min_ev          : minimum EV per unit for VALUE_CANDIDATE
    """

    def __init__(
        self,
        min_edge: float = settings.min_edge_pct_for_value_candidate,
        min_ev: float = settings.min_ev_for_value_candidate,
    ) -> None:
        self._min_edge = min_edge
        self._min_ev = min_ev

    def assess(
        self,
        *,
        home_team: str,
        away_team: str,
        competition: str,
        kickoff: datetime,
        status: MatchStatus,
        bundle: MarketBundle,
        odds_record: OddsRecord,
        prediction: Optional[PredictionOutput],
        model_probability: Optional[float],
        model_agreement: Optional[float] = None,
        calibration_ece: Optional[float] = None,
        data_quality_signals: Optional[dict] = None,
        lineup_confirmed: bool = False,
        key_reasons: Optional[list[str]] = None,
        main_risks: Optional[list[str]] = None,
    ) -> MarketAnalysisResult:
        """
        Produce the full decision output for one selection.

        Returns
        -------
        MarketAnalysisResult
        """
        now = datetime.now(timezone.utc)
        dqs = data_quality_signals or {}
        reasons = list(key_reasons or [])
        risks = list(main_risks or [])

        # ── Penalties ─────────────────────────────────────────────────────────
        from agents.odds_ingestion_agent import OddsIngestionAgent
        staleness_penalty = OddsIngestionAgent.freshness_penalty(bundle)

        calibration_error = min(calibration_ece or 0.0, 0.20)
        disagreement = 1.0 - (model_agreement or 1.0)
        sample_size_penalty = max(0.0, 1.0 - (dqs.get("sample_size", 0) or 200) / 500) if not dqs.get("sample_size_ok", True) else 0.0

        # ── Rapid path: LIVE_UNAVAILABLE ──────────────────────────────────────
        if odds_record.market_type == MarketType.LIVE or status == MatchStatus.LIVE:
            return self._build_result(
                home_team, away_team, competition, kickoff, status,
                bundle, odds_record, prediction,
                model_probability=None,
                decision=DecisionStatus.LIVE_UNAVAILABLE,
                risk=RiskLevel.VERY_HIGH,
                reasons=["Live market analysis requires a real-time feed. Not available."],
                risks=["Live odds change second-by-second; stale probabilities are worthless."],
                data_quality=0.0,
                confidence=ConfidenceLevel.LOW,
                now=now,
            )

        # ── Rapid path: PASS_PENDING_LINEUP ──────────────────────────────────
        if odds_record.market_type == MarketType.PLAYER_PROP and not lineup_confirmed:
            return self._build_result(
                home_team, away_team, competition, kickoff, status,
                bundle, odds_record, prediction,
                model_probability=None,
                decision=DecisionStatus.PASS_PENDING_LINEUP,
                risk=RiskLevel.HIGH,
                reasons=["Player prop requires confirmed lineup and expected minutes."],
                risks=["Player may not start; void bet or zero-probability selection."],
                data_quality=40.0,
                confidence=ConfidenceLevel.LOW,
                now=now,
            )

        # ── No model probability ──────────────────────────────────────────────
        if model_probability is None:
            return self._build_result(
                home_team, away_team, competition, kickoff, status,
                bundle, odds_record, prediction,
                model_probability=None,
                decision=DecisionStatus.NO_BET,
                risk=RiskLevel.VERY_HIGH,
                reasons=["Model probability could not be computed for this selection."],
                risks=["Insufficient model coverage for this market type."],
                data_quality=20.0,
                confidence=ConfidenceLevel.LOW,
                now=now,
            )

        # ── Market fair probability ───────────────────────────────────────────
        fair_market_prob = odds_record.__dict__.get("_fair_probability")
        raw_implied_prob = 1.0 / odds_record.decimal_odds

        if fair_market_prob is None:
            risks.append("Margin removal may not have been applied — using raw implied probability.")
            fair_market_prob = raw_implied_prob

        # ── Core mathematics ──────────────────────────────────────────────────
        raw_edge = model_probability - fair_market_prob
        adj_edge = uncertainty_adjusted_edge(
            raw_edge=raw_edge,
            calibration_error=calibration_error,
            interval_width=0.0,  # Could use p_interval here
            model_disagreement=disagreement,
            staleness_penalty=staleness_penalty,
            sample_size_penalty=sample_size_penalty,
        )

        ev = expected_value(model_probability, odds_record.decimal_odds)
        fair_price = fair_odds(model_probability)
        alert_threshold = price_alert_threshold(
            model_probability, self._min_ev
        )

        # ── Data quality score ────────────────────────────────────────────────
        dq_score = compute_data_quality_score(
            lineup_confirmed=dqs.get("lineup_confirmed", lineup_confirmed),
            odds_freshness_ok=dqs.get("odds_freshness_ok", True),
            stats_freshness_ok=dqs.get("stats_freshness_ok", True),
            no_source_conflicts=dqs.get("no_source_conflicts", True),
            sample_size_ok=dqs.get("sample_size_ok", True),
            calibration_ok=(calibration_ece is not None and calibration_ece < 0.07),
        )

        # ── Confidence level ──────────────────────────────────────────────────
        confidence = self._compute_confidence(
            adj_edge, dq_score, calibration_error, model_agreement
        )

        # ── Decision logic ────────────────────────────────────────────────────
        decision = self._make_decision(
            adj_edge, ev, dq_score, lineup_confirmed,
            odds_record.market_type, reasons, risks,
        )

        # ── Risk level ────────────────────────────────────────────────────────
        risk = self._compute_risk(adj_edge, confidence, dq_score)

        # ── Recommendation strength ───────────────────────────────────────────
        rec_strength = self._recommendation_strength(
            decision, adj_edge, dq_score, confidence
        )

        return self._build_result(
            home_team, away_team, competition, kickoff, status,
            bundle, odds_record, prediction,
            model_probability=round(model_probability, 6),
            decision=decision,
            risk=risk,
            reasons=reasons,
            risks=risks,
            data_quality=dq_score,
            confidence=confidence,
            now=now,
            raw_implied_prob=raw_implied_prob,
            fair_market_prob=fair_market_prob,
            overround=bundle.overround,
            edge=raw_edge,
            adj_edge=adj_edge,
            ev=ev,
            fair_price=fair_price,
            alert_threshold=alert_threshold,
            rec_strength=rec_strength,
            model_agreement=model_agreement,
            calibration_ece=calibration_ece,
        )

    # ── Decision helpers ──────────────────────────────────────────────────────

    def _make_decision(
        self,
        adj_edge: float,
        ev: float,
        dq_score: float,
        lineup_confirmed: bool,
        market_type: MarketType,
        reasons: list[str],
        risks: list[str],
    ) -> DecisionStatus:
        # Check data quality floor
        if dq_score < 40.0:
            reasons.append(f"Data quality too low ({dq_score:.0f}/100) for a recommendation.")
            risks.append("Unreliable data makes any edge estimate meaningless.")
            return DecisionStatus.NO_BET

        if adj_edge <= 0.0:
            if adj_edge > -0.02:
                reasons.append(f"Marginal adjusted edge ({adj_edge:.1%}) — not sufficient for value.")
                return DecisionStatus.LEAN
            reasons.append(f"Negative adjusted edge ({adj_edge:.1%}). The market has the better of this selection.")
            return DecisionStatus.NO_BET

        if adj_edge >= self._min_edge and ev >= self._min_ev:
            reasons.append(
                f"Positive uncertainty-adjusted edge ({adj_edge:.1%}) above threshold "
                f"({self._min_edge:.1%}) with EV {ev:+.2%}."
            )
            if not lineup_confirmed:
                risks.append("Lineups unconfirmed — recheck before placing this selection.")
            return DecisionStatus.VALUE_CANDIDATE

        if adj_edge > 0:
            reasons.append(
                f"Edge is positive ({adj_edge:.1%}) but below the minimum threshold "
                f"({self._min_edge:.1%}) or EV threshold ({self._min_ev:.1%}). "
                "Directional preference only."
            )
            return DecisionStatus.LEAN

        return DecisionStatus.NO_BET

    @staticmethod
    def _compute_confidence(
        adj_edge: float,
        dq_score: float,
        calibration_error: float,
        model_agreement: Optional[float],
    ) -> ConfidenceLevel:
        score = 0
        score += 2 if dq_score >= 80 else (1 if dq_score >= 55 else 0)
        score += 1 if calibration_error < 0.05 else 0
        score += 1 if (model_agreement or 0.0) > 0.85 else 0
        score += 1 if abs(adj_edge) > 0.04 else 0

        if score >= 4:
            return ConfidenceLevel.HIGH
        elif score >= 2:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @staticmethod
    def _compute_risk(
        adj_edge: float,
        confidence: ConfidenceLevel,
        dq_score: float,
    ) -> RiskLevel:
        if confidence == ConfidenceLevel.LOW or dq_score < 50:
            return RiskLevel.VERY_HIGH
        if confidence == ConfidenceLevel.MEDIUM and adj_edge < 0.04:
            return RiskLevel.HIGH
        if confidence == ConfidenceLevel.HIGH and adj_edge >= 0.05:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    @staticmethod
    def _recommendation_strength(
        decision: DecisionStatus,
        adj_edge: float,
        dq_score: float,
        confidence: ConfidenceLevel,
    ) -> Optional[float]:
        if decision == DecisionStatus.NO_BET:
            return None
        if decision == DecisionStatus.LEAN:
            return round(min(adj_edge / 0.05 * 0.4, 0.4), 3)
        # VALUE_CANDIDATE
        edge_score = min(adj_edge / 0.10, 1.0) * 0.5
        dq_score_norm = (dq_score / 100.0) * 0.3
        conf_score = {"low": 0.0, "medium": 0.1, "high": 0.2}[confidence.value] if confidence else 0.0
        return round(min(edge_score + dq_score_norm + conf_score, 1.0), 3)

    # ── Builder ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_result(
        home_team, away_team, competition, kickoff, status,
        bundle, odds_record, prediction,
        model_probability, decision, risk, reasons, risks,
        data_quality, confidence, now,
        raw_implied_prob=None,
        fair_market_prob=None,
        overround=None,
        edge=None,
        adj_edge=None,
        ev=None,
        fair_price=None,
        alert_threshold=None,
        rec_strength=None,
        model_agreement=None,
        calibration_ece=None,
    ) -> MarketAnalysisResult:
        model_version = prediction.model_version if prediction else ""
        calibration_status = (
            f"ECE={calibration_ece:.3f}" if calibration_ece is not None else "not_calibrated"
        )

        return MarketAnalysisResult(
            # Event
            home_team=home_team,
            away_team=away_team,
            competition=competition,
            kickoff=kickoff,
            status=status,
            # Market
            provider="1xBet",
            market_type=odds_record.market_type,
            selection=odds_record.selection,
            line=odds_record.line,
            decimal_odds=odds_record.decimal_odds,
            odds_timestamp=odds_record.odds_timestamp,
            settlement_rule=odds_record.settlement_rule,
            # Prediction
            model_probability=model_probability,
            fair_odds=round(fair_price, 4) if fair_price else None,
            raw_implied_probability=round(raw_implied_prob, 6) if raw_implied_prob else None,
            fair_market_probability=round(fair_market_prob, 6) if fair_market_prob else None,
            overround=round(overround, 4) if overround is not None else None,
            edge_percentage_points=round(adj_edge * 100, 3) if adj_edge is not None else None,
            expected_value_per_unit=round(ev, 6) if ev is not None else None,
            # Decision
            decision_status=decision,
            recommendation_strength=rec_strength,
            risk=risk,
            stake=None,
            stake_method=StakeMethod.NONE,
            key_reasons=reasons,
            main_risks=risks,
            price_alert_threshold=round(alert_threshold, 4) if alert_threshold else None,
            # Quality
            data_quality_score=round(data_quality, 1),
            confidence=confidence,
            model_agreement=round(model_agreement, 4) if model_agreement is not None else None,
            calibration_status=calibration_status,
            sources=[odds_record.source] if odds_record.source else [],
            model_version=model_version,
            generated_at=now,
        )
