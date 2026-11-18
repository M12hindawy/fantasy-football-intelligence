"""
output/formatter.py
===================
Generates the human-readable recommendation report defined in
Section 12 of the master spec.

Format:
    MATCH: Team A vs Team B
    MARKET: Over 2.5 Goals
    1xBet ODDS: 1.85
    ODDS TIME: YYYY-MM-DD HH:MM TZ
    MODEL PROBABILITY: 57.4%
    FAIR ODDS: 1.74
    MARKET-IMPLIED PROBABILITY: 54.1% raw / 52.8% margin-adjusted
    EDGE: +4.6 percentage points
    EXPECTED VALUE: +6.2% per unit stake
    DECISION: VALUE CANDIDATE — subject to lineup and price verification
    CONFIDENCE: Medium
    DATA QUALITY: 88/100
    RISK: High variance
    WHY: [evidence-based reasons]
    MAIN RISKS: [uncertainties]
    PRICE ALERT: Recalculate if odds move below X.XX.
    RESPONSIBLE GAMBLING: ...

All computed numbers come from the MarketAnalysisResult object.
No example numbers are copied from the spec.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from data.schemas import (
    ConfidenceLevel,
    DecisionStatus,
    MarketAnalysisResult,
    RiskLevel,
)

SEPARATOR = "─" * 60

RESPONSIBLE_NOTICE = (
    "Betting involves financial risk. Only use money you can afford to lose, "
    "follow local laws and age requirements, and consider setting limits or "
    "stopping if gambling is becoming difficult to control."
)

# ── Decision labels ───────────────────────────────────────────────────────────

_DECISION_LABELS = {
    DecisionStatus.NO_BET: "❌ NO BET — Insufficient edge, data quality, or model confidence.",
    DecisionStatus.LEAN: "↗ LEAN — Directional preference without sufficient value edge.",
    DecisionStatus.VALUE_CANDIDATE: "✅ VALUE CANDIDATE — Positive uncertainty-adjusted edge. Subject to lineup and price verification.",
    DecisionStatus.PASS_PENDING_LINEUP: "⏳ PASS PENDING LINEUP — Analysis must wait for confirmed lineup or key news.",
    DecisionStatus.LIVE_UNAVAILABLE: "🔴 LIVE UNAVAILABLE — Live market analysis requires a real-time feed.",
}

_RISK_LABELS = {
    RiskLevel.LOW: "Low",
    RiskLevel.MEDIUM: "Medium",
    RiskLevel.HIGH: "High variance",
    RiskLevel.VERY_HIGH: "Very high variance — treat with extreme caution",
}

_CONFIDENCE_LABELS = {
    ConfidenceLevel.LOW: "Low",
    ConfidenceLevel.MEDIUM: "Medium",
    ConfidenceLevel.HIGH: "High",
}


def format_report(result: MarketAnalysisResult) -> str:
    """
    Produce the full Section 12 human-readable report.

    Parameters
    ----------
    result : MarketAnalysisResult from ValueAssessmentAgent

    Returns
    -------
    str — formatted text block
    """
    lines: list[str] = []
    a = lines.append   # convenience alias

    a(SEPARATOR)
    a("  FOOTBALL VALUE & PROBABILITY ANALYZER — 1xBet MARKETS")
    a("  Analytical decision support only. Not a guarantee of profit.")
    a(SEPARATOR)

    # ── Event ─────────────────────────────────────────────────────────────────
    a(f"MATCH      : {result.home_team} vs {result.away_team}")
    a(f"COMPETITION: {result.competition}")
    kickoff_str = result.kickoff.strftime("%Y-%m-%d %H:%M %Z") if result.kickoff else "Unknown"
    a(f"KICKOFF    : {kickoff_str}")
    a(f"STATUS     : {result.status.value.replace('_', ' ').upper()}")
    a("")

    # ── Market ────────────────────────────────────────────────────────────────
    market_label = result.market_type.value.replace("_", " ")
    line_str = f" {result.line:+.2f}" if result.line is not None else ""
    a(f"MARKET     : {market_label}{line_str}")
    a(f"SELECTION  : {result.selection}")

    if result.decimal_odds is not None:
        a(f"1xBet ODDS : {result.decimal_odds:.2f}")
    else:
        a("1xBet ODDS : Not provided")

    if result.odds_timestamp:
        ts_str = result.odds_timestamp.strftime("%Y-%m-%d %H:%M UTC")
        a(f"ODDS TIME  : {ts_str}")
    else:
        a("ODDS TIME  : Not provided")

    if result.settlement_rule:
        a(f"SETTLEMENT : {result.settlement_rule}")
    a("")

    # ── Model output ──────────────────────────────────────────────────────────
    if result.model_probability is not None:
        a(f"MODEL PROBABILITY       : {result.model_probability * 100:.2f}%")
    else:
        a("MODEL PROBABILITY       : Not computed")

    if result.fair_odds is not None:
        a(f"FAIR ODDS               : {result.fair_odds:.2f}")

    if result.raw_implied_probability is not None and result.fair_market_probability is not None:
        a(
            f"MARKET-IMPLIED PROB     : {result.raw_implied_probability * 100:.2f}% raw"
            f" / {result.fair_market_probability * 100:.2f}% margin-adjusted"
        )

    if result.overround is not None:
        margin_method = ""  # Would need to pass from bundle — shown in JSON
        a(f"BOOKMAKER OVERROUND     : {result.overround * 100:.2f}%")

    a("")

    # ── Value metrics ─────────────────────────────────────────────────────────
    if result.edge_percentage_points is not None:
        sign = "+" if result.edge_percentage_points >= 0 else ""
        a(f"EDGE                    : {sign}{result.edge_percentage_points:.2f} percentage points")
    else:
        a("EDGE                    : Not computed")

    if result.expected_value_per_unit is not None:
        sign = "+" if result.expected_value_per_unit >= 0 else ""
        a(f"EXPECTED VALUE          : {sign}{result.expected_value_per_unit * 100:.2f}% per unit stake")
    else:
        a("EXPECTED VALUE          : Not computed")

    a("")

    # ── Decision ──────────────────────────────────────────────────────────────
    decision_label = _DECISION_LABELS.get(
        result.decision_status,
        result.decision_status.value
    )
    a(f"DECISION   : {decision_label}")

    if result.recommendation_strength is not None:
        a(f"REC STRENGTH: {result.recommendation_strength:.2f} / 1.00")

    a(f"CONFIDENCE : {_CONFIDENCE_LABELS.get(result.confidence, result.confidence.value)}")
    a(f"DATA QUALITY: {result.data_quality_score:.0f} / 100" if result.data_quality_score is not None else "DATA QUALITY: Unknown")
    a(f"RISK       : {_RISK_LABELS.get(result.risk, result.risk.value)}")

    if result.model_agreement is not None:
        a(f"MODEL AGREEMENT: {result.model_agreement * 100:.1f}%")

    if result.calibration_status:
        a(f"CALIBRATION: {result.calibration_status}")
    a("")

    # ── Reasons ───────────────────────────────────────────────────────────────
    if result.key_reasons:
        a("WHY:")
        for i, reason in enumerate(result.key_reasons, 1):
            a(f"  {i}. {reason}")
    a("")

    # ── Risks ─────────────────────────────────────────────────────────────────
    if result.main_risks:
        a("MAIN RISKS:")
        for i, risk_item in enumerate(result.main_risks, 1):
            a(f"  {i}. {risk_item}")
    a("")

    # ── Stake (if applicable) ─────────────────────────────────────────────────
    if result.stake is not None and result.stake > 0:
        a(f"STAKE      : {result.stake:.2f} (method: {result.stake_method.value})")
        a("           ⚠️  STAKE NOT REQUIRED. 'No bet' is always a valid choice.")
        a("")

    # ── Price alert ───────────────────────────────────────────────────────────
    if result.price_alert_threshold is not None:
        a(
            f"PRICE ALERT: Recalculate immediately if odds move below "
            f"{result.price_alert_threshold:.2f}. The edge may disappear."
        )
        a("")

    # ── Sources ───────────────────────────────────────────────────────────────
    if result.sources:
        a(f"SOURCES    : {', '.join(result.sources)}")
    if result.model_version:
        a(f"MODEL VER  : {result.model_version}")
    generated_str = result.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    a(f"GENERATED  : {generated_str}")
    a("")

    # ── Responsible gambling (always last) ────────────────────────────────────
    a(SEPARATOR)
    a("⚠️  RESPONSIBLE GAMBLING NOTICE")
    a(result.responsible_gambling_notice or RESPONSIBLE_NOTICE)
    a("")
    a("This output is probabilistic and NOT guaranteed. It is for analytical")
    a("decision-support only. No wager has been placed automatically.")
    a(SEPARATOR)

    return "\n".join(lines)


def format_no_data_report(
    home_team: str,
    away_team: str,
    reason: str,
    generated_at: Optional[datetime] = None,
) -> str:
    """Return a minimal report when data is insufficient to analyse."""
    ts = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"{SEPARATOR}\n"
        f"MATCH      : {home_team} vs {away_team}\n"
        f"DECISION   : ❌ NO BET — Insufficient data\n"
        f"REASON     : {reason}\n"
        f"GENERATED  : {ts}\n"
        f"{SEPARATOR}\n"
        f"⚠️  RESPONSIBLE GAMBLING NOTICE\n"
        f"{RESPONSIBLE_NOTICE}\n"
        f"{SEPARATOR}"
    )
