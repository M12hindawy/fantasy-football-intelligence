"""
api.py
======
Optional FastAPI REST wrapper around the analysis pipeline.

Endpoints
---------
GET  /health          — Liveness check
POST /analyze         — Full market analysis (JSON body)
POST /odds-check      — Quick overround check
POST /accumulator     — Joint probability for multiple legs

Start with: uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Football Value & Probability Analyzer",
    description=(
        "Analytical decision-support for 1xBet football markets. "
        "Not a guarantee of profit. Betting involves financial risk."
    ),
    version="1.0.0",
)

RESPONSIBLE_NOTICE = (
    "Betting involves financial risk. Only use money you can afford to lose, "
    "follow local laws and age requirements, and consider setting limits or "
    "stopping if gambling is becoming difficult to control."
)


# ── Request / Response models ─────────────────────────────────────────────────

class SelectionInput(BaseModel):
    name: str
    decimal_odds: float = Field(gt=1.0)


class AnalyzeRequest(BaseModel):
    home_team: str
    away_team: str
    competition: str = ""
    kickoff: str = ""
    market_type: str = "1X2"
    selection: str = "Home"
    line: Optional[float] = None
    selections: list[SelectionInput] = Field(
        default_factory=list,
        description="All odds for the market (e.g. all three 1X2 prices)"
    )
    odds_timestamp: Optional[str] = None

    # Optional stake sizing
    bankroll: Optional[float] = Field(default=None, gt=0.0)
    currency: Optional[str] = None
    risk_profile: Optional[str] = None
    jurisdiction: Optional[str] = None


class OddsCheckRequest(BaseModel):
    odds: list[float] = Field(min_length=2)


class AccumulatorRequest(BaseModel):
    legs: list[dict[str, Any]] = Field(
        description="List of AnalyzeRequest-compatible dicts, one per accumulator leg"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "generated_at": datetime.now(timezone.utc).isoformat()}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    """
    Full market analysis pipeline.

    Returns a JSON object matching the Section 11 spec contract.
    """
    from datetime import datetime, timezone
    from data.schemas import MatchStatus, MarketType, BankrollContext
    from data.fetchers.odds_api_fetcher import OddsApiFetcher
    from markets.onebet_adapter import OneBetAdapter
    from agents.odds_ingestion_agent import OddsIngestionAgent
    from agents.margin_removal_agent import MarginRemovalAgent
    from agents.market_normalization_agent import MarketNormalizationAgent
    from agents.market_probability_agent import MarketProbabilityAgent
    from agents.value_assessment_agent import ValueAssessmentAgent
    from agents.risk_and_stake_agent import RiskAndStakeAgent
    from agents.responsible_use_guard_agent import ResponsibleUseGuardAgent
    from output.json_contract import AnalysisOutputJSON

    # Parse kickoff
    try:
        kickoff = datetime.fromisoformat(req.kickoff.replace("Z", "+00:00"))
    except Exception:
        kickoff = datetime.now(timezone.utc)

    # Parse odds timestamp
    try:
        odds_ts = datetime.fromisoformat(req.odds_timestamp.replace("Z", "+00:00")) if req.odds_timestamp else datetime.now(timezone.utc)
    except Exception:
        odds_ts = datetime.now(timezone.utc)

    # Market type
    alias = {"1X2": "ONE_X_TWO", "AH": "ASIAN_HANDICAP", "EH": "EUROPEAN_HANDICAP"}
    mt_str = alias.get(req.market_type.upper(), req.market_type.upper().replace("-", "_"))
    try:
        market_type = MarketType[mt_str]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown market type: {req.market_type}")

    # Build selections list
    if req.selections:
        sel_list = [(s.name, s.decimal_odds) for s in req.selections]
    else:
        raise HTTPException(status_code=400, detail="At least one selection with decimal_odds required.")

    # Build bundle
    fixture_id = f"{req.home_team}_{req.away_team}".replace(" ", "_")
    try:
        bundle = OddsApiFetcher.from_manual_input(
            fixture_id=fixture_id,
            bookmaker="1xBet",
            market_type=market_type,
            selections=sel_list,
            line=req.line,
            odds_timestamp=odds_ts,
            event_start_time=kickoff,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Pipeline
    ingestion_agent = OddsIngestionAgent()
    bundle, _ = ingestion_agent.validate(bundle, kickoff)

    margin_agent = MarginRemovalAgent()
    bundle, _, _ = margin_agent.apply(bundle, "auto")

    # Naive baseline model
    from models.dixon_coles import score_probability_matrix, matrix_to_1x2, matrix_to_btts, matrix_to_over_under
    from data.schemas import PredictionOutput, ScoreMatrix
    mu_h, mu_a, rho = 1.35, 1.10, -0.10
    mat = score_probability_matrix(mu_h, mu_a, rho)
    p_home, p_draw, p_away = matrix_to_1x2(mat)
    p_btts = matrix_to_btts(mat)
    p_o25, _ = matrix_to_over_under(mat, 2.5)
    prediction = PredictionOutput(
        fixture_id=fixture_id,
        model_name="BaselineDixonColes",
        model_version="baseline-league-avg",
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

    # Model probability for target selection
    mp_agent = MarketProbabilityAgent()
    target = next((r for r in bundle.odds if r.selection.lower() == req.selection.lower()), bundle.odds[0])
    model_prob, _ = mp_agent.get_model_probability(prediction, bundle, req.selection)

    # Value assessment
    va_agent = ValueAssessmentAgent()
    result = va_agent.assess(
        home_team=req.home_team,
        away_team=req.away_team,
        competition=req.competition,
        kickoff=kickoff,
        status=MatchStatus.PRE_MATCH,
        bundle=bundle,
        odds_record=target,
        prediction=prediction,
        model_probability=model_prob,
    )

    # Optional stake sizing
    if req.bankroll and req.currency and req.risk_profile and req.jurisdiction:
        try:
            ctx = BankrollContext(
                bankroll=req.bankroll,
                currency=req.currency.upper(),
                risk_profile=req.risk_profile,
                legal_jurisdiction=req.jurisdiction,
            )
            stake_agent = RiskAndStakeAgent()
            stake_rec = stake_agent.recommend(result, ctx)
            result = stake_agent.apply_to_result(result, stake_rec)
        except Exception:
            pass

    # Guard
    guard = ResponsibleUseGuardAgent()
    guard_result = guard.check(result)
    result = guard_result.modified_result or result

    return AnalysisOutputJSON.from_result(result).to_dict()


@app.post("/odds-check")
def odds_check(req: OddsCheckRequest) -> dict:
    """Quick overround and fair-probability check."""
    from markets.market_math import overround, normalize_shin, raw_implied_probability
    vig = overround(req.odds)
    fair = normalize_shin(req.odds)
    raw = [raw_implied_probability(o) for o in req.odds]
    return {
        "odds": req.odds,
        "overround_pct": round(vig * 100, 3),
        "fair_probabilities_shin": [round(p, 4) for p in fair],
        "raw_implied_probabilities": [round(p, 4) for p in raw],
        "responsible_gambling_notice": RESPONSIBLE_NOTICE,
    }


@app.post("/accumulator")
def accumulator_check(req: AccumulatorRequest) -> dict:
    """Compute joint probability for an accumulator (with full disclosure)."""
    guard = ResponsibleUseGuardAgent()
    # Simplified: compute product of user-supplied model probs
    probs = [leg.get("model_probability") for leg in req.legs]
    if any(p is None for p in probs):
        raise HTTPException(
            status_code=422,
            detail="Each leg must include 'model_probability'."
        )
    joint = 1.0
    for p in probs:
        joint *= p
    _, disclosure = guard.check_accumulator_request(
        [],  # results not available here
        assume_independence=True,
    )
    return {
        "n_legs": len(probs),
        "individual_model_probs": probs,
        "joint_probability_independence_assumed": round(joint, 6),
        "disclosure": disclosure,
        "responsible_gambling_notice": RESPONSIBLE_NOTICE,
    }
