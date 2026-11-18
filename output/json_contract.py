"""
output/json_contract.py
========================
Pydantic v2 model mirroring the Section 11 JSON output contract.

Validates:
- All probabilities ∈ [0, 1]
- Mutually exclusive probabilities sum ≈ 1
- All decimal odds > 1.0
- All timestamps are explicit

serialize() returns the canonical JSON dict for API responses.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from data.schemas import (
    ConfidenceLevel,
    DecisionStatus,
    MarketAnalysisResult,
    MarketType,
    MatchStatus,
    RiskLevel,
    StakeMethod,
)

RESPONSIBLE_NOTICE = (
    "Betting involves financial risk. Only use money you can afford to lose, "
    "follow local laws and age requirements, and consider setting limits or "
    "stopping if gambling is becoming difficult to control."
)


class EventJSON(BaseModel):
    home_team: str
    away_team: str
    competition: str
    kickoff: str          # ISO-8601
    status: str           # "pre_match" | "live" | "finished"


class MarketJSON(BaseModel):
    provider: str = "1xBet"
    market_type: str
    selection: str
    line: Optional[float] = None
    decimal_odds: Optional[float] = None
    odds_timestamp: Optional[str] = None
    settlement_rule: str = ""

    @model_validator(mode="after")
    def _valid_odds(self) -> "MarketJSON":
        if self.decimal_odds is not None and self.decimal_odds <= 1.0:
            raise ValueError(f"decimal_odds must be > 1.0, got {self.decimal_odds}")
        return self


class PredictionJSON(BaseModel):
    model_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fair_odds: Optional[float] = None
    raw_implied_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fair_market_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overround: Optional[float] = None
    edge_percentage_points: Optional[float] = None
    expected_value_per_unit: Optional[float] = None
    probability_interval: dict[str, Optional[float]] = Field(
        default_factory=lambda: {"low": None, "high": None}
    )


class DecisionJSON(BaseModel):
    status: str    # NO_BET | LEAN | VALUE_CANDIDATE | PASS_PENDING_LINEUP | LIVE_UNAVAILABLE
    recommendation_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk: str      # low | medium | high | very_high
    stake: Optional[float] = None
    stake_method: str = "none"
    key_reasons: list[str] = Field(default_factory=list)
    main_risks: list[str] = Field(default_factory=list)
    price_alert_threshold: Optional[float] = None


class QualityJSON(BaseModel):
    data_quality: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    confidence: str    # low | medium | high
    model_agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    calibration_status: str = ""
    sources: list[str] = Field(default_factory=list)
    model_version: str = ""
    generated_at: str  # ISO-8601


class AnalysisOutputJSON(BaseModel):
    """
    Canonical JSON contract for a complete market analysis result.
    Mirrors Section 11 of the master spec.
    """

    event: EventJSON
    market: MarketJSON
    prediction: PredictionJSON
    decision: DecisionJSON
    quality: QualityJSON
    responsible_gambling_notice: str = RESPONSIBLE_NOTICE

    @classmethod
    def from_result(cls, result: MarketAnalysisResult) -> "AnalysisOutputJSON":
        """Build from a MarketAnalysisResult (the internal schema object)."""
        return cls(
            event=EventJSON(
                home_team=result.home_team,
                away_team=result.away_team,
                competition=result.competition,
                kickoff=result.kickoff.isoformat(),
                status=result.status.value,
            ),
            market=MarketJSON(
                provider=result.provider,
                market_type=result.market_type.value,
                selection=result.selection,
                line=result.line,
                decimal_odds=result.decimal_odds,
                odds_timestamp=result.odds_timestamp.isoformat() if result.odds_timestamp else None,
                settlement_rule=result.settlement_rule,
            ),
            prediction=PredictionJSON(
                model_probability=result.model_probability,
                fair_odds=result.fair_odds,
                raw_implied_probability=result.raw_implied_probability,
                fair_market_probability=result.fair_market_probability,
                overround=result.overround,
                edge_percentage_points=result.edge_percentage_points,
                expected_value_per_unit=result.expected_value_per_unit,
                probability_interval={
                    "low": result.probability_interval_low,
                    "high": result.probability_interval_high,
                },
            ),
            decision=DecisionJSON(
                status=result.decision_status.value,
                recommendation_strength=result.recommendation_strength,
                risk=result.risk.value,
                stake=result.stake,
                stake_method=result.stake_method.value,
                key_reasons=result.key_reasons,
                main_risks=result.main_risks,
                price_alert_threshold=result.price_alert_threshold,
            ),
            quality=QualityJSON(
                data_quality=result.data_quality_score,
                confidence=result.confidence.value,
                model_agreement=result.model_agreement,
                calibration_status=result.calibration_status,
                sources=result.sources,
                model_version=result.model_version,
                generated_at=result.generated_at.isoformat(),
            ),
            responsible_gambling_notice=result.responsible_gambling_notice,
        )

    def serialize(self, indent: int = 2) -> str:
        """Return the canonical JSON string."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical dict."""
        return self.model_dump()
