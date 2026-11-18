"""
data/schemas.py
===============
Pydantic v2 data models for every record flowing through the system.
These are the canonical contracts between agents.  All timestamps must
be timezone-aware (UTC preferred).  All probabilities ∈ [0, 1].
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class MatchStatus(str, enum.Enum):
    PRE_MATCH = "pre_match"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class MarketType(str, enum.Enum):
    ONE_X_TWO = "1X2"
    DOUBLE_CHANCE = "DOUBLE_CHANCE"
    DRAW_NO_BET = "DRAW_NO_BET"
    ASIAN_HANDICAP = "ASIAN_HANDICAP"
    EUROPEAN_HANDICAP = "EUROPEAN_HANDICAP"
    TOTALS = "TOTALS"
    BTTS = "BTTS"
    TEAM_TOTALS = "TEAM_TOTALS"
    CORRECT_SCORE = "CORRECT_SCORE"
    HT_1X2 = "HT_1X2"
    HT_TOTALS = "HT_TOTALS"
    HT_BTTS = "HT_BTTS"
    CORNERS = "CORNERS"
    CARDS = "CARDS"
    PLAYER_PROP = "PLAYER_PROP"
    LIVE = "LIVE"
    OTHER = "OTHER"


class DecisionStatus(str, enum.Enum):
    NO_BET = "NO_BET"
    LEAN = "LEAN"
    VALUE_CANDIDATE = "VALUE_CANDIDATE"
    PASS_PENDING_LINEUP = "PASS_PENDING_LINEUP"
    LIVE_UNAVAILABLE = "LIVE_UNAVAILABLE"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ConfidenceLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FreshnessStatusEnum(str, enum.Enum):
    FRESH = "fresh"
    WARNING = "warning"
    STALE = "stale"
    EXPIRED = "expired"


class StakeMethod(str, enum.Enum):
    NONE = "none"
    FLAT = "flat"
    FRACTIONAL_KELLY_CAPPED = "fractional_kelly_capped"


# ── Source record ─────────────────────────────────────────────────────────────

class SourceRecord(BaseModel):
    """Attribution record attached to every piece of ingested data."""

    provider: str = Field(description="Data provider name, e.g. 'football-data.org'")
    source_url: Optional[str] = None
    retrieved_at: datetime = Field(description="UTC datetime when data was fetched")
    valid_from: Optional[datetime] = None
    freshness_status: FreshnessStatusEnum = FreshnessStatusEnum.FRESH
    notes: Optional[str] = None


# ── Fixture / event ───────────────────────────────────────────────────────────

class TeamRecord(BaseModel):
    id: Optional[str] = None
    name: str
    short_name: Optional[str] = None


class FixtureRecord(BaseModel):
    """A single football match."""

    fixture_id: str
    home_team: TeamRecord
    away_team: TeamRecord
    competition: str
    season: str
    kickoff: datetime                  # Must be timezone-aware
    status: MatchStatus = MatchStatus.PRE_MATCH
    venue: Optional[str] = None
    referee: Optional[str] = None
    source: SourceRecord
    minute: Optional[int] = None       # For live matches
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_red_cards: int = 0
    away_red_cards: int = 0


# ── Team stats ────────────────────────────────────────────────────────────────

class TeamStats(BaseModel):
    team: TeamRecord
    season: str
    competition: str
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    xg_for: Optional[float] = None
    xg_against: Optional[float] = None
    shots_on_target_for: Optional[float] = None
    shots_on_target_against: Optional[float] = None
    home_goals_scored: Optional[float] = None
    home_goals_conceded: Optional[float] = None
    away_goals_scored: Optional[float] = None
    away_goals_conceded: Optional[float] = None
    # Rolling form (last N games, most recent first)
    form_results: list[str] = Field(default_factory=list)  # "W", "D", "L"
    rest_days: Optional[int] = None
    source: SourceRecord


# ── Player availability ───────────────────────────────────────────────────────

class PlayerAvailability(BaseModel):
    player_id: Optional[str] = None
    player_name: str
    team: TeamRecord
    status: str   # "available", "injured", "suspended", "doubtful", "confirmed"
    return_date: Optional[datetime] = None
    probability_of_playing: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: Optional[str] = None
    source: SourceRecord


class LineupRecord(BaseModel):
    fixture_id: str
    home_lineup_confirmed: bool = False
    away_lineup_confirmed: bool = False
    home_players: list[PlayerAvailability] = Field(default_factory=list)
    away_players: list[PlayerAvailability] = Field(default_factory=list)
    source: SourceRecord


# ── Odds ─────────────────────────────────────────────────────────────────────

class OddsRecord(BaseModel):
    """
    A single odds entry from a bookmaker.

    All fields required by spec Section 3 are mandatory.
    """

    bookmaker: str
    market_id: Optional[str] = None
    market_type: MarketType
    selection: str
    line: Optional[float] = None       # For handicap / totals
    decimal_odds: float = Field(gt=1.0, description="Must be > 1.0 for valid decimal odds")
    odds_format: str = "decimal"
    odds_timestamp: datetime           # When these odds were observed
    event_start_time: Optional[datetime] = None
    source: str
    market_status: str = "open"        # "open", "suspended", "closed"
    settlement_rule: str = ""
    freshness_status: FreshnessStatusEnum = FreshnessStatusEnum.FRESH

    @field_validator("decimal_odds")
    @classmethod
    def _positive_odds(cls, v: float) -> float:
        if v <= 1.0:
            raise ValueError(f"Decimal odds must be > 1.0, got {v}")
        return v


class MarketBundle(BaseModel):
    """All odds for one market on one event (e.g., all three 1X2 prices)."""

    fixture_id: str
    market_type: MarketType
    line: Optional[float] = None
    bookmaker: str
    odds: list[OddsRecord]
    bundle_timestamp: datetime
    overround: Optional[float] = None
    margin_method_used: Optional[str] = None


# ── Prediction ────────────────────────────────────────────────────────────────

class ScoreMatrix(BaseModel):
    """Probability matrix P(home=i, away=j) for i,j ∈ [0, max_goals]."""

    max_goals: int = 8
    matrix: list[list[float]]          # [home_goals][away_goals]

    @model_validator(mode="after")
    def _check_sum(self) -> "ScoreMatrix":
        total = sum(cell for row in self.matrix for cell in row)
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"Score matrix probabilities sum to {total:.4f}, expected ≈1.0")
        return self


class PredictionOutput(BaseModel):
    """Raw model output before market normalisation."""

    fixture_id: str
    model_name: str
    model_version: str
    generated_at: datetime

    # 1X2
    p_home_win: float = Field(ge=0.0, le=1.0)
    p_draw: float = Field(ge=0.0, le=1.0)
    p_away_win: float = Field(ge=0.0, le=1.0)

    # Goals
    expected_goals_home: float
    expected_goals_away: float

    # Derived markets
    p_btts: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    p_over_05: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    p_over_15: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    p_over_25: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    p_over_35: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    p_over_45: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    score_matrix: Optional[ScoreMatrix] = None

    # Intervals
    p_home_win_low: Optional[float] = None
    p_home_win_high: Optional[float] = None

    @model_validator(mode="after")
    def _check_1x2_sum(self) -> "PredictionOutput":
        total = self.p_home_win + self.p_draw + self.p_away_win
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"1X2 probabilities sum to {total:.4f}, expected ≈1.0")
        return self


# ── Decision ──────────────────────────────────────────────────────────────────

class MarketAnalysisResult(BaseModel):
    """
    Full analysis result for a single market selection.
    This mirrors the Section 11 JSON contract.
    """

    # Event
    home_team: str
    away_team: str
    competition: str
    kickoff: datetime
    status: MatchStatus

    # Market
    provider: str = "1xBet"
    market_type: MarketType
    selection: str
    line: Optional[float] = None
    decimal_odds: Optional[float] = None
    odds_timestamp: Optional[datetime] = None
    settlement_rule: str = ""

    # Prediction
    model_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fair_odds: Optional[float] = None
    raw_implied_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fair_market_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overround: Optional[float] = None
    edge_percentage_points: Optional[float] = None
    expected_value_per_unit: Optional[float] = None
    probability_interval_low: Optional[float] = None
    probability_interval_high: Optional[float] = None

    # Decision
    decision_status: DecisionStatus
    recommendation_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk: RiskLevel
    stake: Optional[float] = None
    stake_method: StakeMethod = StakeMethod.NONE
    key_reasons: list[str] = Field(default_factory=list)
    main_risks: list[str] = Field(default_factory=list)
    price_alert_threshold: Optional[float] = None

    # Quality
    data_quality_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    confidence: ConfidenceLevel
    model_agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    calibration_status: str = ""
    sources: list[str] = Field(default_factory=list)
    model_version: str = ""
    generated_at: datetime

    # Responsible gambling notice (always populated)
    responsible_gambling_notice: str = (
        "Betting involves financial risk. Only use money you can afford to lose, "
        "follow local laws and age requirements, and consider setting limits or "
        "stopping if gambling is becoming difficult to control."
    )


# ── Bankroll request (optional stake sizing) ──────────────────────────────────

class BankrollContext(BaseModel):
    bankroll: float = Field(gt=0.0, description="Current bankroll amount")
    currency: str = Field(min_length=3, max_length=3, description="ISO-4217 currency code")
    risk_profile: str   # "conservative", "moderate", "aggressive"
    legal_jurisdiction: str
    max_single_stake_pct: float = Field(default=0.02, ge=0.0, le=0.10)
    daily_loss_limit: Optional[float] = None

    @field_validator("risk_profile")
    @classmethod
    def _valid_risk(cls, v: str) -> str:
        allowed = {"conservative", "moderate", "aggressive"}
        if v.lower() not in allowed:
            raise ValueError(f"risk_profile must be one of {allowed}")
        return v.lower()

# ═══════════════════════════════════════════════════════════════════════
# v3 EXTENSIONS: Players, Injuries, Fantasy, Simulation
# ═══════════════════════════════════════════════════════════════════════

class PlayerRecord(BaseModel):
    id: str
    name: str
    short_name: Optional[str] = None
    position: str = "UNK"
    nationality: Optional[str] = None
    team_id: str
    is_active: bool = True

class PlayerStats(BaseModel):
    player_id: str
    season: str
    competition: str
    appearances: int = 0
    starts: int = 0
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    xg: Optional[float] = None
    xa: Optional[float] = None
    source: SourceRecord

class InjuryRecord(BaseModel):
    player_id: str
    player_name: str
    team_id: str
    injury_type: str
    status: str = "unknown"
    expected_return_date: Optional[datetime] = None
    source: SourceRecord

class SuspensionRecord(BaseModel):
    player_id: str
    player_name: str
    team_id: str
    reason: str
    matches_remaining: int
    source: SourceRecord

class FantasyPlayer(BaseModel):
    player_id: str
    platform: str = "fpl"
    position: str
    price: float
    total_points: int = 0
    ownership_pct: Optional[float] = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class FantasyProjection(BaseModel):
    fantasy_player_id: str
    gameweek: int
    expected_minutes: float
    starting_probability: float
    expected_goals: float
    expected_assists: float
    expected_fantasy_points: float
    recommendation: str = "HOLD"

class SimulationResult(BaseModel):
    fixture_id: str
    n_simulations: int
    p_home_win: float
    p_draw: float
    p_away_win: float
    mean_home_goals: float
    mean_away_goals: float
    p_btts: float
    p_over_25: float
    top_scorelines: list[dict]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
