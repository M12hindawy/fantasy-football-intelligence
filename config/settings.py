"""
config/settings.py
==================
Central configuration for the Football Intelligence & Fantasy Decision Agent.
All secrets and tunable parameters are loaded from environment variables
(or a .env file).  Never hard-code API keys here.

v3: Extended with feature flags, simulation, fantasy, DB, and ML settings.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from .env or environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────
    football_data_api_key: str = Field(
        default="",
        description="Football-Data.org API key (free tier available).",
    )
    odds_api_key: str = Field(
        default="",
        description="The Odds API key (free tier available).",
    )

    # ── Data provider URLs ────────────────────────────────────────────────
    football_data_base_url: str = "https://api.football-data.org/v4"
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # ── Cache ─────────────────────────────────────────────────────────────
    cache_db_path: str = "football_agent_cache.db"
    cache_enabled: bool = True

    # ── Freshness thresholds (seconds) ────────────────────────────────────
    # Overrideable per environment; see also config/freshness.py
    live_score_max_age_s: int = 45
    lineup_max_age_s: int = 900          # 15 min
    prematch_odds_max_age_s: int = 300   # 5 min
    team_news_max_age_s: int = 21_600    # 6 hours

    # ── Margin removal method ─────────────────────────────────────────────
    # "proportional" | "shin" | "power" | "additive"
    default_margin_method: str = "shin"

    # ── Ensemble & model ─────────────────────────────────────────────────
    dixon_coles_weight: float = 0.55
    elo_weight: float = 0.45
    # Weights are overridden at runtime by walk-forward calibration when
    # sufficient historical data is available.

    # ── Kelly staking ─────────────────────────────────────────────────────
    kelly_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    max_stake_pct_of_bankroll: float = Field(default=0.02, ge=0.0, le=0.10)
    max_daily_exposure_pct: float = Field(default=0.05, ge=0.0, le=0.20)

    # ── Edge & EV thresholds ─────────────────────────────────────────────
    min_edge_pct_for_value_candidate: float = 0.03   # 3 pp
    min_ev_for_value_candidate: float = 0.02          # +2% per unit

    # ── Output ────────────────────────────────────────────────────────────
    default_odds_format: str = "decimal"  # "decimal" only for now
    default_timezone: str = "UTC"
    default_currency: str = "USD"
    default_language: str = "en"

    # ── Backtesting ───────────────────────────────────────────────────────
    backtest_min_samples: int = 200
    backtest_train_seasons: int = 2   # seasons in rolling training window

    # ═══════════════════════════════════════════════════════════════════════
    # v3 SETTINGS
    # ═══════════════════════════════════════════════════════════════════════

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = "sqlite:///football_intelligence.db"

    # ── Feature flags ─────────────────────────────────────────────────────
    enable_xgboost: bool = False
    enable_lightgbm: bool = False
    enable_rag: bool = False
    enable_fantasy: bool = True
    enable_simulation: bool = True
    enable_monitoring: bool = True
    enable_db: bool = True
    enable_backtesting: bool = True
    enable_api: bool = True

    # ── Simulation ────────────────────────────────────────────────────────
    simulation_count: int = 10_000
    simulation_seed: int = 42

    # ── Data Quality ──────────────────────────────────────────────────────
    min_data_quality: float = Field(default=0.40, ge=0.0, le=1.0)
    min_sample_size: int = 10

    # ── Feature engineering ───────────────────────────────────────────────
    recency_decay: float = 0.0065
    form_window_short: int = 3
    form_window_medium: int = 5
    form_window_long: int = 10

    # ── Fantasy (FPL) ─────────────────────────────────────────────────────
    fantasy_platform: str = "fpl"
    fantasy_budget: float = 100.0
    fantasy_max_per_club: int = 3

    # ── Model registry ────────────────────────────────────────────────────
    model_registry_path: str = "model_registry"
    model_drift_threshold: float = 0.05

    # ── RAG ───────────────────────────────────────────────────────────────
    rag_collection_name: str = "football_knowledge"
    rag_embed_model: str = "all-MiniLM-L6-v2"

    # ── ML weights (only active when ENABLE_* flags = True) ───────────────
    xgboost_weight: float = 0.20
    lightgbm_weight: float = 0.15

    @field_validator("default_margin_method")
    @classmethod
    def _valid_margin_method(cls, v: str) -> str:
        allowed = {"proportional", "shin", "power", "additive"}
        if v not in allowed:
            raise ValueError(f"margin method must be one of {allowed}")
        return v

    @field_validator("default_odds_format")
    @classmethod
    def _valid_odds_format(cls, v: str) -> str:
        if v not in {"decimal"}:
            raise ValueError("Only 'decimal' odds format is supported currently.")
        return v


# Module-level singleton — import this everywhere
settings = Settings()
