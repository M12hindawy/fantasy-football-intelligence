"""
db/models.py - SQLAlchemy ORM for Football Intelligence database.
Works with SQLite (dev) and PostgreSQL (production) - change DATABASE_URL only.
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, JSON, String, Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Competition(Base):
    __tablename__ = "competitions"
    id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    country = Column(String(50))
    tier = Column(Integer, default=1)
    provider_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    seasons = relationship("Season", back_populates="competition")
    matches = relationship("Match", back_populates="competition_rel")


class Season(Base):
    __tablename__ = "seasons"
    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(String(20), ForeignKey("competitions.id"), nullable=False)
    year = Column(Integer, nullable=False)
    label = Column(String(20))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    is_current = Column(Boolean, default=False)
    __table_args__ = (UniqueConstraint("competition_id", "year"),)
    competition = relationship("Competition", back_populates="seasons")


class Venue(Base):
    __tablename__ = "venues"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), nullable=False)
    city = Column(String(100))
    country = Column(String(50))
    capacity = Column(Integer)


class Team(Base):
    __tablename__ = "teams"
    id = Column(String(50), primary_key=True)
    name = Column(String(150), nullable=False)
    short_name = Column(String(20))
    tla = Column(String(5))
    country = Column(String(50))
    founded = Column(Integer)
    venue_id = Column(Integer, ForeignKey("venues.id"))
    logo_url = Column(String(500))
    provider_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    players = relationship("Player", back_populates="team")
    injuries = relationship("Injury", back_populates="team")
    suspensions = relationship("Suspension", back_populates="team")


class Player(Base):
    __tablename__ = "players"
    id = Column(String(50), primary_key=True)
    name = Column(String(150), nullable=False)
    short_name = Column(String(50))
    position = Column(String(5), default="UNK")
    nationality = Column(String(50))
    date_of_birth = Column(DateTime)
    shirt_number = Column(Integer)
    team_id = Column(String(50), ForeignKey("teams.id"))
    provider_id = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    team = relationship("Team", back_populates="players")
    season_stats = relationship("PlayerSeasonStats", back_populates="player")
    match_stats = relationship("PlayerMatchStats", back_populates="player")
    injuries = relationship("Injury", back_populates="player")
    fantasy_profiles = relationship("FantasyPlayer", back_populates="player")


class Match(Base):
    __tablename__ = "matches"
    id = Column(String(50), primary_key=True)
    competition_id = Column(String(20), ForeignKey("competitions.id"))
    season_year = Column(Integer)
    home_team_id = Column(String(50), ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(String(50), ForeignKey("teams.id"), nullable=False)
    kickoff_utc = Column(DateTime, nullable=False)
    status = Column(String(20), default="pre_match")
    venue_id = Column(Integer, ForeignKey("venues.id"))
    matchday = Column(Integer)
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_score_ht = Column(Integer)
    away_score_ht = Column(Integer)
    home_red_cards = Column(Integer, default=0)
    away_red_cards = Column(Integer, default=0)
    provider_id = Column(String(50))
    data_source = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    competition_rel = relationship("Competition", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    predictions = relationship("Prediction", back_populates="match")
    team_stats = relationship("TeamMatchStats", back_populates="match")
    player_stats = relationship("PlayerMatchStats", back_populates="match")
    lineups = relationship("Lineup", back_populates="match")


class TeamMatchStats(Base):
    __tablename__ = "team_match_stats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(50), ForeignKey("matches.id"), nullable=False)
    team_id = Column(String(50), ForeignKey("teams.id"), nullable=False)
    is_home = Column(Boolean, nullable=False)
    goals = Column(Integer)
    shots = Column(Integer)
    shots_on_target = Column(Integer)
    xg = Column(Float)
    possession_pct = Column(Float)
    corners = Column(Integer)
    fouls = Column(Integer)
    yellow_cards = Column(Integer)
    red_cards = Column(Integer)
    __table_args__ = (UniqueConstraint("match_id", "team_id"),)
    match = relationship("Match", back_populates="team_stats")


class PlayerSeasonStats(Base):
    __tablename__ = "player_season_stats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey("players.id"), nullable=False)
    competition_id = Column(String(20), ForeignKey("competitions.id"))
    season_year = Column(Integer, nullable=False)
    appearances = Column(Integer, default=0)
    starts = Column(Integer, default=0)
    minutes = Column(Integer, default=0)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    xg = Column(Float)
    xa = Column(Float)
    key_passes = Column(Integer)
    clean_sheets = Column(Integer, default=0)
    saves = Column(Integer)
    bonus_points = Column(Integer, default=0)
    __table_args__ = (UniqueConstraint("player_id", "competition_id", "season_year"),)
    player = relationship("Player", back_populates="season_stats")


class PlayerMatchStats(Base):
    __tablename__ = "player_match_stats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey("players.id"), nullable=False)
    match_id = Column(String(50), ForeignKey("matches.id"), nullable=False)
    team_id = Column(String(50), ForeignKey("teams.id"))
    minutes_played = Column(Integer)
    started = Column(Boolean)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    xg = Column(Float)
    xa = Column(Float)
    key_passes = Column(Integer)
    saves = Column(Integer)
    clean_sheet = Column(Boolean)
    bonus = Column(Integer, default=0)
    rating = Column(Float)
    __table_args__ = (UniqueConstraint("player_id", "match_id"),)
    player = relationship("Player", back_populates="match_stats")
    match = relationship("Match", back_populates="player_stats")


class Lineup(Base):
    __tablename__ = "lineups"
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(50), ForeignKey("matches.id"), nullable=False)
    team_id = Column(String(50), ForeignKey("teams.id"), nullable=False)
    formation = Column(String(10))
    is_confirmed = Column(Boolean, default=False)
    confirmed_at = Column(DateTime)
    source = Column(String(100))
    __table_args__ = (UniqueConstraint("match_id", "team_id"),)
    match = relationship("Match", back_populates="lineups")
    players = relationship("LineupPlayer", back_populates="lineup")


class LineupPlayer(Base):
    __tablename__ = "lineup_players"
    id = Column(Integer, primary_key=True, autoincrement=True)
    lineup_id = Column(Integer, ForeignKey("lineups.id"), nullable=False)
    player_id = Column(String(50), ForeignKey("players.id"), nullable=False)
    shirt_number = Column(Integer)
    position = Column(String(10))
    is_starter = Column(Boolean, default=True)
    sub_order = Column(Integer)
    lineup = relationship("Lineup", back_populates="players")


class Injury(Base):
    __tablename__ = "injuries"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey("players.id"), nullable=False)
    team_id = Column(String(50), ForeignKey("teams.id"), nullable=False)
    injury_type = Column(String(100))
    status = Column(String(20), default="unknown")
    expected_return_date = Column(DateTime)
    probability_of_playing = Column(Float)
    source = Column(String(100))
    source_reliability = Column(String(20), default="unknown")
    reported_at = Column(DateTime)
    retrieved_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    player = relationship("Player", back_populates="injuries")
    team = relationship("Team", back_populates="injuries")


class Suspension(Base):
    __tablename__ = "suspensions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey("players.id"), nullable=False)
    team_id = Column(String(50), ForeignKey("teams.id"), nullable=False)
    reason = Column(String(200))
    matches_remaining = Column(Integer)
    source = Column(String(100))
    retrieved_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    player = relationship("Player")
    team = relationship("Team", back_populates="suspensions")


class Standing(Base):
    __tablename__ = "standings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(String(20), ForeignKey("competitions.id"))
    season_year = Column(Integer)
    team_id = Column(String(50), ForeignKey("teams.id"))
    position = Column(Integer)
    played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    goals_for = Column(Integer, default=0)
    goals_against = Column(Integer, default=0)
    goal_difference = Column(Integer, default=0)
    points = Column(Integer, default=0)
    form = Column(String(20))
    retrieved_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("competition_id", "season_year", "team_id"),)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(50), ForeignKey("matches.id"), nullable=False)
    model_version = Column(String(50))
    feature_version = Column(String(50))
    generated_at = Column(DateTime, default=datetime.utcnow)
    home_probability = Column(Float, nullable=False)
    draw_probability = Column(Float, nullable=False)
    away_probability = Column(Float, nullable=False)
    expected_home_goals = Column(Float)
    expected_away_goals = Column(Float)
    p_btts = Column(Float)
    p_over_25 = Column(Float)
    uncertainty = Column(Float)
    confidence = Column(String(20))
    data_quality = Column(Float)
    model_agreement = Column(Float)
    recommendation = Column(String(20))
    recommendation_strength = Column(Float)
    key_factors = Column(JSON)
    risks = Column(JSON)
    actual_result = Column(String(10))
    brier_score = Column(Float)
    log_loss = Column(Float)
    match = relationship("Match", back_populates="predictions")


class FantasyPlayer(Base):
    __tablename__ = "fantasy_players"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), ForeignKey("players.id"), nullable=False)
    platform = Column(String(20), default="fpl")
    season_year = Column(Integer)
    fantasy_id = Column(String(50))
    position = Column(String(5))
    price = Column(Float)
    total_points = Column(Integer, default=0)
    ownership_pct = Column(Float)
    form = Column(Float)
    points_per_game = Column(Float)
    retrieved_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("player_id", "platform", "season_year"),)
    player = relationship("Player", back_populates="fantasy_profiles")
    projections = relationship("FantasyProjection", back_populates="fantasy_player")


class FantasyProjection(Base):
    __tablename__ = "fantasy_projections"
    id = Column(Integer, primary_key=True, autoincrement=True)
    fantasy_player_id = Column(Integer, ForeignKey("fantasy_players.id"), nullable=False)
    match_id = Column(String(50), ForeignKey("matches.id"))
    gameweek = Column(Integer)
    season_year = Column(Integer)
    generated_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50))
    expected_minutes = Column(Float)
    starting_probability = Column(Float)
    expected_goals = Column(Float)
    expected_assists = Column(Float)
    expected_clean_sheet_prob = Column(Float)
    expected_saves = Column(Float)
    expected_bonus = Column(Float)
    expected_fantasy_points = Column(Float)
    ceiling = Column(Float)
    floor = Column(Float)
    risk_score = Column(Float)
    fixture_difficulty = Column(Integer)
    captain_score = Column(Float)
    recommendation = Column(String(20))
    actual_minutes = Column(Integer)
    actual_points = Column(Integer)
    prediction_error = Column(Float)
    fantasy_player = relationship("FantasyPlayer", back_populates="projections")


class Simulation(Base):
    __tablename__ = "simulations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_type = Column(String(50))
    competition_id = Column(String(20), ForeignKey("competitions.id"))
    match_id = Column(String(50), ForeignKey("matches.id"))
    season_year = Column(Integer)
    n_simulations = Column(Integer)
    seed = Column(Integer)
    model_version = Column(String(50))
    generated_at = Column(DateTime, default=datetime.utcnow)
    results = Column(JSON)
    summary = Column(JSON)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    training_date = Column(DateTime)
    n_training_matches = Column(Integer)
    features = Column(JSON)
    hyperparameters = Column(JSON)
    brier_score = Column(Float)
    log_loss = Column(Float)
    ece = Column(Float)
    accuracy = Column(Float)
    is_active = Column(Boolean, default=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("model_name", "version"),)
    runs = relationship("ModelRun", back_populates="model_version_rel")


class ModelRun(Base):
    __tablename__ = "model_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    run_type = Column(String(30))
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    n_records = Column(Integer)
    metrics = Column(JSON)
    error = Column(Text)
    model_version_rel = relationship("ModelVersion", back_populates="runs")


class DataQualityRecord(Base):
    __tablename__ = "data_quality_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(50), ForeignKey("matches.id"))
    generated_at = Column(DateTime, default=datetime.utcnow)
    dq_score = Column(Float)
    completeness = Column(Float)
    freshness = Column(Float)
    source_reliability = Column(Float)
    lineup_confirmed = Column(Boolean)
    sample_size = Column(Integer)
    warnings = Column(JSON)
    errors = Column(JSON)
