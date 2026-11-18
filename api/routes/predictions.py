"""
api/routes/predictions.py
=========================
FastAPI routes for match predictions and analysis.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timezone
import json

from api.dependencies import get_db
from agents.orchestrator import orchestrator, OrchestratorInput

router = APIRouter(prefix="/predictions", tags=["Predictions"])

@router.get("/{fixture_id}")
def get_prediction(
    fixture_id: str,
    home_team_id: str,
    away_team_id: str,
    competition_code: str = "PL",
    db: Session = Depends(get_db)
):
    """
    On-demand analysis for a single fixture using the Master Orchestrator.
    In a fully featured app, match records and odds would be loaded from DB here.
    """
    # Create an orchestrator input 
    inp = OrchestratorInput(
        fixture_id=fixture_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        competition_code=competition_code,
        # Default mock Elo for live testing
        home_elo=1600.0,
        away_elo=1500.0,
        match_records=[]  # In prod, fetch from DB
    )
    
    result = orchestrator.analyze(inp)
    
    # Check for hard pipeline crash
    pipeline_errors = [e for e in result.errors if "Pipeline error" in e]
    if pipeline_errors:
        raise HTTPException(status_code=500, detail={"errors": pipeline_errors})
        
    return result.as_dict()
