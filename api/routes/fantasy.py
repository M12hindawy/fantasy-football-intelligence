"""
api/routes/fantasy.py
=====================
FastAPI routes for FPL recommendations.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional

from api.dependencies import get_db
from agents.fantasy_agent import fantasy_agent

router = APIRouter(prefix="/fantasy", tags=["Fantasy"])

@router.get("/recommendation")
def get_recommendation(
    gameweek: Optional[int] = Query(None, description="Target gameweek. Defaults to next GW."),
    db: Session = Depends(get_db)
):
    """
    Generates a full fantasy recommendation squad via the Fantasy Agent.
    """
    # In a full app, we would load records, injuries, and prices from the DB.
    # We pass empty data here so the agent relies on defaults for testing.
    recommendation = fantasy_agent.recommend(
        player_match_records={},
        injuries=[],
        suspensions=[],
        team_clean_sheet_probs={},
        team_mu_goals={},
        team_matches_played={},
        player_positions={},
        gameweek=gameweek
    )
    return recommendation.as_dict()
