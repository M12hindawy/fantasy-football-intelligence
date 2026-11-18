"""
data/elo_store.py
=================
Singleton store for the DynamicEloModel.
Trains on all historical matches in the DB on first use and caches the model.
"""
import logging
from datetime import timezone
from typing import Tuple

from models.elo_model import DynamicEloModel
from db.repository import get_session
from db.models import Match, TeamMatchStats

logger = logging.getLogger(__name__)

class EloStore:
    def __init__(self):
        self.model = DynamicEloModel()
        self._is_trained = False
        
    def _build_records(self) -> list[dict]:
        records = []
        try:
            with get_session() as s:
                matches = s.query(Match).filter(Match.home_score != None).order_by(Match.kickoff_utc).all()
                logger.info(f"Building Elo training records from {len(matches)} historical matches...")
                for m in matches:
                    hs = s.query(TeamMatchStats).filter_by(match_id=m.id, is_home=True).first()
                    aws = s.query(TeamMatchStats).filter_by(match_id=m.id, is_home=False).first()
                    if not hs or not aws: continue
                    
                    kickoff = m.kickoff_utc
                    if kickoff and kickoff.tzinfo is None:
                        kickoff = kickoff.replace(tzinfo=timezone.utc)
                        
                    records.append({
                        "home_id": m.home_team_id,
                        "home_name": m.home_team_id,
                        "away_id": m.away_team_id,
                        "away_name": m.away_team_id,
                        "home_goals": hs.goals or 0,
                        "away_goals": aws.goals or 0,
                        "date": kickoff
                    })
        except Exception as e:
            logger.error(f"Error building Elo records: {e}")
        return records
        
    def _ensure_trained(self):
        if not self._is_trained:
            logger.info("Training DynamicEloModel on historical data...")
            records = self._build_records()
            if records:
                self.model.fit(records)
            self._is_trained = True
            logger.info(f"DynamicEloModel trained on {self.model.n_matches_trained} matches.")
            
    def get_elo(self, team_id: str) -> float:
        self._ensure_trained()
        return self.model.get_rating(team_id)
        
    def get_home_win_prob(self, home_id: str, away_id: str) -> Tuple[float, float, float]:
        self._ensure_trained()
        home_rating = self.model.get_rating(home_id)
        away_rating = self.model.get_rating(away_id)
        return self.model.win_probability(home_rating, away_rating)
        
    def refresh(self):
        self._is_trained = False
        self.model = DynamicEloModel()
        self._ensure_trained()

elo_store = EloStore()
